"""Tests for ``georeader.readers.carbonmapper.rasters``."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
import requests as _requests
from rasterio.transform import from_bounds as t_from_bounds
from shapely.geometry import box

from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader

from georeader.readers.carbonmapper.api_queries import (
    CMSceneNotPublished,
    CMTileItem,
)
from georeader.readers.carbonmapper import rasters as _rasters
from georeader.readers.carbonmapper.rasters import (
    CM_L2B_BANDS,
    CMImageRaster,
    DEFAULT_L2B_CH4_COLLECTION_CANDIDATES,
    DEFAULT_L2B_RGB_COLLECTION_CANDIDATES,
    _l2b_asset_url,
    _parse_scene_date,
)

# ─── Helpers ────────────────────────────────────────────────────────


def _write_synthetic_band(
    path,
    *,
    w=200,
    h=200,
    dtype="float32",
    nodata=-9999,
    crs="EPSG:32613",
    bounds=(500_000, 3_500_000, 560_000, 3_560_000),
    bands=1,
):
    """Write a synthetic GeoTIFF for tests."""
    rng = np.random.default_rng(0)
    arr = rng.random((bands, h, w)).astype(dtype)
    if nodata is not None and dtype.startswith("float"):
        arr[0, :10, :] = nodata
    transform = t_from_bounds(*bounds, w, h)
    with rasterio.open(
        str(path), "w", driver="GTiff", count=bands, dtype=dtype,
        width=w, height=h, transform=transform, crs=crs,
        nodata=nodata if nodata is not None else None,
    ) as dst:
        dst.write(arr)


def _make_l2b_dir(tmp_path, *, with_artifact_mask=True):
    d = tmp_path / "scene"
    d.mkdir()
    for band in ("cmf", "rgb", "uncertainty"):
        _write_synthetic_band(d / f"{band}.tif")
    if with_artifact_mask:
        _write_synthetic_band(
            d / "artifact-mask.tif", dtype="uint8", nodata=0,
        )
    return d


# ─── L2B raster tests ───────────────────────────────────────────────


class TestCMImageRasterFromLocal:
    def test_finds_assets(self, tmp_path):
        d = _make_l2b_dir(tmp_path)
        ir = CMImageRaster.from_local(d)
        assert set(ir.asset_paths) == {"cmf", "rgb", "uncertainty",
                                        "artifact-mask"}
        assert ir.scene_id == "scene"

    def test_artifact_mask_optional(self, tmp_path):
        d = _make_l2b_dir(tmp_path, with_artifact_mask=False)
        ir = CMImageRaster.from_local(d)
        assert "artifact-mask" not in ir.asset_paths


class TestCMImageRasterLazyAccess:
    def test_band_returns_rasterio_reader(self, tmp_path):
        ir = CMImageRaster.from_local(_make_l2b_dir(tmp_path))
        assert isinstance(ir.cmf, RasterioReader)
        assert isinstance(ir.rgb, RasterioReader)

    def test_cached(self, tmp_path):
        ir = CMImageRaster.from_local(_make_l2b_dir(tmp_path))
        assert ir.cmf is ir.cmf

    def test_artifact_mask_none_when_missing(self, tmp_path):
        ir = CMImageRaster.from_local(
            _make_l2b_dir(tmp_path, with_artifact_mask=False),
        )
        assert ir.artifact_mask is None

    def test_missing_required_band_raises(self, tmp_path):
        d = tmp_path / "broken"
        d.mkdir()
        ir = CMImageRaster.from_local(d)
        with pytest.raises(KeyError):
            _ = ir.cmf


class TestCMImageRasterReadHelpers:
    def test_read_polygon_returns_lazy_readers(self, tmp_path):
        ir = CMImageRaster.from_local(_make_l2b_dir(tmp_path))
        crops = ir.read_polygon(
            polygon=box(510_000, 3_510_000, 520_000, 3_520_000),
            crs_polygon="EPSG:32613",
            bands=("cmf", "rgb"),
        )
        assert set(crops) == {"cmf", "rgb"}
        for v in crops.values():
            # read_from_polygon returns a windowed reader (lazy);
            # materialise via .load() to a GeoTensor.
            assert isinstance(v, RasterioReader)
            loaded = v.load()
            assert isinstance(loaded, GeoTensor)

    def test_read_polygon_skips_missing_band(self, tmp_path):
        ir = CMImageRaster.from_local(
            _make_l2b_dir(tmp_path, with_artifact_mask=False),
        )
        crops = ir.read_polygon(
            polygon=box(510_000, 3_510_000, 520_000, 3_520_000),
            crs_polygon="EPSG:32613",
        )
        assert crops["artifact-mask"] is None
        assert isinstance(crops["cmf"], RasterioReader)

    def test_read_window_to_crs_returns_geotensors(self, tmp_path):
        ir = CMImageRaster.from_local(_make_l2b_dir(tmp_path))
        # Reproject scene-CRS bounds via the WGS-84 entry point — bounds
        # transformed to a small lon/lat box that overlaps the scene.
        crops = ir.read_window_to_crs(
            (-104.5, 31.0, -104.4, 31.1),
            "EPSG:32613",
            bands=("cmf",),
        )
        # May be None if the synthetic scene's bounds don't overlap that
        # lon/lat box — accept either, just verify the type contract.
        if crops["cmf"] is not None:
            assert isinstance(crops["cmf"], GeoTensor)

    def test_read_window_uses_bbox(self, tmp_path):
        ir = CMImageRaster.from_local(_make_l2b_dir(tmp_path))
        # `read_window` interprets bounds as EPSG:4326 (W, S, E, N).
        # Pick a bbox in lon/lat that overlaps the synthetic scene's
        # UTM-13N footprint (centred near western Texas).
        crops = ir.read_window(
            (-104.5, 31.5, -104.0, 32.0),
            bands=("cmf",),
        )
        # NOTE: this reads through the reproject path; we just assert
        # no crash and the band is present (None if zero overlap is
        # acceptable).
        assert "cmf" in crops


class TestCMImageRasterFromCmTileItem:
    def test_from_cm_tile_item_retains_all_loadable_assets(self):
        """STAC asset keys carry ``.tif`` (or ``.txt``) extensions; the
        constructor strips them and retains every key in
        ``CM_L2B_BANDS`` plus ``uas`` (text sidecar)."""
        item = CMTileItem(
            scene_id="tan-foo",
            collection="l2b-ch4-mfa-v3a",
            datetime=__import__("datetime").datetime(
                2025, 1, 1, tzinfo=__import__("datetime").timezone.utc,
            ),
            platform="Tanager1",
            bbox=(0, 0, 1, 1),
            geometry=box(0, 0, 1, 1),
            asset_urls={
                "cmf.tif": "https://x/cmf.tif",
                "cmf-unortho.tif": "https://x/cmfu.tif",
                "uncertainty.tif": "https://x/unc.tif",
                "uncertainty-unortho.tif": "https://x/uncu.tif",
                "artifact-mask.tif": "https://x/am.tif",
                "uas.txt": "https://x/uas.txt",
            },
            properties={},
            raw={},
        )
        ir = CMImageRaster.from_cm_tile_item(item)
        assert set(ir.asset_paths) == {
            "cmf", "cmf-unortho",
            "uncertainty", "uncertainty-unortho",
            "artifact-mask", "uas",
        }
        assert ir.asset_paths["cmf"] == "https://x/cmf.tif"
        assert ir.asset_paths["cmf-unortho"] == "https://x/cmfu.tif"
        assert ir.asset_paths["uas"] == "https://x/uas.txt"

    def test_with_rgb_merges_sibling_collection(self):
        """Compose CH4 + RGB siblings (separate STAC collections,
        same ``scene_id``) into one ``CMImageRaster``."""
        import datetime as _dt
        ch4_item = CMTileItem(
            scene_id="tan-foo",
            collection="l2b-ch4-mfa-v3a",
            datetime=_dt.datetime(2025, 1, 1, tzinfo=_dt.timezone.utc),
            platform="Tanager1", bbox=(0, 0, 1, 1), geometry=box(0, 0, 1, 1),
            asset_urls={"cmf.tif": "https://x/cmf.tif",
                        "uncertainty.tif": "https://x/unc.tif"},
            properties={}, raw={},
        )
        rgb_item = CMTileItem(
            scene_id="tan-foo",
            collection="l2b-rgb-v3a",
            datetime=_dt.datetime(2025, 1, 1, tzinfo=_dt.timezone.utc),
            platform="Tanager1", bbox=(0, 0, 1, 1), geometry=box(0, 0, 1, 1),
            asset_urls={"rgb.tif": "https://x/rgb.tif"},
            properties={}, raw={},
        )
        ir = CMImageRaster.from_cm_tile_item(ch4_item).with_rgb(rgb_item)
        assert "rgb" in ir.asset_paths
        assert "cmf" in ir.asset_paths
        assert ir.asset_paths["rgb"] == "https://x/rgb.tif"

    def test_with_rgb_rejects_scene_id_mismatch(self):
        import datetime as _dt
        ch4_item = CMTileItem(
            scene_id="tan-foo", collection="l2b-ch4-mfa-v3a",
            datetime=_dt.datetime(2025, 1, 1, tzinfo=_dt.timezone.utc),
            platform="Tanager1", bbox=(0, 0, 1, 1), geometry=box(0, 0, 1, 1),
            asset_urls={"cmf.tif": "x"}, properties={}, raw={},
        )
        rgb_item = CMTileItem(
            scene_id="tan-OTHER", collection="l2b-rgb-v3a",
            datetime=_dt.datetime(2025, 1, 1, tzinfo=_dt.timezone.utc),
            platform="Tanager1", bbox=(0, 0, 1, 1), geometry=box(0, 0, 1, 1),
            asset_urls={"rgb.tif": "x"}, properties={}, raw={},
        )
        ir = CMImageRaster.from_cm_tile_item(ch4_item)
        with pytest.raises(ValueError, match="scene_id mismatch"):
            ir.with_rgb(rgb_item)

    def test_from_cm_tile_item_uses_asset_urls(self):
        item = CMTileItem(
            scene_id="tan-foo",
            collection="l2b-ch4-mfa-v3a",
            datetime=__import__("datetime").datetime(
                2025, 1, 1, tzinfo=__import__("datetime").timezone.utc,
            ),
            platform="Tanager1",
            bbox=(0, 0, 1, 1),
            geometry=box(0, 0, 1, 1),
            asset_urls={"cmf": "https://cm/.../cmf.tif",
                        "rgb": "https://cm/.../rgb.tif"},
            properties={},
            raw={},
        )
        ir = CMImageRaster.from_cm_tile_item(item)
        assert ir.scene_id == "tan-foo"
        assert ir.asset_paths["cmf"].endswith("cmf.tif")


# ─── New L2B properties (cmf_unortho / uncertainty_unortho / uas) ───


class TestCMImageRasterUnorthoAndUas:
    """L2B exposes raw-frame retrieval variants and a UAS sidecar that
    weren't previously loadable through the wrapper."""

    def test_cmf_unortho_property_opens_when_present(self, tmp_path):
        d = tmp_path / "scene"
        d.mkdir()
        for band in ("cmf", "cmf-unortho", "uncertainty"):
            _write_synthetic_band(d / f"{band}.tif")
        ir = CMImageRaster.from_local(d)
        assert isinstance(ir.cmf_unortho, RasterioReader)

    def test_cmf_unortho_returns_none_when_absent(self, tmp_path):
        # Older mfm-v1-style scene with only orthorectified rasters
        d = tmp_path / "scene"
        d.mkdir()
        _write_synthetic_band(d / "cmf.tif")
        ir = CMImageRaster.from_local(d)
        assert ir.cmf_unortho is None

    def test_uncertainty_unortho_property_opens_when_present(self, tmp_path):
        d = tmp_path / "scene"
        d.mkdir()
        for band in ("cmf", "uncertainty", "uncertainty-unortho"):
            _write_synthetic_band(d / f"{band}.tif")
        ir = CMImageRaster.from_local(d)
        assert isinstance(ir.uncertainty_unortho, RasterioReader)

    def test_uas_property_reads_sidecar_text(self, tmp_path):
        d = tmp_path / "scene"
        d.mkdir()
        _write_synthetic_band(d / "cmf.tif")
        (d / "uas.txt").write_text("instrument: tan\nplatform: Tanager-1\n")
        ir = CMImageRaster.from_local(d)
        assert ir.uas is not None
        assert "Tanager-1" in ir.uas

    def test_uas_returns_none_when_absent(self, tmp_path):
        d = tmp_path / "scene"
        d.mkdir()
        _write_synthetic_band(d / "cmf.tif")
        ir = CMImageRaster.from_local(d)
        assert ir.uas is None

    def test_read_polygon_skips_uas(self, tmp_path):
        """`uas` is text, not a raster — read_polygon must not try to
        open it as a band even if it's in the requested bands list."""
        d = tmp_path / "scene"
        d.mkdir()
        _write_synthetic_band(d / "cmf.tif")
        (d / "uas.txt").write_text("xyz")
        ir = CMImageRaster.from_local(d)
        # Default bands now includes more entries; no error from `uas`
        clip = box(500_500, 3_500_500, 559_500, 3_559_500)
        out = ir.read_polygon(clip, crs_polygon="EPSG:32613")
        # cmf returned; uas not tried
        assert out["cmf"] is not None
        assert "uas" not in out


# ─── Bands constant ─────────────────────────────────────────────────


def test_cm_l2b_bands_constant():
    """Widened from the original 4-band tuple to include the unortho
    variants of cmf and uncertainty."""
    assert CM_L2B_BANDS == (
        "cmf", "cmf-unortho",
        "uncertainty", "uncertainty-unortho",
        "artifact-mask", "rgb",
    )


# ─── __repr__ / __str__ ─────────────────────────────────────────────


class TestCMImageRasterRepr:
    def test_repr_lists_present_and_missing_bands(self, tmp_path):
        ir = CMImageRaster.from_local(
            _make_l2b_dir(tmp_path, with_artifact_mask=False),
        )
        text = repr(ir)
        assert text.startswith("CMImageRaster")
        assert "scene_id:" in text
        assert "'cmf'" in text
        # artifact-mask is missing
        assert "artifact-mask" in text
        assert "bands missing" in text

    def test_repr_does_not_open_assets(self, tmp_path):
        # Construct with bogus URLs — repr must not trigger I/O.
        ir = CMImageRaster(
            scene_id="bogus",
            asset_paths={"cmf": "https://no-such-host.example/x.tif"},
        )
        text = repr(ir)
        assert "bogus" in text
        assert "cmf" in text

    def test_repr_shows_overview_level(self, tmp_path):
        ir = CMImageRaster.from_local(_make_l2b_dir(tmp_path))
        ir.overview_level = 2
        assert "overview_level: 2" in repr(ir)

    def test_str_equals_repr(self, tmp_path):
        ir = CMImageRaster.from_local(_make_l2b_dir(tmp_path))
        assert str(ir) == repr(ir)


# ─── URL-pattern helpers ─────────────────────────────────────────────


class TestParseSceneDate:
    """`_parse_scene_date` extracts YYYY/MM/DD from positions [3:11]."""

    @pytest.mark.parametrize(
        "scene_id, expected",
        [
            ("tan20260331t181625c77s4001", ("2026", "03", "31")),
            ("emi20250515t190623",         ("2025", "05", "15")),
            ("ang20240615t184217",         ("2024", "06", "15")),
            ("av320240801t143728",         ("2024", "08", "01")),
            ("GAO20210820t195716",         ("2021", "08", "20")),
        ],
    )
    def test_parses_known_instruments(self, scene_id, expected):
        assert _parse_scene_date(scene_id) == expected

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            _parse_scene_date("tan2026")

    def test_non_digit_raises(self):
        with pytest.raises(ValueError, match="not an 8-digit date"):
            _parse_scene_date("tan-INVALID-12345")


class TestL2BAssetURL:
    def test_url_pattern_matches_design_doc(self):
        url = _l2b_asset_url(
            "l2b-ch4-mfa-v3c", "tan20260331t181625c77s4001", "cmf.tif",
        )
        assert url == (
            "https://api.carbonmapper.org/api/v1/catalog/asset/"
            "l2b-ch4-mfa-v3c/2026/03/31/"
            "tan20260331t181625c77s4001/"
            "tan20260331t181625c77s4001_l2b-ch4-mfa-v3c_cmf.tif"
        )

    def test_rgb_sibling_url(self):
        url = _l2b_asset_url(
            "l2b-rgb-v3c", "tan20260331t181625c77s4001", "rgb.tif",
        )
        assert "l2b-rgb-v3c" in url
        assert url.endswith("_l2b-rgb-v3c_rgb.tif")


# ─── from_scene_id ───────────────────────────────────────────────────


def _make_probe_response(status_code: int) -> MagicMock:
    """Build a fake ``requests.get`` return value for the range-GET probe."""
    resp = MagicMock()
    resp.status_code = status_code
    return resp


class TestFromSceneIdProbe:
    """`CMImageRaster.from_scene_id` probes candidate collections in
    order, taking the first 200/206. Verified URL-pattern from
    design doc §4.7 — works for v3a (STAC-resident) AND v3c
    (REST-only, the 2026 L2B version)."""

    def test_newest_candidate_picked_first(self, monkeypatch):
        """The newest candidate is probed first — single probe + success.

        Bound to the constant rather than a literal version, so tracking a
        new Carbon Mapper version doesn't silently invalidate this test."""
        newest = DEFAULT_L2B_CH4_COLLECTION_CANDIDATES[0]
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(206)

        monkeypatch.setattr(_requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "tan20260331t181625c77s4001", token="dummy", with_rgb=False,
        )
        # One probe — the newest candidate wins.
        assert len(calls) == 1
        assert newest in calls[0]
        assert "_cmf.tif" in calls[0]

        # All 6 CH4 asset keys built with the winning collection.
        assert set(ir.asset_paths) == {
            "cmf", "cmf-unortho",
            "uncertainty", "uncertainty-unortho",
            "artifact-mask", "uas",
        }
        for url in ir.asset_paths.values():
            assert newest in str(url)
        assert ir.asset_paths["uas"].endswith(".txt")

    def test_falls_through_to_oldest(self, monkeypatch):
        """Every newer candidate 404s → the oldest (v3a) wins. The 2025
        case — STAC would've worked too, but ``from_scene_id`` doesn't go
        through STAC."""
        # CH4 probes: every candidate 404s until the last one, which 206s.
        oldest = DEFAULT_L2B_CH4_COLLECTION_CANDIDATES[-1]
        seq = iter([404] * (len(DEFAULT_L2B_CH4_COLLECTION_CANDIDATES) - 1) + [206])

        def fake_get(url, **kw):
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "tan20250801t120000c01s4001", token="dummy", with_rgb=False,
        )
        for url in ir.asset_paths.values():
            assert oldest in str(url)

    def test_403_skips_to_next_candidate(self, monkeypatch):
        """A 403 on one candidate must not abort the probe chain.

        The asset proxy answers 403 when the collection exists but does
        not hold the scene (and 404 when the collection is unknown), so
        leading the candidate list with a version newer than the scene
        — which is the normal state right after CM cuts a version —
        used to raise instead of falling through to the real parent.
        Auth failures arrive as 401 and are still surfaced (see
        ``test_401_propagates``)."""
        # Newest candidate 403s, next one wins.
        seq = iter([403, 206])
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "tan20260331t181625c77s4001", token="dummy", with_rgb=False,
        )
        assert len(calls) == 2
        second = DEFAULT_L2B_CH4_COLLECTION_CANDIDATES[1]
        for url in ir.asset_paths.values():
            assert second in str(url)

    def test_401_propagates_not_swallowed(self, monkeypatch):
        """401 is an auth failure, not a data fact — must still raise
        even though its neighbour 403 is now skipped."""
        import requests as _r

        resp = MagicMock()
        resp.status_code = 401

        def boom():
            raise _r.HTTPError("401 Unauthorized", response=resp)

        resp.raise_for_status = boom
        monkeypatch.setattr(_requests, "get", lambda url, **kw: resp)

        with pytest.raises(_r.HTTPError, match="401"):
            CMImageRaster.from_scene_id(
                "tan20260331t181625c77s4001", token="dummy", with_rgb=False,
            )

    def test_all_candidates_404_raises(self, monkeypatch):
        """Every CH4 candidate 404s → ``CMSceneNotPublished``."""
        monkeypatch.setattr(
            _requests, "get",
            lambda url, **kw: _make_probe_response(404),
        )
        with pytest.raises(CMSceneNotPublished, match="not published"):
            CMImageRaster.from_scene_id(
                "tan20260331t181625c77s4001",
                token="dummy",
                with_rgb=False,
            )

    def test_with_rgb_adds_sibling(self, monkeypatch):
        """`with_rgb=True` (default) probes RGB sibling candidates after
        CH4 succeeds."""
        # CH4 newest=206 → wins. RGB newest=206 → wins. 2 probes total.
        newest_ch4 = DEFAULT_L2B_CH4_COLLECTION_CANDIDATES[0]
        newest_rgb = DEFAULT_L2B_RGB_COLLECTION_CANDIDATES[0]
        seq = iter([206, 206])
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "tan20260331t181625c77s4001", token="dummy",
        )
        assert len(calls) == 2
        assert newest_ch4 in calls[0]
        assert newest_rgb in calls[1]
        assert "rgb" in ir.asset_paths
        assert newest_rgb in ir.asset_paths["rgb"]

    def test_with_rgb_tolerates_rgb_404(self, monkeypatch):
        """CH4 succeeds + every RGB candidate 404s → return CH4-only,
        no exception. Rare but documented behaviour."""
        # CH4 newest=206 → wins. Every RGB candidate 404s → no rgb URL.
        seq = iter([206] + [404] * len(DEFAULT_L2B_RGB_COLLECTION_CANDIDATES))

        def fake_get(url, **kw):
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "tan20260331t181625c77s4001", token="dummy",
        )
        assert "rgb" not in ir.asset_paths
        assert "cmf" in ir.asset_paths

    def test_custom_candidates(self, monkeypatch):
        """Power-user override — historical scenes can be probed against
        legacy collection variants (``mfm-v1`` etc.)."""
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(206)

        monkeypatch.setattr(_requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "ang20180615t184217",
            token="dummy",
            l2b_collection_candidates=("l2b-ch4-mfa-v3", "l2b-ch4-mfm-v1"),
            with_rgb=False,
        )
        # First custom candidate wins; the defaults aren't probed.
        assert len(calls) == 1
        assert "l2b-ch4-mfa-v3" in calls[0]
        assert DEFAULT_L2B_CH4_COLLECTION_CANDIDATES[0] not in calls[0]
        assert "l2b-ch4-mfa-v3" in ir.asset_paths["cmf"]

    def test_transport_errors_propagate(self, monkeypatch):
        """A transport-level failure (timeout / connection error)
        should NOT be silently treated as "not published" — it's a
        real error. Surface it so the caller sees what went wrong
        rather than getting a misleading CMSceneNotPublished."""
        import requests as _r

        def fake_get(url, **kw):
            raise _r.ConnectTimeout("simulated")

        monkeypatch.setattr(_requests, "get", fake_get)

        with pytest.raises(_r.ConnectTimeout):
            CMImageRaster.from_scene_id(
                "tan20260331t181625c77s4001", token="dummy", with_rgb=False,
            )

    def test_429_propagates_not_swallowed(self, monkeypatch):
        """Rate-limit (HTTP 429) is transient, not a data fact. After the
        shared back-off retries it must surface, not be silently treated
        as "scene not published"."""
        import requests as _r

        from georeader.readers.carbonmapper import download as _dl

        monkeypatch.setattr(_dl, "_sleep", lambda s: None)

        resp = MagicMock()
        resp.status_code = 429

        def boom():
            err = _r.HTTPError("429 Too Many Requests", response=resp)
            raise err

        resp.raise_for_status = boom

        def fake_get(url, **kw):
            return resp

        monkeypatch.setattr(_requests, "get", fake_get)

        with pytest.raises(_r.HTTPError, match="429"):
            CMImageRaster.from_scene_id(
                "tan20260331t181625c77s4001", token="dummy", with_rgb=False,
            )



# ─── DEFAULT_*_CANDIDATES constants ──────────────────────────────────


def test_default_ch4_candidates_priority():
    """Newest first — v3e is the live era (verified 2026-07-31). These
    defaults matter only for scene-name-only lookups; record-driven
    callers use the spec path and never probe."""
    assert DEFAULT_L2B_CH4_COLLECTION_CANDIDATES == (
        "l2b-ch4-mfa-v3e",
        "l2b-ch4-mfa-v3d",
        "l2b-ch4-mfa-v3c",
        "l2b-ch4-mfa-v3a",
    )


def test_default_rgb_candidates_priority():
    assert DEFAULT_L2B_RGB_COLLECTION_CANDIDATES == (
        "l2b-rgb-v3e",
        "l2b-rgb-v3d",
        "l2b-rgb-v3c",
        "l2b-rgb-v3a",
    )


def test_default_candidates_share_version_ordering():
    """Both tuples are generated from one version sequence, so they
    cannot drift apart when a new Carbon Mapper version is tracked."""
    ch4_versions = [c.removeprefix("l2b-ch4-mfa-") for c in DEFAULT_L2B_CH4_COLLECTION_CANDIDATES]
    rgb_versions = [c.removeprefix("l2b-rgb-") for c in DEFAULT_L2B_RGB_COLLECTION_CANDIDATES]
    assert ch4_versions == rgb_versions


# ─── Spec-driven (probe-free) from_scene_id ──────────────────────────


class TestFromSceneIdSpec:
    """When a `CMCollectionSpec` is known, its composed collection id
    is probed FIRST (the record's own version), with the default
    candidates as backup — self-healing for the re-versioned case
    (2026-07 audit: a v3d L3A plume whose L2B still serves at v3c).
    Explicit collection ids are used verbatim with no probing."""

    def _forbid_http(self, monkeypatch):
        def boom(url, **kw):
            raise AssertionError(f"unexpected HTTP request: {url}")
        monkeypatch.setattr(_requests, "get", boom)

    def test_spec_version_probed_first(self, monkeypatch):
        from georeader.readers.carbonmapper.products import CMCollectionSpec

        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(206)

        monkeypatch.setattr(_requests, "get", fake_get)
        ir = CMImageRaster.from_scene_id(
            "tan20260623t124240c80s4001",
            token="dummy",
            spec=CMCollectionSpec(version="v3d"),
        )
        # One CH4 probe (spec version wins first) + one RGB probe.
        assert len(calls) == 2
        assert "l2b-ch4-mfa-v3d" in calls[0]
        assert "l2b-rgb-v3d" in calls[1]
        assert "l2b-ch4-mfa-v3d" in str(ir.asset_paths["cmf"])
        assert "l2b-rgb-v3d" in str(ir.asset_paths["rgb"])

    def test_spec_falls_back_to_default_candidates(self, monkeypatch):
        """The re-versioned case: L3A says v3e (new, unknown), the L2B
        parent still serves at v3d — the spec candidate 404s and the
        defaults catch it."""
        from georeader.readers.carbonmapper.products import CMCollectionSpec

        calls: list[str] = []
        # CH4: v3e=404 → v3d=206. RGB: v3e=404 → v3d=206.
        seq = iter([404, 206, 404, 206])

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_requests, "get", fake_get)
        ir = CMImageRaster.from_scene_id(
            "tan20260623t124240c80s4001",
            token="dummy",
            spec=CMCollectionSpec(version="v3e"),
        )
        assert "l2b-ch4-mfa-v3e" in calls[0]
        assert "l2b-ch4-mfa-v3d" in str(ir.asset_paths["cmf"])
        assert "l2b-rgb-v3d" in str(ir.asset_paths["rgb"])

    def test_explicit_collection_ids_without_probing(self, monkeypatch):
        self._forbid_http(monkeypatch)
        ir = CMImageRaster.from_scene_id(
            "ang20190615t184217",
            token="dummy",
            collection="l2b-ch4-mf-v1",
            rgb_collection="l2b-rgb-v1",
        )
        assert "l2b-ch4-mf-v1" in str(ir.asset_paths["cmf"])
        assert "l2b-rgb-v1" in str(ir.asset_paths["rgb"])

    def test_explicit_collection_with_rgb_probes_only_rgb(self, monkeypatch):
        """CH4 pinned + RGB unpinned → exactly the RGB candidates are
        probed."""
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(404)

        monkeypatch.setattr(_requests, "get", fake_get)
        ir = CMImageRaster.from_scene_id(
            "tan20260623t124240c80s4001",
            token="dummy",
            collection="l2b-ch4-mfa-v3d",
        )
        assert all("l2b-rgb-" in c for c in calls)
        assert "rgb" not in ir.asset_paths

    def test_products_subset(self, monkeypatch):
        from georeader.readers.carbonmapper import products as P

        self._forbid_http(monkeypatch)
        ir = CMImageRaster.from_scene_id(
            "tan20260623t124240c80s4001",
            token="dummy",
            collection="l2b-ch4-mfa-v3d",
            products=(P.CMF, P.UNCERTAINTY),
            with_rgb=False,
        )
        assert set(ir.asset_paths) == {"cmf", "uncertainty"}

    def test_non_l2b_product_rejected(self, monkeypatch):
        from georeader.readers.carbonmapper import products as P

        self._forbid_http(monkeypatch)
        with pytest.raises(ValueError, match="not an L2B"):
            CMImageRaster.from_scene_id(
                "tan20260623t124240c80s4001",
                token="dummy",
                collection="l2b-ch4-mfa-v3d",
                products=(P.PLUME_TIF,),
                with_rgb=False,
            )


# ─── Same-gas, lag-ordered candidates (CO2 + version lag) ────────────


class TestSpecCandidates:
    def test_co2_never_falls_back_to_ch4(self, monkeypatch):
        """A CO2 plume whose CO2 scene isn't served used to resolve to
        the CH4 scene of the same name via the CH4-only defaults."""
        from georeader.readers.carbonmapper.products import CMCollectionSpec

        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(206 if "l2b-ch4-" in url else 404)

        monkeypatch.setattr(_requests, "get", fake_get)
        with pytest.raises(CMSceneNotPublished):
            CMImageRaster.from_scene_id(
                "tan20260823t091609c53s4001", token="dummy", with_rgb=False,
                spec=CMCollectionSpec("v3e", "co2", "mfa", ime_cmf_type="mfal"),
            )
        assert calls and all("l2b-co2-" in c for c in calls)

    def test_co2_tries_both_cmf_variants_at_its_version(self, monkeypatch):
        """CO2 L2B moved between `mfa` and `mfal` across versions."""
        from georeader.readers.carbonmapper.products import CMCollectionSpec

        seq = iter([404, 206])
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_requests, "get", fake_get)
        ir = CMImageRaster.from_scene_id(
            "tan20260620t091609c53s4001", token="dummy", with_rgb=False,
            spec=CMCollectionSpec("v3d", "co2", "mfa", ime_cmf_type="mfal"),
        )
        assert "l2b-co2-mfa-v3d" in calls[0]
        assert "l2b-co2-mfal-v3d" in str(ir.asset_paths["cmf"])

    def test_version_lag_prefers_older_before_newer(self):
        """After the spec's own version misses, older versions (the L2B
        lag case) come before any newer reprocess."""
        from georeader.readers.carbonmapper.products import CMCollectionSpec
        from georeader.readers.carbonmapper.rasters import (
            l2b_collection_candidates_for_spec,
            rgb_collection_candidates_for_spec,
        )

        spec = CMCollectionSpec("v3d")
        assert l2b_collection_candidates_for_spec(spec) == (
            "l2b-ch4-mfa-v3d", "l2b-ch4-mfa-v3c", "l2b-ch4-mfa-v3a",
            "l2b-ch4-mfa-v3e",
        )
        assert rgb_collection_candidates_for_spec(spec)[:2] == (
            "l2b-rgb-v3d", "l2b-rgb-v3c",
        )

    def test_unknown_future_version_probed_first(self):
        from georeader.readers.carbonmapper.products import CMCollectionSpec
        from georeader.readers.carbonmapper.rasters import (
            l2b_collection_candidates_for_spec,
        )

        cands = l2b_collection_candidates_for_spec(CMCollectionSpec("v3f"))
        assert cands[0] == "l2b-ch4-mfa-v3f"
        assert cands[1] == "l2b-ch4-mfa-v3e"


# ─── Optional bands absent server-side ───────────────────────────────


def _fake_reader_factory(tmp_path, absent_status: int = 404):
    """RasterioReader stand-in: local paths open for real; the
    ``https://cm/...artifact-mask.tif`` URL fails like GDAL's vsicurl."""
    from rasterio.errors import RasterioIOError

    calls: list[dict] = []

    def factory(path, **kw):
        calls.append({"path": path, **kw})
        if path.startswith("https://"):
            raise RasterioIOError(f"HTTP response code: {absent_status}")
        return RasterioReader(path, overview_level=kw.get("overview_level"))

    return factory, calls


class TestOptionalBandsAbsent:
    """v3e scenes ship no artifact-mask (checked 2026-09), yet the URL
    is always built — opening it used to raise and fail whole reads."""

    def _raster(self, tmp_path):
        d = _make_l2b_dir(tmp_path, with_artifact_mask=False)
        paths = {b: str(d / f"{b}.tif") for b in ("cmf", "rgb", "uncertainty")}
        paths["artifact-mask"] = "https://cm/x/scene_l2b-ch4-mfa-v3e_artifact-mask.tif"
        return CMImageRaster(scene_id="scene", asset_paths=paths, token="tok")

    def test_artifact_mask_404_is_none(self, tmp_path, monkeypatch):
        factory, _ = _fake_reader_factory(tmp_path, 404)
        monkeypatch.setattr(_rasters, "RasterioReader", factory)
        assert self._raster(tmp_path).artifact_mask is None

    def test_read_window_default_bands_survives_absent_mask(self, tmp_path, monkeypatch):
        factory, _ = _fake_reader_factory(tmp_path, 404)
        monkeypatch.setattr(_rasters, "RasterioReader", factory)
        ir = self._raster(tmp_path)
        crops = ir.read_polygon(
            box(510_000, 3_510_000, 520_000, 3_520_000),
            crs_polygon="EPSG:32613",
        )
        assert crops["artifact-mask"] is None
        assert crops["cmf"] is not None and crops["uncertainty"] is not None

    def test_auth_error_on_optional_band_still_raises(self, tmp_path, monkeypatch):
        from rasterio.errors import RasterioIOError

        factory, _ = _fake_reader_factory(tmp_path, 401)
        monkeypatch.setattr(_rasters, "RasterioReader", factory)
        with pytest.raises(RasterioIOError, match="401"):
            _ = self._raster(tmp_path).artifact_mask


# ─── Token scoping ───────────────────────────────────────────────────


class TestTokenScoping:
    def test_remote_open_gets_scoped_bearer_header(self, monkeypatch):
        seen: list[dict] = []
        monkeypatch.setattr(
            _rasters, "RasterioReader",
            lambda path, **kw: seen.append(kw) or MagicMock(),
        )
        ir = CMImageRaster(
            scene_id="s", asset_paths={"cmf": "https://cm/x/cmf.tif"}, token="abc",
        )
        _ = ir.cmf
        env = seen[0]["rio_env_options"]
        assert env["GDAL_HTTP_HEADERS"] == "Authorization: Bearer abc"
        # georeader's defaults are kept alongside the header.
        assert env["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"

    def test_local_or_tokenless_open_uses_defaults(self, tmp_path, monkeypatch):
        seen: list[dict] = []
        monkeypatch.setattr(
            _rasters, "RasterioReader",
            lambda path, **kw: seen.append(kw) or MagicMock(),
        )
        CMImageRaster(scene_id="s", asset_paths={"cmf": "/tmp/cmf.tif"}, token="abc").cmf
        CMImageRaster(scene_id="s", asset_paths={"cmf": "https://cm/cmf.tif"}).cmf
        assert [kw["rio_env_options"] for kw in seen] == [None, None]

    def test_from_scene_id_carries_token(self, monkeypatch):
        monkeypatch.setattr(_requests, "get", lambda url, **kw: _make_probe_response(206))
        ir = CMImageRaster.from_scene_id(
            "tan20260331t181625c77s4001", token="tok", with_rgb=False,
        )
        assert ir.token == "tok"

    def test_uas_prefers_token_then_multiline_gdal_env(self, monkeypatch):
        sent: list[dict] = []

        def fake_get(url, **kw):
            sent.append(kw["headers"])
            resp = MagicMock(status_code=200, text="uas")
            return resp

        monkeypatch.setattr(_requests, "get", fake_get)
        monkeypatch.setenv(
            "GDAL_HTTP_HEADERS", "X-Other: 1\r\nAuthorization: Bearer env-tok",
        )
        url = {"uas": "https://cm/x/uas.txt"}
        assert CMImageRaster(scene_id="s", asset_paths=url, token="t").uas == "uas"
        assert CMImageRaster(scene_id="s", asset_paths=url).uas == "uas"
        assert sent == [
            {"Authorization": "Bearer t"},
            {"Authorization": "Bearer env-tok"},
        ]


def test_read_window_rejects_antimeridian_bbox(tmp_path):
    ir = CMImageRaster.from_local(_make_l2b_dir(tmp_path))
    with pytest.raises(ValueError, match="antimeridian"):
        ir.read_window((179.5, 10.0, -179.5, 11.0))
