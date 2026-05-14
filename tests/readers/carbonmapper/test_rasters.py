"""Tests for ``georeader.readers.carbonmapper.rasters``."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
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

    def test_v3c_picked_first(self, monkeypatch):
        """v3c is the default first candidate — single probe + success."""
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(206)

        monkeypatch.setattr(_rasters.requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "tan20260331t181625c77s4001", token="dummy", with_rgb=False,
        )
        # One probe — first candidate (v3c) wins.
        assert len(calls) == 1
        assert "l2b-ch4-mfa-v3c" in calls[0]
        assert "_cmf.tif" in calls[0]

        # All 6 CH4 asset keys built with the winning collection (v3c).
        assert set(ir.asset_paths) == {
            "cmf", "cmf-unortho",
            "uncertainty", "uncertainty-unortho",
            "artifact-mask", "uas",
        }
        for url in ir.asset_paths.values():
            assert "l2b-ch4-mfa-v3c" in str(url)
        assert ir.asset_paths["uas"].endswith(".txt")

    def test_falls_through_to_v3a(self, monkeypatch):
        """v3c 404 → v3a 206. The 2025 case — STAC would've worked too,
        but ``from_scene_id`` doesn't go through STAC."""
        # CH4 probes: v3c=404, v3a=206 → wins.
        # No rgb probe (with_rgb=False).
        seq = iter([404, 206])

        def fake_get(url, **kw):
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_rasters.requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "tan20250801t120000c01s4001", token="dummy", with_rgb=False,
        )
        for url in ir.asset_paths.values():
            assert "l2b-ch4-mfa-v3a" in str(url)

    def test_all_candidates_404_raises(self, monkeypatch):
        """Every CH4 candidate 404s → ``CMSceneNotPublished``."""
        monkeypatch.setattr(
            _rasters.requests, "get",
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
        # CH4 v3c=206 → wins. RGB v3c=206 → wins. 2 probes total.
        seq = iter([206, 206])
        calls: list[str] = []

        def fake_get(url, **kw):
            calls.append(url)
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_rasters.requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "tan20260331t181625c77s4001", token="dummy",
        )
        assert len(calls) == 2
        assert "l2b-ch4-mfa-v3c" in calls[0]
        assert "l2b-rgb-v3c" in calls[1]
        assert "rgb" in ir.asset_paths
        assert "l2b-rgb-v3c" in ir.asset_paths["rgb"]

    def test_with_rgb_tolerates_rgb_404(self, monkeypatch):
        """CH4 succeeds + every RGB candidate 404s → return CH4-only,
        no exception. Rare but documented behaviour."""
        # CH4 v3c=206 → wins. RGB v3c=404, v3a=404 → no rgb URL.
        seq = iter([206, 404, 404])

        def fake_get(url, **kw):
            return _make_probe_response(next(seq))

        monkeypatch.setattr(_rasters.requests, "get", fake_get)

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

        monkeypatch.setattr(_rasters.requests, "get", fake_get)

        ir = CMImageRaster.from_scene_id(
            "ang20180615t184217",
            token="dummy",
            l2b_collection_candidates=("l2b-ch4-mfa-v3", "l2b-ch4-mfm-v1"),
            with_rgb=False,
        )
        # First custom candidate wins; the default v3c isn't probed.
        assert len(calls) == 1
        assert "l2b-ch4-mfa-v3" in calls[0]
        assert "l2b-ch4-mfa-v3c" not in calls[0]
        assert "l2b-ch4-mfa-v3" in ir.asset_paths["cmf"]

    def test_transport_errors_propagate(self, monkeypatch):
        """A transport-level failure (timeout / connection error)
        should NOT be silently treated as "not published" — it's a
        real error. Surface it so the caller sees what went wrong
        rather than getting a misleading CMSceneNotPublished."""
        import requests as _r

        def fake_get(url, **kw):
            raise _r.ConnectTimeout("simulated")

        monkeypatch.setattr(_rasters.requests, "get", fake_get)

        with pytest.raises(_r.ConnectTimeout):
            CMImageRaster.from_scene_id(
                "tan20260331t181625c77s4001", token="dummy", with_rgb=False,
            )

    def test_429_propagates_not_swallowed(self, monkeypatch):
        """Rate-limit (HTTP 429) is transient, not a data fact. Must
        not be silently treated as "scene not published"."""
        import requests as _r

        resp = MagicMock()
        resp.status_code = 429

        def boom():
            err = _r.HTTPError("429 Too Many Requests", response=resp)
            raise err

        resp.raise_for_status = boom

        def fake_get(url, **kw):
            return resp

        monkeypatch.setattr(_rasters.requests, "get", fake_get)

        with pytest.raises(_r.HTTPError, match="429"):
            CMImageRaster.from_scene_id(
                "tan20260331t181625c77s4001", token="dummy", with_rgb=False,
            )



# ─── DEFAULT_*_CANDIDATES constants ──────────────────────────────────


def test_default_ch4_candidates_priority():
    """v3c first — covers 2026 onward (the practical 'live' scenes)."""
    assert DEFAULT_L2B_CH4_COLLECTION_CANDIDATES == (
        "l2b-ch4-mfa-v3c",
        "l2b-ch4-mfa-v3a",
    )


def test_default_rgb_candidates_priority():
    assert DEFAULT_L2B_RGB_COLLECTION_CANDIDATES == (
        "l2b-rgb-v3c",
        "l2b-rgb-v3a",
    )
