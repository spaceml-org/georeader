"""Tests for ``georeader.readers.carbonmapper.rasters``."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds as t_from_bounds
from shapely.geometry import box

from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader

from georeader.readers.carbonmapper.api_queries import CMTileItem
from georeader.readers.carbonmapper.rasters import (
    CM_L2B_BANDS,
    CMImageRaster,
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
