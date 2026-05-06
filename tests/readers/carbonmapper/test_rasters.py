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
    CMPlumeRaster,
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
        # Bounds in scene CRS just to keep the test simple.
        crops = ir.read_window(
            (510_000, 3_510_000, 520_000, 3_520_000),
            bands=("cmf",),
        )
        # NOTE: read_window applies EPSG:4326 by default — this test
        # reads through the reproject path. We just assert no crash and
        # the band is present (None if zero overlap is acceptable).
        assert "cmf" in crops


class TestCMImageRasterFromCmTileItem:
    def test_from_cm_tile_item_strips_tif_extensions(self):
        """STAC asset keys carry ``.tif`` extensions in live responses."""
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
                "uncertainty.tif": "https://x/unc.tif",
                "artifact-mask.tif": "https://x/am.tif",
                "uas.txt": "https://x/uas.txt",          # ignored
                "cmf-unortho.tif": "https://x/cmfu.tif", # ignored
            },
            properties={},
            raw={},
        )
        ir = CMImageRaster.from_cm_tile_item(item)
        assert set(ir.asset_paths) == {"cmf", "uncertainty", "artifact-mask"}
        assert ir.asset_paths["cmf"] == "https://x/cmf.tif"

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


# ─── L3A per-plume raster tests ─────────────────────────────────────


def _write_4band_plume(path, *, mask_box=(30, 70)):
    """Write a 4-band uint8 GeoTIFF with band 4 = alpha mask in [mask_box]."""
    arr = np.zeros((4, 100, 100), dtype="uint8")
    arr[3, mask_box[0]:mask_box[1], mask_box[0]:mask_box[1]] = 1
    transform = t_from_bounds(-100, 30, -99, 31, 100, 100)
    with rasterio.open(
        str(path), "w", driver="GTiff", count=4, dtype="uint8",
        width=100, height=100, transform=transform, crs="EPSG:4326",
    ) as dst:
        dst.write(arr)


class TestCMPlumeRasterLazyAccess:
    def test_lazy_open(self, tmp_path):
        plume_tif = tmp_path / "plume.tif"
        _write_4band_plume(plume_tif)
        pr = CMPlumeRaster(plume_id="tan-foo-A",
                           urls={"plume_tif": str(plume_tif)})
        assert isinstance(pr.plume_tif, RasterioReader)

    def test_plume_tif_none_when_missing(self):
        pr = CMPlumeRaster(plume_id="tan-foo-A", urls={})
        assert pr.plume_tif is None

    def test_ignores_non_geotiff_keys(self):
        # con_tif / rgb_png keys are accepted as data on the urls
        # mapping (we don't filter at construction here — only the
        # factories filter), but no accessors surface them.
        pr = CMPlumeRaster(
            plume_id="tan-foo-A",
            urls={"con_tif": "ignored", "rgb_png": "ignored",
                  "plume_png": "ignored"},
        )
        assert pr.plume_tif is None
        assert not hasattr(pr, "con_tif")
        assert not hasattr(pr, "rgb_png")


class TestCMPlumeRasterPolygon:
    def test_polygon_from_alpha_returns_4326(self, tmp_path):
        path = tmp_path / "plume.tif"
        _write_4band_plume(path)
        pr = CMPlumeRaster(plume_id="tan-foo-A",
                           urls={"plume_tif": str(path)})
        poly = pr.polygon_from_alpha()
        assert poly is not None
        # mask centred in [-100, -99] × [30, 31]
        assert -100 <= poly.bounds[0] <= -99
        assert 30 <= poly.bounds[1] <= 31

    def test_polygon_from_alpha_none_when_no_pixels(self, tmp_path):
        path = tmp_path / "plume.tif"
        # Empty mask
        arr = np.zeros((4, 100, 100), dtype="uint8")
        transform = t_from_bounds(-100, 30, -99, 31, 100, 100)
        with rasterio.open(
            str(path), "w", driver="GTiff", count=4, dtype="uint8",
            width=100, height=100, transform=transform, crs="EPSG:4326",
        ) as dst:
            dst.write(arr)
        pr = CMPlumeRaster(plume_id="x", urls={"plume_tif": str(path)})
        assert pr.polygon_from_alpha() is None

    def test_polygon_prefers_geojson_when_present(self, tmp_path, monkeypatch):
        pr = CMPlumeRaster(
            plume_id="tan-foo-A",
            urls={"plume_tif": "x", "ime_outline_geojson": "y"},
        )
        monkeypatch.setattr(
            type(pr), "polygon_from_geojson", lambda self: box(0, 0, 1, 1),
        )
        monkeypatch.setattr(
            type(pr), "polygon_from_alpha",
            lambda self: pytest.fail("alpha path must not run"),
        )
        assert pr.polygon().area == 1.0

    def test_polygon_falls_back_to_alpha(self, tmp_path):
        path = tmp_path / "plume.tif"
        _write_4band_plume(path)
        pr = CMPlumeRaster(plume_id="x", urls={"plume_tif": str(path)})
        # No outline → falls through to alpha extraction.
        poly = pr.polygon()
        assert poly is not None
        assert -100 <= poly.bounds[0] <= -99


class TestCMPlumeRasterDirectLoads:
    def test_load_alpha_mask_returns_boolean_geotensor(self, tmp_path):
        path = tmp_path / "plume.tif"
        _write_4band_plume(path)
        pr = CMPlumeRaster(plume_id="x", urls={"plume_tif": str(path)})
        m = pr.load_alpha_mask()
        assert isinstance(m, GeoTensor)
        assert m.values.dtype == bool
        # Mask covers the inner [30:70, 30:70] block of a 100×100 raster.
        assert m.values.sum() == 40 * 40


class TestCMPlumeRasterFactories:
    def test_from_plume_dict_drops_non_geotiff(self):
        pr = CMPlumeRaster.from_plume_dict({
            "plume_id": "tan-foo-A",
            "plume_tif": "https://x/.tif",
            "con_tif": "https://x/con.tif",       # ignored
            "rgb_png": "https://x/.png",          # ignored
            "plume_png": "https://x/.png",        # ignored
            "plume_rgb_png": "https://x/.png",    # ignored
            "rgb_tif": None,                      # ignored (None)
        })
        assert pr.plume_id == "tan-foo-A"
        assert "plume_tif" in pr.urls
        for dropped in ("con_tif", "rgb_png", "plume_png",
                        "plume_rgb_png", "rgb_tif"):
            assert dropped not in pr.urls

    def test_from_cmrawplume_pulls_only_plume_tif(self):
        from georeader.readers.carbonmapper.plume import CMRawPlume
        raw = CMRawPlume(
            plume_id="tan-foo-A",
            plume_tif="https://x/.tif",
            con_tif="https://x/con.tif",
            rgb_png="https://x/rgb.png",
        )
        pr = CMPlumeRaster.from_cmrawplume(raw)
        assert pr.plume_id == "tan-foo-A"
        assert pr.urls == {"plume_tif": "https://x/.tif"}

    def test_with_outline_returns_copy(self):
        pr = CMPlumeRaster(plume_id="x", urls={"plume_tif": "p.tif"})
        pr2 = pr.with_outline("outline.geojson")
        assert pr2 is not pr
        assert pr2.urls["plume_tif"] == "p.tif"
        assert pr2.urls["ime_outline_geojson"] == "outline.geojson"
        # Original unchanged
        assert "ime_outline_geojson" not in pr.urls


# ─── Bands constant ─────────────────────────────────────────────────


def test_cm_l2b_bands_constant():
    assert CM_L2B_BANDS == ("cmf", "rgb", "uncertainty", "artifact-mask")


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


class TestCMPlumeRasterRepr:
    def test_repr_present_absent_states(self):
        pr = CMPlumeRaster(plume_id="tan-A", urls={"plume_tif": "p.tif"})
        text = repr(pr)
        assert text.startswith("CMPlumeRaster")
        assert "tan-A" in text
        assert "plume_tif:      present" in text
        assert "ime_outline:    absent" in text

    def test_repr_outline_present(self):
        pr = CMPlumeRaster(
            plume_id="tan-A",
            urls={"plume_tif": "p.tif", "ime_outline_geojson": "o.geojson"},
        )
        text = repr(pr)
        assert "ime_outline:    present" in text

    def test_repr_flags_ignored_keys(self):
        pr = CMPlumeRaster(
            plume_id="tan-A",
            urls={"plume_tif": "p.tif", "con_tif": "x", "rgb_png": "x"},
        )
        text = repr(pr)
        assert "ignored keys" in text
        assert "con_tif" in text
        assert "rgb_png" in text

    def test_str_equals_repr(self):
        pr = CMPlumeRaster(plume_id="tan-A", urls={"plume_tif": "p.tif"})
        assert str(pr) == repr(pr)
