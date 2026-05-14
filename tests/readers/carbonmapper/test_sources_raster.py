"""Tests for ``georeader.readers.carbonmapper.sources_raster``."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from shapely.geometry import Point

from georeader.readers.carbonmapper.source import CMSource
from georeader.readers.carbonmapper.sources_raster import (
    CMSourceRaster,
    rasterize_sources,
)


# ─── Helpers ──────────────────────────────────────────────────────────


def _utm_grid(
    *,
    origin_xy: tuple[float, float] = (500_000.0, 4_000_000.0),
    res: float = 30.0,
    shape: tuple[int, int] = (100, 100),
    crs: str = "EPSG:32613",  # UTM 13N — Permian basin
) -> dict:
    """A small UTM grid spec for tests."""
    transform = rasterio.Affine(res, 0, origin_xy[0], 0, -res, origin_xy[1])
    return {"transform": transform, "shape": shape, "crs": crs}


def _make_source(lon: float, lat: float, name: str = "S") -> CMSource:
    return CMSource(
        source_name=name,
        gas="CH4",
        sector="1B2",
        point=Point(lon, lat),
        plume_count=1,
        persistence=0.5,
    )


def _grid_lonlat_at(grid: dict, row: int, col: int) -> tuple[float, float]:
    """Lon/Lat of pixel-centre at (row, col) on a UTM grid."""
    t = grid["transform"]
    x = t.c + (col + 0.5) * t.a
    y = t.f + (row + 0.5) * t.e
    from pyproj import Transformer
    tr = Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return float(lon), float(lat)


# ─── rasterize_sources — happy paths ─────────────────────────────────


def test_empty_sources_returns_zero_mask():
    g = _utm_grid()
    out = rasterize_sources([], **g)
    assert out.shape == g["shape"]
    assert out.dtype == np.uint8
    assert int(out.sum()) == 0


def test_single_source_no_buffer_stamps_one_pixel():
    g = _utm_grid()
    lon, lat = _grid_lonlat_at(g, row=42, col=17)
    out = rasterize_sources([_make_source(lon, lat)], **g)
    assert int(out.sum()) == 1
    assert int(np.asarray(out)[42, 17]) == 1


def test_multiple_sources_stamp_distinct_pixels():
    g = _utm_grid()
    pts = [_grid_lonlat_at(g, r, c) for r, c in [(10, 10), (50, 50), (80, 20)]]
    out = rasterize_sources([_make_source(lo, la) for lo, la in pts], **g)
    assert int(out.sum()) == 3
    for r, c in [(10, 10), (50, 50), (80, 20)]:
        assert int(np.asarray(out)[r, c]) == 1


def test_source_outside_grid_is_dropped():
    g = _utm_grid()
    # Way outside (different hemisphere)
    out = rasterize_sources([_make_source(0.0, -45.0)], **g)
    assert int(out.sum()) == 0


def test_accepts_shapely_point():
    g = _utm_grid()
    lon, lat = _grid_lonlat_at(g, row=5, col=5)
    out = rasterize_sources([Point(lon, lat)], **g)
    assert int(np.asarray(out)[5, 5]) == 1


def test_accepts_lonlat_tuple():
    g = _utm_grid()
    lon, lat = _grid_lonlat_at(g, row=5, col=5)
    out = rasterize_sources([(lon, lat)], **g)
    assert int(np.asarray(out)[5, 5]) == 1


# ─── rasterize_sources — buffer ───────────────────────────────────────


def test_buffer_in_metres_paints_disk():
    g = _utm_grid(res=30.0, shape=(100, 100))
    lon, lat = _grid_lonlat_at(g, row=50, col=50)
    out = rasterize_sources(
        [_make_source(lon, lat)], buffer_m=120.0, **g
    )
    # 120 m disk on 30 m grid: diameter ~8 px, area ~ pi*4^2 ≈ 50 px.
    n_set = int(out.sum())
    assert 30 <= n_set <= 90, f"unexpected pixel count {n_set}"
    # Centre is set, far corner is not.
    assert int(np.asarray(out)[50, 50]) == 1
    assert int(np.asarray(out)[0, 0]) == 0


def test_buffer_requires_projected_crs():
    transform = rasterio.Affine(0.001, 0, -100, 0, -0.001, 30)
    with pytest.raises(ValueError, match="projected CRS"):
        rasterize_sources(
            [_make_source(-100.0, 30.0)],
            transform=transform,
            shape=(100, 100),
            crs="EPSG:4326",
            buffer_m=100.0,
        )


def test_zero_buffer_allows_geographic_crs():
    """buffer_m=0 should work in lon/lat too."""
    transform = rasterio.Affine(0.001, 0, -100.0, 0, -0.001, 30.1)
    out = rasterize_sources(
        [(-99.95, 30.05)],
        transform=transform,
        shape=(100, 100),
        crs="EPSG:4326",
        buffer_m=0.0,
    )
    assert int(out.sum()) == 1


# ─── rasterize_sources — output type ─────────────────────────────────


def test_output_is_geotensor_with_crs_and_transform():
    g = _utm_grid()
    out = rasterize_sources([], **g)
    assert out.transform == g["transform"]
    # CRS may be stored as rasterio.CRS — compare via to_epsg.
    assert rasterio.crs.CRS.from_user_input(out.crs).to_epsg() == 32613
    assert out.fill_value_default == 0


def test_invalid_shape_raises():
    g = _utm_grid()
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        rasterize_sources(
            [],
            transform=g["transform"],
            shape=(1, 100, 100),  # type: ignore[arg-type]
            crs=g["crs"],
        )


# ─── CMSourceRaster ──────────────────────────────────────────────────


def test_cmsourceraster_load_matches_function():
    g = _utm_grid()
    lon, lat = _grid_lonlat_at(g, row=10, col=20)
    sources = [_make_source(lon, lat)]
    raster = CMSourceRaster(
        sources=sources,
        transform=g["transform"],
        shape=g["shape"],
        crs=rasterio.crs.CRS.from_user_input(g["crs"]),
    )
    out = raster.load()
    assert int(np.asarray(out)[10, 20]) == 1
    assert int(out.sum()) == 1


def test_cmsourceraster_repr_mentions_count():
    g = _utm_grid()
    raster = CMSourceRaster(
        sources=[_make_source(-100.0, 30.0)],
        transform=g["transform"],
        shape=g["shape"],
        crs=rasterio.crs.CRS.from_user_input(g["crs"]),
        buffer_m=50.0,
    )
    s = repr(raster)
    assert "n_sources=1" in s
    assert "buffer_m=50" in s


def test_cmsourceraster_from_geodata_inherits_grid():
    """``from_geodata`` should copy transform/shape/crs from a template."""
    g = _utm_grid()
    template = rasterize_sources([], **g)  # zero GeoTensor as a template
    raster = CMSourceRaster.from_geodata(
        sources=[_make_source(-100.0, 30.0)],
        template=template,
        buffer_m=15.0,
    )
    assert raster.transform == g["transform"]
    assert raster.shape == g["shape"]
    assert rasterio.crs.CRS.from_user_input(raster.crs).to_epsg() == 32613
    assert raster.buffer_m == 15.0
