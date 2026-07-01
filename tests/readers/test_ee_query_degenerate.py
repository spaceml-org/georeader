"""Tests for the degenerate-footprint guard in georeader.readers.ee_query.

Run with the earthengine-api extra installed (``ee_query`` imports ``ee`` at module load):

    poetry run pytest tests/readers/test_ee_query_degenerate.py -v
"""
import warnings

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from georeader.readers.ee_query import (
    DEGENERATE_FOOTPRINT_MAX_LAT_SPAN,
    _filter_degenerate_footprints,
)

# Real ~185 km scene (northern Fennoscandia) — ~2.4 deg latitude span.
NORMAL = box(21.28, 67.04, 27.88, 69.46)
# Corrupt whole-globe footprint returned by GEE for LC08_193012_20260627.
WORLD = box(-180, -90, 180, 90)


def _gdf(rows):
    return gpd.GeoDataFrame(
        {"geometry": [g for _, g in rows]},
        index=[t for t, _ in rows],
        crs="EPSG:4326",
    )


def test_default_threshold_is_5_degrees():
    assert DEGENERATE_FOOTPRINT_MAX_LAT_SPAN == 5.0


def test_drops_world_footprint_and_keeps_normal():
    gdf = _gdf([("normal", NORMAL), ("corrupt", WORLD)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _filter_degenerate_footprints(gdf)
    assert list(out.index) == ["normal"]


def test_warns_on_dropped_tile():
    gdf = _gdf([("corrupt", WORLD)])
    with pytest.warns(UserWarning, match="degenerate footprint"):
        out = _filter_degenerate_footprints(gdf)
    assert len(out) == 0


def test_keeps_wide_low_latitude_span_scene():
    # A large-longitude / high-latitude scene (e.g. S1) with small latitude span is kept.
    wide = box(10.0, 60.0, 40.0, 62.5)  # 2.5 deg lat span
    assert list(_filter_degenerate_footprints(_gdf([("s1", wide)])).index) == ["s1"]


def test_empty_geodataframe_returned_unchanged():
    assert len(_filter_degenerate_footprints(gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326"))) == 0


def test_empty_geometry_is_kept():
    assert list(_filter_degenerate_footprints(_gdf([("empty", Polygon())])).index) == ["empty"]


def test_custom_threshold():
    span3 = box(0.0, 40.0, 1.0, 43.0)  # 3 deg lat span
    assert list(_filter_degenerate_footprints(_gdf([("s", span3)])).index) == ["s"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = _filter_degenerate_footprints(_gdf([("s", span3)]), max_lat_span=2.0)
    assert len(out) == 0
