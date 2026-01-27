"""
Tests for the georeader.rasterize module.

These tests verify rasterization of geometries and geodataframes.
"""

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from georeader.geotensor import GeoTensor
from georeader.rasterize import (
    rasterize_from_geometry,
    rasterize_from_geopandas,
    rasterize_geometry_like,
    rasterize_geopandas_like,
)


@pytest.fixture
def sample_geodata():
    """Create a sample GeoTensor for testing."""
    data = np.zeros((100, 100), dtype=np.float32)
    transform = from_origin(0, 100, 1, 1)  # 1m resolution
    return GeoTensor(data, transform=transform, crs="EPSG:32631")


@pytest.fixture
def sample_polygon():
    """Create a sample polygon for testing."""
    return Polygon([(20, 20), (80, 20), (80, 80), (20, 80), (20, 20)])


@pytest.fixture
def sample_geodataframe():
    """Create a sample GeoDataFrame for testing."""
    polygons = [
        Polygon([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)]),
        Polygon([(50, 50), (90, 50), (90, 90), (50, 90), (50, 50)]),
    ]
    gdf = gpd.GeoDataFrame({"value": [1, 2], "geometry": polygons}, crs="EPSG:32631")
    return gdf


class TestRasterizeGeometryLike:
    """Tests for rasterize_geometry_like function."""

    def test_basic_rasterization(self, sample_geodata, sample_polygon):
        """Test basic polygon rasterization."""
        result = rasterize_geometry_like(sample_polygon, sample_geodata)

        assert isinstance(result, GeoTensor)
        assert result.shape == (100, 100)
        # Polygon should be rasterized
        assert result.values.sum() > 0

    def test_custom_value(self, sample_geodata, sample_polygon):
        """Test rasterization with custom value."""
        result = rasterize_geometry_like(sample_polygon, sample_geodata, value=5)

        # Non-zero pixels should have value 5
        assert np.all((result.values == 0) | (result.values == 5))
        assert (result.values == 5).sum() > 0

    def test_custom_dtype(self, sample_geodata, sample_polygon):
        """Test rasterization with custom dtype."""
        result = rasterize_geometry_like(sample_polygon, sample_geodata, dtype=np.float32)

        assert result.values.dtype == np.float32

    def test_fill_value(self, sample_geodata, sample_polygon):
        """Test rasterization with custom fill value."""
        result = rasterize_geometry_like(sample_polygon, sample_geodata, fill=255, value=1)

        # Background should be 255, polygon should be 1
        assert (result.values == 255).sum() > 0
        assert (result.values == 1).sum() > 0

    def test_return_only_data(self, sample_geodata, sample_polygon):
        """Test returning only numpy array."""
        result = rasterize_geometry_like(sample_polygon, sample_geodata, return_only_data=True)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100)

    def test_all_touched(self, sample_geodata):
        """Test all_touched parameter."""
        # Small polygon that might miss pixel centers
        small_polygon = Polygon([(0.1, 99.1), (0.9, 99.1), (0.9, 99.9), (0.1, 99.9), (0.1, 99.1)])

        result_default = rasterize_geometry_like(small_polygon, sample_geodata, all_touched=False)
        result_all_touched = rasterize_geometry_like(small_polygon, sample_geodata, all_touched=True)

        # all_touched should rasterize more pixels
        assert result_all_touched.values.sum() >= result_default.values.sum()

    def test_line_geometry(self, sample_geodata):
        """Test rasterizing a line geometry."""
        line = LineString([(10, 50), (90, 50)])

        result = rasterize_geometry_like(line, sample_geodata)

        assert isinstance(result, GeoTensor)
        assert result.values.sum() > 0


class TestRasterizeFromGeometry:
    """Tests for rasterize_from_geometry function."""

    def test_with_bounds_and_resolution(self, sample_polygon):
        """Test rasterization with bounds and resolution."""
        result = rasterize_from_geometry(
            sample_polygon, bounds=(0, 0, 100, 100), resolution=1.0, crs_geom_bounds="EPSG:32631"
        )

        assert isinstance(result, GeoTensor)
        assert result.shape[0] == 100
        assert result.shape[1] == 100

    def test_with_transform(self, sample_polygon):
        """Test rasterization with transform."""
        transform = from_origin(0, 100, 1, 1)

        result = rasterize_from_geometry(
            sample_polygon, transform=transform, bounds=(0, 0, 100, 100), crs_geom_bounds="EPSG:32631"
        )

        assert result.transform == transform

    def test_with_window_out(self, sample_polygon):
        """Test rasterization with explicit output window."""
        import rasterio.windows

        transform = from_origin(0, 100, 1, 1)
        window_out = rasterio.windows.Window(0, 0, width=50, height=50)

        result = rasterize_from_geometry(
            sample_polygon, transform=transform, window_out=window_out, crs_geom_bounds="EPSG:32631"
        )

        assert result.shape == (50, 50)

    def test_multipolygon(self):
        """Test rasterizing a MultiPolygon."""
        poly1 = Polygon([(10, 10), (30, 10), (30, 30), (10, 30), (10, 10)])
        poly2 = Polygon([(60, 60), (80, 60), (80, 80), (60, 80), (60, 60)])
        multipolygon = MultiPolygon([poly1, poly2])

        result = rasterize_from_geometry(
            multipolygon, bounds=(0, 0, 100, 100), resolution=1.0, crs_geom_bounds="EPSG:32631"
        )

        assert result.values.sum() > 0


class TestRasterizeGeopandasLike:
    """Tests for rasterize_geopandas_like function."""

    def test_basic_rasterization(self, sample_geodata, sample_geodataframe):
        """Test basic GeoDataFrame rasterization."""
        result = rasterize_geopandas_like(sample_geodataframe, sample_geodata, column="value")

        assert isinstance(result, GeoTensor)
        assert result.shape == (100, 100)
        # Should have values from the column
        assert (result.values == 1).sum() > 0
        assert (result.values == 2).sum() > 0

    def test_return_only_data(self, sample_geodata, sample_geodataframe):
        """Test returning only numpy array."""
        result = rasterize_geopandas_like(sample_geodataframe, sample_geodata, column="value", return_only_data=True)

        assert isinstance(result, np.ndarray)

    def test_fill_value(self, sample_geodata, sample_geodataframe):
        """Test custom fill value."""
        result = rasterize_geopandas_like(sample_geodataframe, sample_geodata, column="value", fill=255)

        # Background should be 255
        assert (result.values == 255).sum() > 0


class TestRasterizeFromGeopandas:
    """Tests for rasterize_from_geopandas function."""

    def test_with_bounds_and_resolution(self, sample_geodataframe):
        """Test rasterization with bounds and resolution."""
        result = rasterize_from_geopandas(sample_geodataframe, column="value", bounds=(0, 0, 100, 100), resolution=1.0)

        assert isinstance(result, GeoTensor)
        assert result.shape[0] == 100
        assert result.shape[1] == 100

    def test_crs_transformation(self, sample_geodataframe):
        """Test CRS transformation during rasterization."""
        # Create GeoDataFrame in WGS84
        polygons = [
            Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001), (0, 0)]),
        ]
        gdf_wgs84 = gpd.GeoDataFrame({"value": [1], "geometry": polygons}, crs="EPSG:4326")

        # Rasterize to UTM
        result = rasterize_from_geopandas(
            gdf_wgs84, column="value", bounds=(0, 0, 1000, 1000), resolution=10.0, crs_out="EPSG:32631"
        )

        assert result.crs is not None

    def test_preserve_dtype(self, sample_geodataframe):
        """Test that column dtype is preserved."""
        # Add float column
        sample_geodataframe["float_value"] = [1.5, 2.5]

        result = rasterize_from_geopandas(
            sample_geodataframe, column="float_value", bounds=(0, 0, 100, 100), resolution=1.0
        )

        assert result.values.dtype == np.float64

    def test_empty_geodataframe(self):
        """Test rasterizing empty GeoDataFrame."""
        gdf = gpd.GeoDataFrame({"value": [], "geometry": []}, crs="EPSG:32631")

        result = rasterize_from_geopandas(gdf, column="value", bounds=(0, 0, 100, 100), resolution=1.0)

        # Should return array filled with fill value (0 by default)
        assert result.values.sum() == 0
