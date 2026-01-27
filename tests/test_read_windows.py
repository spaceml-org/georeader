"""
Tests for the georeader.read module - window computation functions.

These tests verify window computation from various inputs (bounds, polygon, center coords, tile).
"""

import numpy as np
import pytest
import rasterio.windows
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon, box

from georeader import read
from georeader.geotensor import GeoTensor


@pytest.fixture
def sample_geodata():
    """Create a sample GeoTensor for testing."""
    data = np.zeros((3, 100, 100), dtype=np.float32)
    transform = from_origin(0, 100, 1, 1)  # 1m resolution, origin at (0, 100)
    return GeoTensor(data, transform=transform, crs="EPSG:32631")


@pytest.fixture
def sample_geodata_10m():
    """Create a sample GeoTensor with 10m resolution."""
    data = np.zeros((3, 100, 100), dtype=np.float32)
    transform = from_origin(500000, 4500000, 10, 10)  # 10m resolution
    return GeoTensor(data, transform=transform, crs="EPSG:32631")


class TestWindowFromBounds:
    """Tests for window_from_bounds function."""

    def test_basic_bounds(self, sample_geodata):
        """Test window from simple bounds."""
        bounds = (10, 50, 60, 90)  # xmin, ymin, xmax, ymax

        window = read.window_from_bounds(sample_geodata, bounds)

        assert isinstance(window, rasterio.windows.Window)
        # Check dimensions: 50 x 40 in geographic coords = 50 x 40 pixels at 1m res
        assert window.width == pytest.approx(50, abs=1)
        assert window.height == pytest.approx(40, abs=1)

    def test_bounds_same_as_data(self, sample_geodata):
        """Test window that matches data bounds."""
        bounds = sample_geodata.bounds

        window = read.window_from_bounds(sample_geodata, bounds)

        assert window.width == pytest.approx(100, abs=1)
        assert window.height == pytest.approx(100, abs=1)

    def test_bounds_with_different_crs(self, sample_geodata_10m):
        """Test window from bounds in different CRS."""
        # Bounds in WGS84 that cover part of the data
        bounds_wgs84 = (3.0, 40.0, 3.1, 40.1)

        window = read.window_from_bounds(sample_geodata_10m, bounds_wgs84, crs_bounds="EPSG:4326")

        assert isinstance(window, rasterio.windows.Window)


class TestWindowFromPolygon:
    """Tests for window_from_polygon function."""

    def test_simple_polygon(self, sample_geodata):
        """Test window from simple polygon."""
        polygon = box(20, 30, 70, 80)  # xmin, ymin, xmax, ymax

        window = read.window_from_polygon(sample_geodata, polygon)

        assert isinstance(window, rasterio.windows.Window)
        # Width should be ~50 pixels (70-20)
        assert window.width == pytest.approx(50, abs=2)

    def test_multipolygon(self, sample_geodata):
        """Test window from MultiPolygon."""
        poly1 = box(10, 60, 30, 80)
        poly2 = box(50, 20, 80, 50)
        multipolygon = MultiPolygon([poly1, poly2])

        window = read.window_from_polygon(sample_geodata, multipolygon)

        # Should encompass both polygons
        assert window.width >= 70  # From 10 to 80

    def test_polygon_with_crs(self, sample_geodata_10m):
        """Test window from polygon with different CRS."""
        # Small polygon in WGS84
        polygon = box(3.0, 40.5, 3.05, 40.55)

        window = read.window_from_polygon(sample_geodata_10m, polygon, crs_polygon="EPSG:4326")

        assert isinstance(window, rasterio.windows.Window)

    def test_window_surrounding_false(self, sample_geodata):
        """Test window_surrounding=False."""
        polygon = box(20, 30, 70, 80)

        window = read.window_from_polygon(sample_geodata, polygon, window_surrounding=False)

        assert isinstance(window, rasterio.windows.Window)

    def test_window_surrounding_true(self, sample_geodata):
        """Test window_surrounding=True adds extra pixel."""
        polygon = box(20, 30, 70, 80)

        window_false = read.window_from_polygon(sample_geodata, polygon, window_surrounding=False)
        window_true = read.window_from_polygon(sample_geodata, polygon, window_surrounding=True)

        # window_surrounding=True should be 1 pixel larger
        assert window_true.width == window_false.width + 1
        assert window_true.height == window_false.height + 1


class TestWindowFromCenterCoords:
    """Tests for window_from_center_coords function."""

    def test_basic_center(self, sample_geodata):
        """Test window from center coordinates."""
        center_coords = (50, 50)  # Center of the data
        shape = (20, 20)  # 20x20 pixels

        window = read.window_from_center_coords(sample_geodata, center_coords, shape)

        assert isinstance(window, rasterio.windows.Window)
        assert window.width == 20
        assert window.height == 20

    def test_different_shape(self, sample_geodata):
        """Test with non-square shape."""
        center_coords = (50, 50)
        shape = (30, 50)  # 30 rows, 50 cols

        window = read.window_from_center_coords(sample_geodata, center_coords, shape)

        assert window.width == 50
        assert window.height == 30

    def test_with_crs(self, sample_geodata_10m):
        """Test center coords with different CRS."""
        # Center in WGS84
        center_coords = (3.0, 40.5)
        shape = (50, 50)

        window = read.window_from_center_coords(sample_geodata_10m, center_coords, shape, crs_center_coords="EPSG:4326")

        assert window.width == 50
        assert window.height == 50


class TestWindowFromTile:
    """Tests for window_from_tile function."""

    def test_basic_tile(self, sample_geodata_10m):
        """Test window from TMS tile."""
        # Get a tile that intersects the data
        # For UTM data, this may not intersect, but test the function works
        window = read.window_from_tile(sample_geodata_10m, x=0, y=0, z=1)

        assert isinstance(window, rasterio.windows.Window)

    def test_different_zoom(self, sample_geodata_10m):
        """Test with different zoom levels."""
        window_z10 = read.window_from_tile(sample_geodata_10m, x=500, y=300, z=10)
        window_z12 = read.window_from_tile(sample_geodata_10m, x=2000, y=1200, z=12)

        # Higher zoom = smaller area = smaller window
        assert isinstance(window_z10, rasterio.windows.Window)
        assert isinstance(window_z12, rasterio.windows.Window)


class TestReadFromWindow:
    """Tests for read_from_window function."""

    def test_basic_read(self, sample_geodata):
        """Test basic window reading."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)

        result = read.read_from_window(sample_geodata, window)

        assert isinstance(result, GeoTensor)
        assert result.shape == (3, 50, 50)

    def test_boundless_read(self, sample_geodata):
        """Test boundless window reading."""
        window = rasterio.windows.Window(col_off=-10, row_off=10, width=50, height=50)

        result = read.read_from_window(sample_geodata, window, boundless=True)

        assert result.shape == (3, 50, 50)

    def test_non_boundless_out_of_bounds(self, sample_geodata):
        """Test non-boundless read with window extending outside."""
        window = rasterio.windows.Window(col_off=-10, row_off=10, width=50, height=50)

        result = read.read_from_window(sample_geodata, window, boundless=False)

        # Should be clipped to intersection
        assert result is not None
        assert result.width < 50


class TestReadFromBounds:
    """Tests for read_from_bounds function."""

    def test_basic_read(self, sample_geodata):
        """Test reading from bounds."""
        bounds = (20, 40, 70, 80)

        result = read.read_from_bounds(sample_geodata, bounds)

        assert isinstance(result, GeoTensor)

    def test_with_padding(self, sample_geodata):
        """Test reading with padding."""
        bounds = (20, 40, 70, 80)

        result_no_pad = read.read_from_bounds(sample_geodata, bounds, pad_add=(0, 0))
        result_with_pad = read.read_from_bounds(sample_geodata, bounds, pad_add=(10, 10))

        # Padded result should be larger
        assert result_with_pad.width > result_no_pad.width
        assert result_with_pad.height > result_no_pad.height


class TestReadFromPolygon:
    """Tests for read_from_polygon function."""

    def test_basic_read(self, sample_geodata):
        """Test reading from polygon."""
        polygon = box(20, 40, 70, 80)

        result = read.read_from_polygon(sample_geodata, polygon)

        assert isinstance(result, GeoTensor)

    def test_with_padding(self, sample_geodata):
        """Test reading with padding."""
        polygon = box(30, 40, 60, 70)

        result_no_pad = read.read_from_polygon(sample_geodata, polygon, pad_add=(0, 0))
        result_with_pad = read.read_from_polygon(sample_geodata, polygon, pad_add=(5, 5))

        assert result_with_pad.width > result_no_pad.width


class TestReadFromCenterCoords:
    """Tests for read_from_center_coords function."""

    def test_basic_read(self, sample_geodata):
        """Test reading from center coordinates."""
        center_coords = (50, 50)
        shape = (30, 30)

        result = read.read_from_center_coords(sample_geodata, center_coords, shape)

        assert isinstance(result, GeoTensor)
        assert result.shape == (3, 30, 30)

    def test_with_crs(self, sample_geodata_10m):
        """Test reading with coordinates in different CRS."""
        center_coords = (3.0, 40.5)
        shape = (20, 20)

        result = read.read_from_center_coords(sample_geodata_10m, center_coords, shape, crs_center_coords="EPSG:4326")

        assert result.shape == (3, 20, 20)


class TestCalculateTransformWindow:
    """Tests for calculate_transform_window function."""

    def test_same_crs(self, sample_geodata):
        """Test calculating transform for same CRS."""
        dst_crs = "EPSG:32631"

        transform, window = read.calculate_transform_window(sample_geodata, dst_crs, resolution_dst_crs=1.0)

        assert transform is not None
        assert window is not None

    def test_different_crs(self, sample_geodata_10m):
        """Test calculating transform for different CRS."""
        dst_crs = "EPSG:4326"

        transform, window = read.calculate_transform_window(sample_geodata_10m, dst_crs)

        assert transform is not None


class TestResize:
    """Tests for resize function."""

    def test_resize_to_half(self, sample_geodata):
        """Test resizing to half size."""
        # Need to fill with some data
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)

        result = read.resize(sample_geodata, resolution_dst=2.0)

        # Half resolution = half the pixels
        assert result.shape == (3, 50, 50)

    def test_resize_to_double(self, sample_geodata):
        """Test resizing to double size."""
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)

        result = read.resize(sample_geodata, resolution_dst=0.5)

        # Double resolution = double the pixels
        assert result.shape == (3, 200, 200)

    def test_resize_with_window(self, sample_geodata):
        """Test resize with specific output window."""
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)

        window_out = rasterio.windows.Window(0, 0, width=50, height=50)
        result = read.resize(sample_geodata, resolution_dst=2.0, window_out=window_out)

        assert result.shape == (3, 50, 50)
