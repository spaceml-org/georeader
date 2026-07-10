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
        """Test calculating transform for same CRS.

        Pins the return ORDER — ``(window, transform)`` — which two call
        sites historically unpacked backwards (the function's annotation
        used to promise the reverse).
        """
        dst_crs = "EPSG:32631"

        window, transform = read.calculate_transform_window(sample_geodata, dst_crs, resolution_dst_crs=1.0)

        assert isinstance(window, rasterio.windows.Window)
        assert isinstance(transform, rasterio.Affine)

    def test_different_crs(self, sample_geodata_10m):
        """Test calculating transform for different CRS."""
        dst_crs = "EPSG:4326"

        window, transform = read.calculate_transform_window(sample_geodata_10m, dst_crs)

        assert isinstance(window, rasterio.windows.Window)
        assert isinstance(transform, rasterio.Affine)


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


class TestReadReprojectLike:
    """Tests for read_reproject_like function."""

    def test_basic_reproject_like(self, sample_geodata):
        """Test basic reprojection to match another GeoTensor."""
        # Create source data with different extent
        source_data = np.ones((3, 50, 50), dtype=np.float32)
        source_transform = from_origin(20, 80, 1, 1)  # Different origin
        source_geodata = GeoTensor(source_data, transform=source_transform, crs="EPSG:32631")

        result = read.read_reproject_like(source_geodata, sample_geodata)

        assert isinstance(result, GeoTensor)
        assert result.shape[-2:] == sample_geodata.shape[-2:]
        assert result.crs == sample_geodata.crs

    def test_reproject_like_with_resolution(self, sample_geodata):
        """Test reprojection with custom resolution."""
        source_data = np.ones((3, 50, 50), dtype=np.float32)
        source_transform = from_origin(20, 80, 1, 1)
        source_geodata = GeoTensor(source_data, transform=source_transform, crs="EPSG:32631")

        result = read.read_reproject_like(source_geodata, sample_geodata, resolution_dst=2.0)

        assert isinstance(result, GeoTensor)
        # Output should be smaller due to coarser resolution
        assert result.shape[-2] == pytest.approx(50, abs=1)  # Half the height
        assert result.shape[-1] == pytest.approx(50, abs=1)  # Half the width

    def test_reproject_like_return_only_data(self, sample_geodata):
        """Test reprojection returning only data array."""
        source_data = np.ones((3, 50, 50), dtype=np.float32)
        source_transform = from_origin(20, 80, 1, 1)
        source_geodata = GeoTensor(source_data, transform=source_transform, crs="EPSG:32631")

        result = read.read_reproject_like(source_geodata, sample_geodata, return_only_data=True)

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, GeoTensor)

    def test_reproject_like_different_crs(self, sample_geodata_10m):
        """Test reprojection between different CRS."""
        # Create source in WGS84
        source_data = np.ones((3, 50, 50), dtype=np.float32)
        source_transform = from_origin(3.0, 40.5, 0.001, 0.001)
        source_geodata = GeoTensor(source_data, transform=source_transform, crs="EPSG:4326")

        result = read.read_reproject_like(source_geodata, sample_geodata_10m)

        assert isinstance(result, GeoTensor)
        assert result.crs == sample_geodata_10m.crs

    def test_reproject_like_with_resampling(self, sample_geodata):
        """Test reprojection with different resampling method."""
        import rasterio.warp

        source_data = np.ones((3, 50, 50), dtype=np.float32)
        source_transform = from_origin(20, 80, 1, 1)
        source_geodata = GeoTensor(source_data, transform=source_transform, crs="EPSG:32631")

        result = read.read_reproject_like(source_geodata, sample_geodata, resampling=rasterio.warp.Resampling.nearest)

        assert isinstance(result, GeoTensor)
        assert result.shape[-2:] == sample_geodata.shape[-2:]


class TestApplyAntiAliasing:
    """Tests for apply_anti_aliasing function.

    These tests require scipy which is an optional dependency.
    """

    @pytest.fixture(autouse=True)
    def _skip_without_scipy(self):
        """Skip all tests in this class if scipy is not available."""
        pytest.importorskip("scipy")

    def test_basic_anti_aliasing(self, sample_geodata):
        """Test basic anti-aliasing application."""
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)

        result = read.apply_anti_aliasing(sample_geodata, resolution_dst=2.0)

        assert isinstance(result, GeoTensor)
        assert result.shape == sample_geodata.shape
        # Anti-aliasing should smooth the data
        assert result.values is not None

    def test_anti_aliasing_with_custom_sigma(self, sample_geodata):
        """Test anti-aliasing with custom sigma."""
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)

        result = read.apply_anti_aliasing(sample_geodata, anti_aliasing_sigma=2.0, resolution_dst=2.0)

        assert isinstance(result, GeoTensor)
        assert result.shape == sample_geodata.shape

    def test_anti_aliasing_no_downscale(self, sample_geodata):
        """Test anti-aliasing when not downscaling (should return unchanged)."""
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)
        original_values = sample_geodata.values.copy()

        # Resolution same as original (1.0) - no downscale needed
        result = read.apply_anti_aliasing(sample_geodata, resolution_dst=1.0)

        # Should be unchanged since no downscaling
        assert np.allclose(result.values, original_values)

    def test_anti_aliasing_upscale(self, sample_geodata):
        """Test anti-aliasing when upscaling (should return unchanged)."""
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)
        original_values = sample_geodata.values.copy()

        # Resolution finer than original - upscaling, no anti-aliasing needed
        result = read.apply_anti_aliasing(sample_geodata, resolution_dst=0.5)

        # Should be unchanged since upscaling
        assert np.allclose(result.values, original_values)

    def test_anti_aliasing_tuple_resolution(self, sample_geodata):
        """Test anti-aliasing with tuple resolution."""
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)

        result = read.apply_anti_aliasing(sample_geodata, resolution_dst=(2.0, 3.0))

        assert isinstance(result, GeoTensor)
        assert result.shape == sample_geodata.shape

    def test_anti_aliasing_2d_array(self):
        """Test anti-aliasing with 2D array."""
        data = np.random.rand(100, 100).astype(np.float32)
        transform = from_origin(0, 100, 1, 1)
        geodata = GeoTensor(data, transform=transform, crs="EPSG:32631")

        result = read.apply_anti_aliasing(geodata, resolution_dst=2.0)

        assert isinstance(result, GeoTensor)
        assert result.shape == geodata.shape

    def test_anti_aliasing_4d_array(self):
        """Test anti-aliasing with 4D array."""
        data = np.random.rand(2, 3, 50, 50).astype(np.float32)
        transform = from_origin(0, 100, 1, 1)
        geodata = GeoTensor(data, transform=transform, crs="EPSG:32631")

        result = read.apply_anti_aliasing(geodata, resolution_dst=2.0)

        assert isinstance(result, GeoTensor)
        assert result.shape == geodata.shape


class TestReadToCrs:
    """Tests for read_to_crs function."""

    def test_basic_crs_conversion(self, sample_geodata_10m):
        """Test basic CRS conversion."""
        sample_geodata_10m.values[:] = np.random.rand(*sample_geodata_10m.shape)

        result = read.read_to_crs(sample_geodata_10m, dst_crs="EPSG:4326")

        assert isinstance(result, GeoTensor)
        assert result.crs == "EPSG:4326"

    def test_same_crs_returns_same(self, sample_geodata):
        """Test that same CRS returns the input unchanged."""
        sample_geodata.values[:] = np.random.rand(*sample_geodata.shape)

        result = read.read_to_crs(sample_geodata, dst_crs="EPSG:32631")

        # Should return the same object since CRS is identical
        assert result is sample_geodata

    def test_crs_conversion_with_resolution(self, sample_geodata_10m):
        """Test CRS conversion with custom resolution."""
        sample_geodata_10m.values[:] = np.random.rand(*sample_geodata_10m.shape)

        result = read.read_to_crs(sample_geodata_10m, dst_crs="EPSG:4326", resolution_dst_crs=0.0001)

        assert isinstance(result, GeoTensor)
        assert result.crs == "EPSG:4326"

    def test_crs_conversion_return_only_data(self, sample_geodata_10m):
        """Test CRS conversion returning only data."""
        sample_geodata_10m.values[:] = np.random.rand(*sample_geodata_10m.shape)

        result = read.read_to_crs(sample_geodata_10m, dst_crs="EPSG:4326", return_only_data=True)

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, GeoTensor)

    def test_crs_conversion_to_web_mercator(self, sample_geodata_10m):
        """Test CRS conversion to Web Mercator."""
        sample_geodata_10m.values[:] = np.random.rand(*sample_geodata_10m.shape)

        result = read.read_to_crs(sample_geodata_10m, dst_crs="EPSG:3857")

        assert isinstance(result, GeoTensor)
        assert result.crs == "EPSG:3857"


@pytest.fixture
def sample_rpcs():
    """Create a sample RPC model for testing.

    This creates an RPC that represents a simple linear transformation,
    which allows the geotransform to be invertible.
    """
    import rasterio.rpc

    # RPC coefficients that create a simple, invertible transformation
    # line_num has coefficients for: 1, lat, lon, height, lat*lon, ...
    # For a simple linear transform: line = (lat - lat_off) / lat_scale * line_scale + line_off
    # We need line_num[0] = 0, line_num[1] = 1 (lat coefficient) to get line = lat_scale_factor * lat
    return rasterio.rpc.RPC(
        height_off=0,
        height_scale=1,
        lat_off=40.0,
        lat_scale=0.01,  # Small scale so 100 pixels covers 1 degree
        line_off=50,
        line_scale=50,
        long_off=3.0,
        long_scale=0.01,  # Small scale so 100 pixels covers 1 degree
        samp_off=50,
        samp_scale=50,
        # Coefficients: [1, lat, lon, h, lat*lon, lat*h, lon*h, lat^2, lon^2, h^2, ...]
        line_num_coeff=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        line_den_coeff=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        samp_num_coeff=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        samp_den_coeff=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )


class TestReadRpcs:
    """Tests for read_rpcs function."""

    def test_basic_rpc_georeferencing(self, sample_rpcs):
        """Test basic RPC-based georeferencing."""
        data = np.random.rand(100, 100).astype(np.float32)

        result = read.read_rpcs(data, sample_rpcs)

        assert isinstance(result, GeoTensor)
        assert result.crs is not None
        assert result.transform is not None

    def test_rpc_georeferencing_3d_array(self, sample_rpcs):
        """Test RPC georeferencing with 3D array."""
        data = np.random.rand(3, 100, 100).astype(np.float32)

        result = read.read_rpcs(data, sample_rpcs)

        assert isinstance(result, GeoTensor)
        assert result.shape[0] == 3

    def test_rpc_georeferencing_4d_array(self, sample_rpcs):
        """Test RPC georeferencing with 4D array."""
        data = np.random.rand(2, 3, 100, 100).astype(np.float32)

        result = read.read_rpcs(data, sample_rpcs)

        assert isinstance(result, GeoTensor)
        assert result.shape[:2] == (2, 3)

    def test_rpc_georeferencing_return_only_data(self, sample_rpcs):
        """Test RPC georeferencing returning only data."""
        data = np.random.rand(100, 100).astype(np.float32)

        result = read.read_rpcs(data, sample_rpcs, return_only_data=True)

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, GeoTensor)

    def test_rpc_georeferencing_with_dst_crs(self, sample_rpcs):
        """Test RPC georeferencing with destination CRS."""
        data = np.random.rand(100, 100).astype(np.float32)

        result = read.read_rpcs(data, sample_rpcs, dst_crs="EPSG:32631")

        assert isinstance(result, GeoTensor)
        assert result.crs == "EPSG:32631"

    def test_rpc_georeferencing_with_resolution(self, sample_rpcs):
        """Test RPC georeferencing with custom resolution."""
        data = np.random.rand(100, 100).astype(np.float32)

        result = read.read_rpcs(data, sample_rpcs, resolution_dst_crs=0.001)

        assert isinstance(result, GeoTensor)

    def test_rpc_georeferencing_with_fill_value(self, sample_rpcs):
        """Test RPC georeferencing with custom fill value."""
        data = np.random.rand(100, 100).astype(np.float32)

        result = read.read_rpcs(data, sample_rpcs, fill_value_default=-9999)

        assert isinstance(result, GeoTensor)
        assert result.fill_value_default == -9999


class TestReadRpcsErrors:
    """Tests for read_rpcs error cases."""

    def test_invalid_dimensions_1d(self, sample_rpcs):
        """Test that 1D array raises error."""
        data = np.random.rand(100).astype(np.float32)  # 1D array

        with pytest.raises(AssertionError, match="2, 3 or 4 dimensions"):
            read.read_rpcs(data, sample_rpcs)

    def test_invalid_dimensions_5d(self, sample_rpcs):
        """Test that 5D array raises error."""
        data = np.random.rand(2, 2, 3, 50, 50).astype(np.float32)  # 5D array

        with pytest.raises(AssertionError, match="2, 3 or 4 dimensions"):
            read.read_rpcs(data, sample_rpcs)


class TestReadFromTile:
    """Tests for read_from_tile function."""

    def test_basic_read_from_tile(self, sample_geodata_10m):
        """Test basic tile reading."""
        sample_geodata_10m.values[:] = np.random.rand(*sample_geodata_10m.shape)

        # Try to read from a tile - may return None if tile doesn't intersect
        result = read.read_from_tile(sample_geodata_10m, x=2048, y=1024, z=12)

        # Result may be None if tile doesn't intersect data
        assert result is None or isinstance(result, GeoTensor)

    def test_read_from_tile_with_out_shape(self, sample_geodata_10m):
        """Test tile reading with specific output shape."""
        sample_geodata_10m.values[:] = np.random.rand(*sample_geodata_10m.shape)

        result = read.read_from_tile(sample_geodata_10m, x=2048, y=1024, z=12, out_shape=(128, 128))

        # Result may be None if tile doesn't intersect data
        assert result is None or isinstance(result, GeoTensor)

    def test_read_from_tile_dst_crs_none(self, sample_geodata_10m):
        """Test tile reading with dst_crs=None."""
        sample_geodata_10m.values[:] = np.random.rand(*sample_geodata_10m.shape)

        result = read.read_from_tile(sample_geodata_10m, x=2048, y=1024, z=12, dst_crs=None)

        # Result may be None if tile doesn't intersect data
        assert result is None or isinstance(result, GeoTensor)
