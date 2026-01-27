"""
Tests for the georeader.rasterio_reader.RasterioReader class.

These tests verify the core reading functionality of RasterioReader including:
- Reading specific band indexes
- Reading with output shape transformations
- Handling boundless reads (reading outside raster extent)
- Using isel for band selection

Uses a temporary GeoTiff test file created via the test_raster_path fixture.
Properties: 15 bands, height=200, width=250, CRS=EPSG:32738, resolution=10m
"""

import numpy as np
import rasterio
import rasterio.windows

from georeader import rasterio_reader, read

# Window for testing - within image bounds (height=200, width=250)
WINDOW_TEST = rasterio.windows.Window(col_off=50, row_off=50, width=100, height=64)


def test_read_indexes(test_raster_path):
    """
    Test reading specific band indexes from a raster.

    This test verifies that:
    1. set_indexes correctly configures which bands to read
    2. Reading returns the expected shape for stacked multi-band reads
    3. Reading with a specific index returns the correct single band
    4. The data values match what rasterio would read directly
    5. The same behavior works with stack=False mode
    """
    window, file = WINDOW_TEST, test_raster_path
    reader = rasterio_reader.RasterioReader([file], window_focus=window)
    # Select bands 2 and 3 (relative to the reader's band selection)
    reader.set_indexes([2, 3], relative=True)

    # Read all selected bands - should return 4D array (time, band, height, width)
    data = reader.read()
    assert data.shape == (1, 2, window.height, window.width), (
        f"Expected {(1, 2, window.height, window.width)} found {data.shape}"
    )

    # Read single band by index - should return 3D array (time, height, width)
    data = reader.read(indexes=2)
    assert data.shape == (1, window.height, window.width), (
        f"Expected {(1, window.height, window.width)} found {data.shape}"
    )

    # Compare with direct rasterio read - index 3 in original file (2+1)
    with rasterio.open(file) as src:
        data_expected = src.read(window=window, indexes=3)

    assert np.allclose(data, data_expected), "Content of the array is different"

    # Test same behavior with stack=False mode
    reader = rasterio_reader.RasterioReader([file], window_focus=window, stack=False)
    reader = reader.isel({"band": [1, 2]})  # Select bands 2 and 3 (0-indexed)
    data = reader.values

    assert data.shape == (2, window.height, window.width), (
        f"Expected {(2, window.height, window.width)} found {data.shape}"
    )

    data = reader.read(indexes=2)
    assert data.shape == (window.height, window.width), f"Expected {(window.height, window.width)} found {data.shape}"

    assert np.allclose(data, data_expected), "Content of the array is different"


def test_read_out_shape(test_raster_path):
    """
    Test reading with output shape transformation.

    This test verifies that the RasterioReader correctly resamples data
    to a specified output shape when reading.
    """
    window, file = WINDOW_TEST, test_raster_path
    reader = rasterio_reader.RasterioReader([file], window_focus=window)
    reader.set_indexes([2, 3], relative=True)

    # Read with a different output shape (64x32 instead of 64x100)
    data = reader.read(indexes=2, out_shape=(64, 32))
    assert data.shape == (1, 64, 32), f"Expected {(1, 64, 32)} found {data.shape}"


def test_read_boundless_false(test_raster_path):
    """
    Test reading with boundless=False when the window extends beyond raster extent.

    This test verifies that:
    1. When boundless=False, reads are clipped to the raster extent
    2. The resulting shape reflects only the valid (non-padded) area
    3. The window_focus is adjusted to represent the valid window
    4. Both direct and loaded reads produce the same result
    """
    # Window that extends beyond the top-left corner of the image
    window = rasterio.windows.Window(col_off=-10, row_off=-10, width=128, height=64)
    file = test_raster_path

    reader = rasterio_reader.RasterioReader(file, window_focus=None)
    reader_subset = read.read_from_window(reader, window=window, boundless=False)
    xr_subset = read.read_from_window(reader, window=window, boundless=True).load(boundless=False)

    # Expected shape: original window minus the out-of-bounds portion
    # height = 64 - 10 = 54, width = 128 - 10 = 118
    expected_shape = (15, window.height + window.row_off, window.width + window.col_off)
    assert reader_subset.shape == expected_shape, f"Unexpected shape {reader_subset.shape} expected {expected_shape}"

    assert reader_subset.shape == xr_subset.shape, "Unexpected shapes"

    # The clipped window should start at (0, 0) with reduced dimensions
    expected_window = rasterio.windows.Window(col_off=0, row_off=0, width=118, height=54)
    assert reader_subset.window_focus == expected_window, "Different windows"

    expected_bounds = rasterio.windows.bounds(expected_window, reader.transform)

    # Transform should be the same (origin unchanged)
    assert reader.transform == reader_subset.transform, "Expected same transform"

    # Bounds should match between methods
    assert xr_subset.transform == reader_subset.transform, "Expected same transform"
    assert xr_subset.bounds == reader_subset.bounds, "Expected same bounds"
    assert xr_subset.bounds == expected_bounds, "Expected same bounds"


def test_isel(test_raster_path):
    """
    Test using isel method for dimension selection.

    This test verifies that the isel method correctly selects along
    dimensions (like time for stacked readers).
    """
    window, file = WINDOW_TEST, test_raster_path
    reader = rasterio_reader.RasterioReader([file], window_focus=window)
    reader.set_indexes([2, 3], relative=True)

    # Select first time slice using isel
    data = reader.isel({"time": 0}).values
    assert data.shape == (2, window.height, window.width), (
        f"Expected {(2, window.height, window.width)} found {data.shape}"
    )


# =============================================================================
# Additional tests for RasterioReader functionality
# =============================================================================


class TestRasterioReaderProperties:
    """Tests for RasterioReader properties and basic functionality."""

    def test_basic_properties(self, test_raster_path):
        """Test basic reader properties."""
        reader = rasterio_reader.RasterioReader(test_raster_path)

        assert reader.width == 250
        assert reader.height == 200
        assert reader.count == 15
        assert reader.crs is not None
        assert reader.transform is not None

    def test_bounds(self, test_raster_path):
        """Test bounds property."""
        reader = rasterio_reader.RasterioReader(test_raster_path)

        bounds = reader.bounds
        assert len(bounds) == 4
        assert bounds[2] > bounds[0]  # xmax > xmin
        assert bounds[3] > bounds[1]  # ymax > ymin

    def test_res(self, test_raster_path):
        """Test resolution property."""
        reader = rasterio_reader.RasterioReader(test_raster_path)

        res = reader.res
        assert res == (10.0, 10.0)  # Test file has 10m resolution

    def test_dtype(self, test_raster_path):
        """Test dtype property."""
        reader = rasterio_reader.RasterioReader(test_raster_path)

        assert reader.dtype is not None


class TestRasterioReaderSetWindow:
    """Tests for RasterioReader set_window method."""

    def test_set_window(self, test_raster_path):
        """Test setting a window focus."""
        reader = rasterio_reader.RasterioReader(test_raster_path)
        window = rasterio.windows.Window(col_off=50, row_off=50, width=100, height=100)

        reader.set_window(window)

        assert reader.width == 100
        assert reader.height == 100

    def test_set_window_changes_shape(self, test_raster_path):
        """Test that set_window changes the reported shape."""
        reader = rasterio_reader.RasterioReader(test_raster_path)
        original_shape = reader.shape

        window = rasterio.windows.Window(col_off=0, row_off=0, width=50, height=50)
        reader.set_window(window)

        assert reader.shape != original_shape
        assert reader.shape == (15, 50, 50)


class TestRasterioReaderLoad:
    """Tests for RasterioReader load method."""

    def test_load_full(self, test_raster_path):
        """Test loading full raster into memory."""
        reader = rasterio_reader.RasterioReader(test_raster_path)

        geotensor = reader.load()

        assert geotensor is not None
        assert geotensor.shape == reader.shape
        assert geotensor.crs == reader.crs
        assert geotensor.transform == reader.transform

    def test_load_with_window(self, test_raster_path):
        """Test loading with window focus."""
        window = rasterio.windows.Window(col_off=50, row_off=50, width=100, height=100)
        reader = rasterio_reader.RasterioReader(test_raster_path, window_focus=window)

        geotensor = reader.load()

        assert geotensor.shape == (15, 100, 100)


class TestRasterioReaderMultiFile:
    """Tests for RasterioReader with multiple files."""

    def test_single_file_as_list(self, test_raster_path):
        """Test creating reader with single file in list."""
        reader = rasterio_reader.RasterioReader([test_raster_path])

        # Should have time dimension when created with list
        assert len(reader.shape) == 4  # (time, band, height, width)
        assert reader.shape[0] == 1  # Single file = 1 time step

    def test_stacked_mode(self, test_raster_path):
        """Test stack=True mode (default for list)."""
        reader = rasterio_reader.RasterioReader([test_raster_path], stack=True)

        assert reader.shape == (1, 15, 200, 250)

    def test_non_stacked_mode(self, test_raster_path):
        """Test stack=False mode."""
        reader = rasterio_reader.RasterioReader([test_raster_path], stack=False)

        assert reader.shape == (15, 200, 250)


class TestRasterioReaderHelperFunctions:
    """Tests for helper functions in rasterio_reader module."""

    def test_get_out_shape(self):
        """Test get_out_shape function."""
        # get_out_shape(shape, size_read) takes a 2D shape and a single int size_read
        shape = (100, 100)
        size_read = 50  # single int for size_read

        result = rasterio_reader.get_out_shape(shape, size_read)

        # Returns shape scaled to fit within size_read while preserving aspect ratio
        assert result == (50, 50)

    def test_get_out_shape_rectangular(self):
        """Test get_out_shape with non-square shape."""
        shape = (200, 100)  # height > width
        size_read = 100

        result = rasterio_reader.get_out_shape(shape, size_read)

        # Should scale to (100, 50) to preserve aspect ratio
        assert result == (100, 50)

    def test_get_out_shape_smaller_than_size(self):
        """Test get_out_shape when shape is smaller than size_read."""
        shape = (50, 50)
        size_read = 100

        result = rasterio_reader.get_out_shape(shape, size_read)

        # Returns None if shape is smaller than size_read
        assert result is None

    def test_needs_boundless(self):
        """Test needs_boundless function."""
        window_data = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)

        # Window fully inside
        window_inside = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)
        assert rasterio_reader.needs_boundless(window_data, window_inside) is False

        # Window extending outside
        window_outside = rasterio.windows.Window(col_off=-10, row_off=10, width=50, height=50)
        assert rasterio_reader.needs_boundless(window_data, window_outside) is True
