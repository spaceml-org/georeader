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

from georeader import rasterio_reader, read
import rasterio
import rasterio.windows
import numpy as np


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
    assert data.shape == (1, 2, window.height, window.width), \
        f"Expected {(1, 2, window.height, window.width)} found {data.shape}"

    # Read single band by index - should return 3D array (time, height, width)
    data = reader.read(indexes=2)
    assert data.shape == (1, window.height, window.width), \
        f"Expected {(1, window.height, window.width)} found {data.shape}"

    # Compare with direct rasterio read - index 3 in original file (2+1)
    with rasterio.open(file) as src:
        data_expected = src.read(window=window, indexes=3)

    assert np.allclose(data, data_expected), "Content of the array is different"

    # Test same behavior with stack=False mode
    reader = rasterio_reader.RasterioReader([file], window_focus=window, stack=False)
    reader = reader.isel({"band": [1, 2]})  # Select bands 2 and 3 (0-indexed)
    data = reader.values

    assert data.shape == (2, window.height, window.width), \
        f"Expected {(2, window.height, window.width)} found {data.shape}"

    data = reader.read(indexes=2)
    assert data.shape == (window.height, window.width), \
        f"Expected {(window.height, window.width)} found {data.shape}"

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
    assert data.shape == (1, 64, 32), \
        f"Expected {(1, 64, 32)} found {data.shape}"


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
    assert reader_subset.shape == expected_shape, \
        f"Unexpected shape {reader_subset.shape} expected {expected_shape}"

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
    assert data.shape == (2, window.height, window.width), \
        f"Expected {(2, window.height, window.width)} found {data.shape}"
