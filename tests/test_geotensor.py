"""
Tests for the georeader.geotensor.GeoTensor class and related operations.

These tests verify that GeoTensor objects correctly handle window-based reading,
maintain proper geospatial metadata (bounds, transform), and produce consistent
results with RasterioReader.

Uses a temporary GeoTiff test file created via the test_raster_path fixture.
"""

from georeader import rasterio_reader, geotensor, read
import rasterio.windows
import numpy as np
import itertools


# Initial window to focus on (within the test image bounds: height=200, width=250)
WINDOW_INITIAL = rasterio.windows.Window(col_off=50, row_off=50, width=100, height=100)

# Sub-windows for testing (relative to the loaded GeoTensor's extent)
WINDOW_NORMAL = rasterio.windows.Window(col_off=10, row_off=5, width=30, height=20)
WINDOW_OUT_1 = rasterio.windows.Window(col_off=-10, row_off=5, width=30, height=20)  # Out left
WINDOW_OUT_2 = rasterio.windows.Window(col_off=1, row_off=-5, width=30, height=20)  # Out top
WINDOW_OUT_3 = rasterio.windows.Window(col_off=80, row_off=5, width=30, height=20)  # Out right
WINDOW_OUT_4 = rasterio.windows.Window(col_off=1, row_off=90, width=30, height=20)  # Out bottom


def test_read_window(test_raster_path):
    """
    Test reading from windows with GeoTensor and RasterioReader objects.

    This test verifies that:
    1. GeoTensor objects correctly maintain shape after loading
    2. The width, height, and count attributes are correctly computed
    3. Reading from windows on GeoTensor produces correct shapes and transforms
    4. Boundless and non-boundless reads produce correct shapes and metadata
    5. Multiple reads from the same GeoTensor produce identical results
    """
    window = WINDOW_INITIAL
    rst_obj = rasterio_reader.RasterioReader(test_raster_path, window_focus=window)
    # C, H, W = "band", "y", "x"
    gtobj = rst_obj.load()

    # Verify shapes match (15 bands from the test file)
    assert rst_obj.shape == (15, window.height, window.width), \
        f"Unexpected RasterioReader shape {rst_obj.shape} expected {(15, window.height, window.width)}"
    assert gtobj.shape == (15, window.height, window.width), \
        f"Unexpected GeoTensor shape {gtobj.shape} expected {(15, window.height, window.width)}"

    # Verify basic properties match between reader and tensor
    assert rst_obj.width == gtobj.width, f"Unexpected width {rst_obj.width} {gtobj.width}"
    assert rst_obj.count == gtobj.count, f"Unexpected count {rst_obj.count} {gtobj.count}"
    assert rst_obj.height == gtobj.height, f"Unexpected height {rst_obj.height} {gtobj.height}"

    # Test reading from various sub-windows with both boundless modes
    for subwindow, boundless in itertools.product(
        [WINDOW_NORMAL, WINDOW_OUT_1, WINDOW_OUT_2, WINDOW_OUT_3, WINDOW_OUT_4],
        [True, False]
    ):
        # Read from GeoTensor using read_from_window (two separate calls)
        gtobj_isel_1 = read.read_from_window(gtobj, window=subwindow, boundless=boundless)
        gtobj_isel_2 = read.read_from_window(gtobj, window=subwindow, boundless=boundless)

        # Skip if window doesn't intersect
        if gtobj_isel_1 is None:
            assert gtobj_isel_2 is None, "Inconsistent None results"
            continue

        # Verify shapes match between the two reads
        assert gtobj_isel_1.shape == gtobj_isel_2.shape, \
            f"Different shapes {subwindow} {boundless}"

        if boundless:
            # For boundless reads, verify output has expected shape
            assert gtobj_isel_1.shape[-2:] == (subwindow.height, subwindow.width), \
                f"Unexpected shape {gtobj_isel_1.shape} for window {subwindow}"

        # Verify transforms match
        assert gtobj_isel_1.transform == gtobj_isel_2.transform, \
            f"Different transforms {subwindow} {boundless}"
        assert gtobj_isel_1.bounds == gtobj_isel_2.bounds, \
            f"Different bounds {subwindow} {boundless}"

        # Verify data values match between the two reads
        assert np.allclose(gtobj_isel_1.values, gtobj_isel_2.values), \
            f"Content of the array is different {subwindow} {boundless}"


def test_geotensor_properties(test_raster_path):
    """
    Test that GeoTensor correctly exposes geospatial properties.

    This test verifies that the GeoTensor object correctly computes and exposes
    its geospatial properties (bounds, transform, crs, resolution) after loading.
    """
    window = WINDOW_INITIAL
    rst_obj = rasterio_reader.RasterioReader(test_raster_path, window_focus=window)
    gtobj = rst_obj.load()

    # Verify properties match between reader and tensor
    assert rst_obj.bounds == gtobj.bounds, \
        f"Bounds mismatch: {rst_obj.bounds} vs {gtobj.bounds}"
    assert rst_obj.transform == gtobj.transform, \
        f"Transform mismatch: {rst_obj.transform} vs {gtobj.transform}"
    assert rst_obj.crs == gtobj.crs, \
        f"CRS mismatch: {rst_obj.crs} vs {gtobj.crs}"
    assert rst_obj.res == gtobj.res, \
        f"Resolution mismatch: {rst_obj.res} vs {gtobj.res}"
