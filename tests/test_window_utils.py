"""
Tests for the georeader.window_utils module.

These tests verify window manipulation, transform operations, and geometry utilities.
"""

import numpy as np
import pytest
import rasterio.windows
from rasterio.transform import Affine, from_origin
from shapely.geometry import MultiPolygon, Point, Polygon

from georeader import window_utils


class TestPadWindow:
    """Tests for pad_window function."""

    def test_pad_window_symmetric(self):
        """Test padding a window with symmetric padding."""
        window = rasterio.windows.Window(col_off=10, row_off=20, width=100, height=50)
        pad_size = (5, 10)  # (rows, cols)

        result = window_utils.pad_window(window, pad_size)

        assert result.col_off == 0  # 10 - 10
        assert result.row_off == 15  # 20 - 5
        assert result.width == 120  # 100 + 2*10
        assert result.height == 60  # 50 + 2*5

    def test_pad_window_zero_padding(self):
        """Test padding with zero values returns same dimensions."""
        window = rasterio.windows.Window(col_off=10, row_off=20, width=100, height=50)
        pad_size = (0, 0)

        result = window_utils.pad_window(window, pad_size)

        assert result.col_off == window.col_off
        assert result.row_off == window.row_off
        assert result.width == window.width
        assert result.height == window.height

    def test_pad_window_asymmetric(self):
        """Test padding with different row and column padding."""
        window = rasterio.windows.Window(col_off=50, row_off=50, width=100, height=100)
        pad_size = (10, 20)  # 10 rows, 20 cols

        result = window_utils.pad_window(window, pad_size)

        assert result.col_off == 30  # 50 - 20
        assert result.row_off == 40  # 50 - 10
        assert result.width == 140  # 100 + 2*20
        assert result.height == 120  # 100 + 2*10


class TestPadWindowToSize:
    """Tests for pad_window_to_size function."""

    def test_pad_to_larger_size(self):
        """Test padding window to a larger size."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)
        size = (100, 100)  # (height, width)

        result = window_utils.pad_window_to_size(window, size)

        assert result.width == 100
        assert result.height == 100
        # Check centering: should add 25 on each side
        assert result.col_off == 10 - 25  # -15
        assert result.row_off == 10 - 25  # -15

    def test_pad_to_same_size(self):
        """Test padding to same size returns equivalent window."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)
        size = (50, 50)

        result = window_utils.pad_window_to_size(window, size)

        assert result.width == window.width
        assert result.height == window.height
        assert result.col_off == window.col_off
        assert result.row_off == window.row_off

    def test_pad_to_smaller_size(self):
        """Test padding to smaller size (crops center)."""
        window = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)
        size = (50, 50)

        result = window_utils.pad_window_to_size(window, size)

        assert result.width == 50
        assert result.height == 50
        # Should be centered: (100-50)/2 = 25
        assert result.col_off == 25
        assert result.row_off == 25


class TestFigureOutTransform:
    """Tests for figure_out_transform function."""

    def test_with_transform_only(self):
        """Test with only transform provided."""
        transform = from_origin(0, 100, 10, 10)

        result = window_utils.figure_out_transform(transform=transform)

        assert result == transform

    def test_with_bounds_and_resolution(self):
        """Test computing transform from bounds and resolution."""
        bounds = (0, 0, 100, 100)  # xmin, ymin, xmax, ymax
        resolution = 10.0

        result = window_utils.figure_out_transform(bounds=bounds, resolution_dst=resolution)

        # Should create transform at top-left corner with given resolution
        assert result.c == 0  # x origin
        assert result.f == 100  # y origin (max y)
        assert abs(result.a) == 10  # x resolution
        assert abs(result.e) == 10  # y resolution

    def test_with_transform_and_resolution(self):
        """Test changing resolution of existing transform."""
        transform = from_origin(0, 100, 10, 10)
        resolution_dst = 20.0

        result = window_utils.figure_out_transform(transform=transform, resolution_dst=resolution_dst)

        # Resolution should be changed
        res_result = window_utils.res(result)
        assert abs(res_result[0] - 20.0) < 0.001
        assert abs(res_result[1] - 20.0) < 0.001

    def test_with_transform_bounds_resolution(self):
        """Test with all parameters provided."""
        transform = from_origin(0, 100, 10, 10)
        bounds = (50, 0, 150, 100)
        resolution_dst = 5.0

        result = window_utils.figure_out_transform(transform=transform, bounds=bounds, resolution_dst=resolution_dst)

        # Resolution should match requested
        res_result = window_utils.res(result)
        assert abs(res_result[0] - 5.0) < 0.001

    def test_with_tuple_resolution(self):
        """Test with tuple resolution (different x and y)."""
        bounds = (0, 0, 100, 200)
        resolution_dst = (10, 20)

        result = window_utils.figure_out_transform(bounds=bounds, resolution_dst=resolution_dst)

        res_result = window_utils.res(result)
        assert abs(res_result[0] - 10) < 0.001
        assert abs(res_result[1] - 20) < 0.001


class TestTransformToResolutionDst:
    """Tests for transform_to_resolution_dst function."""

    def test_double_resolution(self):
        """Test doubling the resolution."""
        transform = from_origin(0, 100, 10, 10)

        result = window_utils.transform_to_resolution_dst(transform, 20.0)

        res_result = window_utils.res(result)
        assert abs(res_result[0] - 20.0) < 0.001
        assert abs(res_result[1] - 20.0) < 0.001

    def test_half_resolution(self):
        """Test halving the resolution."""
        transform = from_origin(0, 100, 10, 10)

        result = window_utils.transform_to_resolution_dst(transform, 5.0)

        res_result = window_utils.res(result)
        assert abs(res_result[0] - 5.0) < 0.001
        assert abs(res_result[1] - 5.0) < 0.001

    def test_tuple_resolution(self):
        """Test with different x and y resolutions."""
        transform = from_origin(0, 100, 10, 10)

        result = window_utils.transform_to_resolution_dst(transform, (15, 25))

        res_result = window_utils.res(result)
        assert abs(res_result[0] - 15.0) < 0.001
        assert abs(res_result[1] - 25.0) < 0.001


class TestRoundOuterWindow:
    """Tests for round_outer_window function."""

    def test_round_fractional_window(self):
        """Test rounding a window with fractional values."""
        window = rasterio.windows.Window(col_off=10.3, row_off=20.7, width=50.2, height=30.8)

        result = window_utils.round_outer_window(window)

        # Should floor offsets and ceil extents
        assert result.col_off == 10
        assert result.row_off == 20
        assert result.width == 51  # ceil(10.3 + 50.2) - floor(10.3) = 61 - 10 = 51
        assert result.height == 32  # ceil(20.7 + 30.8) - floor(20.7) = 52 - 20 = 32

    def test_round_integer_window(self):
        """Test rounding an already-integer window."""
        window = rasterio.windows.Window(col_off=10, row_off=20, width=50, height=30)

        result = window_utils.round_outer_window(window)

        assert result.col_off == 10
        assert result.row_off == 20
        assert result.width == 50
        assert result.height == 30


class TestGetSlicePad:
    """Tests for get_slice_pad function."""

    def test_window_fully_inside(self):
        """Test when read window is fully inside data window."""
        window_data = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)
        window_read = rasterio.windows.Window(col_off=10, row_off=10, width=30, height=30)

        slice_dict, pad_width = window_utils.get_slice_pad(window_data, window_read)

        assert slice_dict["x"] == slice(10, 40)
        assert slice_dict["y"] == slice(10, 40)
        assert pad_width["x"] == (0, 0)
        assert pad_width["y"] == (0, 0)

    def test_window_extends_left(self):
        """Test when read window extends past left edge."""
        window_data = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)
        window_read = rasterio.windows.Window(col_off=-10, row_off=10, width=30, height=30)

        slice_dict, pad_width = window_utils.get_slice_pad(window_data, window_read)

        assert pad_width["x"][0] == 10  # Padding on left
        assert pad_width["x"][1] == 0  # No padding on right

    def test_window_extends_right(self):
        """Test when read window extends past right edge."""
        window_data = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)
        window_read = rasterio.windows.Window(col_off=80, row_off=10, width=30, height=30)

        slice_dict, pad_width = window_utils.get_slice_pad(window_data, window_read)

        assert pad_width["x"][0] == 0  # No padding on left
        assert pad_width["x"][1] == 10  # Padding on right

    def test_non_intersecting_windows_raises(self):
        """Test that non-intersecting windows raise an error."""
        window_data = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)
        window_read = rasterio.windows.Window(col_off=200, row_off=200, width=30, height=30)

        with pytest.raises(rasterio.windows.WindowError):
            window_utils.get_slice_pad(window_data, window_read)


class TestWindowPolygon:
    """Tests for window_polygon function."""

    def test_basic_window_polygon(self):
        """Test creating polygon from window."""
        window = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=50)
        transform = from_origin(0, 100, 10, 10)  # 10m resolution, origin at (0, 100)

        result = window_utils.window_polygon(window, transform)

        assert isinstance(result, Polygon)
        bounds = result.bounds
        # Origin is at (0, 100), resolution is 10m
        # x goes from 0 to 100*10 = 1000
        # y goes from 100 to 100 - 50*10 = -400 (negative y direction)
        assert bounds[0] == 0  # xmin
        assert bounds[1] == -400  # ymin (100 - 50*10)
        assert bounds[2] == 1000  # xmax (100 * 10)
        assert bounds[3] == 100  # ymax (origin y)

    def test_window_polygon_with_offset(self):
        """Test polygon from offset window."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)
        transform = from_origin(0, 100, 1, 1)

        result = window_utils.window_polygon(window, transform)

        assert isinstance(result, Polygon)
        # Check it's a valid polygon
        assert result.is_valid


class TestWindowBounds:
    """Tests for window_bounds function."""

    def test_basic_window_bounds(self):
        """Test computing bounds from window."""
        window = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=50)
        transform = from_origin(0, 100, 10, 10)

        xmin, ymin, xmax, ymax = window_utils.window_bounds(window, transform)

        assert xmin == 0
        assert xmax == 1000  # 100 * 10
        assert ymax == 100
        assert ymin == 100 - 500  # 100 - 50*10 = -400

    def test_window_bounds_with_offset(self):
        """Test bounds from offset window."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)
        transform = from_origin(0, 100, 1, 1)

        xmin, ymin, xmax, ymax = window_utils.window_bounds(window, transform)

        assert xmin == 10
        assert xmax == 60
        assert ymax == 90  # 100 - 10
        assert ymin == 40  # 100 - 60


class TestNormalizeBounds:
    """Tests for normalize_bounds function."""

    def test_already_normalized(self):
        """Test bounds that are already normalized."""
        bounds = (0, 0, 100, 100)

        result = window_utils.normalize_bounds(bounds)

        assert result == (0, 0, 100, 100)

    def test_swapped_bounds(self):
        """Test bounds with swapped min/max."""
        bounds = (100, 100, 0, 0)

        result = window_utils.normalize_bounds(bounds)

        assert result[0] < result[2]  # xmin < xmax
        assert result[1] < result[3]  # ymin < ymax

    def test_equal_x_bounds(self):
        """Test bounds with equal x values get margin added."""
        bounds = (50, 0, 50, 100)

        result = window_utils.normalize_bounds(bounds)

        assert result[0] < result[2]  # xmin < xmax after margin


class TestPolygonToCrs:
    """Tests for polygon_to_crs function."""

    def test_same_crs_returns_same(self):
        """Test that same CRS returns the same polygon."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        result = window_utils.polygon_to_crs(polygon, "EPSG:4326", "EPSG:4326")

        assert result.equals(polygon)

    def test_transform_to_different_crs(self):
        """Test transforming polygon to different CRS."""
        # Create a polygon in WGS84
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        result = window_utils.polygon_to_crs(polygon, "EPSG:4326", "EPSG:32631")

        # Result should be valid but with different coordinates
        assert isinstance(result, (Polygon, MultiPolygon))
        assert result.is_valid


class TestExteriorPixelCoords:
    """Tests for exterior_pixel_coords function."""

    def test_basic_polygon(self):
        """Test getting pixel coordinates of polygon exterior."""
        transform = from_origin(0, 100, 10, 10)
        polygon = Polygon([(0, 100), (100, 100), (100, 0), (0, 0), (0, 100)])

        result = window_utils.exterior_pixel_coords(transform, "EPSG:4326", polygon)

        assert isinstance(result, list)
        assert len(result) == 1  # Single polygon
        assert len(result[0]) == 5  # 5 vertices (closed ring)

    def test_multipolygon(self):
        """Test getting pixel coordinates of multipolygon."""
        transform = from_origin(0, 100, 10, 10)
        poly1 = Polygon([(0, 100), (50, 100), (50, 50), (0, 50), (0, 100)])
        poly2 = Polygon([(50, 50), (100, 50), (100, 0), (50, 0), (50, 50)])
        multipolygon = MultiPolygon([poly1, poly2])

        result = window_utils.exterior_pixel_coords(transform, "EPSG:4326", multipolygon)

        assert len(result) == 2  # Two polygons


class TestRowColEnd:
    """Tests for row_end and col_end functions."""

    def test_row_end(self):
        """Test row_end computation."""
        window = rasterio.windows.Window(col_off=10, row_off=20, width=100, height=50)

        assert window_utils.row_end(window) == 70  # 20 + 50

    def test_col_end(self):
        """Test col_end computation."""
        window = rasterio.windows.Window(col_off=10, row_off=20, width=100, height=50)

        assert window_utils.col_end(window) == 110  # 10 + 100


class TestSliceSaveForPred:
    """Tests for slice_save_for_pred function."""

    def test_centered_padding(self):
        """Test slice computation for centered prediction window."""
        w_read = rasterio.windows.Window(col_off=0, row_off=0, width=128, height=128)
        w_write = rasterio.windows.Window(col_off=16, row_off=16, width=96, height=96)

        row_slice, col_slice = window_utils.slice_save_for_pred(w_read, w_write)

        assert row_slice.start == 16
        assert col_slice.start == 16

    def test_edge_window(self):
        """Test slice computation at edge."""
        w_read = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)
        w_write = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)

        row_slice, col_slice = window_utils.slice_save_for_pred(w_read, w_write)

        assert row_slice.start == 0
        assert col_slice.start == 0


class TestPadListNumpy:
    """Tests for pad_list_numpy function."""

    def test_basic_conversion(self):
        """Test converting pad dict to numpy format."""
        pad_width = {"x": (5, 10), "y": (3, 7)}

        result = window_utils.pad_list_numpy(pad_width)

        assert result == [(3, 7), (5, 10)]  # y first, then x

    def test_missing_dimension(self):
        """Test with missing dimension."""
        pad_width = {"x": (5, 10)}

        result = window_utils.pad_list_numpy(pad_width)

        assert result == [(0, 0), (5, 10)]  # y defaults to (0, 0)

    def test_empty_dict(self):
        """Test with empty dict."""
        pad_width = {}

        result = window_utils.pad_list_numpy(pad_width)

        assert result == [(0, 0), (0, 0)]


# =============================================================================
# Tests for window_utils error handling (Phase 2 Sprint 1)
# =============================================================================


class TestFigureOutTransformErrors:
    """Tests for figure_out_transform function error handling."""

    def test_no_transform_no_bounds_raises(self):
        """Test that providing no transform and no bounds raises AssertionError."""
        with pytest.raises(AssertionError, match="Transform and bounds not provided"):
            window_utils.figure_out_transform(resolution_dst=10.0)

    def test_no_transform_no_resolution_raises(self):
        """Test that providing no transform and no resolution raises AssertionError."""
        bounds = (0, 0, 100, 100)
        with pytest.raises(AssertionError, match="Transform and resolution not provided"):
            window_utils.figure_out_transform(bounds=bounds)

    def test_all_none_raises(self):
        """Test that providing all None parameters raises AssertionError."""
        with pytest.raises(AssertionError):
            window_utils.figure_out_transform()


class TestGetSlicePadErrors:
    """Tests for get_slice_pad function error handling."""

    def test_completely_disjoint_windows_raises(self):
        """Test that completely disjoint windows raise WindowError."""
        window_data = rasterio.windows.Window(col_off=0, row_off=0, width=100, height=100)
        window_read = rasterio.windows.Window(col_off=500, row_off=500, width=50, height=50)

        with pytest.raises(rasterio.windows.WindowError):
            window_utils.get_slice_pad(window_data, window_read)

    def test_negative_disjoint_windows_raises(self):
        """Test that negatively disjoint windows raise WindowError."""
        window_data = rasterio.windows.Window(col_off=100, row_off=100, width=100, height=100)
        window_read = rasterio.windows.Window(col_off=0, row_off=0, width=50, height=50)

        with pytest.raises(rasterio.windows.WindowError):
            window_utils.get_slice_pad(window_data, window_read)


class TestExteriorPixelCoordsErrors:
    """Tests for exterior_pixel_coords function error handling."""

    def test_invalid_geometry_type_raises(self):
        """Test that invalid geometry type raises NotImplementedError."""
        transform = from_origin(0, 100, 10, 10)
        point = Point(50, 50)  # Point is not Polygon or MultiPolygon

        with pytest.raises(NotImplementedError, match="different from"):
            window_utils.exterior_pixel_coords(transform, "EPSG:4326", point)


class TestNormalizeBoundsEdgeCases:
    """Tests for normalize_bounds edge cases and error handling."""

    def test_inf_values(self):
        """Test handling of infinite values in bounds."""
        bounds = (0, 0, float("inf"), 100)

        # Function should handle inf values (may produce inf in output)
        result = window_utils.normalize_bounds(bounds)
        assert result[2] == float("inf")

    def test_negative_inf_values(self):
        """Test handling of negative infinite values in bounds."""
        bounds = (float("-inf"), 0, 100, 100)

        result = window_utils.normalize_bounds(bounds)
        assert result[0] == float("-inf")

    def test_nan_values(self):
        """Test handling of NaN values in bounds."""
        bounds = (0, 0, float("nan"), 100)

        # NaN comparison results in margin being added since min(0, nan) returns nan
        # which causes xmin >= xmax to be False (nan comparisons are always False)
        # so margin is added, leading to (-0.0005, ..., 0.0005, ...)
        result = window_utils.normalize_bounds(bounds)
        # Result depends on NaN comparison behavior in min/max
        assert len(result) == 4  # Function completes without error


class TestPadWindowEdgeCases:
    """Tests for pad_window edge cases."""

    def test_negative_padding_produces_smaller_window(self):
        """Test that negative padding produces a smaller window."""
        window = rasterio.windows.Window(col_off=50, row_off=50, width=100, height=100)
        pad_size = (-10, -20)  # Negative padding

        result = window_utils.pad_window(window, pad_size)

        # Negative padding shrinks the window
        assert result.width == 100 + 2 * (-20)  # 60
        assert result.height == 100 + 2 * (-10)  # 80
        assert result.col_off == 50 - (-20)  # 70
        assert result.row_off == 50 - (-10)  # 60

    def test_large_negative_padding_raises_on_negative_dimensions(self):
        """Test that very large negative padding raises ValueError from rasterio."""
        window = rasterio.windows.Window(col_off=50, row_off=50, width=100, height=100)
        pad_size = (-60, -60)  # Padding larger than half dimensions

        # rasterio.windows.Window validates that width/height must be non-negative
        with pytest.raises(ValueError, match="non-negative"):
            window_utils.pad_window(window, pad_size)


class TestPadWindowToSizeEdgeCases:
    """Tests for pad_window_to_size edge cases."""

    def test_zero_size(self):
        """Test padding to zero size."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)
        size = (0, 0)

        result = window_utils.pad_window_to_size(window, size)

        assert result.width == 0
        assert result.height == 0

    def test_negative_size_raises(self):
        """Test padding to negative size raises ValueError from rasterio."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)
        size = (-10, -10)

        # rasterio.windows.Window validates that width/height must be non-negative
        with pytest.raises(ValueError, match="non-negative"):
            window_utils.pad_window_to_size(window, size)


class TestRoundOuterWindowEdgeCases:
    """Tests for round_outer_window edge cases."""

    def test_zero_dimensions(self):
        """Test rounding window with zero dimensions."""
        window = rasterio.windows.Window(col_off=10, row_off=20, width=0, height=0)

        result = window_utils.round_outer_window(window)

        assert result.width == 0
        assert result.height == 0

    def test_very_small_fractional(self):
        """Test rounding window with very small fractional values."""
        window = rasterio.windows.Window(col_off=10.0001, row_off=20.0001, width=50.0001, height=30.0001)

        result = window_utils.round_outer_window(window)

        # Very small fractions should be rounded to integers
        assert result.col_off == 10
        assert result.row_off == 20
        assert result.width == 50
        assert result.height == 30

    def test_negative_offsets(self):
        """Test rounding window with negative offsets."""
        window = rasterio.windows.Window(col_off=-10.5, row_off=-20.5, width=50.5, height=30.5)

        result = window_utils.round_outer_window(window)

        assert result.col_off == -11  # floor(-10.5)
        assert result.row_off == -21  # floor(-20.5)


class TestPolygonToCrsEdgeCases:
    """Tests for polygon_to_crs edge cases."""

    def test_empty_polygon(self):
        """Test transforming empty polygon."""
        polygon = Polygon()  # Empty polygon

        result = window_utils.polygon_to_crs(polygon, "EPSG:4326", "EPSG:4326")

        assert result.is_empty


class TestWindowFromBoundsEdgeCases:
    """Tests for window operations with unusual bounds."""

    def test_window_polygon_zero_dimensions(self):
        """Test window_polygon with zero-dimension window."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=0, height=0)
        transform = from_origin(0, 100, 1, 1)

        result = window_utils.window_polygon(window, transform)

        # Zero-dimension window produces a degenerate polygon (all vertices at same point)
        # This creates an invalid polygon by Shapely standards
        assert isinstance(result, Polygon)
        # The polygon has area 0 since width and height are 0
        assert result.area == 0

    def test_window_bounds_negative_offset(self):
        """Test window_bounds with negative offsets."""
        window = rasterio.windows.Window(col_off=-10, row_off=-10, width=50, height=50)
        transform = from_origin(0, 100, 1, 1)

        xmin, ymin, xmax, ymax = window_utils.window_bounds(window, transform)

        # Should handle negative offsets correctly
        assert xmin == -10
        assert xmax == 40
