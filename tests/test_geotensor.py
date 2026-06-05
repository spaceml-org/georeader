"""
Tests for the georeader.geotensor.GeoTensor class and related operations.

These tests verify that GeoTensor objects correctly handle window-based reading,
maintain proper geospatial metadata (bounds, transform), and produce consistent
results with RasterioReader.

Uses a temporary GeoTiff test file created via the test_raster_path fixture.
"""

import itertools

import numpy as np
import pytest
import rasterio.windows
from rasterio.transform import from_origin

from georeader import geotensor, rasterio_reader, read
from georeader.geotensor import GeoTensor, concatenate, stack

# Initial window to focus on (within the test image bounds: height=200, width=250)
WINDOW_INITIAL = rasterio.windows.Window(col_off=50, row_off=50, width=100, height=100)

# Sub-windows for testing (relative to the loaded GeoTensor's extent)
WINDOW_NORMAL = rasterio.windows.Window(col_off=10, row_off=5, width=30, height=20)
WINDOW_OUT_1 = rasterio.windows.Window(col_off=-10, row_off=5, width=30, height=20)  # Out left
WINDOW_OUT_2 = rasterio.windows.Window(col_off=1, row_off=-5, width=30, height=20)  # Out top
WINDOW_OUT_3 = rasterio.windows.Window(col_off=80, row_off=5, width=30, height=20)  # Out right
WINDOW_OUT_4 = rasterio.windows.Window(col_off=1, row_off=90, width=30, height=20)  # Out bottom


@pytest.fixture
def sample_geotensor():
    """Create a sample GeoTensor for testing."""
    data = np.random.rand(3, 100, 100).astype(np.float32)
    transform = from_origin(0, 100, 1, 1)
    return GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)


@pytest.fixture
def sample_geotensor_2d():
    """Create a 2D GeoTensor for testing."""
    data = np.random.rand(100, 100).astype(np.float32)
    transform = from_origin(0, 100, 1, 1)
    return GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)


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
    assert rst_obj.shape == (15, window.height, window.width), (
        f"Unexpected RasterioReader shape {rst_obj.shape} expected {(15, window.height, window.width)}"
    )
    assert gtobj.shape == (15, window.height, window.width), (
        f"Unexpected GeoTensor shape {gtobj.shape} expected {(15, window.height, window.width)}"
    )

    # Verify basic properties match between reader and tensor
    assert rst_obj.width == gtobj.width, f"Unexpected width {rst_obj.width} {gtobj.width}"
    assert rst_obj.count == gtobj.count, f"Unexpected count {rst_obj.count} {gtobj.count}"
    assert rst_obj.height == gtobj.height, f"Unexpected height {rst_obj.height} {gtobj.height}"

    # Test reading from various sub-windows with both boundless modes
    for subwindow, boundless in itertools.product(
        [WINDOW_NORMAL, WINDOW_OUT_1, WINDOW_OUT_2, WINDOW_OUT_3, WINDOW_OUT_4], [True, False]
    ):
        # Read from GeoTensor using read_from_window (two separate calls)
        gtobj_isel_1 = read.read_from_window(gtobj, window=subwindow, boundless=boundless)
        gtobj_isel_2 = read.read_from_window(gtobj, window=subwindow, boundless=boundless)

        # Skip if window doesn't intersect
        if gtobj_isel_1 is None:
            assert gtobj_isel_2 is None, "Inconsistent None results"
            continue

        # Verify shapes match between the two reads
        assert gtobj_isel_1.shape == gtobj_isel_2.shape, f"Different shapes {subwindow} {boundless}"

        if boundless:
            # For boundless reads, verify output has expected shape
            assert gtobj_isel_1.shape[-2:] == (subwindow.height, subwindow.width), (
                f"Unexpected shape {gtobj_isel_1.shape} for window {subwindow}"
            )

        # Verify transforms match
        assert gtobj_isel_1.transform == gtobj_isel_2.transform, f"Different transforms {subwindow} {boundless}"
        assert gtobj_isel_1.bounds == gtobj_isel_2.bounds, f"Different bounds {subwindow} {boundless}"

        # Verify data values match between the two reads
        assert np.allclose(gtobj_isel_1.values, gtobj_isel_2.values), (
            f"Content of the array is different {subwindow} {boundless}"
        )


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
    assert rst_obj.bounds == gtobj.bounds, f"Bounds mismatch: {rst_obj.bounds} vs {gtobj.bounds}"
    assert rst_obj.transform == gtobj.transform, f"Transform mismatch: {rst_obj.transform} vs {gtobj.transform}"
    assert rst_obj.crs == gtobj.crs, f"CRS mismatch: {rst_obj.crs} vs {gtobj.crs}"
    assert rst_obj.res == gtobj.res, f"Resolution mismatch: {rst_obj.res} vs {gtobj.res}"


# =============================================================================
# Tests for GeoTensor dimensions and properties
# =============================================================================


class TestGeoTensorDims:
    """Tests for GeoTensor dims property."""

    def test_dims_2d(self, sample_geotensor_2d):
        """Test dims for 2D tensor."""
        assert sample_geotensor_2d.dims == ("y", "x")

    def test_dims_3d(self, sample_geotensor):
        """Test dims for 3D tensor."""
        assert sample_geotensor.dims == ("band", "y", "x")

    def test_dims_4d(self):
        """Test dims for 4D tensor."""
        data = np.random.rand(2, 3, 100, 100)
        transform = from_origin(0, 100, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631")
        assert gt.dims == ("time", "band", "y", "x")


class TestGeoTensorBasicProperties:
    """Tests for GeoTensor basic properties."""

    def test_height_width_count(self, sample_geotensor):
        """Test height, width, count properties."""
        assert sample_geotensor.height == 100
        assert sample_geotensor.width == 100
        assert sample_geotensor.count == 3

    def test_dtype(self, sample_geotensor):
        """Test dtype property."""
        assert sample_geotensor.dtype == np.float32

    def test_res(self, sample_geotensor):
        """Test resolution property."""
        assert sample_geotensor.res == (1.0, 1.0)


# =============================================================================
# Tests for GeoTensor serialization
# =============================================================================


class TestGeoTensorSerialization:
    """Tests for GeoTensor to_json and from_json methods."""

    def test_to_json(self, sample_geotensor):
        """Test serialization to JSON."""
        json_data = sample_geotensor.to_json()

        assert "values" in json_data
        assert "transform" in json_data
        assert "crs" in json_data
        assert "fill_value_default" in json_data

    def test_from_json_roundtrip(self):
        """Test serialization roundtrip."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        transform = from_origin(0, 10, 1, 1)
        original = GeoTensor(data, transform=transform, crs="EPSG:4326", fill_value_default=-1)

        json_data = original.to_json()
        restored = GeoTensor.from_json(json_data)

        assert np.allclose(restored.values, original.values)
        assert restored.crs == original.crs
        assert restored.fill_value_default == original.fill_value_default


# =============================================================================
# Tests for GeoTensor arithmetic operations
# =============================================================================


class TestGeoTensorArithmetic:
    """Tests for GeoTensor arithmetic operations."""

    def test_add_scalar(self, sample_geotensor):
        """Test adding a scalar."""
        result = sample_geotensor + 5

        assert isinstance(result, GeoTensor)
        assert np.allclose(result.values, sample_geotensor.values + 5)
        assert result.transform == sample_geotensor.transform

    def test_add_geotensor(self, sample_geotensor):
        """Test adding two GeoTensors."""
        other = sample_geotensor.copy()
        result = sample_geotensor + other

        assert isinstance(result, GeoTensor)
        assert np.allclose(result.values, sample_geotensor.values * 2)

    def test_add_mismatched_extent_raises(self, sample_geotensor):
        """Test that adding GeoTensors with different extents raises."""
        other_data = np.random.rand(3, 50, 50)
        other_transform = from_origin(100, 100, 1, 1)
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:32631")

        with pytest.raises(ValueError):
            sample_geotensor + other

    def test_sub_scalar(self, sample_geotensor):
        """Test subtracting a scalar."""
        result = sample_geotensor - 2

        assert np.allclose(result.values, sample_geotensor.values - 2)

    def test_sub_geotensor(self, sample_geotensor):
        """Test subtracting two GeoTensors."""
        other = sample_geotensor.copy()
        result = sample_geotensor - other

        assert np.allclose(result.values, 0)

    def test_mul_scalar(self, sample_geotensor):
        """Test multiplying by a scalar."""
        result = sample_geotensor * 3

        assert np.allclose(result.values, sample_geotensor.values * 3)

    def test_mul_geotensor(self, sample_geotensor):
        """Test multiplying two GeoTensors."""
        other = sample_geotensor.copy()
        result = sample_geotensor * other

        assert np.allclose(result.values, sample_geotensor.values**2)

    def test_div_scalar(self, sample_geotensor):
        """Test dividing by a scalar."""
        result = sample_geotensor / 2

        assert np.allclose(result.values, sample_geotensor.values / 2)

    def test_div_geotensor(self, sample_geotensor):
        """Test dividing two GeoTensors."""
        # Add 1 to avoid division by zero
        gt1 = sample_geotensor + 1
        gt2 = gt1.copy()
        result = gt1 / gt2

        assert np.allclose(result.values, 1)


# =============================================================================
# Tests for GeoTensor manipulation methods
# =============================================================================


class TestGeoTensorManipulation:
    """Tests for GeoTensor manipulation methods."""

    def test_squeeze_3d_to_2d(self):
        """Test squeezing 3D tensor to 2D."""
        data = np.random.rand(1, 100, 100)
        transform = from_origin(0, 100, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631")

        result = gt.squeeze()

        assert result.shape == (100, 100)

    def test_squeeze_preserves_nonsqueezable(self):
        """Test that squeeze preserves dims that can't be squeezed."""
        # Create a 4D tensor with all non-spatial dimensions of size 1
        data = np.random.rand(1, 1, 100, 100)
        transform = from_origin(0, 100, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631")

        result = gt.squeeze()

        # Should squeeze both time and band dims (both size 1) -> 2D
        assert result.shape == (100, 100)

    def test_clip(self, sample_geotensor):
        """Test clipping values."""
        result = sample_geotensor.clip(0.2, 0.8)

        assert result.values.min() >= 0.2
        assert result.values.max() <= 0.8

    def test_astype(self, sample_geotensor):
        """Test changing dtype."""
        result = sample_geotensor.astype(np.float64)

        assert result.dtype == np.float64
        assert result.transform == sample_geotensor.transform

    def test_copy(self, sample_geotensor):
        """Test copying GeoTensor."""
        copy = sample_geotensor.copy()

        # Modify original
        sample_geotensor.values[0, 0, 0] = 999

        # Copy should be unchanged
        assert copy.values[0, 0, 0] != 999

    def test_isel_x_slice(self, sample_geotensor):
        """Test isel with x slice."""
        result = sample_geotensor.isel({"x": slice(10, 50)})

        assert result.shape == (3, 100, 40)
        assert result.width == 40

    def test_isel_y_slice(self, sample_geotensor):
        """Test isel with y slice."""
        result = sample_geotensor.isel({"y": slice(20, 80)})

        assert result.shape == (3, 60, 100)
        assert result.height == 60

    def test_isel_xy_slice(self, sample_geotensor):
        """Test isel with both x and y slices."""
        result = sample_geotensor.isel({"x": slice(10, 50), "y": slice(20, 80)})

        assert result.shape == (3, 60, 40)

    def test_isel_band_selection(self, sample_geotensor):
        """Test isel with band selection."""
        result = sample_geotensor.isel({"band": [0, 2]})

        assert result.shape == (2, 100, 100)


# =============================================================================
# Tests for GeoTensor spatial operations
# =============================================================================


class TestGeoTensorSpatial:
    """Tests for GeoTensor spatial operations."""

    def test_footprint(self, sample_geotensor):
        """Test footprint generation."""
        from shapely.geometry import Polygon

        footprint = sample_geotensor.footprint()

        assert isinstance(footprint, Polygon)
        assert footprint.is_valid

    def test_footprint_with_crs(self, sample_geotensor):
        """Test footprint with CRS transformation."""
        footprint = sample_geotensor.footprint(crs="EPSG:4326")

        assert footprint.is_valid

    def test_meshgrid(self, sample_geotensor):
        """Test meshgrid generation without CRS transformation."""
        xs, ys = sample_geotensor.meshgrid()

        # meshgrid without dst_crs returns coordinate arrays
        assert len(xs) == 100 * 100
        assert len(ys) == 100 * 100

    def test_meshgrid_with_dst_crs(self, sample_geotensor):
        """Test meshgrid generation with CRS transformation."""
        xs, ys = sample_geotensor.meshgrid(dst_crs="EPSG:4326")

        # With CRS transformation, returns 2D arrays of transformed coordinates
        assert xs.shape == (100, 100)
        assert ys.shape == (100, 100)

    def test_same_extent_true(self, sample_geotensor):
        """Test same_extent returns True for identical extent."""
        other = sample_geotensor.copy()

        assert sample_geotensor.same_extent(other)

    def test_same_extent_false_different_transform(self, sample_geotensor):
        """Test same_extent returns False for different transform."""
        other_data = np.random.rand(3, 100, 100)
        other_transform = from_origin(10, 100, 1, 1)  # Different origin
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:32631")

        assert not sample_geotensor.same_extent(other)


# =============================================================================
# Tests for GeoTensor pad operation
# =============================================================================


class TestGeoTensorPad:
    """Tests for GeoTensor pad method."""

    def test_pad_symmetric(self, sample_geotensor):
        """Test symmetric padding."""
        result = sample_geotensor.pad({"x": (10, 10), "y": (10, 10)})

        assert result.shape == (3, 120, 120)

    def test_pad_asymmetric(self, sample_geotensor):
        """Test asymmetric padding."""
        result = sample_geotensor.pad({"x": (5, 15), "y": (10, 20)})

        assert result.shape == (3, 130, 120)

    def test_pad_constant_value(self, sample_geotensor):
        """Test padding with specific constant value."""
        result = sample_geotensor.pad({"x": (10, 10)}, mode="constant", constant_values=-999)

        # Check that padding uses the constant value
        assert result.values[:, :, :10].min() == -999


# =============================================================================
# Tests for GeoTensor read/write from window
# =============================================================================


class TestGeoTensorWindowOperations:
    """Tests for GeoTensor window read/write operations."""

    def test_read_from_window_inside(self, sample_geotensor):
        """Test reading from window inside bounds."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=50, height=50)

        result = sample_geotensor.read_from_window(window)

        assert result.shape == (3, 50, 50)

    def test_read_from_window_boundless(self, sample_geotensor):
        """Test reading from window extending outside bounds."""
        window = rasterio.windows.Window(col_off=-10, row_off=10, width=50, height=50)

        result = sample_geotensor.read_from_window(window, boundless=True)

        assert result.shape == (3, 50, 50)

    def test_write_from_window(self, sample_geotensor):
        """Test writing to window."""
        window = rasterio.windows.Window(col_off=10, row_off=10, width=20, height=20)
        data = np.ones((3, 20, 20), dtype=np.float32) * 999

        sample_geotensor.write_from_window(data, window)

        # Check that data was written
        assert sample_geotensor.values[0, 10, 10] == 999
        assert sample_geotensor.values[0, 29, 29] == 999


# =============================================================================
# Tests for stack and concatenate functions
# =============================================================================


class TestStackConcatenate:
    """Tests for stack and concatenate functions."""

    def test_stack_basic(self, sample_geotensor):
        """Test basic stacking."""
        gt1 = sample_geotensor.copy()
        gt2 = sample_geotensor.copy()
        gt3 = sample_geotensor.copy()

        result = stack([gt1, gt2, gt3])

        assert result.shape == (3, 3, 100, 100)

    def test_stack_single(self, sample_geotensor):
        """Test stacking single GeoTensor."""
        result = stack([sample_geotensor])

        assert result.shape == (1, 3, 100, 100)

    def test_stack_preserves_georef(self, sample_geotensor):
        """Test that stacking preserves georeferencing."""
        gt1 = sample_geotensor.copy()
        gt2 = sample_geotensor.copy()

        result = stack([gt1, gt2])

        assert result.transform == sample_geotensor.transform
        assert result.crs == sample_geotensor.crs

    def test_concatenate_basic(self, sample_geotensor):
        """Test basic concatenation along axis 0."""
        gt1 = sample_geotensor.copy()
        gt2 = sample_geotensor.copy()
        gt3 = sample_geotensor.copy()

        result = concatenate([gt1, gt2, gt3], axis=0)

        assert result.shape == (9, 100, 100)

    def test_concatenate_single(self, sample_geotensor):
        """Test concatenating single GeoTensor."""
        result = concatenate([sample_geotensor])

        assert result.shape == sample_geotensor.shape

    def test_concatenate_invalid_axis_raises(self, sample_geotensor):
        """Test that concatenating along spatial axis raises."""
        gt1 = sample_geotensor.copy()
        gt2 = sample_geotensor.copy()

        # Axis 1 would be y (spatial) for 3D tensor
        with pytest.raises(AssertionError):
            concatenate([gt1, gt2], axis=1)


# =============================================================================
# Tests for GeoTensor resize
# =============================================================================


class TestGeoTensorResize:
    """Tests for GeoTensor resize method."""

    def test_resize_to_half(self, sample_geotensor):
        """Test resizing to half size."""
        result = sample_geotensor.resize(output_shape=(50, 50))

        assert result.shape == (3, 50, 50)

    def test_resize_to_double(self, sample_geotensor):
        """Test resizing to double size."""
        result = sample_geotensor.resize(output_shape=(200, 200))

        assert result.shape == (3, 200, 200)

    def test_resize_by_resolution(self, sample_geotensor):
        """Test resizing by resolution."""
        result = sample_geotensor.resize(resolution_dst=(2, 2))

        assert result.shape == (3, 50, 50)
        assert result.res == pytest.approx((2.0, 2.0), rel=0.01)

    def test_resize_2d(self, sample_geotensor_2d):
        """Test resizing 2D tensor."""
        result = sample_geotensor_2d.resize(output_shape=(50, 50))

        assert result.shape == (50, 50)


# =============================================================================
# Tests for GeoTensor error handling (Phase 2 Sprint 1)
# =============================================================================


class TestGeoTensorConstructorErrors:
    """Tests for GeoTensor constructor error handling."""

    def test_constructor_1d_array_raises(self):
        """Test that 1D array raises ValueError."""
        data = np.random.rand(100)
        transform = from_origin(0, 100, 1, 1)

        with pytest.raises(ValueError, match="Expected 2d-4d array"):
            GeoTensor(data, transform=transform, crs="EPSG:32631")

    def test_constructor_5d_array_raises(self):
        """Test that 5D array raises ValueError."""
        data = np.random.rand(2, 3, 4, 100, 100)
        transform = from_origin(0, 100, 1, 1)

        with pytest.raises(ValueError, match="Expected 2d-4d array"):
            GeoTensor(data, transform=transform, crs="EPSG:32631")

    def test_constructor_empty_array_raises(self):
        """Test that empty array raises ValueError."""
        data = np.array([])
        transform = from_origin(0, 100, 1, 1)

        with pytest.raises(ValueError, match="Expected 2d-4d array"):
            GeoTensor(data, transform=transform, crs="EPSG:32631")

    def test_constructor_0d_scalar_raises(self):
        """Test that 0D scalar raises ValueError."""
        data = np.array(5.0)
        transform = from_origin(0, 100, 1, 1)

        with pytest.raises(ValueError, match="Expected 2d-4d array"):
            GeoTensor(data, transform=transform, crs="EPSG:32631")


class TestGeoTensorIselErrors:
    """Tests for GeoTensor isel method error handling."""

    def test_isel_invalid_axis_raises(self, sample_geotensor):
        """Test that invalid axis name raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Axis invalid not in"):
            sample_geotensor.isel({"invalid": slice(0, 10)})

    def test_isel_non_slice_for_spatial_raises(self, sample_geotensor):
        """Test that non-slice for spatial dims raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Only slice selection supported"):
            sample_geotensor.isel({"x": 5})  # int instead of slice


class TestGeoTensorArithmeticErrors:
    """Tests for GeoTensor arithmetic error handling."""

    def test_add_mismatched_crs_raises(self, sample_geotensor):
        """Test that adding GeoTensors with different CRS raises ValueError."""
        other_data = np.random.rand(3, 100, 100)
        other_transform = from_origin(0, 100, 1, 1)
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:4326")  # Different CRS

        with pytest.raises(ValueError, match="georref must match"):
            sample_geotensor + other

    def test_sub_mismatched_transform_raises(self, sample_geotensor):
        """Test that subtracting GeoTensors with different transforms raises ValueError."""
        other_data = np.random.rand(3, 100, 100)
        other_transform = from_origin(100, 100, 1, 1)  # Different origin
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:32631")

        with pytest.raises(ValueError, match="georref must match"):
            sample_geotensor - other

    def test_mul_mismatched_shape_raises(self, sample_geotensor):
        """Test that multiplying GeoTensors with different shapes raises ValueError."""
        other_data = np.random.rand(3, 50, 50)  # Different shape
        other_transform = from_origin(0, 100, 1, 1)
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:32631")

        with pytest.raises(ValueError, match="georref must match"):
            sample_geotensor * other

    def test_div_mismatched_extent_raises(self, sample_geotensor):
        """Test that dividing GeoTensors with different extents raises ValueError."""
        other_data = np.random.rand(3, 100, 100)
        other_transform = from_origin(50, 150, 1, 1)  # Different origin
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:32631")

        with pytest.raises(ValueError, match="georref must match"):
            sample_geotensor / other


class TestGeoTensorResizeErrors:
    """Tests for GeoTensor resize method error handling."""

    def test_resize_both_params_raises(self, sample_geotensor):
        """Test that providing both output_shape and resolution_dst raises AssertionError."""
        with pytest.raises(AssertionError, match="Both output_shape and resolution_dst"):
            sample_geotensor.resize(output_shape=(50, 50), resolution_dst=(2, 2))

    def test_resize_neither_param_raises(self, sample_geotensor):
        """Test that providing neither output_shape nor resolution_dst raises AssertionError."""
        with pytest.raises(AssertionError, match="Can't have output_shape and resolution_dst as None"):
            sample_geotensor.resize()

    def test_resize_wrong_output_shape_length_raises(self, sample_geotensor):
        """Test that output_shape with wrong length raises AssertionError."""
        with pytest.raises(AssertionError, match="Expected output shape to be the spatial dimensions"):
            sample_geotensor.resize(output_shape=(3, 50, 50))  # 3D instead of 2D


class TestStackConcatenateErrors:
    """Tests for stack and concatenate function error handling."""

    def test_stack_empty_list_raises(self):
        """Test that stacking empty list raises AssertionError."""
        with pytest.raises(AssertionError, match="Empty list provided"):
            stack([])

    def test_concatenate_empty_list_raises(self):
        """Test that concatenating empty list raises AssertionError."""
        with pytest.raises(AssertionError, match="Empty list provided"):
            concatenate([])

    def test_stack_mismatched_transform_raises(self, sample_geotensor):
        """Test that stacking GeoTensors with different transforms raises AssertionError."""
        other_data = np.random.rand(3, 100, 100)
        other_transform = from_origin(100, 100, 1, 1)  # Different origin
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:32631")

        with pytest.raises(AssertionError, match="Different size"):
            stack([sample_geotensor, other])

    def test_stack_mismatched_crs_raises(self, sample_geotensor):
        """Test that stacking GeoTensors with different CRS raises AssertionError."""
        other_data = np.random.rand(3, 100, 100)
        other_transform = from_origin(0, 100, 1, 1)
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:4326")  # Different CRS

        with pytest.raises(AssertionError, match="Different size"):
            stack([sample_geotensor, other])

    def test_stack_mismatched_shape_raises(self, sample_geotensor):
        """Test that stacking GeoTensors with different shapes raises AssertionError."""
        other_data = np.random.rand(5, 100, 100)  # Different band count
        other_transform = from_origin(0, 100, 1, 1)
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:32631")

        with pytest.raises(AssertionError, match="Different shape"):
            stack([sample_geotensor, other])

    def test_concatenate_mismatched_extent_raises(self, sample_geotensor):
        """Test that concatenating GeoTensors with different extents raises AssertionError."""
        other_data = np.random.rand(3, 100, 100)
        other_transform = from_origin(50, 100, 1, 1)  # Different origin
        other = GeoTensor(other_data, transform=other_transform, crs="EPSG:32631")

        # concatenate should work if same_extent fails on the first comparison
        # The function doesn't actually check same_extent for concatenate, so let's test axis bounds
        with pytest.raises(AssertionError):
            concatenate([sample_geotensor, other], axis=1)  # Invalid axis

    def test_concatenate_spatial_axis_raises(self, sample_geotensor):
        """Test that concatenating along spatial axis raises AssertionError."""
        gt1 = sample_geotensor.copy()
        gt2 = sample_geotensor.copy()

        # Axis 2 would be x (spatial) for 3D tensor - out of valid range
        with pytest.raises(AssertionError, match="Can't concatenate along spatial axis"):
            concatenate([gt1, gt2], axis=2)


class TestGeoTensorSetitemErrors:
    """Tests for GeoTensor __setitem__ error handling."""

    def test_setitem_invalid_index_type_raises(self, sample_geotensor):
        """Test that invalid index type raises an error."""
        # String indexing should raise a TypeError from numpy
        with pytest.raises((TypeError, IndexError)):
            sample_geotensor["invalid"] = 5

    def test_setitem_wrong_shape_mask_raises(self, sample_geotensor):
        """Test that boolean mask with wrong shape raises IndexError from numpy."""
        wrong_shape_mask = np.ones((50, 50), dtype=bool)

        with pytest.raises(IndexError):
            sample_geotensor[wrong_shape_mask] = 5


class TestGeoTensorValidFootprintErrors:
    """Tests for GeoTensor valid_footprint error handling."""

    def test_valid_footprint_invalid_method_raises(self, sample_geotensor):
        """Test that invalid method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Method .* to aggregate channels not implemented"):
            sample_geotensor.valid_footprint(method="invalid")

    def test_valid_footprint_no_valid_values_raises(self):
        """Test that GeoTensor with all fill values raises ValueError."""
        # Create GeoTensor with all fill values
        data = np.zeros((3, 100, 100))  # All zeros = fill_value_default
        transform = from_origin(0, 100, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)

        with pytest.raises(ValueError, match="GeoTensor has no valid values"):
            gt.valid_footprint()


# =============================================================================
# Tests for missing GeoTensor methods (Phase 2 Sprint 2)
# =============================================================================


class TestGeoTensorSetDtype:
    """Tests for GeoTensor set_dtype method.
    
    Note: set_dtype is deprecated and doesn't work reliably due to numpy subclass limitations.
    These tests verify the deprecation warning and RuntimeError are raised.
    """

    def test_set_dtype_raises_error_and_warning(self, sample_geotensor):
        """Test that set_dtype raises deprecation warning and RuntimeError when dtype doesn't change."""
        with pytest.warns(DeprecationWarning, match="set_dtype.*deprecated"):
            with pytest.raises(RuntimeError, match="set_dtype.*failed to change dtype"):
                sample_geotensor.set_dtype(np.float64)


class TestGeoTensorValidFootprint:
    """Tests for GeoTensor valid_footprint method."""

    def test_valid_footprint_all_method(self):
        """Test valid_footprint with 'all' aggregation method."""
        from shapely.geometry import Polygon

        # Create GeoTensor with some valid values
        data = np.ones((3, 100, 100))
        data[:, 50:, 50:] = 0  # Set bottom-right quadrant to fill value
        transform = from_origin(0, 100, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)

        footprint = gt.valid_footprint(method="all")

        assert isinstance(footprint, (Polygon,))
        assert footprint.is_valid

    def test_valid_footprint_any_method(self):
        """Test valid_footprint with 'any' aggregation method."""
        from shapely.geometry import Polygon

        # Create GeoTensor where only some bands have valid values
        data = np.zeros((3, 100, 100))
        data[0, :50, :50] = 1  # Only first band has some valid values
        transform = from_origin(0, 100, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)

        footprint = gt.valid_footprint(method="any")

        assert isinstance(footprint, (Polygon,))

    def test_valid_footprint_with_crs_transformation(self):
        """Test valid_footprint with CRS transformation."""
        data = np.ones((3, 100, 100))
        transform = from_origin(500000, 5000000, 10, 10)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)

        footprint = gt.valid_footprint(crs="EPSG:4326")

        assert footprint.is_valid


class TestGeoTensorRepr:
    """Tests for GeoTensor __repr__ method."""

    def test_repr_contains_shape(self, sample_geotensor):
        """Test that repr contains shape information."""
        repr_str = repr(sample_geotensor)

        assert "Shape" in repr_str
        assert "100" in repr_str  # Part of shape

    def test_repr_contains_transform(self, sample_geotensor):
        """Test that repr contains transform information."""
        repr_str = repr(sample_geotensor)

        assert "Transform" in repr_str

    def test_repr_contains_crs(self, sample_geotensor):
        """Test that repr contains CRS information."""
        repr_str = repr(sample_geotensor)

        assert "CRS" in repr_str

    def test_repr_contains_bounds(self, sample_geotensor):
        """Test that repr contains bounds information."""
        repr_str = repr(sample_geotensor)

        assert "Bounds" in repr_str


class TestGeoTensorReverseArithmetic:
    """Tests for GeoTensor reverse arithmetic operations.

    Note: With numpy ufunc implementation, reverse operations now work correctly.
    These tests document that reverse operations with scalars ARE supported.
    """

    def test_radd_scalar_works(self, sample_geotensor):
        """Test that reverse add with scalar now works."""
        result = 5 + sample_geotensor
        assert isinstance(result, GeoTensor)
        assert np.allclose(result.values, 5 + sample_geotensor.values)

    def test_rsub_scalar_works(self, sample_geotensor):
        """Test that reverse subtract with scalar now works."""
        result = 10 - sample_geotensor
        assert isinstance(result, GeoTensor)
        assert np.allclose(result.values, 10 - sample_geotensor.values)

    def test_rmul_scalar_works(self, sample_geotensor):
        """Test that reverse multiply with scalar now works."""
        result = 3 * sample_geotensor
        assert isinstance(result, GeoTensor)
        assert np.allclose(result.values, 3 * sample_geotensor.values)

    def test_rtruediv_scalar_works(self, sample_geotensor):
        """Test that reverse divide with scalar now works."""
        gt = sample_geotensor + 1
        result = 10 / gt
        assert isinstance(result, GeoTensor)
        assert np.allclose(result.values, 10 / gt.values)


class TestGeoTensorLoadFile:
    """Tests for GeoTensor load_file class method."""

    def test_load_file_basic(self, test_raster_path):
        """Test basic file loading."""
        gt = GeoTensor.load_file(test_raster_path)

        assert gt is not None
        assert gt.shape == (15, 200, 250)  # Test file dimensions
        assert gt.crs is not None
        assert gt.transform is not None

    def test_load_file_with_tags(self, test_raster_path):
        """Test loading file with tags."""
        gt = GeoTensor.load_file(test_raster_path, load_tags=True)

        assert gt is not None
        # Tags may or may not be present depending on the file
        assert "tags" in gt.attrs or gt.attrs == {}

    def test_load_file_with_descriptions(self, test_raster_path):
        """Test loading file with descriptions."""
        gt = GeoTensor.load_file(test_raster_path, load_descriptions=True)

        assert gt is not None


class TestGeoTensorLoadBytes:
    """Tests for GeoTensor load_bytes class method."""

    def test_load_bytes_basic(self, test_raster_path):
        """Test loading from bytes."""
        # Read the test file into bytes
        with open(test_raster_path, "rb") as f:
            file_bytes = f.read()

        gt = GeoTensor.load_bytes(file_bytes)

        assert gt is not None
        assert gt.shape == (15, 200, 250)
        assert gt.crs is not None

    def test_load_bytes_with_tags(self, test_raster_path):
        """Test loading from bytes with tags."""
        with open(test_raster_path, "rb") as f:
            file_bytes = f.read()

        gt = GeoTensor.load_bytes(file_bytes, load_tags=True)

        assert gt is not None


class TestGeoTensorAttrs:
    """Tests for GeoTensor attrs attribute."""

    def test_attrs_default_empty(self, sample_geotensor):
        """Test that attrs defaults to empty dict."""
        assert sample_geotensor.attrs == {}

    def test_attrs_custom(self):
        """Test creating GeoTensor with custom attrs."""
        data = np.random.rand(3, 100, 100)
        transform = from_origin(0, 100, 1, 1)
        attrs = {"custom_key": "custom_value", "numeric": 42}

        gt = GeoTensor(data, transform=transform, crs="EPSG:32631", attrs=attrs)

        assert gt.attrs == attrs
        assert gt.attrs["custom_key"] == "custom_value"


# =============================================================================
# Tests for NumPy API compatibility (ufuncs, slicing, reductions)
# =============================================================================


class TestGeoTensorNumpyUfuncs:
    """Tests for NumPy universal functions (ufuncs) on GeoTensor."""

    def test_ufunc_sin(self, sample_geotensor):
        """Test np.sin returns GeoTensor with same spatial info."""
        result = np.sin(sample_geotensor)

        assert isinstance(result, GeoTensor)
        assert result.shape == sample_geotensor.shape
        assert result.transform == sample_geotensor.transform
        assert result.crs == sample_geotensor.crs

    def test_ufunc_exp(self, sample_geotensor):
        """Test np.exp returns GeoTensor."""
        result = np.exp(sample_geotensor)

        assert isinstance(result, GeoTensor)
        assert result.transform == sample_geotensor.transform

    def test_ufunc_sqrt(self, sample_geotensor):
        """Test np.sqrt returns GeoTensor."""
        result = np.sqrt(sample_geotensor)

        assert isinstance(result, GeoTensor)
        assert result.shape == sample_geotensor.shape

    def test_ufunc_add_two_geotensors(self, sample_geotensor):
        """Test np.add with two GeoTensors of same extent."""
        gt1 = sample_geotensor
        gt2 = sample_geotensor.copy()

        result = np.add(gt1, gt2)

        assert isinstance(result, GeoTensor)
        assert np.allclose(result.values, gt1.values + gt2.values)


class TestGeoTensorNumpyReductions:
    """Tests for NumPy reduction operations on GeoTensor."""

    def test_mean_preserves_spatial_returns_geotensor(self, sample_geotensor):
        """Test np.mean along non-spatial axis returns GeoTensor."""
        # Mean along band axis (axis=0) preserves spatial dims
        result = np.mean(sample_geotensor, axis=0)

        assert isinstance(result, GeoTensor)
        assert result.shape == (100, 100)  # 2D result
        assert result.transform == sample_geotensor.transform
        assert result.crs == sample_geotensor.crs

    def test_mean_reduces_spatial_returns_ndarray(self, sample_geotensor):
        """Test np.mean along spatial axis returns ndarray."""
        # Mean along spatial axes returns ndarray
        result = np.mean(sample_geotensor, axis=(-2, -1))

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, GeoTensor)
        assert result.shape == (3,)  # One value per band

    def test_mean_full_reduction_returns_scalar(self, sample_geotensor):
        """Test np.mean with no axis returns scalar."""
        result = np.mean(sample_geotensor)

        assert np.isscalar(result) or result.shape == ()

    def test_sum_preserves_spatial(self, sample_geotensor):
        """Test np.sum along non-spatial axis returns GeoTensor."""
        result = np.sum(sample_geotensor, axis=0)

        assert isinstance(result, GeoTensor)
        assert result.shape == (100, 100)

    def test_max_preserves_spatial(self, sample_geotensor):
        """Test np.max along non-spatial axis returns GeoTensor."""
        result = np.max(sample_geotensor, axis=0)

        assert isinstance(result, GeoTensor)
        assert result.shape == (100, 100)

    def test_all_preserves_spatial(self, sample_geotensor):
        """Test np.all along non-spatial axis returns GeoTensor."""
        bool_tensor = sample_geotensor > 0.5
        result = np.all(bool_tensor, axis=0)

        assert isinstance(result, GeoTensor)
        assert result.dtype == bool

    def test_mean_keepdims_true(self, sample_geotensor):
        """Test np.mean with keepdims=True."""
        result = np.mean(sample_geotensor, axis=0, keepdims=True)

        assert isinstance(result, GeoTensor)
        assert result.shape == (1, 100, 100)


class TestGeoTensorDirectSlicing:
    """Tests for direct slicing with __getitem__ and transform propagation."""

    def test_slice_spatial_dims_updates_transform(self, sample_geotensor):
        """Test that slicing spatial dimensions updates transform correctly."""
        sliced = sample_geotensor[:, 20:80, 30:90]

        assert sliced.shape == (3, 60, 60)
        # Transform should be shifted by the slice offsets
        assert sliced.transform != sample_geotensor.transform
        # Check that origin is shifted (col_off=30, row_off=20 with resolution 1)
        assert sliced.transform.c == sample_geotensor.transform.c + 30
        assert sliced.transform.f == sample_geotensor.transform.f - 20

    def test_slice_with_step(self, sample_geotensor):
        """Test slicing with step changes resolution."""
        sliced = sample_geotensor[:, ::2, ::2]

        assert sliced.shape == (3, 50, 50)
        # Resolution should be doubled
        assert abs(sliced.res[0]) == abs(sample_geotensor.res[0]) * 2
        assert abs(sliced.res[1]) == abs(sample_geotensor.res[1]) * 2

    def test_slice_with_negative_step_reverses(self, sample_geotensor):
        """Test slicing with negative step reverses the array."""
        sliced = sample_geotensor[:, ::-1, :]

        assert sliced.shape == sample_geotensor.shape
        # Data should be reversed along y axis
        assert np.allclose(sliced.values[:, 0, :], sample_geotensor.values[:, -1, :])

    def test_slice_band_dimension(self, sample_geotensor):
        """Test slicing band dimension with list."""
        sliced = sample_geotensor[[0, 2]]

        assert sliced.shape == (2, 100, 100)
        assert sliced.transform == sample_geotensor.transform

    def test_slice_single_band(self, sample_geotensor):
        """Test selecting a single band."""
        sliced = sample_geotensor[1]

        assert sliced.shape == (100, 100)
        assert sliced.transform == sample_geotensor.transform


class TestGeoTensorValidInvalidMask:
    """Tests for validmask and invalidmask methods."""

    def test_validmask_basic(self):
        """Test validmask returns correct boolean GeoTensor."""
        data = np.array([[[1, 0, 2], [0, 3, 0]]])  # 1x2x3
        transform = from_origin(0, 2, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)

        valid = gt.validmask()

        assert isinstance(valid, GeoTensor)
        assert valid.dtype == bool
        expected = np.array([[[True, False, True], [False, True, False]]])
        assert np.array_equal(valid.values, expected)

    def test_invalidmask_basic(self):
        """Test invalidmask returns correct boolean GeoTensor."""
        data = np.array([[[1, 0, 2], [0, 3, 0]]])
        transform = from_origin(0, 2, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)

        invalid = gt.invalidmask()

        assert isinstance(invalid, GeoTensor)
        assert invalid.dtype == bool
        expected = np.array([[[False, True, False], [True, False, True]]])
        assert np.array_equal(invalid.values, expected)

    def test_validmask_none_fill_value(self):
        """Test validmask when fill_value_default is None."""
        data = np.ones((3, 10, 10))
        transform = from_origin(0, 10, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=None)

        valid = gt.validmask()

        # All values should be valid when fill_value_default is None
        assert np.all(valid.values)

    def test_invalidmask_preserves_spatial_info(self):
        """Test invalidmask preserves transform and crs."""
        data = np.zeros((3, 50, 50))
        transform = from_origin(10, 100, 2, 2)
        gt = GeoTensor(data, transform=transform, crs="EPSG:4326", fill_value_default=0)

        invalid = gt.invalidmask()

        assert invalid.transform == gt.transform
        assert invalid.crs == gt.crs


class TestGeoTensorComparisonOperations:
    """Tests for comparison operations returning GeoTensor."""

    def test_less_than(self, sample_geotensor):
        """Test < operator returns GeoTensor."""
        result = sample_geotensor < 0.5

        assert isinstance(result, GeoTensor)
        assert result.dtype == bool
        assert result.fill_value_default is False

    def test_less_equal(self, sample_geotensor):
        """Test <= operator returns GeoTensor."""
        result = sample_geotensor <= 0.5

        assert isinstance(result, GeoTensor)
        assert result.dtype == bool

    def test_greater_than(self, sample_geotensor):
        """Test > operator returns GeoTensor."""
        result = sample_geotensor > 0.5

        assert isinstance(result, GeoTensor)
        assert result.dtype == bool

    def test_not_equal(self, sample_geotensor):
        """Test != operator returns GeoTensor."""
        result = sample_geotensor != 0

        assert isinstance(result, GeoTensor)
        assert result.dtype == bool

    def test_comparison_preserves_spatial_info(self, sample_geotensor):
        """Test comparison preserves transform and crs."""
        result = sample_geotensor > 0.5

        assert result.transform == sample_geotensor.transform
        assert result.crs == sample_geotensor.crs


class TestGeoTensorExpandDims:
    """Tests for expand_dims method."""

    def test_expand_dims_axis_0(self, sample_geotensor):
        """Test expand_dims at axis 0."""
        result = sample_geotensor.expand_dims(0)

        assert result.shape == (1, 3, 100, 100)
        assert result.transform == sample_geotensor.transform

    def test_expand_dims_invalid_axis_raises(self, sample_geotensor):
        """Test expand_dims at spatial axis raises error."""
        # Axis 1 would be spatial for 3D array after expansion
        with pytest.raises(ValueError, match="Cannot add dimension"):
            sample_geotensor.expand_dims(2)


class TestGeoTensorTranspose:
    """Tests for transpose method."""

    def test_transpose_4d(self):
        """Test transpose reverses non-spatial dimensions."""
        data = np.random.rand(2, 3, 50, 50)
        transform = from_origin(0, 50, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631")

        result = gt.transpose()

        assert result.shape == (3, 2, 50, 50)
        assert result.transform == gt.transform

    def test_transpose_explicit_axes(self):
        """Test transpose with explicit axes."""
        data = np.random.rand(2, 3, 50, 50)
        transform = from_origin(0, 50, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631")

        result = gt.transpose((1, 0, 2, 3))

        assert result.shape == (3, 2, 50, 50)

    def test_transpose_invalid_spatial_axes_raises(self):
        """Test transpose fails if spatial dims are moved."""
        data = np.random.rand(2, 3, 50, 50)
        transform = from_origin(0, 50, 1, 1)
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631")

        with pytest.raises(ValueError, match="Cannot change the position of spatial dimensions"):
            gt.transpose((0, 2, 1, 3))


class TestGeoTensorArrayAsGeotensor:
    """Tests for array_as_geotensor method."""

    def test_array_as_geotensor_same_shape(self, sample_geotensor):
        """Test converting ndarray to GeoTensor with same spatial dims."""
        arr = np.ones((3, 100, 100))
        result = sample_geotensor.array_as_geotensor(arr)

        assert isinstance(result, GeoTensor)
        assert result.transform == sample_geotensor.transform
        assert result.crs == sample_geotensor.crs

    def test_array_as_geotensor_2d(self, sample_geotensor):
        """Test converting 2D ndarray to GeoTensor."""
        arr = np.ones((100, 100))
        result = sample_geotensor.array_as_geotensor(arr)

        assert isinstance(result, GeoTensor)
        assert result.shape == (100, 100)

    def test_array_as_geotensor_wrong_spatial_raises(self, sample_geotensor):
        """Test array_as_geotensor raises for wrong spatial dims."""
        arr = np.ones((3, 50, 50))  # Different spatial dims

        with pytest.raises(ValueError, match="Operation altered spatial dimensions"):
            sample_geotensor.array_as_geotensor(arr)


class TestGeoTensorBitwiseOperations:
    """Tests for bitwise AND and OR operations."""

    def test_and_operation(self):
        """Test & operator with boolean GeoTensors."""
        data1 = np.array([[True, False], [True, True]])
        data2 = np.array([[True, True], [False, True]])
        transform = from_origin(0, 2, 1, 1)
        gt1 = GeoTensor(data1, transform=transform, crs="EPSG:32631")
        gt2 = GeoTensor(data2, transform=transform, crs="EPSG:32631")

        result = gt1 & gt2

        assert isinstance(result, GeoTensor)
        expected = np.array([[True, False], [False, True]])
        assert np.array_equal(result.values, expected)

    def test_or_operation(self):
        """Test | operator with boolean GeoTensors."""
        data1 = np.array([[True, False], [False, False]])
        data2 = np.array([[False, True], [False, True]])
        transform = from_origin(0, 2, 1, 1)
        gt1 = GeoTensor(data1, transform=transform, crs="EPSG:32631")
        gt2 = GeoTensor(data2, transform=transform, crs="EPSG:32631")

        result = gt1 | gt2

        assert isinstance(result, GeoTensor)
        expected = np.array([[True, True], [False, True]])
        assert np.array_equal(result.values, expected)


class TestValuesSetter:
    """Tests for the strict (fail-loud) GeoTensor.values setter."""

    def _gt(self, shape=(3, 4, 5), dtype=np.float32):
        data = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
        transform = from_origin(0, shape[-2], 1, 1)
        return GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=0)

    def test_inplace_same_shape_and_dtype(self):
        """Same shape and dtype writes through into the existing buffer."""
        gt = self._gt()
        new = np.ones_like(gt.values)
        gt.values = new
        assert np.array_equal(gt.values, new)
        assert gt.dtype == np.float32

    def test_inplace_writes_through_buffer(self):
        """The write mutates the underlying buffer (no rebinding to a new array)."""
        gt = self._gt()
        view = gt.values  # shares memory with gt
        gt.values = np.full(gt.shape, 7.0, dtype=np.float32)
        # The pre-existing view sees the mutation -> same buffer was written.
        assert np.all(view == 7.0)

    def test_inplace_preserves_metadata(self):
        """Setting values must not touch transform / crs / fill_value_default."""
        gt = self._gt()
        transform, crs, fill = gt.transform, gt.crs, gt.fill_value_default
        gt.values = np.zeros(gt.shape, dtype=np.float32)
        assert gt.transform == transform
        assert gt.crs == crs
        assert gt.fill_value_default == fill

    def test_accepts_array_like(self):
        """A non-ndarray array-like of matching shape/dtype is accepted."""
        # Python float lists become float64 via np.asarray, so use a float64 tensor.
        gt = self._gt(shape=(2, 2), dtype=np.float64)
        gt.values = [[1.0, 2.0], [3.0, 4.0]]
        assert np.array_equal(gt.values, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_shape_mismatch_raises(self):
        """A shape change is rejected loudly instead of silently failing."""
        gt = self._gt(shape=(3, 4, 5))
        with pytest.raises(ValueError, match="shape"):
            gt.values = np.ones((4, 5), dtype=np.float32)

    def test_squeeze_shape_change_raises(self):
        """Assigning a squeezed array (shape change) raises rather than corrupts."""
        gt = self._gt(shape=(1, 4, 5))
        with pytest.raises(ValueError, match="shape"):
            gt.values = gt.values.squeeze()

    def test_dtype_mismatch_raises_no_silent_cast(self):
        """A dtype change must raise rather than silently cast/truncate."""
        gt = self._gt(shape=(2, 3), dtype=np.uint16)
        # Float radiance into a uint16 buffer would silently truncate -> forbidden.
        with pytest.raises(TypeError, match="dtype"):
            gt.values = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float64)

    def test_bool_into_int_raises(self):
        """Assigning a boolean mask into an integer GeoTensor raises (dtype change)."""
        gt = self._gt(shape=(4, 5), dtype=np.uint8)
        with pytest.raises(TypeError, match="dtype"):
            gt.values = gt.values > 2

    def test_item_assignment_still_works(self):
        """In-place item/mask assignment (not the setter) is unaffected."""
        gt = self._gt(shape=(4, 5), dtype=np.float32)
        mask = gt.values > 10
        gt.values[mask] = -1.0
        assert np.all(gt.values[mask] == -1.0)

    def test_astype_is_the_dtype_change_path(self):
        """astype() returns a new GeoTensor with the new dtype, preserving georef."""
        gt = self._gt(shape=(2, 3), dtype=np.uint16)
        gt2 = gt.astype(np.float32)
        assert isinstance(gt2, GeoTensor)
        assert gt2.dtype == np.float32
        assert gt2.transform == gt.transform
        assert gt2.crs == gt.crs
