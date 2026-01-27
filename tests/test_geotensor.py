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
