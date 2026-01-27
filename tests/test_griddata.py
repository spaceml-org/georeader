"""
Tests for the georeader.griddata module.

These tests verify gridded data operations including:
- Footprint generation from lon/lat arrays
- Meshgrid generation
- Reprojection of irregularly gridded data
- Georeferencing using geolocation arrays
"""

import numpy as np
import pytest
from rasterio.transform import from_origin

from georeader import griddata
from georeader.geotensor import GeoTensor


class TestFootprint:
    """Tests for footprint function."""

    def test_basic_footprint(self):
        """Test footprint generation from regular lon/lat grid."""
        # Create simple regular grid
        lons = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        lats = np.array([[2, 2, 2], [1, 1, 1], [0, 0, 0]])

        footprint = griddata.footprint(lons, lats)

        assert footprint is not None
        assert footprint.is_valid
        # Should cover roughly 2x2 degrees
        bounds = footprint.bounds
        assert bounds[2] - bounds[0] >= 1.5  # Width

    def test_irregular_footprint(self):
        """Test footprint from irregular grid."""
        # Create slightly irregular grid
        lons = np.array([[0, 1.1, 2], [0.1, 1, 2.1], [0, 1, 2]])
        lats = np.array([[2, 2.1, 2], [1, 1, 1], [0, 0.1, 0]])

        footprint = griddata.footprint(lons, lats)

        assert footprint.is_valid


class TestMeshgrid:
    """Tests for meshgrid function."""

    def test_basic_meshgrid(self):
        """Test meshgrid generation from transform without CRS transformation."""
        transform = from_origin(0, 100, 10, 10)  # 10m resolution
        width, height = 50, 50

        # meshgrid without dst_crs returns coordinate arrays
        xs, ys = griddata.meshgrid(transform, width, height)

        # Verify correct number of coordinate values
        assert len(xs) == width * height
        assert len(ys) == width * height

    def test_meshgrid_with_dst_crs(self):
        """Test meshgrid with destination CRS transformation."""
        transform = from_origin(500000, 4500000, 10, 10)
        width, height = 50, 50

        # When dst_crs is provided, coordinates are transformed and returned as 2D
        xs, ys = griddata.meshgrid(transform, width, height, source_crs="EPSG:32631", dst_crs="EPSG:4326")

        # Output should be in WGS84 (lon/lat) as 2D arrays when CRS transform occurs
        assert xs.shape == (height, width)
        assert ys.shape == (height, width)
        # xs should be longitudes - reasonable for UTM zone 31
        assert -10 < xs.mean() < 10
        # ys should be latitudes
        assert 30 < ys.mean() < 50


class TestGetShapeTransformCrs:
    """Tests for get_shape_transform_crs function."""

    def test_basic_shape_transform(self):
        """Test computing shape and transform from lon/lat arrays."""
        # Create 10x10 grid
        lons = np.linspace(0, 1, 10)
        lats = np.linspace(45, 46, 10)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # get_shape_transform_crs returns (width, height, transform, crs)
        width, height, transform, crs = griddata.get_shape_transform_crs(
            lon_grid, lat_grid, resolution_dst=0.1, dst_crs="EPSG:4326"
        )

        assert width > 0
        assert height > 0
        assert transform is not None

    def test_with_utm_output(self):
        """Test with UTM output CRS."""
        lons = np.linspace(3, 3.5, 10)
        lats = np.linspace(45, 45.5, 10)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # get_shape_transform_crs returns (width, height, transform, crs)
        width, height, transform, crs = griddata.get_shape_transform_crs(
            lon_grid, lat_grid, resolution_dst=100, dst_crs="EPSG:32631"
        )

        assert width > 0
        assert height > 0


class TestReproject:
    """Tests for reproject function."""

    def test_basic_reproject(self):
        """Test basic reprojection of gridded data."""
        # Create simple data with coordinates
        data = np.random.rand(10, 10).astype(np.float32)
        lons = np.linspace(3, 3.5, 10)
        lats = np.linspace(45, 45.5, 10)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Define output grid
        transform = from_origin(3, 45.5, 0.05, 0.05)
        width, height = 10, 10

        result = griddata.reproject(data, lon_grid, lat_grid, width, height, transform, dst_crs="EPSG:4326")

        # reproject returns a GeoTensor, access shape through .shape
        assert result.shape == (height, width)
        assert isinstance(result, GeoTensor)

    def test_reproject_3d(self):
        """Test reprojection of 3D data (height, width, bands) -> (bands, height, width)."""
        # reproject expects data in (H, W, C) format for 3D
        data = np.random.rand(10, 10, 3).astype(np.float32)
        lons = np.linspace(3, 3.5, 10)
        lats = np.linspace(45, 45.5, 10)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        transform = from_origin(3, 45.5, 0.05, 0.05)
        width, height = 10, 10

        result = griddata.reproject(data, lon_grid, lat_grid, width, height, transform, dst_crs="EPSG:4326")

        # Output is transposed to (C, H, W) format
        assert result.shape == (3, height, width)


class TestReadReprojectLike:
    """Tests for read_reproject_like function."""

    def test_basic_reproject_like(self):
        """Test reprojecting gridded data like another GeoData."""
        # Source data with coordinates - reproject expects (H, W) or (H, W, C) format
        data = np.random.rand(20, 20, 3).astype(np.float32)
        lons = np.linspace(3, 3.5, 20)
        lats = np.linspace(45, 45.5, 20)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Target GeoTensor to match
        target_data = np.zeros((3, 10, 10))
        target_transform = from_origin(3.1, 45.4, 0.03, 0.03)
        target = GeoTensor(target_data, transform=target_transform, crs="EPSG:4326", fill_value_default=0)

        result = griddata.read_reproject_like(data, lon_grid, lat_grid, target)

        assert isinstance(result, GeoTensor)
        assert result.shape[-2:] == target.shape[-2:]


class TestReadToCrs:
    """Tests for read_to_crs function."""

    def test_basic_read_to_crs(self):
        """Test reading gridded data to a specific CRS."""
        # read_to_crs expects data in (H, W) or (H, W, C) format
        data = np.random.rand(20, 20, 3).astype(np.float32)
        lons = np.linspace(3, 3.5, 20)
        lats = np.linspace(45, 45.5, 20)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        result = griddata.read_to_crs(data, lon_grid, lat_grid, resolution_dst=0.01, dst_crs="EPSG:4326")

        assert isinstance(result, GeoTensor)


class TestGeorreference:
    """Tests for georreference function (GLT-based)."""

    def test_basic_georeferencing(self):
        """Test basic georeferencing using geolocation table."""
        # Create data - the raw sensor data
        data = np.random.rand(3, 10, 10).astype(np.float32)

        # Create simple GLT (geolocation table) as a GeoTensor
        # GLT has shape (2, H', W') and maps output pixel to input pixel:
        # glt[0] = column indices, glt[1] = row indices
        glt_values = np.zeros((2, 10, 10), dtype=np.int32)
        # Simple 1:1 mapping
        for i in range(10):
            for j in range(10):
                glt_values[0, i, j] = j  # column mapping (x)
                glt_values[1, i, j] = i  # row mapping (y)

        transform = from_origin(0, 100, 10, 10)
        glt = GeoTensor(glt_values, transform=transform, crs="EPSG:32631", fill_value_default=-1)

        result = griddata.georreference(glt, data)

        assert result.shape == data.shape
        assert isinstance(result, GeoTensor)

    def test_georeferencing_with_invalid(self):
        """Test georeferencing with invalid GLT values."""
        data = np.random.rand(3, 10, 10).astype(np.float32)

        # Create GLT with some invalid values (negative = invalid)
        glt_values = np.zeros((2, 10, 10), dtype=np.int32)
        for i in range(10):
            for j in range(10):
                if i < 5:
                    glt_values[0, i, j] = j  # column
                    glt_values[1, i, j] = i  # row
                else:
                    glt_values[0, i, j] = -1  # Invalid
                    glt_values[1, i, j] = -1

        transform = from_origin(0, 100, 10, 10)
        glt = GeoTensor(glt_values, transform=transform, crs="EPSG:32631", fill_value_default=-1)

        valid_glt = glt_values[0] >= 0

        result = griddata.georreference(glt, data, valid_glt=valid_glt, fill_value_default=-9999)

        # Invalid areas should have fill value
        assert result.shape == data.shape
        assert isinstance(result, GeoTensor)
