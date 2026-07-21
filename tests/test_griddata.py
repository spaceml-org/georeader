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
from shapely.geometry import Polygon, MultiPolygon, box

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


class TestPolygonToImageCoords:
    """Tests for polygon_to_image_coords function."""

    @staticmethod
    def _grid(H=20, W=30, lon_min=0.0, lon_max=1.0, lat_min=45.0, lat_max=46.0):
        """
        Build a regular lon/lat grid where the inverse mapping is closed-form:

            col = (lon - lon_min) / (lon_max - lon_min) * (W - 1)
            row = (lat - lat_min) / (lat_max - lat_min) * (H - 1)
        """
        lons, lats = np.meshgrid(
            np.linspace(lon_min, lon_max, W),
            np.linspace(lat_min, lat_max, H),
        )
        return lons, lats

    def test_polygon_basic(self):
        """Polygon vertices on grid lines should map to integer pixel coords."""
        H, W = 20, 30
        lons, lats = self._grid(H, W)

        # box covering exactly half the grid extent in each axis
        pol = box(0.0, 45.0, 0.5, 45.5)
        result = griddata.polygon_to_image_coords(pol, lons, lats)

        assert isinstance(result, Polygon)
        assert not result.is_empty

        xs, ys = result.exterior.coords.xy
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        # Expected pixel extent: col in [0, 14.5], row in [0, 9.5]
        assert np.isclose(xs.min(), 0.0, atol=1e-6)
        assert np.isclose(xs.max(), 14.5, atol=1e-6)
        assert np.isclose(ys.min(), 0.0, atol=1e-6)
        assert np.isclose(ys.max(), 9.5, atol=1e-6)

    def test_polygon_preserves_holes(self):
        """Interior rings (holes) should be transformed too."""
        H, W = 20, 30
        lons, lats = self._grid(H, W)

        shell = [(0.0, 45.0), (1.0, 45.0), (1.0, 46.0), (0.0, 46.0)]
        hole = [(0.25, 45.25), (0.75, 45.25), (0.75, 45.75), (0.25, 45.75)]
        pol = Polygon(shell, [hole])

        result = griddata.polygon_to_image_coords(pol, lons, lats)

        assert isinstance(result, Polygon)
        assert len(result.interiors) == 1

        # Shell should span the full image extent
        sx, sy = result.exterior.coords.xy
        assert np.isclose(min(sx), 0.0, atol=1e-6)
        assert np.isclose(max(sx), W - 1, atol=1e-6)
        assert np.isclose(min(sy), 0.0, atol=1e-6)
        assert np.isclose(max(sy), H - 1, atol=1e-6)

        # Hole should be the inner box at 25%-75% of each axis
        hx, hy = result.interiors[0].coords.xy
        assert np.isclose(min(hx), 0.25 * (W - 1), atol=1e-6)
        assert np.isclose(max(hx), 0.75 * (W - 1), atol=1e-6)
        assert np.isclose(min(hy), 0.25 * (H - 1), atol=1e-6)
        assert np.isclose(max(hy), 0.75 * (H - 1), atol=1e-6)

    def test_multipolygon(self):
        """MultiPolygon input should yield MultiPolygon output with same arity."""
        H, W = 20, 30
        lons, lats = self._grid(H, W)

        a = box(0.0, 45.0, 0.25, 45.25)
        b = box(0.75, 45.75, 1.0, 46.0)
        mp = MultiPolygon([a, b])

        result = griddata.polygon_to_image_coords(mp, lons, lats)

        assert isinstance(result, MultiPolygon)
        assert len(result.geoms) == 2

        # First part: top-left quadrant of the image
        ax, ay = result.geoms[0].exterior.coords.xy
        assert np.isclose(min(ax), 0.0, atol=1e-6)
        assert np.isclose(max(ax), 0.25 * (W - 1), atol=1e-6)
        assert np.isclose(min(ay), 0.0, atol=1e-6)
        assert np.isclose(max(ay), 0.25 * (H - 1), atol=1e-6)

        # Second part: bottom-right quadrant
        bx, by = result.geoms[1].exterior.coords.xy
        assert np.isclose(min(bx), 0.75 * (W - 1), atol=1e-6)
        assert np.isclose(max(bx), W - 1, atol=1e-6)
        assert np.isclose(min(by), 0.75 * (H - 1), atol=1e-6)
        assert np.isclose(max(by), H - 1, atol=1e-6)

    def test_empty_polygon(self):
        """Empty Polygon in -> empty Polygon out."""
        lons, lats = self._grid()
        result = griddata.polygon_to_image_coords(Polygon(), lons, lats)
        assert isinstance(result, Polygon)
        assert result.is_empty

    def test_empty_multipolygon(self):
        """Empty MultiPolygon in -> empty MultiPolygon out."""
        lons, lats = self._grid()
        result = griddata.polygon_to_image_coords(MultiPolygon(), lons, lats)
        assert isinstance(result, MultiPolygon)
        assert result.is_empty

    def test_nearest_method_snaps_to_pixels(self):
        """method='nearest' should yield integer-valued coordinates."""
        H, W = 20, 30
        lons, lats = self._grid(H, W)

        # Vertices deliberately off-grid (between pixels)
        pol = Polygon([
            (0.012, 45.012),
            (0.488, 45.012),
            (0.488, 45.488),
            (0.012, 45.488),
        ])

        result = griddata.polygon_to_image_coords(pol, lons, lats, method="nearest")
        coords = np.asarray(result.exterior.coords)
        # All values must be integer pixel indices
        assert np.allclose(coords, np.round(coords))
        # And within the image bounds
        assert coords[:, 0].min() >= 0
        assert coords[:, 0].max() <= W - 1
        assert coords[:, 1].min() >= 0
        assert coords[:, 1].max() <= H - 1

    def test_vertex_outside_convex_hull_falls_back(self):
        """A vertex outside the LUT extent should fall back to nearest pixel
        instead of becoming NaN under method='linear'."""
        H, W = 20, 30
        lons, lats = self._grid(H, W)

        # Far outside the grid (lon=2.0, lat=47.0 vs grid [0,1]x[45,46])
        pol = Polygon([
            (0.0, 45.0),
            (2.0, 47.0),
            (0.5, 45.5),
        ])

        result = griddata.polygon_to_image_coords(pol, lons, lats, method="linear")
        coords = np.asarray(result.exterior.coords)
        # No NaNs
        assert np.isfinite(coords).all()
        # The out-of-hull vertex snaps to the corner pixel (W-1, H-1)
        assert np.isclose(coords[1, 0], W - 1, atol=1e-6)
        assert np.isclose(coords[1, 1], H - 1, atol=1e-6)

    def test_invalid_method_raises(self):
        lons, lats = self._grid()
        with pytest.raises(ValueError, match="method"):
            griddata.polygon_to_image_coords(box(0, 45, 0.5, 45.5), lons, lats, method="cubic")

    def test_shape_mismatch_raises(self):
        lons = np.zeros((10, 20))
        lats = np.zeros((20, 10))
        with pytest.raises(ValueError, match="shape"):
            griddata.polygon_to_image_coords(box(0, 0, 1, 1), lons, lats)

    def test_non_2d_raises(self):
        lons = np.zeros(10)
        lats = np.zeros(10)
        with pytest.raises(ValueError, match="2D"):
            griddata.polygon_to_image_coords(box(0, 0, 1, 1), lons, lats)

    def test_wrong_geometry_type_raises(self):
        from shapely.geometry import Point
        lons, lats = self._grid()
        with pytest.raises(TypeError, match="Polygon or MultiPolygon"):
            griddata.polygon_to_image_coords(Point(0.5, 45.5), lons, lats)
