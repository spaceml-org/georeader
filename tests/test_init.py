"""
Tests for the georeader.__init__ module.

These tests verify CRS utilities and coordinate conversion functions.
"""

import math

import pytest
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from shapely.geometry import Point, Polygon

import georeader


class TestCompareCrs:
    """Tests for compare_crs function."""

    def test_same_epsg_string(self):
        """Test comparing identical EPSG strings."""
        assert georeader.compare_crs("EPSG:4326", "EPSG:4326") is True

    def test_different_case(self):
        """Test comparing EPSG strings with different case."""
        assert georeader.compare_crs("EPSG:4326", "epsg:4326") is True

    def test_with_init_prefix(self):
        """Test comparing with +init= prefix."""
        assert georeader.compare_crs("+init=EPSG:4326", "EPSG:4326") is True

    def test_different_epsg(self):
        """Test comparing different EPSG codes."""
        assert georeader.compare_crs("EPSG:4326", "EPSG:32631") is False

    def test_crs_objects(self):
        """Test comparing CRS objects."""
        crs1 = CRS.from_epsg(4326)
        crs2 = CRS.from_epsg(4326)
        assert georeader.compare_crs(crs1, crs2) is True


class TestGetUtmEpsg:
    """Tests for get_utm_epsg function."""

    def test_northern_hemisphere_point(self):
        """Test UTM zone for point in northern hemisphere."""
        # London (approximately 0 degrees longitude, 51 degrees latitude)
        result = georeader.get_utm_epsg((0, 51))

        assert result.startswith("EPSG:326")  # Northern hemisphere UTM

    def test_southern_hemisphere_point(self):
        """Test UTM zone for point in southern hemisphere."""
        # Sydney (approximately 151 degrees longitude, -34 degrees latitude)
        result = georeader.get_utm_epsg((151, -34))

        assert result.startswith("EPSG:327")  # Southern hemisphere UTM

    def test_zone_calculation(self):
        """Test correct zone calculation."""
        # Zone 31 should be around 0-6 degrees east
        result = georeader.get_utm_epsg((3, 45))

        assert result == "EPSG:32631"  # UTM zone 31N

    def test_with_geometry(self):
        """Test with shapely geometry."""
        point = Point(3, 45)
        result = georeader.get_utm_epsg(point)

        assert result == "EPSG:32631"

    def test_with_polygon_uses_centroid(self):
        """Test with polygon uses centroid."""
        polygon = Polygon([(0, 44), (6, 44), (6, 46), (0, 46), (0, 44)])
        result = georeader.get_utm_epsg(polygon)

        # Centroid is at (3, 45) which is zone 31N
        assert result == "EPSG:32631"

    def test_different_input_crs(self):
        """Test with non-WGS84 input CRS."""
        # Point in UTM coordinates that we convert
        result = georeader.get_utm_epsg((500000, 5000000), crs_point_or_geom="EPSG:32631")

        # Should return a valid UTM EPSG
        assert result.startswith("EPSG:32")


class TestGetUtmFromMgrs:
    """Tests for get_utm_from_mgrs function."""

    def test_northern_hemisphere_tile(self):
        """Test MGRS tile in northern hemisphere."""
        result = georeader.get_utm_from_mgrs("31T")

        assert isinstance(result, CRS)
        # Zone 31, northern hemisphere
        assert "utm" in result.to_proj4().lower()

    def test_southern_hemisphere_tile(self):
        """Test MGRS tile in southern hemisphere."""
        result = georeader.get_utm_from_mgrs("56H")  # H is south of N

        assert isinstance(result, CRS)

    def test_full_mgrs_tile(self):
        """Test with full MGRS tile ID."""
        result = georeader.get_utm_from_mgrs("31TFN")

        assert isinstance(result, CRS)


class TestRasterioCrs:
    """Tests for rasterio_crs function."""

    def test_from_epsg_string(self):
        """Test converting EPSG string to CRS."""
        result = georeader.rasterio_crs("EPSG:4326")

        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_from_epsg_int(self):
        """Test converting EPSG integer to CRS."""
        result = georeader.rasterio_crs(4326)

        assert isinstance(result, CRS)
        assert result.to_epsg() == 4326

    def test_from_crs_object(self):
        """Test passing CRS object returns same."""
        crs = CRS.from_epsg(4326)
        result = georeader.rasterio_crs(crs)

        assert result == crs

    def test_invalid_type_raises(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError):
            georeader.rasterio_crs([4326])  # List is not valid


class TestRes:
    """Tests for res function."""

    def test_rectilinear_transform(self):
        """Test resolution from rectilinear transform."""
        transform = from_origin(0, 100, 10, 10)

        result = georeader.res(transform)

        assert abs(result[0] - 10) < 0.001
        assert abs(result[1] - 10) < 0.001

    def test_different_xy_resolution(self):
        """Test with different x and y resolution."""
        transform = from_origin(0, 100, 10, 20)

        result = georeader.res(transform)

        assert abs(result[0] - 10) < 0.001
        assert abs(result[1] - 20) < 0.001

    def test_rotated_transform(self):
        """Test resolution from rotated transform."""
        # Create a rotated transform
        angle = math.pi / 6  # 30 degrees
        scale = 10
        transform = Affine(
            scale * math.cos(angle), -scale * math.sin(angle), 0, scale * math.sin(angle), -scale * math.cos(angle), 100
        )

        result = georeader.res(transform)

        # Resolution should still be ~10 in both directions
        assert abs(result[0] - 10) < 0.1
        assert abs(result[1] - 10) < 0.1


class TestDistanceMeters:
    """Tests for distance_meters function."""

    def test_same_point(self):
        """Test distance between same point is zero."""
        point = Point(0, 0)

        result = georeader.distance_meters(point, point)

        assert abs(result) < 1  # Should be essentially zero

    def test_known_distance(self):
        """Test distance between known points."""
        # Two points approximately 111km apart (1 degree at equator)
        point1 = Point(0, 0)
        point2 = Point(1, 0)

        result = georeader.distance_meters(point1, point2)

        # Should be approximately 111km (111000m)
        assert 100000 < result < 120000

    def test_north_south_distance(self):
        """Test distance in north-south direction."""
        point1 = Point(0, 0)
        point2 = Point(0, 1)

        result = georeader.distance_meters(point1, point2)

        # 1 degree latitude is approximately 111km
        assert 100000 < result < 120000


class TestPixelSizeMeters:
    """Tests for pixel_size_meters function."""

    def test_projected_crs(self):
        """Test pixel size for projected CRS."""
        point = Point(500000, 5000000)
        transform = from_origin(400000, 5100000, 10, 10)

        result = georeader.pixel_size_meters(point, crs_transform="EPSG:32631", transform=transform)

        # UTM is in meters, so 10m pixels
        assert abs(result[0] - 10) < 0.1
        assert abs(result[1] - 10) < 0.1

    def test_geographic_crs(self):
        """Test pixel size for geographic CRS."""
        point = Point(0, 45)
        # Create a transform with ~0.0001 degree pixels
        transform = from_origin(-1, 46, 0.0001, 0.0001)

        result = georeader.pixel_size_meters(
            point, crs_transform="EPSG:4326", transform=transform, crs_point="EPSG:4326"
        )

        # 0.0001 degree at 45 degrees latitude should be ~8-11 meters
        assert 5 < result[0] < 15
        assert 5 < result[1] < 15
