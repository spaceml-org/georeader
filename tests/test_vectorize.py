"""
Tests for the georeader.vectorize module.

These tests verify vectorization of binary masks and polygon transformation.
"""

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon

from georeader.vectorize import get_polygons, transform_polygon


class TestGetPolygons:
    """Tests for get_polygons function."""

    def test_single_polygon(self):
        """Test vectorizing a simple single polygon mask."""
        # Create a simple square mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1

        result = get_polygons(mask, min_area=0)

        assert len(result) >= 1
        # At least one polygon should represent the square
        total_area = sum(p.area for p in result)
        assert total_area > 0

    def test_min_area_filter(self):
        """Test that small polygons are filtered by min_area."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Large polygon
        mask[10:90, 10:90] = 1
        # Small polygon (5x5 = 25 pixels)
        mask[2:7, 2:7] = 1

        # Filter out polygons smaller than 30 pixels
        result = get_polygons(mask, min_area=30)

        # Only the large polygon should remain
        assert len(result) == 1
        assert result[0].area > 30

    def test_multiple_polygons(self):
        """Test vectorizing mask with multiple polygons."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Two separate squares
        mask[10:30, 10:30] = 1
        mask[60:80, 60:80] = 1

        result = get_polygons(mask, min_area=0)

        assert len(result) >= 2

    def test_with_transform(self):
        """Test vectorizing with geographic transform."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1
        transform = from_origin(0, 100, 1, 1)  # 1m resolution

        result = get_polygons(mask, min_area=0, transform=transform)

        assert len(result) >= 1
        # Polygon should be in geographic coordinates
        bounds = result[0].bounds
        # With transform, coordinates should be transformed
        assert bounds[0] >= 0  # xmin

    def test_polygon_buffer(self):
        """Test polygon buffering."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1  # 20x20 square

        result_no_buffer = get_polygons(mask, min_area=0, polygon_buffer=0)
        result_with_buffer = get_polygons(mask, min_area=0, polygon_buffer=5)

        # Buffered polygon should be larger
        assert result_with_buffer[0].area > result_no_buffer[0].area

    def test_tolerance_simplification(self):
        """Test polygon simplification with tolerance."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Create a complex shape
        mask[20:80, 20:80] = 1
        mask[30:70, 30:70] = 0  # Hole

        result_low_tol = get_polygons(mask, min_area=0, tolerance=0.1)
        result_high_tol = get_polygons(mask, min_area=0, tolerance=10)

        # Higher tolerance should produce simpler polygons (fewer vertices)
        if len(result_low_tol) > 0 and len(result_high_tol) > 0:
            # Get vertex count from exterior
            low_tol_vertices = len(result_low_tol[0].exterior.coords)
            high_tol_vertices = len(result_high_tol[0].exterior.coords)
            assert high_tol_vertices <= low_tol_vertices

    def test_empty_mask(self):
        """Test vectorizing an empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = get_polygons(mask, min_area=0)

        assert len(result) == 0


class TestTransformPolygon:
    """Tests for transform_polygon function."""

    def test_identity_transform(self):
        """Test with identity transform."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        transform = from_origin(0, 0, 1, -1)  # Identity-like

        result = transform_polygon(polygon, transform)

        assert isinstance(result, (Polygon, MultiPolygon))
        assert result.is_valid

    def test_scale_transform(self):
        """Test transformation with scaling."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        transform = from_origin(0, 100, 10, 10)  # 10m resolution

        result = transform_polygon(polygon, transform)

        assert isinstance(result, Polygon)
        # Polygon should be scaled
        bounds = result.bounds
        # Original was 10x10 in pixel coords, should be 100x100 in geo coords
        assert bounds[2] - bounds[0] == pytest.approx(100, rel=0.1)

    def test_offset_transform(self):
        """Test transformation with offset."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        transform = from_origin(100, 200, 1, 1)

        result = transform_polygon(polygon, transform)

        bounds = result.bounds
        assert bounds[0] >= 100  # Offset applied

    def test_relative_coordinates(self):
        """Test transformation to relative coordinates."""
        polygon = Polygon([(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)])
        transform = from_origin(0, 100, 1, 1)
        shape_raster = (100, 100)

        result = transform_polygon(polygon, transform, relative=True, shape_raster=shape_raster)

        bounds = result.bounds
        # Relative coordinates should be between 0 and 1
        assert bounds[0] >= 0
        assert bounds[2] <= 1
        assert bounds[1] >= -1  # y can be negative due to transform direction
        assert bounds[3] <= 1

    def test_multipolygon(self):
        """Test transformation of MultiPolygon."""
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        multipolygon = MultiPolygon([poly1, poly2])
        transform = from_origin(0, 10, 1, 1)

        result = transform_polygon(multipolygon, transform)

        assert isinstance(result, MultiPolygon)
        assert len(result.geoms) == 2

    def test_polygon_with_hole(self):
        """Test transformation of polygon with hole."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7), (3, 3)]
        polygon = Polygon(exterior, [hole])
        transform = from_origin(0, 100, 1, 1)

        result = transform_polygon(polygon, transform)

        assert isinstance(result, Polygon)
        # Should preserve the hole
        assert len(result.interiors) == 1
