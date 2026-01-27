"""
Tests for the georeader.slices module.

These tests verify slice and window generation for tiling operations.
"""

import pytest
import rasterio.windows

from georeader.slices import _slices, create_slices, create_windows


class TestSlices:
    """Tests for the internal _slices function."""

    def test_basic_slicing(self):
        """Test basic slicing without overlap."""
        result = _slices(dimsize=100, size=25)

        assert len(result) == 4
        assert result[0] == slice(0, 25)
        assert result[1] == slice(25, 50)
        assert result[2] == slice(50, 75)
        assert result[3] == slice(75, 100)

    def test_slicing_with_overlap(self):
        """Test slicing with overlap."""
        result = _slices(dimsize=100, size=30, overlap=10)

        # stride = 30 - 10 = 20
        # starts: 0, 20, 40, 60, 80
        assert len(result) >= 4
        assert result[0] == slice(0, 30)
        assert result[1] == slice(20, 50)

    def test_dimsize_smaller_than_size(self):
        """Test when dimension is smaller than window size."""
        result = _slices(dimsize=10, size=25)

        assert len(result) == 1
        assert result[0] == slice(0, 25)  # Extends beyond dimsize

    def test_exclude_incomplete(self):
        """Test excluding incomplete slices at borders."""
        result = _slices(dimsize=100, size=30, include_incomplete=False)

        # Only complete windows: 0-30, 30-60, 60-90
        assert len(result) == 3
        for s in result:
            assert s.stop - s.start == 30

    def test_trim_incomplete(self):
        """Test trimming incomplete slices."""
        result = _slices(dimsize=100, size=30, include_incomplete=True, trim_incomplete=True)

        # Last slice should be trimmed to dimsize
        assert result[-1].stop == 100

    def test_start_negative_if_padding(self):
        """Test starting with negative offset for padding."""
        result = _slices(dimsize=100, size=30, overlap=10, start_negative_if_padding=True)

        # Should start at -overlap//2 = -5
        assert result[0].start == -5


class TestCreateSlices:
    """Tests for create_slices function."""

    def test_basic_2d_slicing(self):
        """Test 2D slicing without overlap."""
        named_shape = {"x": 100, "y": 100}
        dims = {"x": 50, "y": 50}

        result = create_slices(named_shape, dims)

        # Should have 4 windows (2x2)
        assert len(result) == 4
        # Each result should be a dict with x and y slices
        assert "x" in result[0] and "y" in result[0]

    def test_with_overlap(self):
        """Test 2D slicing with overlap."""
        named_shape = {"x": 100, "y": 100}
        dims = {"x": 50, "y": 50}
        overlap = {"x": 10, "y": 10}

        result = create_slices(named_shape, dims, overlap=overlap)

        # More windows due to overlap
        assert len(result) > 4

    def test_single_dimension(self):
        """Test slicing single dimension."""
        named_shape = {"x": 100, "y": 100}
        dims = {"x": 50}  # Only slice in x

        result = create_slices(named_shape, dims)

        assert len(result) == 2
        assert "x" in result[0]
        assert "y" not in result[0]

    def test_exclude_incomplete_2d(self):
        """Test excluding incomplete slices in 2D."""
        named_shape = {"x": 100, "y": 100}
        dims = {"x": 30, "y": 30}

        result = create_slices(named_shape, dims, include_incomplete=False)

        # 3x3 = 9 complete windows
        assert len(result) == 9


class TestCreateWindows:
    """Tests for create_windows function."""

    def test_basic_window_creation(self):
        """Test basic window creation."""
        geodata_shape = (100, 100)  # (height, width)
        window_size = (50, 50)

        result = create_windows(geodata_shape, window_size)

        assert len(result) == 4  # 2x2 windows
        assert all(isinstance(w, rasterio.windows.Window) for w in result)

    def test_window_dimensions(self):
        """Test that windows have correct dimensions."""
        geodata_shape = (100, 100)
        window_size = (50, 50)

        result = create_windows(geodata_shape, window_size, trim_incomplete=False)

        for window in result:
            assert window.height == 50
            assert window.width == 50

    def test_with_overlap(self):
        """Test window creation with overlap."""
        geodata_shape = (100, 100)
        window_size = (50, 50)
        overlap = (10, 10)

        result = create_windows(geodata_shape, window_size, overlap=overlap)

        # More windows due to overlap
        assert len(result) > 4

    def test_trim_incomplete_windows(self):
        """Test trimming incomplete windows at edges."""
        geodata_shape = (110, 110)
        window_size = (50, 50)

        result = create_windows(geodata_shape, window_size, trim_incomplete=True)

        # Check that edge windows are trimmed
        # Last window in each direction should end at 110
        last_window = result[-1]
        assert last_window.row_off + last_window.height <= 110
        assert last_window.col_off + last_window.width <= 110

    def test_exclude_incomplete_windows(self):
        """Test excluding incomplete windows."""
        geodata_shape = (100, 100)
        window_size = (30, 30)

        result = create_windows(geodata_shape, window_size, include_incomplete=False, trim_incomplete=False)

        # Only 3x3 = 9 complete windows
        assert len(result) == 9
        for window in result:
            assert window.height == 30
            assert window.width == 30

    def test_window_coverage(self):
        """Test that windows cover the entire area."""
        geodata_shape = (100, 100)
        window_size = (50, 50)

        result = create_windows(geodata_shape, window_size, include_incomplete=True)

        # Check coverage
        rows_covered = set()
        cols_covered = set()
        for window in result:
            for r in range(int(window.row_off), int(window.row_off + window.height)):
                if 0 <= r < 100:
                    rows_covered.add(r)
            for c in range(int(window.col_off), int(window.col_off + window.width)):
                if 0 <= c < 100:
                    cols_covered.add(c)

        assert len(rows_covered) == 100
        assert len(cols_covered) == 100

    def test_negative_start_for_prediction(self):
        """Test windows with negative start for prediction mode."""
        geodata_shape = (100, 100)
        window_size = (50, 50)
        overlap = (10, 10)

        result = create_windows(geodata_shape, window_size, overlap=overlap, start_negative_if_padding=True)

        # First window should have negative offset
        assert result[0].row_off < 0 or result[0].col_off < 0
