"""
Tests for the georeader.io module.

These tests verify I/O utilities including:
- URL detection
- Safe NetCDF opening with fallback engines
"""

import os
import tempfile

import pytest

from georeader import io


class TestIsUrl:
    """Tests for is_url function."""

    def test_http_url(self):
        """Test HTTP URL detection."""
        assert io.is_url("http://example.com/data.nc") is True

    def test_https_url(self):
        """Test HTTPS URL detection."""
        assert io.is_url("https://example.com/data.nc") is True

    def test_ftp_url(self):
        """Test FTP URL detection."""
        assert io.is_url("ftp://example.com/data.nc") is True

    def test_local_path_not_url(self):
        """Test that local paths are not URLs."""
        assert io.is_url("/home/user/data.nc") is False

    def test_relative_path_not_url(self):
        """Test that relative paths are not URLs."""
        assert io.is_url("data/file.nc") is False

    def test_windows_path_not_url(self):
        """Test that Windows paths are not URLs."""
        assert io.is_url("C:\\Users\\data.nc") is False

    def test_non_string_not_url(self):
        """Test that non-string inputs return False."""
        assert io.is_url(123) is False
        assert io.is_url(None) is False
        assert io.is_url(["http://example.com"]) is False


class TestSafeOpenNetcdf:
    """Tests for safe_open_netcdf function."""

    @pytest.fixture
    def simple_netcdf(self):
        """Create a simple NetCDF file for testing."""
        pytest.importorskip("xarray")
        pytest.importorskip("scipy")  # Needed for writing netcdf

        import numpy as np
        import xarray as xr

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.nc")

            # Create simple dataset
            ds = xr.Dataset({
                "temperature": (["x", "y"], np.random.rand(10, 10)),
                "pressure": (["x", "y"], np.random.rand(10, 10)),
            })
            ds.to_netcdf(path, engine="scipy")

            yield path

    def test_open_local_file(self, simple_netcdf):
        """Test opening a local NetCDF file."""
        ds = io.safe_open_netcdf(simple_netcdf)

        assert ds is not None
        assert "temperature" in ds
        assert "pressure" in ds

    def test_open_with_load_false(self, simple_netcdf):
        """Test opening without loading into memory."""
        ds = io.safe_open_netcdf(simple_netcdf, load=False)

        assert ds is not None
        # Data should still be accessible
        assert ds["temperature"].shape == (10, 10)

    def test_invalid_file_raises(self):
        """Test that invalid file raises IOError."""
        with pytest.raises(IOError):
            io.safe_open_netcdf("/nonexistent/path/file.nc")

    def test_requires_xarray(self, monkeypatch):
        """Test that missing xarray raises ImportError."""
        # Skip if we can't import xarray at all
        pytest.importorskip("xarray")

        # Mock HAS_XARRAY to be False
        monkeypatch.setattr(io, "HAS_XARRAY", False)

        with pytest.raises(ImportError):
            io.safe_open_netcdf("test.nc")
