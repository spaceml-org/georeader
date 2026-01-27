"""
Tests for the georeader.save module.

These tests verify file saving operations including:
- Saving GeoTensors as tiled GeoTIFFs
- Saving as Cloud Optimized GeoTIFFs (COG)
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from georeader import save
from georeader.geotensor import GeoTensor


@pytest.fixture
def sample_geotensor():
    """Create a sample GeoTensor for saving tests."""
    data = np.random.randint(0, 10000, (3, 100, 100), dtype=np.int16)
    transform = from_origin(500000, 4500000, 10, 10)
    return GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=-9999)


@pytest.fixture
def sample_geotensor_float():
    """Create a float GeoTensor for saving tests."""
    data = np.random.rand(3, 100, 100).astype(np.float32)
    transform = from_origin(500000, 4500000, 10, 10)
    return GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=-9999)


class TestSaveTiledGeotiff:
    """Tests for save_tiled_geotiff function."""

    def test_basic_save(self, sample_geotensor):
        """Test basic GeoTIFF saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")

            save.save_tiled_geotiff(sample_geotensor, path)

            assert os.path.exists(path)

            # Verify the saved file
            with rasterio.open(path) as src:
                assert src.count == 3
                assert src.width == 100
                assert src.height == 100
                assert src.crs is not None

    def test_save_with_descriptions(self, sample_geotensor):
        """Test saving with band descriptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")
            descriptions = ["Red", "Green", "Blue"]

            save.save_tiled_geotiff(sample_geotensor, path, descriptions=descriptions)

            with rasterio.open(path) as src:
                assert src.descriptions == tuple(descriptions)

    def test_save_with_tags(self, sample_geotensor):
        """Test saving with custom tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")
            tags = {"AUTHOR": "test", "DATE": "2023-01-01"}

            save.save_tiled_geotiff(sample_geotensor, path, tags=tags)

            with rasterio.open(path) as src:
                file_tags = src.tags()
                assert file_tags.get("AUTHOR") == "test"

    def test_save_float_data(self, sample_geotensor_float):
        """Test saving float data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")

            save.save_tiled_geotiff(sample_geotensor_float, path)

            with rasterio.open(path) as src:
                assert src.dtypes[0] == "float32"

    def test_save_and_read_roundtrip(self, sample_geotensor):
        """Test that saved data can be read back correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.tif")

            save.save_tiled_geotiff(sample_geotensor, path)

            with rasterio.open(path) as src:
                data_read = src.read()

            assert np.allclose(sample_geotensor.values, data_read)


class TestSaveCog:
    """Tests for save_cog (Cloud Optimized GeoTIFF) function."""

    def test_basic_cog_save(self, sample_geotensor):
        """Test basic COG saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_cog.tif")

            save.save_cog(sample_geotensor, path)

            assert os.path.exists(path)

            # Verify the saved file
            with rasterio.open(path) as src:
                assert src.count == 3
                assert src.width == 100
                assert src.height == 100

    def test_cog_with_descriptions(self, sample_geotensor):
        """Test COG saving with band descriptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_cog.tif")
            descriptions = ["Band1", "Band2", "Band3"]

            save.save_cog(sample_geotensor, path, descriptions=descriptions)

            with rasterio.open(path) as src:
                assert src.descriptions == tuple(descriptions)

    def test_cog_roundtrip(self, sample_geotensor):
        """Test COG save and read roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_cog.tif")

            save.save_cog(sample_geotensor, path)

            with rasterio.open(path) as src:
                data_read = src.read()
                transform_read = src.transform

            assert np.allclose(sample_geotensor.values, data_read)
            assert sample_geotensor.transform.almost_equals(transform_read)


class TestSave2DData:
    """Tests for saving 2D GeoTensor data."""

    def test_save_2d_geotiff(self):
        """Test saving 2D data as GeoTIFF."""
        data = np.random.rand(100, 100).astype(np.float32)
        transform = from_origin(500000, 4500000, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_2d.tif")

            save.save_tiled_geotiff(geotensor, path)

            with rasterio.open(path) as src:
                assert src.count == 1
                data_read = src.read(1)
                assert data_read.shape == (100, 100)
