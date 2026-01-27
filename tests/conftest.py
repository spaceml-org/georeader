"""
Pytest configuration and fixtures for georeader tests.

This module provides a shared test fixture that creates a temporary GeoTiff file
for testing. The file is created once per test session to minimize overhead.
"""

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
import tempfile
import os


@pytest.fixture(scope="session")
def test_raster_path():
    """
    Create a temporary GeoTiff test file for the test session.
    
    The test file has the following properties (similar to the WorldFloodsv2 dataset):
    - 15 bands
    - Height: 200 pixels
    - Width: 250 pixels
    - CRS: EPSG:32738
    - Resolution: 10.0m x 10.0m
    - Data type: int16
    
    Returns:
        str: Path to the temporary test GeoTiff file
    """
    # Create a temporary directory that persists for the session
    tmpdir = tempfile.mkdtemp()
    test_file = os.path.join(tmpdir, "test_data.tif")
    
    # Create transform matching WorldFloodsv2 format
    transform = from_origin(537430.0, 7844180.0, 10.0, 10.0)
    
    # Create reproducible test data
    np.random.seed(42)
    data = np.random.randint(0, 10000, (15, 200, 250), dtype=np.int16)
    
    with rasterio.open(
        test_file, 'w',
        driver='GTiff',
        height=200,
        width=250,
        count=15,
        dtype=data.dtype,
        crs='EPSG:32738',
        transform=transform,
        tiled=True,
        blockxsize=128,
        blockysize=128,
        compress='deflate'
    ) as dst:
        dst.write(data)
    
    yield test_file
    
    # Cleanup after tests complete
    try:
        os.remove(test_file)
        os.rmdir(tmpdir)
    except OSError:
        pass
