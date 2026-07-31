"""
Tests for interpolate_20mbands_s2ee in georeader.readers.ee_image.

Regression tests for https://github.com/spaceml-org/georeader/issues/43
"""

import numpy as np
import pytest
import warnings
from rasterio.transform import from_origin

from georeader.geotensor import GeoTensor
from georeader.readers.ee_image import interpolate_20mbands_s2ee


# All 13 Sentinel-2 L1C bands
S2_CHANNELS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']


def _make_s2_geotensor(dtype=np.uint16, shape=(13, 64, 64)):
    """Helper to create a synthetic Sentinel-2 GeoTensor."""
    if dtype == np.uint8:
        data = np.random.randint(0, 255, size=shape, dtype=dtype)
    else:
        data = np.random.randint(0, 10000, size=shape, dtype=dtype)
    return GeoTensor(
        data,
        crs="EPSG:32633",
        transform=from_origin(900260.0, 3173500.0, 10.0, 10.0),
    )


class TestInterpolate20mBandsUint8:
    """Tests for uint8 handling in interpolate_20mbands_s2ee (Issue #43)."""

    def test_uint16_no_warning(self):
        """Standard uint16 input should not emit a warning."""
        geotensor = _make_s2_geotensor(dtype=np.uint16)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = interpolate_20mbands_s2ee(geotensor, S2_CHANNELS, inplace=False)
        assert result.dtype == np.uint16

    def test_uint8_casts_with_warning(self):
        """uint8 input should be cast to uint16 and emit a UserWarning.

        Regression test for https://github.com/spaceml-org/georeader/issues/43
        """
        geotensor = _make_s2_geotensor(dtype=np.uint8)

        with pytest.warns(UserWarning, match="Expected np.uint16, found uint8"):
            result = interpolate_20mbands_s2ee(geotensor, S2_CHANNELS, inplace=False)

        assert result.dtype == np.uint16
        assert result.shape == (13, 64, 64)

    def test_uint8_inplace_casts_with_warning(self):
        """uint8 input with inplace=True should also work.

        Regression test for https://github.com/spaceml-org/georeader/issues/43
        """
        geotensor = _make_s2_geotensor(dtype=np.uint8)

        with pytest.warns(UserWarning, match="Expected np.uint16, found uint8"):
            result = interpolate_20mbands_s2ee(geotensor, S2_CHANNELS, inplace=True)

        assert result.dtype == np.uint16

    def test_uint8_preserves_values(self):
        """uint8 values should be correctly preserved after casting to uint16.

        Regression test for https://github.com/spaceml-org/georeader/issues/43
        """
        geotensor = _make_s2_geotensor(dtype=np.uint8)
        original_10m_values = geotensor.values[1].copy()  # B02 is a 10m band

        with pytest.warns(UserWarning):
            result = interpolate_20mbands_s2ee(geotensor, S2_CHANNELS, inplace=False)

        # 10m bands (B02, B03, B04, B08) should not be modified by interpolation
        # After cast, B02 values should match the original uint8 values
        np.testing.assert_array_equal(result.values[1], original_10m_values.astype(np.uint16))
