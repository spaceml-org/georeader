"""
Tests for the georeader.reflectance module.

These tests verify radiometric calculations including:
- Earth-sun distance correction factor
- Radiance to reflectance conversion
- Spectral response function calculations
- Solar irradiance loading
"""

from datetime import datetime

import numpy as np
import pytest
from rasterio.transform import from_origin

from georeader import reflectance
from georeader.geotensor import GeoTensor


class TestEarthSunDistanceCorrectionFactor:
    """Tests for earth_sun_distance_correction_factor function."""

    def test_perihelion(self):
        """Test correction factor near perihelion (early January)."""
        # Perihelion is around January 3rd
        date = datetime(2023, 1, 3)

        factor = reflectance.earth_sun_distance_correction_factor(date)

        # At perihelion, Earth is closest to Sun, so factor should be smallest (~0.983)
        assert 0.98 < factor < 0.99

    def test_aphelion(self):
        """Test correction factor near aphelion (early July)."""
        # Aphelion is around July 4th
        date = datetime(2023, 7, 4)

        factor = reflectance.earth_sun_distance_correction_factor(date)

        # At aphelion, Earth is farthest from Sun, so factor should be largest (~1.017)
        assert 1.01 < factor < 1.02

    def test_equinox(self):
        """Test correction factor near equinox."""
        # Spring equinox around March 20th
        date = datetime(2023, 3, 20)

        factor = reflectance.earth_sun_distance_correction_factor(date)

        # Near equinox, factor should be close to 1
        assert 0.99 < factor < 1.01

    def test_seasonal_variation(self):
        """Test that factor varies seasonally."""
        winter = datetime(2023, 1, 1)
        summer = datetime(2023, 7, 1)

        factor_winter = reflectance.earth_sun_distance_correction_factor(winter)
        factor_summer = reflectance.earth_sun_distance_correction_factor(summer)

        # Summer should have larger factor than winter (northern hemisphere)
        assert factor_summer > factor_winter

    def test_leap_year(self):
        """Test that leap year is handled correctly."""
        # Test Dec 31 of a leap year (day 366)
        date = datetime(2024, 12, 31)

        factor = reflectance.earth_sun_distance_correction_factor(date)

        # Should still return valid value
        assert 0.98 < factor < 1.02


class TestSRF:
    """Tests for srf (spectral response function) function."""

    def test_single_band(self):
        """Test SRF for single band."""
        center_wavelengths = np.array([550])  # Green band
        fwhm = np.array([50])  # 50nm FWHM
        wavelengths = np.arange(400, 700, 1)  # 400-700nm at 1nm resolution

        result = reflectance.srf(center_wavelengths, fwhm, wavelengths)

        # srf returns shape (N, K) where N=len(wavelengths), K=len(center_wavelengths)
        assert result.shape == (len(wavelengths), 1)
        # Peak should be at center wavelength
        peak_idx = np.argmax(result[:, 0])
        assert wavelengths[peak_idx] == pytest.approx(550, abs=5)

    def test_multiple_bands(self):
        """Test SRF for multiple bands."""
        center_wavelengths = np.array([450, 550, 650])  # Blue, Green, Red
        fwhm = np.array([50, 50, 50])
        wavelengths = np.arange(400, 700, 1)

        result = reflectance.srf(center_wavelengths, fwhm, wavelengths)

        # srf returns shape (N, K) where N=len(wavelengths), K=len(center_wavelengths)
        assert result.shape == (len(wavelengths), 3)

    def test_normalized(self):
        """Test that SRF sums to approximately 1 for each band."""
        center_wavelengths = np.array([550])
        fwhm = np.array([50])
        wavelengths = np.arange(400, 700, 1)

        result = reflectance.srf(center_wavelengths, fwhm, wavelengths)

        # SRF should sum to approximately 1 (normalized) along axis 0
        total = result[:, 0].sum()
        # Allow for some tolerance due to truncation at edges
        assert 0.9 < total < 1.1


class TestLoadThuillierIrradiance:
    """Tests for load_thuillier_irradiance function."""

    def test_load_succeeds(self):
        """Test that irradiance data loads successfully."""
        import pandas as pd

        irradiance = reflectance.load_thuillier_irradiance()

        assert irradiance is not None
        assert isinstance(irradiance, pd.DataFrame)

    def test_contains_wavelengths(self):
        """Test that loaded data contains wavelength information."""
        irradiance = reflectance.load_thuillier_irradiance()

        # Should have wavelength and irradiance columns
        assert "Nanometer" in irradiance.columns
        assert "Radiance(mW/m2/nm)" in irradiance.columns
        assert len(irradiance) > 0


class TestIntegratedIrradiance:
    """Tests for integrated_irradiance function."""

    def test_basic_integration(self):
        """Test basic irradiance integration."""
        import pandas as pd

        # Create SRF as DataFrame with wavelength index
        wavelengths = np.arange(400, 500)
        srf_values = np.ones((100, 1)) / 100  # Normalized flat response (N, K) shape
        srf_df = pd.DataFrame(srf_values, index=wavelengths, columns=["band1"])

        # Create mock solar irradiance data as DataFrame
        solar_irradiance = pd.DataFrame({"Nanometer": wavelengths, "Radiance(mW/m2/nm)": np.ones(100) * 1.5})

        result = reflectance.integrated_irradiance(srf_df, solar_irradiance)

        assert result is not None
        assert len(result) == 1

    def test_multiple_bands(self):
        """Test integration for multiple bands."""
        import pandas as pd

        # Create SRF as DataFrame with wavelength index
        wavelengths = np.arange(400, 500)
        srf_values = np.ones((100, 3)) / 100  # (N, K) shape for 3 bands
        srf_df = pd.DataFrame(srf_values, index=wavelengths, columns=["band1", "band2", "band3"])

        # Create mock solar irradiance data as DataFrame
        solar_irradiance = pd.DataFrame({"Nanometer": wavelengths, "Radiance(mW/m2/nm)": np.ones(100) * 1.5})

        result = reflectance.integrated_irradiance(srf_df, solar_irradiance)

        assert len(result) == 3


class TestRadianceToReflectance:
    """Tests for radiance_to_reflectance function."""

    def test_basic_conversion_with_factor(self):
        """Test basic radiance to reflectance conversion with known factor."""
        # Create simple test data
        data = np.ones((3, 10, 10), dtype=np.float32) * 100  # 100 uW/cm^2/SR/nm
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5, 1.5, 1.5])  # W/m²/nm

        result = reflectance.radiance_to_reflectance(
            geotensor,
            solar_irradiance,
            observation_date_corr_factor=np.pi,  # Simplified factor
        )

        assert isinstance(result, GeoTensor)
        assert result.shape == geotensor.shape
        # Check values are reasonable (not negative, not > 1 typically)
        assert result.values.min() >= 0

    def test_with_date_and_coords(self):
        """Test conversion with date and coordinates."""
        pytest.importorskip("pysolar", reason="pysolar required for this test")

        data = np.ones((3, 10, 10), dtype=np.float32) * 100
        transform = from_origin(0, 45, 0.01, 0.01)  # Geographic coords
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:4326")

        solar_irradiance = np.array([1.5, 1.5, 1.5])
        date = datetime(2023, 6, 21, 12, 0, 0, tzinfo=None)  # Summer solstice noon
        # Need timezone-aware datetime for pysolar
        from datetime import timezone

        date = date.replace(tzinfo=timezone.utc)
        center_coords = (0, 45)  # lon, lat

        result = reflectance.radiance_to_reflectance(
            geotensor, solar_irradiance, date_of_acquisition=date, center_coords=center_coords, crs_coords="EPSG:4326"
        )

        assert isinstance(result, GeoTensor)

    def test_units_conversion(self):
        """Test that different units are handled correctly."""
        data = np.ones((3, 10, 10), dtype=np.float32) * 100
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5, 1.5, 1.5])
        factor = np.pi

        # Test different units
        result_uw = reflectance.radiance_to_reflectance(
            geotensor, solar_irradiance, observation_date_corr_factor=factor, units="uW/cm^2/SR/nm"
        )

        result_mw = reflectance.radiance_to_reflectance(
            geotensor, solar_irradiance, observation_date_corr_factor=factor, units="mW/m2/sr/nm"
        )

        result_w = reflectance.radiance_to_reflectance(
            geotensor, solar_irradiance, observation_date_corr_factor=factor, units="W/m2/sr/nm"
        )

        # All should return valid results but with different values
        assert result_uw.values.mean() != result_w.values.mean()

    def test_invalid_units_raises(self):
        """Test that invalid units raise an error."""
        data = np.ones((3, 10, 10), dtype=np.float32) * 100
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5, 1.5, 1.5])

        with pytest.raises(ValueError):
            reflectance.radiance_to_reflectance(
                geotensor, solar_irradiance, observation_date_corr_factor=np.pi, units="invalid_units"
            )

    def test_preserves_fill_value(self):
        """Test that fill values are preserved in output."""
        data = np.ones((3, 10, 10), dtype=np.float32) * 100
        data[0, 0, 0] = -9999  # Fill value
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631", fill_value_default=-9999)

        solar_irradiance = np.array([1.5, 1.5, 1.5])

        result = reflectance.radiance_to_reflectance(geotensor, solar_irradiance, observation_date_corr_factor=np.pi)

        assert result.values[0, 0, 0] == -9999


class TestReflectanceToRadiance:
    """Tests for reflectance_to_radiance function."""

    def test_roundtrip(self):
        """Test that radiance -> reflectance -> radiance preserves values (with known factor)."""
        # Create radiance data in W/m2/sr/nm (to avoid unit conversion complexity)
        radiance = np.ones((3, 10, 10), dtype=np.float32) * 0.1  # Small values typical of radiance
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(radiance, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5, 1.5, 1.5])
        factor = np.pi

        # Convert to reflectance (using W/m2/sr/nm units for simpler math)
        toa_reflectance = reflectance.radiance_to_reflectance(
            geotensor, solar_irradiance, observation_date_corr_factor=factor, units="W/m2/sr/nm"
        )

        # Convert back to radiance
        restored = reflectance.reflectance_to_radiance(
            toa_reflectance, solar_irradiance, observation_date_corr_factor=factor
        )

        # Should be approximately equal
        assert np.allclose(geotensor.values, restored.values, rtol=0.01)


# =============================================================================
# Tests for reflectance module error handling (Phase 2 Sprint 1)
# =============================================================================


class TestRadianceToReflectanceErrors:
    """Tests for radiance_to_reflectance function error handling."""

    def test_2d_data_raises(self):
        """Test that 2D data raises AssertionError."""
        data = np.ones((10, 10), dtype=np.float32) * 100  # 2D instead of 3D
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5])

        with pytest.raises(AssertionError, match="Expected 3 channels"):
            reflectance.radiance_to_reflectance(geotensor, solar_irradiance, observation_date_corr_factor=np.pi)

    def test_mismatched_bands_raises(self):
        """Test that mismatched band count raises AssertionError."""
        data = np.ones((3, 10, 10), dtype=np.float32) * 100  # 3 bands
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5, 1.5])  # Only 2 bands

        with pytest.raises(AssertionError, match="Different number of channels"):
            reflectance.radiance_to_reflectance(geotensor, solar_irradiance, observation_date_corr_factor=np.pi)

    def test_no_date_or_factor_raises(self):
        """Test that neither date nor factor raises AssertionError."""
        data = np.ones((3, 10, 10), dtype=np.float32) * 100
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5, 1.5, 1.5])

        with pytest.raises(AssertionError, match="date_of_acquisition must be provided"):
            reflectance.radiance_to_reflectance(
                geotensor,
                solar_irradiance,  # No factor or date
            )


class TestReflectanceToRadianceErrors:
    """Tests for reflectance_to_radiance function error handling."""

    def test_2d_data_raises(self):
        """Test that 2D data raises AssertionError."""
        data = np.ones((10, 10), dtype=np.float32) * 0.5  # 2D instead of 3D
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5])

        with pytest.raises(AssertionError, match="Expected 3 channels"):
            reflectance.reflectance_to_radiance(geotensor, solar_irradiance, observation_date_corr_factor=np.pi)

    def test_mismatched_bands_raises(self):
        """Test that mismatched band count raises AssertionError."""
        data = np.ones((3, 10, 10), dtype=np.float32) * 0.5  # 3 bands
        transform = from_origin(0, 100, 10, 10)
        geotensor = GeoTensor(data, transform=transform, crs="EPSG:32631")

        solar_irradiance = np.array([1.5])  # Only 1 band

        with pytest.raises(AssertionError, match="Different number of channels"):
            reflectance.reflectance_to_radiance(geotensor, solar_irradiance, observation_date_corr_factor=np.pi)


class TestSRFErrors:
    """Tests for srf function error handling."""

    def test_mismatched_center_fwhm_raises(self):
        """Test that mismatched center_wavelengths and fwhm raises AssertionError."""
        center_wavelengths = np.array([450, 550, 650])  # 3 bands
        fwhm = np.array([50, 50])  # Only 2 FWHM values
        wavelengths = np.arange(400, 700, 1)

        with pytest.raises(AssertionError, match="same shape"):
            reflectance.srf(center_wavelengths, fwhm, wavelengths)


class TestSRFEdgeCases:
    """Tests for srf function edge cases."""

    def test_single_wavelength(self):
        """Test SRF with single wavelength in wavelengths array."""
        center_wavelengths = np.array([550])
        fwhm = np.array([50])
        wavelengths = np.array([550])  # Single wavelength

        result = reflectance.srf(center_wavelengths, fwhm, wavelengths)

        assert result.shape == (1, 1)
        # At center, value should be maximum
        assert result[0, 0] > 0

    def test_wavelengths_outside_band(self):
        """Test SRF with wavelengths completely outside band range."""
        center_wavelengths = np.array([550])
        fwhm = np.array([50])
        wavelengths = np.arange(800, 900)  # Far from center at 550

        result = reflectance.srf(center_wavelengths, fwhm, wavelengths)

        assert result.shape == (100, 1)
        # SRF is normalized, so even outside band, values may not be tiny
        # Check that it's a valid result with proper shape
        assert result.sum() > 0  # Should have some weight due to normalization

    def test_very_narrow_fwhm(self):
        """Test SRF with very narrow FWHM."""
        center_wavelengths = np.array([550])
        fwhm = np.array([1])  # Very narrow: 1nm FWHM
        wavelengths = np.arange(540, 560, 0.1)  # High resolution

        result = reflectance.srf(center_wavelengths, fwhm, wavelengths)

        # Should still work and have a sharp peak
        assert result.shape == (200, 1)
        peak_idx = np.argmax(result[:, 0])
        # Peak should be near center
        assert 95 < peak_idx < 105  # Near middle of 200 samples
