"""
Radiometric Conversion Module for Top-of-Atmosphere Reflectance and Radiance.

This module provides functions for converting between radiance and top-of-atmosphere
(ToA) reflectance, handling spectral response functions (SRF), and computing
solar irradiance integrals. It is essential for calibrating satellite imagery
from raw digital numbers to physically meaningful radiometric quantities.

Unit Conventions & Conversion Pipeline
--------------------------------------

The module handles conversions between different unit systems commonly used in
remote sensing:

::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RADIOMETRIC UNIT CONVERSION FLOW                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Raw DN ──────►  Radiance ──────────────────────────►  Reflectance     │
    │   (counts)        (W/m²/sr/nm)                          (unitless 0-1)  │
    │                                                                          │
    │   Supported radiance units:                                              │
    │   ┌────────────────────────────────────────────────────────────────┐    │
    │   │  Unit              │  Factor to W/m²/sr/nm                     │    │
    │   ├────────────────────┼──────────────────────────────────────────┤    │
    │   │  W/m²/sr/nm        │  1.0         (no conversion)             │    │
    │   │  mW/m²/sr/nm       │  ÷ 1000      (milli → base)              │    │
    │   │  µW/cm²/sr/nm      │  ÷ 100       (micro/cm² → base/m²)       │    │
    │   └────────────────────┴──────────────────────────────────────────┘    │
    │                                                                          │
    │   Solar Irradiance: W/m²/nm or mW/m²/nm (at TOA, perpendicular)         │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Physics of ToA Reflectance Conversion
-------------------------------------

ToA reflectance (ρ) accounts for solar illumination geometry and Earth-Sun distance:

::

    ρ = (π × d² × L) / (E_sun × cos(θ_z))

    where:
    - L      = at-sensor radiance (W/m²/sr/nm)
    - E_sun  = solar irradiance at TOA (W/m²/nm)
    - d      = Earth-Sun distance in AU (varies ~3% annually)
    - θ_z    = solar zenith angle (0° = Sun overhead)

The observation_date_correction_factor combines these geometric factors:

::

    obfactor = (π × d²) / cos(θ_z)

    Then:  ρ = L × obfactor / E_sun

Earth-Sun Distance Variation
---------------------------

::

    ┌──────────────────────────────────────────────────────────────────┐
    │       Earth-Sun Distance Throughout the Year                      │
    │                                                                    │
    │  Distance   ▲                                                      │
    │  (AU)       │     Aphelion (~July 4)                               │
    │             │          ╭───────╮                                   │
    │  1.017 ─────┼─────────╱         ╲─────────────────────────         │
    │             │        ╱           ╲                                 │
    │  1.000 ─────┼───────╱─────────────╲──────────────────────         │
    │             │      ╱               ╲                               │
    │  0.983 ─────┼─────╱─────────────────╲────────────────────         │
    │             │    ╱    Perihelion     ╲                             │
    │             │   ╱     (~Jan 3)        ╲                            │
    │             └───┴──────────────────────┴────────────────► Day      │
    │                 0    91   182   273   365                          │
    │                Jan  Apr   Jul   Oct   Jan                          │
    │                                                                    │
    │  d = 1 - 0.01673 × cos(0.0172 × (day_of_year - 4))                │
    │                                                                    │
    │  Impact: ~6.5% variation in irradiance (d² factor)                │
    └──────────────────────────────────────────────────────────────────┘

Spectral Response Functions (SRF)
--------------------------------

When converting hyperspectral to multispectral data, the SRF defines how
each band integrates radiance across wavelengths:

::

    ┌─────────────────────────────────────────────────────────────────┐
    │  Spectral Response Function Convolution                          │
    │                                                                   │
    │  Hyperspectral               SRF for Band X             Result   │
    │  Radiance L(λ)               R(λ)                       L_X     │
    │                                                                   │
    │  L(λ)│     ╱╲                R(λ)│   ╱╲                         │
    │      │    ╱  ╲╱╲╱╲              │  ╱  ╲                         │
    │      │   ╱        ╲             │ ╱    ╲                        │
    │      │  ╱          ╲            │╱      ╲                       │
    │      └──────────────── λ        └──────────── λ                 │
    │            400-2500 nm              λ_center ± FWHM/2            │
    │                                                                   │
    │  Integration:  L_X = ∫ L(λ) × R(λ) dλ  /  ∫ R(λ) dλ             │
    │                                                                   │
    │  The SRF is typically Gaussian:                                  │
    │  R(λ) = exp(-(λ - λ_center)² / (2σ²))                           │
    │  where σ = FWHM / (2 × √(2 × ln(2))) ≈ FWHM / 2.355             │
    └─────────────────────────────────────────────────────────────────┘

Module Functions Overview
------------------------

Core Conversion:
    - :func:`radiance_to_reflectance`: L → ρ with unit handling
    - :func:`reflectance_to_radiance`: ρ → L (inverse)

Correction Factors:
    - :func:`earth_sun_distance_correction_factor`: d from date
    - :func:`observation_date_correction_factor`: Combined π×d²/cos(θ_z)
    - :func:`compute_sza`: Solar zenith angle from location & time

Spectral Integration:
    - :func:`srf`: Build Gaussian spectral response function
    - :func:`integrated_irradiance`: ∫ E_sun(λ) × R(λ) dλ
    - :func:`transform_to_srf`: Hyperspectral → multispectral via SRF

Solar Irradiance Data:
    - :func:`load_thuillier_irradiance`: Standard TOA irradiance (200-2400 nm)

Example Workflow
---------------

Complete conversion from EMIT radiance to Sentinel-2 reflectance bands::

    import numpy as np
    from datetime import datetime
    from georeader.reflectance import (
        radiance_to_reflectance,
        transform_to_srf,
        load_thuillier_irradiance,
        srf
    )
    
    # 1. Load hyperspectral radiance (µW/cm²/sr/nm)
    emit_radiance = ...  # shape: (285, 1242, 1280)
    emit_wavelengths = np.linspace(380, 2500, 285)  # nm
    emit_fwhm = np.full(285, 7.5)  # nm
    
    # 2. Build Sentinel-2 SRF (example for Band 4, Red ~665nm)
    s2_center = [665.0]  # nm
    s2_fwhm = [30.0]     # nm
    wavelengths_fine = np.arange(380, 2501, 1)  # 1nm resolution
    s2_response = srf(s2_center, s2_fwhm, wavelengths_fine)  # (2121, 1)
    
    # 3. Convert hyperspectral radiance to multispectral reflectance
    s2_reflectance = transform_to_srf(
        emit_radiance,
        srf=pd.DataFrame(s2_response, index=wavelengths_fine, columns=['B4']),
        wavelengths_hyperspectral=emit_wavelengths,
        as_reflectance=True,
        date_of_acquisition=datetime(2024, 6, 15, 10, 30),
        units="uW/cm^2/SR/nm"
    )

References
----------
- ESA Sentinel-2 TOA Processing: https://sentiwiki.copernicus.eu/web/s2-processing
- Thuillier Solar Irradiance: Solar Physics, vol. 214, pp. 1-22, 2003
- NASA EMIT L2A Reflectance ATBD: https://lpdaac.usgs.gov/documents/

See Also
--------
georeader.readers.emit : EMIT hyperspectral reader with GLT orthorectification
georeader.readers.prisma : PRISMA reader with built-in radiance calibration
georeader.readers.enmap : EnMAP reader with DN→radiance conversion
"""
from datetime import datetime
from typing import Tuple, Union, List, Optional
from georeader import window_utils
from georeader.geotensor import GeoTensor
from georeader.abstract_reader import GeoData
from georeader import read
import numpy as np
import pandas as pd
import os
from numpy.typing import ArrayLike, NDArray
import numbers


def earth_sun_distance_correction_factor(date_of_acquisition:datetime) -> float:
    """
    Compute the Earth-Sun distance correction factor for a given date.

    The Earth's orbit is slightly elliptical (eccentricity ~0.0167), causing
    solar irradiance at Earth to vary by approximately ±3.4% throughout the year.
    This factor is used to normalize radiance measurements to a standard distance.

    Formula
    -------
    ::

        d = 1 - 0.01673 × cos(0.0172 × (t - 4))

        where:
        - 0.0172 = 2π/365.256363 rad/day (Earth's mean angular velocity)
        - 0.01673 = Earth's orbital eccentricity
        - t = day of year (1-366)
        - The "-4" offset accounts for perihelion occurring ~Jan 3-4

    Unit Analysis
    -------------
    ::

        Input:  date (datetime)
        Output: d (dimensionless, in Astronomical Units)

        Physical interpretation:
        - d ≈ 0.983 AU at perihelion (early January)
        - d ≈ 1.017 AU at aphelion (early July)
        - d² appears in irradiance: E ∝ 1/d² (inverse square law)

    Relationship to Sentinel-2 Metadata
    -----------------------------------
    Sentinel-2 provides ``U`` in the metadata, which is the squared inverse::

        U = 1/d²

    This is directly used in their reflectance formula. To convert::

        d = 1/√U

    Args:
        date_of_acquisition: Date/time of image acquisition. Only the day-of-year
            is used; the time component is ignored for this approximation.

    Returns:
        Earth-Sun distance in AU (typically 0.983 to 1.017).

    Examples
    --------
    >>> from datetime import datetime
    >>> # Perihelion (closest to Sun) around January 3
    >>> d_jan = earth_sun_distance_correction_factor(datetime(2024, 1, 3))
    >>> print(f"January 3: d = {d_jan:.4f} AU")  # ~0.983
    January 3: d = 0.9833 AU

    >>> # Aphelion (farthest from Sun) around July 4
    >>> d_jul = earth_sun_distance_correction_factor(datetime(2024, 7, 4))
    >>> print(f"July 4: d = {d_jul:.4f} AU")  # ~1.017
    July 4: d = 1.0167 AU

    >>> # Irradiance ratio (inverse square)
    >>> irradiance_ratio = (d_jan / d_jul) ** 2
    >>> print(f"Jan/Jul irradiance ratio: {irradiance_ratio:.3f}")  # ~0.935
    Jan/Jul irradiance ratio: 0.935

    See Also
    --------
    observation_date_correction_factor : Combines this with solar zenith angle
    
    References
    ----------
    .. [1] ESA Sentinel-2 Processing:
       https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-TOAReflectanceComputation
    """
    tm_yday = date_of_acquisition.timetuple().tm_yday # from 1 to 365 (or 366!)
    return 1 - 0.01673 * np.cos(0.0172 * (tm_yday - 4))


def compute_sza(center_coords:Tuple[float, float], date_of_acquisition:datetime, crs_coords:Optional[str]=None) -> float:
    """
    This function returns the solar zenith angle for a given location and date of acquisition.

    Args:
        center_coords (Tuple[float, float]): location being considered (x,y) (long, lat if EPSG:4326)
        date_of_acquisition (datetime): date of acquisition to compute the solar zenith angles. It 
            is assumed to be UTC time.
        crs_coords (Optional[str], optional): if None it will assume center_coords are in EPSG:4326. Defaults to None.

    Returns:
        float: solar zenith angle in degrees
    """
    try:
        from pysolar.solar import get_altitude
    except ImportError:
        raise ImportError("pysolar is required to compute the solar zenith angle. Install it with `pip install pysolar`")

    if crs_coords is not None and not window_utils.compare_crs(crs_coords, "EPSG:4326"):
        from rasterio import warp
        centers_long, centers_lat = warp.transform(crs_coords,
                                                   {'init': 'epsg:4326'}, [center_coords[0]], [center_coords[1]])
        centers_long = centers_long[0]
        centers_lat = centers_lat[0]
    else:
        centers_long = center_coords[0]
        centers_lat = center_coords[1]
    
    # Get Solar Altitude (in degrees)
    solar_altitude = get_altitude(latitude_deg=centers_lat, longitude_deg=centers_long,
                                  when=date_of_acquisition)
    return 90 - solar_altitude


def observation_date_correction_factor(center_coords:Tuple[float, float], 
                                       date_of_acquisition:datetime,
                                       crs_coords:Optional[str]=None) -> float:
    """
    This function returns the observation date correction factor given by the formula:
    
    ``
      obfactor = (pi * d^2) / cos(solarzenithangle/180*pi)
    ``
    where:
        - `d` is the Earth-sun distance correction factor. In Sentinel-2 they provide U
            in the metadata which is the square inverse of this factor: `U = 1 / d^2`.
            `d`is computed from the date of acquisition.
        - `solarzenithangle` is obtained from the date of acquisition and location.

    Args:
        center_coords: location being considered (x,y) (long, lat if EPSG:4326) 
        date_of_acquisition: date of acquisition to compute the solar zenith angles.
        crs_coords: if None it will assume center_coords are in EPSG:4326

    Returns:
        correction factor

    """
    sza = compute_sza(center_coords, date_of_acquisition, crs_coords=crs_coords)
    d = earth_sun_distance_correction_factor(date_of_acquisition)

    return np.pi*(d**2) / np.cos(sza/180.*np.pi)


def radiance_to_reflectance(data:Union[GeoTensor, ArrayLike], 
                            solar_irradiance:ArrayLike,
                            date_of_acquisition:Optional[datetime]=None,
                            center_coords:Optional[Tuple[float, float]]=None,
                            crs_coords:Optional[str]=None,
                            observation_date_corr_factor:Optional[float]=None,
                            units:str="uW/cm^2/SR/nm") -> Union[GeoTensor, NDArray]:
    """
    Convert at-sensor radiance to Top-of-Atmosphere (ToA) reflectance.

    This function implements the standard radiometric calibration equation
    that accounts for solar illumination geometry and Earth-Sun distance.

    Physical Equation
    -----------------
    ::

        ρ = (L × π × d²) / (E_sun × cos(θ_z))

        Equivalently using observation_date_correction_factor:
        ρ = L × obfactor / E_sun

        where:
        - ρ      = ToA reflectance (dimensionless, typically 0-1)
        - L      = at-sensor radiance
        - E_sun  = solar spectral irradiance at TOA
        - d      = Earth-Sun distance (AU)
        - θ_z    = solar zenith angle
        - obfactor = π × d² / cos(θ_z)

    Unit Analysis
    -------------
    ::

        ┌─────────────────────────────────────────────────────────────────┐
        │  UNIT CONVERSION FLOW                                            │
        │                                                                   │
        │  Input radiance        Normalized radiance      Output           │
        │  (various units)   →   (W/m²/sr/nm)         →   reflectance     │
        │                                                                   │
        │  ┌─────────────────────────────────────────────────────────────┐ │
        │  │ Input Unit       │ factor_div │ Conversion                  │ │
        │  ├──────────────────┼────────────┼─────────────────────────────┤ │
        │  │ W/m²/sr/nm       │ 1          │ No conversion               │ │
        │  │ mW/m²/sr/nm      │ 1000       │ ×10⁻³ (milli → base)       │ │
        │  │ µW/cm²/sr/nm     │ 100        │ ×10⁻⁶×10⁴ = ×10⁻²         │ │
        │  └──────────────────┴────────────┴─────────────────────────────┘ │
        │                                                                   │
        │  Final calculation:                                              │
        │                                                                   │
        │  L [W/m²/sr/nm] × obfactor [sr⁻¹] / E_sun [W/m²/nm]            │
        │  = dimensionless reflectance                                     │
        │                                                                   │
        │  Note: The steradian cancels with implicit assumptions about     │
        │  the solar disk's solid angle as seen from Earth.               │
        └─────────────────────────────────────────────────────────────────┘

    Args:
        data: Radiance tensor with shape (C, H, W) where C is spectral bands.
            Units must match the ``units`` parameter.
        solar_irradiance: Per-band solar irradiance values with shape (C,).
            **Must be in W/m²/nm** (SI units, NOT mW/m²/nm).
        date_of_acquisition: UTC datetime for computing solar geometry.
            Required if ``observation_date_corr_factor`` is not provided.
        center_coords: Image center as (x, y) or (lon, lat) for solar angle.
            If None and data is GeoTensor, computed from transform.
        crs_coords: CRS of center_coords. If None, assumes EPSG:4326.
        observation_date_corr_factor: Pre-computed π×d²/cos(θ_z). If provided,
            ``date_of_acquisition`` and ``center_coords`` are ignored.
        units: Input radiance units. Must be one of:
            - ``"W/m2/sr/nm"``: SI units (factor=1)
            - ``"mW/m2/sr/nm"``: milliwatts (factor=1000)
            - ``"uW/cm^2/SR/nm"``: EMIT/PRISMA standard (factor=100)

    Returns:
        ToA reflectance with same shape (C, H, W). Values typically 0-1
        for non-saturated pixels over land. Returns GeoTensor if input
        was GeoTensor, preserving georeferencing. Fill values are propagated.

    Raises:
        ValueError: If units string is not recognized.
        AssertionError: If data shape doesn't match solar_irradiance length.

    Examples
    --------
    Basic conversion from EMIT radiance::

        >>> import numpy as np
        >>> from datetime import datetime
        >>> from georeader.reflectance import radiance_to_reflectance
        >>> 
        >>> # Simulated 3-band radiance (µW/cm²/sr/nm)
        >>> radiance = np.array([[[10, 12], [11, 13]],    # Band 1 (blue ~450nm)
        ...                      [[15, 18], [16, 19]],    # Band 2 (green ~550nm)
        ...                      [[20, 24], [21, 25]]])   # Band 3 (red ~650nm)
        >>> 
        >>> # Approximate solar irradiance (W/m²/nm) - decreases with wavelength
        >>> solar_irrad = np.array([1.95, 1.88, 1.55])  # Blue, Green, Red
        >>> 
        >>> # Summer acquisition in Northern Hemisphere
        >>> refl = radiance_to_reflectance(
        ...     radiance,
        ...     solar_irradiance=solar_irrad,
        ...     date_of_acquisition=datetime(2024, 6, 21, 10, 30),
        ...     center_coords=(-122.4, 37.8),  # San Francisco
        ...     units="uW/cm^2/SR/nm"
        ... )
        >>> print(f"Reflectance range: {refl.min():.3f} to {refl.max():.3f}")

    Using pre-computed correction factor::

        >>> obfactor = 3.5  # Pre-computed from metadata
        >>> refl = radiance_to_reflectance(
        ...     radiance,
        ...     solar_irradiance=solar_irrad,
        ...     observation_date_corr_factor=obfactor,
        ...     units="uW/cm^2/SR/nm"
        ... )

    Warning
    -------
    The ``solar_irradiance`` parameter must be in **W/m²/nm**, even when
    input radiance uses different units. The function handles radiance
    unit conversion internally, but solar irradiance is assumed to be
    already in SI units.

    See Also
    --------
    reflectance_to_radiance : Inverse conversion
    observation_date_correction_factor : Compute obfactor from date/location
    transform_to_srf : Combined SRF integration and reflectance conversion

    References
    ----------
    .. [1] ESA Sentinel-2 TOA Reflectance Computation:
       https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-TOAReflectanceComputation
    """

    solar_irradiance = np.array(solar_irradiance)[:, np.newaxis, np.newaxis] # (C, 1, 1)
    assert len(data.shape) == 3, f"Expected 3 channels found {len(data.shape)}"
    assert data.shape[0] == len(solar_irradiance), \
        f"Different number of channels {data.shape[0]} than number of radiances {len(solar_irradiance)}"

    if units == "mW/m2/sr/nm":
        factor_div = 1000
    elif units == "W/m2/sr/nm":
        factor_div = 1
    elif units == "uW/cm^2/SR/nm":
        factor_div = 100 # (10**(-6) / 1) * (1 /10**(-4))
    else:
        raise ValueError(f"Units {units} not recognized must be 'mW/m2/sr/nm', 'W/m2/sr/nm', 'uW/cm^2/SR/nm'")


    if observation_date_corr_factor is None:
        assert date_of_acquisition is not None, "If observation_date_corr_factor is None, date_of_acquisition must be provided"
        # Get latitude and longitude of the center of image to compute the solar angle
        if center_coords is None:
            assert isinstance(data, GeoTensor), "If center_coords is None, data must be a GeoTensor"
            center_coords = data.transform * (data.shape[-1] // 2, data.shape[-2] // 2)
            crs_coords = data.crs
        
        observation_date_corr_factor = observation_date_correction_factor(center_coords, date_of_acquisition, crs_coords=crs_coords)

    if isinstance(data, GeoTensor):
        data_values = data.values
    else:
        data_values = data

    # radiances = data_values * (10**(-6) / 1) * (1 /10**(-4))

    # Convert units to W/m²/sr/nm
    radiances = data_values / factor_div

    # data_toa = data.values / 100 * constant_factor / solar_irradiance
    data_toa_reflectance = radiances * observation_date_corr_factor / solar_irradiance
    if not  isinstance(data, GeoTensor):
        return data_toa_reflectance
    
    mask = data.values == data.fill_value_default
    data_toa_reflectance[mask] = data.fill_value_default

    return GeoTensor(values=data_toa_reflectance, crs=data.crs, transform=data.transform,
                     fill_value_default=data.fill_value_default)


def srf(center_wavelengths:ArrayLike, fwhm:ArrayLike, wavelengths:ArrayLike) -> NDArray:
    """
    Generate Gaussian spectral response functions (SRF) for sensor bands.

    Creates a normalized Gaussian response curve for each band, which describes
    the relative sensitivity of a sensor band to different wavelengths. This is
    essential for simulating how a hyperspectral signal would appear in a
    multispectral sensor.

    Mathematical Definition
    -----------------------
    ::

        For each band k with center wavelength λ_k and FWHM_k:

        σ_k = FWHM_k / (2 × √(2 × ln(2))) ≈ FWHM_k / 2.355

        R_k(λ) = exp(-(λ - λ_k)² / (2σ_k²)) / √(2πσ_k²)

        Then normalized so that: Σ R_k(λ) = 1 over all λ

    ASCII Visualization
    -------------------
    ::

        ┌───────────────────────────────────────────────────────────────┐
        │  Gaussian SRF Bands (example: RGB)                            │
        │                                                                │
        │  Response  ▲                                                   │
        │            │        Blue         Green         Red             │
        │  1.0 ──────┼──────╱╲────────────╱╲────────────╱╲────────      │
        │            │     ╱  ╲          ╱  ╲          ╱  ╲             │
        │  0.5 ──────┼────╱────╲────────╱────╲────────╱────╲────       │
        │            │   ╱      ╲      ╱      ╲      ╱      ╲          │
        │  0.0 ──────┼──╱────────╲────╱────────╲────╱────────╲────     │
        │            └──┴─────────┴───┴─────────┴───┴─────────┴──► λ   │
        │              450       490 520       570 620       680       │
        │                                                                │
        │  ◄──FWHM──►  = Full Width at Half Maximum                     │
        │                                                                │
        │  The response drops to 50% at λ_center ± FWHM/2              │
        └───────────────────────────────────────────────────────────────┘

    Args:
        center_wavelengths: Band center wavelengths in nm. Shape (K,) where
            K is the number of bands.
        fwhm: Full Width at Half Maximum for each band in nm. Shape (K,).
            Typical values: ~30nm for Sentinel-2, ~7-10nm for hyperspectral.
        wavelengths: Wavelength grid where SRF is evaluated, in nm. Shape (N,).
            Should span the range of center_wavelengths with appropriate resolution
            (typically 1nm for accurate integration).

    Returns:
        Normalized SRF matrix of shape (N, K). Each column sums to 1.0 and
        represents the relative weight of each input wavelength for that band.

    Examples
    --------
    Create Sentinel-2 visible bands SRF::

        >>> import numpy as np
        >>> from georeader.reflectance import srf
        >>> 
        >>> # Sentinel-2 Band 2 (Blue), Band 3 (Green), Band 4 (Red)
        >>> s2_centers = np.array([492.4, 559.8, 664.6])  # nm
        >>> s2_fwhm = np.array([66, 36, 31])  # nm
        >>> 
        >>> # Fine wavelength grid for integration
        >>> wavelengths = np.arange(400, 800, 1)  # 400-799 nm at 1nm steps
        >>> 
        >>> response = srf(s2_centers, s2_fwhm, wavelengths)
        >>> print(f"SRF shape: {response.shape}")  # (400, 3)
        SRF shape: (400, 3)
        >>> 
        >>> # Verify normalization
        >>> print(f"Column sums: {response.sum(axis=0)}")  # All ~1.0
        Column sums: [1. 1. 1.]

    Convert hyperspectral radiance to multispectral::

        >>> # Hyperspectral radiance at 1nm resolution
        >>> hyper_radiance = np.random.rand(400)  # 400-799 nm
        >>> 
        >>> # Integrate using SRF weights
        >>> multi_radiance = hyper_radiance @ response  # Shape: (3,)
        >>> print(f"Multispectral radiance: {multi_radiance}")

    Notes
    -----
    - The FWHM-to-sigma conversion uses the exact Gaussian relationship:
      σ = FWHM / (2√(2·ln(2)))
    - Normalization ensures energy conservation when integrating radiance
    - For non-Gaussian SRFs (e.g., from measured sensor response), use
      :func:`integrated_irradiance` with actual SRF data

    See Also
    --------
    integrated_irradiance : Integrate solar irradiance weighted by SRF
    transform_to_srf : Full hyperspectral to multispectral conversion

    References
    ----------
    .. [1] Sentinel-2 Spectral Response Functions:
       https://sentiwiki.copernicus.eu/web/s2-msi-spectral-model
    """
    center_wavelengths = np.array(center_wavelengths) # (K, )
    fwhm = np.array(fwhm) # (K, )
    assert center_wavelengths.shape == fwhm.shape, f"Center wavelengths and FWHM must have the same shape {center_wavelengths.shape} != {fwhm.shape}"

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))) # (K, )
    var = sigma ** 2 # (K, )
    denom = (2 * np.pi * var) ** 0.5 # (K, )
    numer = np.exp(-(wavelengths[:, None] - center_wavelengths[None, :])**2 / (2*var)) # (N, K)
    response = numer / denom # (N, K)
    
    # Normalize each gaussian response to sum to 1.
    response = np.divide(response, response.sum(axis=0), where=response.sum(axis=0) > 0)# (N, K)
    return response


THUILLIER_RADIANCE = None

def load_thuillier_irradiance() -> pd.DataFrame:
    """
    https://oceancolor.gsfc.nasa.gov/docs/rsr/f0.txt

    G. Thuillier et al., "The Solar Spectral Irradiance from 200 to 2400nm as 
        Measured by the SOLSPEC Spectrometer from the Atlas and Eureca Missions",
    Solar Physics, vol. 214, no. 1, pp. 1-22, May 2003, doi: 10.1023/A:1024048429145.


    Returns:
        pandas dataframe with columns: Nanometer, Radiance(mW/m2/nm)
    """
    global THUILLIER_RADIANCE
    
    if THUILLIER_RADIANCE is None:
        THUILLIER_RADIANCE = pd.read_csv(os.path.join(os.path.dirname(__file__), "SolarIrradiance_Thuillier.csv"))

    return THUILLIER_RADIANCE


def integrated_irradiance(srf:pd.DataFrame, 
                          solar_irradiance:Optional[pd.DataFrame]=None,
                          epsilon_srf:float=1e-4) -> NDArray:
    """
    Compute band-integrated solar irradiance weighted by spectral response.

    Calculates the effective solar irradiance for each sensor band by
    convolving the TOA solar spectrum with the band's spectral response
    function. This is necessary for accurate radiance-to-reflectance
    conversion of multispectral data.

    Mathematical Definition
    -----------------------
    ::

        For band k with spectral response function R_k(λ):

        E_k = ∫ E_sun(λ) × R_k(λ) dλ  /  ∫ R_k(λ) dλ

        where:
        - E_sun(λ) = solar spectral irradiance at TOA (W/m²/nm or mW/m²/nm)
        - R_k(λ)   = spectral response function for band k
        - E_k      = band-integrated irradiance (same units as E_sun)

    Visual Representation
    ---------------------
    ::

        ┌────────────────────────────────────────────────────────────────┐
        │  Spectral Integration Process                                   │
        │                                                                  │
        │  E_sun(λ)          R(λ)               E_sun(λ) × R(λ)          │
        │  (Solar)           (SRF)              (Product)                 │
        │                                                                  │
        │    │╲              │ ╱╲                 │  ╱╲                   │
        │    │ ╲╲            │╱  ╲                │ ╱  ╲                  │
        │    │  ╲╲╲          │    ╲               │╱    ╲                 │
        │    │   ╲╲╲╲        │     ╲              │      ╲                │
        │    └────────λ      └──────λ            └────────λ              │
        │                                         ████████ ← Area = E_k  │
        │                                                                  │
        │  Solar spectrum   Band response    Weighted → integrate & norm  │
        │  (~200-2500 nm)   (Gaussian)       gives band-effective E_sun   │
        └────────────────────────────────────────────────────────────────┘

    Args:
        srf: Spectral response function as DataFrame. Index is wavelength (nm),
            columns are band names. Shape (N, K) where N wavelengths, K bands.
            Values should be normalized so each column sums to ~1.
        solar_irradiance: Solar spectrum DataFrame with columns:
            - ``"Nanometer"``: wavelength in nm
            - ``"Radiance(mW/m2/nm)"``: spectral irradiance in mW/m²/nm
            If None, loads Thuillier (2003) standard spectrum.
        epsilon_srf: Threshold below which SRF values are treated as zero.
            Bands/wavelengths with all values < epsilon_srf are excluded.
            Default: 1e-4.

    Returns:
        Band-integrated irradiance array of shape (K,). Units match input
        solar_irradiance (mW/m²/nm if using default Thuillier).

    Examples
    --------
    Compute Sentinel-2 band irradiances::

        >>> import numpy as np
        >>> import pandas as pd
        >>> from georeader.reflectance import srf, integrated_irradiance
        >>> 
        >>> # Create simple SRF for 3 bands
        >>> wavelengths = np.arange(400, 800)
        >>> centers = [490, 560, 665]
        >>> fwhms = [65, 35, 30]
        >>> srf_matrix = srf(centers, fwhms, wavelengths)
        >>> srf_df = pd.DataFrame(srf_matrix, index=wavelengths, 
        ...                       columns=['B2_Blue', 'B3_Green', 'B4_Red'])
        >>> 
        >>> # Integrate with default Thuillier solar spectrum
        >>> band_irradiance = integrated_irradiance(srf_df)
        >>> print("Band irradiances (mW/m²/nm):")
        >>> for name, val in zip(srf_df.columns, band_irradiance):
        ...     print(f"  {name}: {val:.1f}")
        Band irradiances (mW/m²/nm):
          B2_Blue: 1960.5
          B3_Green: 1853.2
          B4_Red: 1535.8

    Using with radiance_to_reflectance::

        >>> # Convert to W/m²/nm for use in radiance_to_reflectance
        >>> band_irradiance_si = band_irradiance / 1000  # mW → W
        >>> # Now use in reflectance conversion...

    Notes
    -----
    - The function interpolates the SRF to the solar spectrum wavelengths,
      not vice versa, to preserve spectral detail in the solar spectrum.
    - Wavelengths outside the SRF range are excluded from integration.
    - Default Thuillier spectrum covers 200-2400 nm at ~1nm resolution.

    Warning
    -------
    The default output is in **mW/m²/nm** (Thuillier units). Divide by 1000
    to convert to **W/m²/nm** for use with :func:`radiance_to_reflectance`.

    See Also
    --------
    srf : Generate Gaussian spectral response functions
    load_thuillier_irradiance : Load Thuillier (2003) solar spectrum
    radiance_to_reflectance : Uses band irradiance for conversion

    References
    ----------
    .. [1] Thuillier, G. et al. (2003). "The Solar Spectral Irradiance
       from 200 to 2400nm as Measured by the SOLSPEC Spectrometer."
       Solar Physics, 214(1), 1-22.
    """
    from scipy import interpolate

    if solar_irradiance is None:
        solar_irradiance = load_thuillier_irradiance()
    
    anybigvalue = (srf>epsilon_srf).any(axis=1)
    srf = srf.loc[anybigvalue, :]

    # Trim the solar irradiance to the min and max wavelengths
    solar_irradiance = solar_irradiance[(solar_irradiance["Nanometer"] >= srf.index.min()) &\
                                        (solar_irradiance["Nanometer"] <= srf.index.max())]

    # interpolate srf to the solar irradiance
    interp = interpolate.interp1d(srf.index, srf, kind="linear", axis=0)
    srf_interp = interp(solar_irradiance["Nanometer"].values) # (D, K)

    # integrate the product of the solar irradiance and the srf
    return np.sum(solar_irradiance["Radiance(mW/m2/nm)"].values[:, np.newaxis] * srf_interp, axis=0) / srf_interp.sum(axis=0)


def reflectance_to_radiance(data:Union[GeoTensor, ArrayLike], 
                            solar_irradiance:ArrayLike,
                            date_of_acquisition:Optional[datetime]=None,
                            center_coords:Optional[Tuple[float, float]]=None,
                            crs_coords:Optional[str]=None,
                            observation_date_corr_factor:Optional[float]=None) -> Union[GeoTensor, NDArray]:
    """
    Convert the ToA reflectance to radiance using the solar irradiance and the date of acquisition.
    The formula is:

    ```
    radianceBandX = (toaBandX * solarIrradianceBandX * cos(solarzenithangle/180*pi)) / (pi * d^2)
    radianceBandX = (toaBandX * solarIrradianceBandX) / observation_date_correction_factor(center_coords, date_of_acquisition)
    ```

    Formula for `observation_date_corr_factor`:
    ```
        obfactor = (pi * d^2) / cos(solarzenithangle/180*pi)
    ```

    [ESA reference of ToA calculation](https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-TOAReflectanceComputation)    

    Args:
        data (Union[GeoTensor, ArrayLike]): data to be converted (C, H, W) tensor in ToA reflectance units
        solar_irradiance (ArrayLike): solar irradiance for each band (C,) in W/m²/nm
        date_of_acquisition (Optional[datetime], optional): Date of acquisition to compute the 
            solar zenith angles and the Earth-Sun distance correction factor.
        center_coords (Optional[Tuple[float, float]], optional): location being considered to compute 
            the solar zenith angles and the Earth-Sun distance correction factor.
        crs_coords (Optional[str], optional): if None it will assume center_coords are in EPSG:4326. 
            Defaults to None.
        observation_date_corr_factor (Optional[float], optional): observation date correction factor. 
            If provided, it will be used instead of computing it from the date of acquisition and the center coordinates.

    Returns:
        Union[GeoTensor, NDArray]: radiance (C, H, W) tensor in W/m²/nm
    """
    solar_irradiance = np.array(solar_irradiance)[:, np.newaxis, np.newaxis] # (C, 1, 1)
    assert len(data.shape) == 3, f"Expected 3 channels found {len(data.shape)}"
    assert data.shape[0] == len(solar_irradiance), \
        f"Different number of channels {data.shape[0]} than number of radiances {len(solar_irradiance)}"

    if observation_date_corr_factor is None:
        assert date_of_acquisition is not None, "If observation_date_corr_factor is None, date_of_acquisition must be provided"
        # Get latitude and longitude of the center of image to compute the solar angle
        if center_coords is None:
            assert isinstance(data, GeoTensor), "If center_coords is None, data must be a GeoTensor"
            center_coords = data.transform * (data.shape[-1] // 2, data.shape[-2] // 2)
            crs_coords = data.crs
        
        observation_date_corr_factor = observation_date_correction_factor(center_coords, 
                                                                          date_of_acquisition, 
                                                                          crs_coords=crs_coords)

    if isinstance(data, GeoTensor):
        data_values = data.values
    else:
        data_values = data

    data_toa_reflectance = data_values
    radiances = data_toa_reflectance / observation_date_corr_factor * solar_irradiance
    
    if not  isinstance(data, GeoTensor):
        return radiances
    
    mask = data.values == data.fill_value_default
    radiances[mask] = data.fill_value_default

    return GeoTensor(values=radiances, crs=data.crs, transform=data.transform,
                     fill_value_default=data.fill_value_default)



def transform_to_srf(hyperspectral_data:Union[GeoData, NDArray], 
                     srf:pd.DataFrame,
                     wavelengths_hyperspectral:List[float],
                     as_reflectance:bool=False,
                     solar_irradiance_bands:Optional[NDArray]=None,
                     observation_date_corr_factor:Optional[float]=None,
                     center_coords:Optional[Tuple[float, float]]=None,
                     date_of_acquisition:Optional[datetime]=None,
                     resolution_dst:Optional[Union[float,Tuple[float,float]]]=None,
                     fill_value_default:float=0.,
                     sigma_bands:Optional[np.array]=None,
                     verbose:bool=False,
                     epsilon_srf:float=1e-4,
                     extrapolate:bool=False,
                     units:Optional[str]=None) -> Union[GeoData, NDArray]:
    """
    Integrates the hyperspectral bands to the multispectral bands using the spectral response function (SRF).

    Args:
        hyperspectral_data (Union[GeoData, NDArray]): hyperspectral data (B, H, W) or GeoData. If as_reflectance is True, the data must be radiance
            and units must be filled in.
        srf (pd.DataFrame): spectral response function (SRF) (N, K). The index is the wavelengths and the columns are the bands.
        wavelengths_hyperspectral (List[float]): wavelengths of the hyperspectral data (B,)
        as_reflectance (bool, optional): if True, the hyperspectral data will be converted to reflectance after integrating. Defaults to False.
        solar_irradiance_bands (Optional[NDArray], optional): solar irradiance for each band to be used for the conversion to reflectance (K,). 
            Defaults to None. Must be provided in W/m²/nm.
        observation_date_corr_factor (Optional[float], optional): observation date correction factor. Defaults to None. 
            Only used if as_reflectance is True.
        center_coords (Optional[Tuple[float, float]], optional): center coordinates of the image. Defaults to None. 
            Only used if as_reflectance is True and observation_date_corr_factor is None.
        date_of_acquisition (Optional[datetime], optional): date of acquisition. Defaults to None.
            Only used if as_reflectance is True and observation_date_corr_factor is None.
        resolution_dst (Optional[Union[float,Tuple[float,float]]], optional): output resolution of the multispectral data. Defaults to None. 
            If None, the output will have the same resolution as the input hyperspectral data.
        fill_value_default (float, optional): fill value for missing data. Defaults to 0.
        sigma_bands (Optional[np.array], optional): sigma for the anti-aliasing filter. Defaults to None.
        verbose (bool, optional): print progress. Defaults to False.
        epsilon_srf (float, optional): threshold to consider a band in the SRF. Defaults to 1e-4.
        extrapolate (bool, optional): if True, it will extrapolate the SRF to the hyperspectral wavelengths. Defaults to False.
        units: if as_reflectance is True, the units of the hyperspectral data must be provided. Defaults to None.
            accepted values: "mW/m2/sr/nm", "W/m2/sr/nm", "uW/cm^2/SR/nm"

    Returns:
        Union[GeoData, NDArray]: multispectral data (C, H, W) or GeoData
    """
    from scipy import interpolate

    assert hyperspectral_data.shape[0] == len(wavelengths_hyperspectral), f"Different number of bands {hyperspectral_data.shape[0]} and band frequency centers {len(wavelengths_hyperspectral)}"
    
    anybigvalue = (srf>epsilon_srf).any(axis=1)
    srf = srf.loc[anybigvalue, :]    
    bands = srf.columns

    if as_reflectance:
        assert units is not None, "If as_reflectance is True, the units of the hyperspectral data must be specified"
        # check observation_date_corr_factor
        if observation_date_corr_factor is None:
            assert date_of_acquisition is not None, "If observation_date_corr_factor is None, date_of_acquisition must be provided"
            if center_coords is None:
                assert isinstance(hyperspectral_data, GeoTensor), "If center_coords is None, data must be a GeoTensor"
                center_coords = hyperspectral_data.transform * (hyperspectral_data.shape[-1] // 2, hyperspectral_data.shape[-2] // 2)
                crs_coords = hyperspectral_data.crs
            else:
                crs_coords = None

            observation_date_corr_factor = observation_date_correction_factor(center_coords, date_of_acquisition,crs_coords=crs_coords)
        
        if solar_irradiance_bands is None:
            solar_irradiance_bands = integrated_irradiance(srf, epsilon_srf=epsilon_srf)
            solar_irradiance_bands/=1_000
    
    # Construct hyperspectral frequencies in the same resolution as srf
    bands_index_hyperspectral = np.arange(0, len(wavelengths_hyperspectral))
    interp = interpolate.interp1d(wavelengths_hyperspectral, bands_index_hyperspectral, kind="nearest",
                                  fill_value="extrapolate" if extrapolate else np.nan)
    y_nearest = interp(srf.index).astype(int)
    table_hyperspectral_as_srf_multispectral = pd.DataFrame({"SR_WL": srf.index, "band": y_nearest})
    table_hyperspectral_as_srf_multispectral = table_hyperspectral_as_srf_multispectral.set_index("SR_WL")

    output_array_spectral = np.full((len(bands),) + hyperspectral_data.shape[-2:],
                                    fill_value=fill_value_default, dtype=np.float32)

    for i,column_name in enumerate(bands):
        if verbose:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}({i}/{len(bands)}) Processing band {column_name}")
        mask_zero = srf[column_name] <= epsilon_srf
        weight_per_wavelength = srf.loc[~mask_zero, [column_name]].copy()

        assert weight_per_wavelength.shape[0] >= 0, f"No weights found! {weight_per_wavelength}"

        # Join with table of previous chunk
        weight_per_wavelength = weight_per_wavelength.join(table_hyperspectral_as_srf_multispectral)

        assert weight_per_wavelength.shape[0] >= 0, "No weights found!"

        # Normalize the SRF to sum one
        column_name_norm = f"{column_name}_norm"
        weight_per_wavelength[column_name_norm] = weight_per_wavelength[column_name] / weight_per_wavelength[
            column_name].sum()
        weight_per_hyperspectral_band = weight_per_wavelength.groupby("band")[[column_name_norm]].sum()

        indexes_read = weight_per_hyperspectral_band.index.tolist()

        # Load bands of hyperspectral image
        if verbose:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t Loading {len(weight_per_hyperspectral_band.index)} bands")
            # print("these ones:", weight_per_aviris_band.index)
        
        if hasattr(hyperspectral_data, "isel"):
            hyperspectral_multispectral_band_i_values = hyperspectral_data.isel({"band":indexes_read}).load().values
            
            missing_values = np.any(hyperspectral_multispectral_band_i_values == hyperspectral_data.fill_value_default, axis=0)
            if not np.any(missing_values):
                missing_values = None
        else:
            hyperspectral_multispectral_band_i_values = hyperspectral_data[indexes_read]
            missing_values = None

        # hyperspectral_multispectral_band_i = hyperspectral_data.isel({"band": weight_per_hyperspectral_band.index}).load()
        if verbose:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t bands loaded, computing tensor")


        output_array_spectral[i] = np.sum(weight_per_hyperspectral_band[column_name_norm].values[:, np.newaxis,
                                          np.newaxis] * hyperspectral_multispectral_band_i_values,
                                          axis=0)
    
        if as_reflectance:
            output_array_spectral[i:(i+1)] = radiance_to_reflectance(output_array_spectral[i:(i+1)],
                                                                     solar_irradiance_bands[i:(i+1)],
                                                                     observation_date_corr_factor=observation_date_corr_factor,
                                                                     units=units)

        if missing_values is not None:
            output_array_spectral[i][missing_values] = fill_value_default

    if hasattr(hyperspectral_data, "load"):
        geotensor_spectral = GeoTensor(output_array_spectral, transform=hyperspectral_data.transform,
                                       crs=hyperspectral_data.crs,
                                       fill_value_default=fill_value_default)

        if (resolution_dst is None) or (resolution_dst == geotensor_spectral.res):
            return geotensor_spectral
        
        if isinstance(resolution_dst, numbers.Number):
            resolution_dst = (abs(resolution_dst), abs(resolution_dst))


        return read.resize(geotensor_spectral, resolution_dst=resolution_dst,
                           anti_aliasing=True, anti_aliasing_sigma=sigma_bands)
    else:
        return output_array_spectral
