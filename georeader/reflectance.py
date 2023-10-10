from datetime import datetime
from typing import Tuple, Union, List, Optional
from georeader import window_utils
from georeader.geotensor import GeoTensor
import numpy as np
import pandas as pd
import pkg_resources


def earth_sun_distance_correction_factor(date_of_acquisition:datetime) -> float:
    """
    returns: (1-0.01673*cos(0.0172*(t-4)))

     0.0172 = 360/365.256363 * np.pi/180.
     0.01673 is the Earth eccentricity

     t = datenum(Y,M,D) - datenum(Y,1,1) + 1;

     tm_yday starts in 1
     > datetime.datetime.strptime("2022-01-01", "%Y-%m-%d").timetuple().tm_yday -> 1

    Args:
        date_of_acquisition: date of acquisition. The day of the year will be used 
            to compute the correction factor

    Returns:
        (1-0.01673*cos(0.0172*(t-4)))
    """
    tm_yday = date_of_acquisition.timetuple().tm_yday # from 1 to 365 (or 366!)
    return 1 - 0.01673 * np.cos(0.0172 * (tm_yday - 4))


def observation_date_correction_factor(center_coords:Tuple[float, float], date_of_acquisition:datetime,
                                       crs_coords:Optional[str]=None,) -> float:
    """
    returns  (pi * d^2) / cos(solarzenithangle/180*pi)

    Args:
        center_coords: location being considered (x,y) (long, lat if EPSG:4326) 
        date_of_acquisition: date of acquisition to compute the solar zenith angles.
        crs_coords: if None it will assume center_coords are in EPSG:4326

    Returns:
        correction factor

    """
    from pysolar.solar import get_altitude
    from rasterio import warp

    if crs_coords is not None and not window_utils.compare_crs(crs_coords, "EPSG:4326"):
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
    sza = 90 - solar_altitude
    d = earth_sun_distance_correction_factor(date_of_acquisition)

    return np.pi*(d**2) / np.cos(sza/180.*np.pi)


def radiance_to_reflectance(data:GeoTensor, solar_irradiance:Union[List[float], np.array],
                            date_of_acquisition:datetime) -> GeoTensor:
    """

    toaBandX = (radianceBandX / 100 * pi * d^2) / (cos(solarzenithangle/180*pi) * solarIrradianceBandX)

    ESA reference of ToA calculation:
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-1c/algorithm

    where:
        d = earth_sun_distance_correction_factor(date_of_acquisition)
        solarzenithangle = is obtained from the date of aquisition and location

    Args:
        data:  (C, H, W) tensor with units: µW /(nm cm² sr)
                microwatts per centimeter_squared per nanometer per steradian
        solar_irradiance: (C,) vector units: W/m²/nm
        date_of_acquisition: date of acquisition to compute the solar zenith angles

    Returns:
        GeoTensor with ToA on each channel
    """

    solar_irradiance = np.array(solar_irradiance)[:, np.newaxis, np.newaxis] # (C, 1, 1)
    assert len(data.shape) == 3, f"Expected 3 channels found {len(data.shape)}"
    assert data.shape[0] == len(solar_irradiance), \
        f"Different number of channels {data.shape[0]} than number of radiances {len(solar_irradiance)}"

    # Get latitude and longitude of the center of image to compute the solar angle
    center_coords = data.transform * (data.shape[-1] // 2, data.shape[-2] // 2)
    constant_factor = observation_date_correction_factor(center_coords, date_of_acquisition, crs_coords=data.crs)

    # µW /(nm cm² sr) to W/(nm m² sr)
    radiances = data.values * (10**(-6) / 1) * (1 /10**(-4))

    # data_toa = data.values / 100 * constant_factor / solar_irradiance
    data_toa = radiances * constant_factor / solar_irradiance
    mask = data.values == data.fill_value_default
    data_toa[mask] = data.fill_value_default

    return GeoTensor(values=data_toa, crs=data.crs, transform=data.transform,
                     fill_value_default=data.fill_value_default)


def srf(center_wavelengths:np.array, fwhm:np.array, wavelengths:np.array) -> np.array:
    """
    Returns the spectral response function (SRF) for the given center wavelengths and full width half maximum (FWHM).

    Args:
        center_wavelengths (np.array): array with center wavelengths. (K, )
        fwhm (np.array): array with full width half maximum (FWHM) values (K,)
        wavelengths (np.array): array with wavelengths where the SRF is evaluated (N,)

    Returns:
        np.array: normalized SRF (N, K)
    """

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
        THUILLIER_RADIANCE = pd.read_csv(pkg_resources.resource_filename("georeader","SolarIrradiance_Thuillier.csv"))

    return THUILLIER_RADIANCE


