from datetime import datetime
from typing import Tuple, Union, List, Optional
from georeader import window_utils
from georeader.geotensor import GeoTensor
from georeader.abstract_reader import GeoData
from georeader import read
import numpy as np
import pandas as pd
import pkg_resources
from numpy.typing import ArrayLike, NDArray
import numbers


def earth_sun_distance_correction_factor(date_of_acquisition:datetime) -> float:
    """
    This function returns the Earth-sun distance correction factor given by the formula:

    ```
    d = 1-0.01673*cos(0.0172*(t-4))

    Where:
    0.0172 = 360/365.256363 * np.pi/180.  # (Earth orbit angular velocity)
    0.01673 is the Earth eccentricity

    # t is the day of the year starting in 1:
    t = datenum(Y,M,D) - datenum(Y,1,1) + 1;

    # tm_yday starts in 1
    datetime.datetime.strptime("2022-01-01", "%Y-%m-%d").timetuple().tm_yday -> 1

    ```

    In the Sentinel-2 metadata they provide `U` which is the square inverse of this factor: `U = 1 / d^2`
    
    [https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-TOAReflectanceComputation](https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-TOAReflectanceComputation)
    
    Args:
        date_of_acquisition: date of acquisition. The day of the year will be used 
            to compute the correction factor

    Returns:
        (1-0.01673*cos(0.0172*(t-4)))
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
    
      obfactor = (pi * d^2) / cos(solarzenithangle/180*pi)

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
    Convert the radiance to ToA reflectance using the solar irradiance and the date of acquisition.

    ```
    toaBandX = (radianceBandX / 100 * pi * d^2) / (cos(solarzenithangle/180*pi) * solarIrradianceBandX)
    toaBandX = (radianceBandX / 100 / solarIrradianceBandX) * observation_date_correction_factor(center_coords, date_of_acquisition)
    ```

    [ESA reference of ToA calculation](https://sentiwiki.copernicus.eu/web/s2-processing#S2Processing-TOAReflectanceComputation)

    where:
        d = earth_sun_distance_correction_factor(date_of_acquisition)
        solarzenithangle = is obtained from the date of aquisition and location

    Args:
        data:  (C, H, W) tensor with units: µW /(nm cm² sr)
                microwatts per centimeter_squared per nanometer per steradian
        solar_irradiance: (C,) vector units: W/m²/nm
        date_of_acquisition: date of acquisition to compute the solar zenith angles and the Earth-Sun distance correction factor.
        center_coords: location being considered to compute the solar zenith angles and the Earth-Sun distance correction factor.
            (x,y) (long, lat if EPSG:4326). If None, it will use the center of the image.
        observation_date_corr_factor: if None, it will be computed using the center_coords and date_of_acquisition.        
        crs_coords: if None it will assume center_coords are in `EPSG:4326`.
        units: if as_reflectance is True, the units of the hyperspectral data must be provided. Defaults to None.
            accepted values: "mW/m2/sr/nm", "W/m2/sr/nm", "uW/cm^2/SR/nm"

    Returns:
        GeoTensor with ToA reflectance values (C, H, W)
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
    Returns the spectral response function (SRF) for the given center wavelengths and full width half maximum (FWHM).

    Args:
        center_wavelengths (np.array): array with center wavelengths. (K, )
        fwhm (np.array): array with full width half maximum (FWHM) values (K,)
        wavelengths (np.array): array with wavelengths where the SRF is evaluated (N,)

    Returns:
        np.array: normalized SRF (N, K)
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
        THUILLIER_RADIANCE = pd.read_csv(pkg_resources.resource_filename("georeader","SolarIrradiance_Thuillier.csv"))

    return THUILLIER_RADIANCE


def integrated_irradiance(srf:pd.DataFrame, 
                          solar_irradiance:Optional[pd.DataFrame]=None,
                          epsilon_srf:float=1e-4) -> NDArray:
    """
    Returns the integrated irradiance for the given spectral response function (SRF) and solar irradiance.

    The output is the integrated irradiance for each band.

    Args:
        srf (pd.DataFrame): dataframe with the spectral response function (SRF) (N, K) where N is the number of wavelengths and K the number of bands.
            The index is the wavelengths in nanometers and the columns are the bands.
        solar_irradiance (Optional[pd.DataFrame], optional): dataframe with the solar irradiance. It must contain the columns "Nanometer" and "Radiance(mW/m2/nm)". (D, 2)
                  Defaults to None. If None, it will load the Thuillier solar irradiance and the output will be in mW/m2/nm.
        epsilon_srf (float, optional): threshold to consider a band in the SRF. Defaults to 1e-4.

    Returns:
        NDArray: integrated irradiance (K,)
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
