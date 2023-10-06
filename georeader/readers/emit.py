"""
Module to read EMIT images.

Requires: netCDF4:
pip install netCDF4 

Some of the functions of this module are based on the official EMIT repo: https://github.com/emit-sds/emit-utils/

"""
import os
import json
from typing import Tuple, Optional, Any, Union
from georeader.readers.download_utils import download_product as download_product_base
from georeader.abstract_reader import AbstractGeoData
import rasterio
import rasterio.windows
import numpy as np
from georeader import window_utils
from shapely.geometry import Polygon
from georeader.geotensor import GeoTensor
import netCDF4
from shapely.ops import unary_union
from georeader import read
import rasterio.warp
from datetime import datetime, timezone


def get_auth() -> Tuple[str, str]:
    home_dir = os.path.join(os.path.expanduser('~'),".georeader")
    json_file = os.path.join(home_dir, "auth_emit.json")
    if not os.path.exists(json_file):
        os.makedirs(home_dir, exist_ok=True)
        with open(json_file, "w") as fh:
            json.dump({"user": "SET-USER", "password": "SET-PASSWORD"}, fh)

        raise FileNotFoundError(f"In order to download Proba-V images add user and password to file : {json_file}")

    with open(json_file, "r") as fh:
        data = json.load(fh)
    
    if data["user"] == "SET-USER":
        raise FileNotFoundError(f"In order to download Proba-V images add user and password to file : {json_file}")

    return (data["user"], data["password"])

def product_name_from_params(scene_fid:str, orbit:str, daac_scene_number:str)-> str:
    """
    Return the product name from the scene_fid, daac_scene_number and orbit

    Args:
        scene_fid (str): scene_fid of the product. e.g. 'emit20220810t064957'
        orbit (str): orbit of the product. e.g. '2222205'
        daac_scene_number (str): daac_scene_number of the product. e.g. '033'

    Returns:
        str: product name. e.g. 'EMIT_L1B_RAD_001_20220810T064957_2222205_033'
    """
    scenedate = scene_fid[4:].replace("t", "T")
    return f"EMIT_L1B_RAD_001_{scenedate}_{orbit}_{daac_scene_number}"


def split_product_name(product_name:str) -> Tuple[str, str, str, datetime]:
    """
    Split the product name into its components

    Args:
        product_name (str): product name. e.g. 'EMIT_L1B_RAD_001_20220810T064957_2222205_033'

    Returns:
        Tuple[str, str, str, str, str]: scene_fid, orbit, daac_scene_number, datetime
            e.g. ('emit20220810t064957', '2222205', '033', datetime('2022-08-10T06:49:57'))
    """
    scene_fid = f"emit{product_name.split('_')[4]}".replace("T", "t")
    date, orbit, daac_scene_number= product_name.split("_")[4:7]

    dt = datetime.strptime(date, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

    return scene_fid, orbit, daac_scene_number, dt

def download_product(link_down:str, filename:Optional[str]=None,
                     display_progress_bar:bool=True) -> str:
    """
    Download a product from the EMIT website.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        link_down: link to the product
        filename: filename to save the product
        display_progress_bar: display tqdm progress bar

    Example:
        >>> link_down = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220828T051941_2224004_006/EMIT_L1B_RAD_001_20220828T051941_2224004_006.nc'
        >>> filename = download_product(link_down)
    """
    auth_emit = get_auth()
    return download_product_base(link_down, filename=filename, auth=auth_emit,
                                  display_progress_bar=display_progress_bar, 
                                  verify=False)


def get_radiance_link(product_path:str) -> str:
    """
    Get the link to download a product from the EMIT website.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        product_path: path to the product or filename of the product or product name without extension.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Example:
        >>> product_path = 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        >>> link = get_radiance_link(product_path)
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220827T060753_2223904_013/EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
    """
    "EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc"
    namefile = os.path.splitext(os.path.basename(product_path))[0]
    namefile = namefile + ".nc"
    product_id = os.path.splitext(namefile)[0]
    content_id = product_id.split("_")
    content_id[2] = "RAD"
    product_id = "_".join(content_id)
    link = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{product_id}/{namefile}"
    return link


def get_obs_link(product_path:str) -> str:
    """
    Get the link to download a product from the EMIT website.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        product_path: path to the product or filename of the product.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Example:
        >>> product_path = 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        >>> link = get_radiance_link(product_path)
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220827T060753_2223904_013/EMIT_L1B_OBS_001_20220827T060753_2223904_013.nc'
    """
    namefile = os.path.splitext(os.path.basename(product_path))[0]
    namefile = namefile + ".nc"

    product_id = os.path.splitext(namefile)[0]
    content_id = product_id.split("_")
    content_id[2] = "RAD"
    product_id = "_".join(content_id)
    link = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{product_id}/{namefile.replace('RAD', 'OBS')}"
    return link


class EMITImage:
    """
    Class to read L1B EMIT images.

    This class has been inspired by: https://github.com/emit-sds/emit-utils/

    Example:
        >>> from georeader.readers.emit import EMITImage, download_product
        >>> link = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220828T051941_2224004_006/EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        >>> filepath = download_product(link)
        >>> emit = EMITImage(filepath)
        >>> # reproject to UTM
        >>> crs_utm = georeader.get_utm_epsg(emit.footprint("EPSG:4326"))
        >>> emit_utm = emit.to_crs(crs_utm)

    """
    attributes_set_if_exists = ["_pol", "_nc_ds_obs", "_mean_sza", "_mean_vza", "_observation_bands"]
    def __init__(self, filename:str, glt:Optional[GeoTensor]=None, 
                 band_selection:Optional[Union[int, Tuple[int, ...],slice]]=slice(None)):
        self.filename = filename
        self.nc_ds = netCDF4.Dataset(self.filename, 'r', format='NETCDF4')
        self._nc_ds_obs:Optional[netCDF4.Dataset] = None
        self._observation_bands = None
        self.nc_ds.set_auto_mask(False) # disable automatic masking when reading data
        # self.real_shape = (self.nc_ds['radiance'].shape[-1],) + self.nc_ds['radiance'].shape[:-1]

        self._mean_sza = None
        self._mean_vza = None

        self.real_transform = rasterio.Affine(self.nc_ds.geotransform[1], self.nc_ds.geotransform[2], self.nc_ds.geotransform[0],
                                              self.nc_ds.geotransform[4], self.nc_ds.geotransform[5], self.nc_ds.geotransform[3])
        
        self.time_coverage_start = datetime.strptime(self.nc_ds.time_coverage_start, "%Y-%m-%dT%H:%M:%S%z")
        self.time_coverage_end = datetime.strptime(self.nc_ds.time_coverage_end, "%Y-%m-%dT%H:%M:%S%z")

        self.dtype = self.nc_ds['radiance'].dtype
        self.dims = ("band", "y", "x")
        self.fill_value_default = self.nc_ds['radiance']._FillValue
        self.nodata = self.nc_ds['radiance']._FillValue
        self.units = self.nc_ds["radiance"].units

        if glt is None:
            glt_arr = np.zeros((2,) + self.nc_ds.groups['location']['glt_x'].shape, dtype=np.int32)
            glt_arr[0] = np.array(self.nc_ds.groups['location']['glt_x'])
            glt_arr[1] = np.array(self.nc_ds.groups['location']['glt_y'])
            # glt_arr -= 1 # account for 1-based indexing

            # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html
            self.glt = GeoTensor(glt_arr, transform=self.real_transform, 
                                 crs=rasterio.crs.CRS.from_wkt(self.nc_ds.spatial_ref),
                                 fill_value_default=0)
        else:
            self.glt = glt
        
        self.valid_glt = np.all(self.glt.values != self.glt.fill_value_default, axis=0)
        xmin, ymin, xmax, ymax = self._bounds_indexes_raw() # values are 1-based!

        # glt has the absolute indexes of the netCDF object
        # glt_relative has the relative indexes
        self.glt_relative = self.glt.values.copy()
        self.glt_relative[0, self.valid_glt] -= xmin
        self.glt_relative[1, self.valid_glt] -= ymin

        self.window_raw = rasterio.windows.Window(col_off=xmin-1, row_off=ymin-1, 
                                                  width=xmax-xmin+1, height=ymax-ymin+1)

        if "wavelengths" in self.nc_ds['sensor_band_parameters'].variables:
            self.bandname_dimension = "wavelengths"
        elif "radiance_wl"  in self.nc_ds['sensor_band_parameters'].variables:
            self.bandname_dimension = "radiance_wl"
        else:
            raise ValueError(f"Cannot find wavelength dimension in {list(self.nc_ds['sensor_band_parameters'].variables.keys())}")
        
        self.band_selection = band_selection
        self.wavelengths = self.nc_ds['sensor_band_parameters'][self.bandname_dimension][self.band_selection]
        self.fwhm = self.nc_ds['sensor_band_parameters']['fwhm'][self.band_selection]

    @property
    def crs(self) -> Any:
        return self.glt.crs

    @property
    def shape(self) -> Tuple:
        try:
            n_bands = len(self.wavelengths)
            return  (n_bands,) + self.glt.shape[1:]
        except Exception:
            return self.glt.shape

    @property
    def transform(self) -> rasterio.Affine:
        return self.glt.transform

    @property
    def res(self) -> Tuple[float, float]:
        return self.glt.res

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.glt.bounds

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        """
        Get the footprint of the image in the given CRS. If no CRS is given, the footprint is returned in the native CRS.
        This function takes into account the valid_glt mask to compute the footprint.

        Args:
            crs (Optional[str], optional): The CRS to return the footprint in. Defaults to None. 
                If None, the footprint is returned in the native CRS.
        
        Returns:
            Polygon: The footprint of the image in the given CRS.
        """
        if not hasattr(self, '_pol'):
            from georeader.vectorize import get_polygons
            pols = get_polygons(self.valid_glt, transform=self.transform)
            self._pol = unary_union(pols)
        if crs is not None:
            pol_crs = window_utils.polygon_to_crs(self._pol, self.crs, crs)
        else:
            pol_crs = self._pol
        
        pol_glt = self.glt.footprint(crs=crs)

        return pol_crs.intersection(pol_glt)
    
    def set_band_selection(self, band_selection:Optional[Union[int, Tuple[int, ...],slice]]=None):
        """
        Set the band selection. Band selection is absolute w.r.t self.nc_ds['radiance']

        Args:
            band_selection (Optional[Union[int, Tuple[int, ...],slice]], optional): slicing or selection of the bands. Defaults to None.
        
        Example:
            >>> emit_image.set_band_selection(slice(0, 3)) # will only load the three first bands
            >>> emit_image.wavelengths # will only return the wavelengths of the three first bands
            >>> emit_image.load() # will only load the three first bands
        """
        if band_selection is None:
            band_selection = slice(None)
        self.band_selection = band_selection
        self.wavelengths = self.nc_ds['sensor_band_parameters'][self.bandname_dimension][self.band_selection]
        self.fwhm = self.nc_ds['sensor_band_parameters']['fwhm'][self.band_selection]
    
    @ property
    def nc_ds_obs(self, obs_file:Optional[str]=None) -> netCDF4.Dataset:
        """
        Loads the observation file. In this file we have information about angles (solar and viewing),
        elevation and ilumination based on elevation and path length.

        This function downloads the observation file if it does not exist from the JPL portal.

        It caches the observation file in the object. (self.nc_ds_obs)

        Args:
            obs_file (Optional[str], optional): Path to the observation file. 
                Defaults to None. If none it will download the observation file 
                from the EMIT server.
        """
        if self._nc_ds_obs is not None:
            return self._nc_ds_obs
        
        if obs_file is None:
            link_obs_file = get_obs_link(self.filename)
            obs_file = os.path.join(os.path.dirname(self.filename), os.path.basename(link_obs_file))
            if not os.path.exists(obs_file):
                download_product(link_obs_file, obs_file)
        
        self._nc_ds_obs = netCDF4.Dataset(obs_file)
        self._nc_ds_obs.set_auto_mask(False)
        self._observation_bands = self._nc_ds_obs['sensor_band_parameters']['observation_bands'][:]
    
    @property
    def observation_bands(self) -> np.array:
        """ Returns the observation bands """
        self.nc_ds_obs
        return self._observation_bands
    
    def observation(self, name:str) -> GeoTensor:
        """ Returns the observation with the given name """
        band_index = self.observation_bands.tolist().index(name)
        slice_y, slice_x = self.window_raw.toslices()
        obs_arr = self.nc_ds_obs['obs'][slice_y, slice_x, band_index]
        return self.georreference(obs_arr, 
                                  fill_value_default=self.nc_ds_obs['obs']._FillValue)

    def sza(self) -> GeoTensor:
        """ Return the solar zenith angle as a GeoTensor """
        return self.observation('To-sun zenith (0 to 90 degrees from zenith)')
    
    def vza(self) -> GeoTensor:
        """ Return the view zenith angle as a GeoTensor """
        return self.observation('To-sensor zenith (0 to 90 degrees from zenith)')
    
    def elevation(self) -> GeoTensor:
        obs_arr = self.nc_ds["location"]["elev"]
        slice_y, slice_x = self.window_raw.toslices()
        return self.georreference(obs_arr[slice_y, slice_x], 
                                  fill_value_default=obs_arr._FillValue)

    @property
    def mean_sza(self) -> float:
        """ Return the mean solar zenith angle """
        if self._mean_sza is not None:
            return self._mean_sza
        
        band_index = self.observation_bands.tolist().index('To-sun zenith (0 to 90 degrees from zenith)')
        sza_arr = self.nc_ds_obs['obs'][..., band_index]
        self._mean_sza = np.mean(sza_arr[sza_arr != self.nc_ds_obs['obs']._FillValue])
        return self._mean_sza
    
    @property
    def mean_vza(self) -> float:
        """ Return the mean view zenith angle """
        if self._mean_vza is not None:
            return self._mean_vza
        band_index = self.observation_bands.tolist().index('To-sensor zenith (0 to 90 degrees from zenith)')
        vza_arr = self.nc_ds_obs['obs'][..., band_index]
        self._mean_vza = np.mean(vza_arr[vza_arr != self.nc_ds_obs['obs']._FillValue])
        return self._mean_vza
        
    def __copy__(self) -> '__class__':
        out = EMITImage(self.filename, glt=self.glt.copy(), band_selection=self.band_selection)
        
        # copy nc_ds_obs if it exists
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, self.nc_ds_obs)

        return out
    def copy(self) -> '__class__':
        return self.__copy__()
    
    def to_crs(self, crs:Any, resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=60) -> '__class__':
        """
        Reproject the image to a new crs

        Args:
            crs (Any): CRS. 

        Returns:
            EmitImage: EMIT image in the new CRS
        
        Example:
            >>> emit_image = EMITImage("path/to/emit_image.nc")
            >>> crs_utm = georeader.get_utm_epsg(emit_image.footprint("EPSG:4326"))
            >>> emit_image_utm = emit_image.to_crs(crs_utm)
        """
        glt = read.read_to_crs(self.glt, crs, resampling=rasterio.warp.Resampling.nearest, 
                               resolution_dst_crs=resolution_dst_crs)

        out = EMITImage(self.filename, glt=glt, band_selection=self.band_selection)
        # Copy _pol attribute if it exists
        if hasattr(self, '_pol'):
            setattr(out, '_pol', self._pol)
        
        return out


    def read_from_window(self, window:Optional[rasterio.windows.Window]=None, boundless:bool=True) -> '__class__':
        glt_window = self.glt.read_from_window(window, boundless=boundless)
        out = EMITImage(self.filename, glt=glt_window, band_selection=self.band_selection)

        # copy attributes as in __copy__ method
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, self.nc_ds_obs)

        return out
    
    def read_from_bands(self, bands:Union[int, Tuple[int, ...], slice]) -> '__class__':
        copy = self.__copy__()
        copy.set_band_selection(bands)
        return copy
  
    def load(self, boundless:bool=True)-> GeoTensor:
        data = self.load_raw() # (C, H, W) or (H, W)
        return self.georreference(data)
    
    @property
    def shape_raw(self) -> Tuple[int, int, int]:
        """ Return the shape of the raw data in (C, H, W) format """
        return (len(self.wavelengths),) + rasterio.windows.shape(self.window_raw)

    def _bounds_indexes_raw(self) -> Tuple[int, int, int, int]:
        """ Return the bounds of the raw data: (min_x, min_y, max_x, max_y) """
        min_x = np.min(self.glt.values[0, self.valid_glt])
        max_x = np.max(self.glt.values[0, self.valid_glt])
        min_y = np.min(self.glt.values[1, self.valid_glt])
        max_y = np.max(self.glt.values[1, self.valid_glt])
        return min_x, min_y, max_x, max_y


    def load_raw(self, transpose:bool=True) -> np.array:
        """
        Load the raw data, without orthorectification

        Args:
            transpose (bool, optional): Transpose the data if it has 3 dimentsions to (C, H, W) 
                Defaults to True. if False return (H, W, C)

        Returns:
            np.array: raw data (C, H, W) or (H, W)
        """

        slice_y, slice_x = self.window_raw.toslices()

        if isinstance(self.band_selection, slice):
            data = np.array(self.nc_ds['radiance'][slice_y, slice_x, self.band_selection])
        else:
            data = np.array(self.nc_ds['radiance'][slice_y, slice_x][..., self.band_selection])
        
        # transpose to (C, H, W)
        if transpose and (len(data.shape) == 3):
            data = np.transpose(data, axes=(2, 0, 1))
        
        return data


    def georreference(self, data:np.array, 
                      fill_value_default:Optional[Union[int,float]]=None) -> GeoTensor:
        """
        Georreference an image in sensor coordinates to coordinates of the current 
        georreferenced object. If you do some processing with the raw data, you can 
        georreference the raw output with this function.

        Args:
            data (np.array): raw data (C, H, W) or (H, W). 

        Returns:
            GeoTensor: georreferenced version of data (C, H', W') or (H', W')
        
        Example:
            >>> emit_image = EMITImage("path/to/emit_image.nc")
            >>> emit_image_rgb = emit_image.read_from_bands([35, 23, 11])
            >>> data_rgb = emit_image_rgb.load_raw() # (3, H, W)
            >>> data_rgb_ortho = emit_image.georreference(data_rgb) # (3, H', W')
        """
        spatial_shape = self.shape[-2:]
        if len(data.shape) == 3:
            shape = data.shape[:-2] + spatial_shape
        elif len(data.shape) == 2:
            shape = spatial_shape
        else:
            raise ValueError(f"Data shape {data.shape} not supported")

        if fill_value_default is None:
            fill_value_default = self.fill_value_default
        outdat = np.full(shape, dtype=data.dtype, 
                         fill_value=fill_value_default)
        
        if len(data.shape) == 3:
            outdat[:, self.valid_glt] = data[:, self.glt_relative[1, self.valid_glt], 
                                             self.glt_relative[0, self.valid_glt]]
        else:
            outdat[self.valid_glt] = data[self.glt_relative[1, self.valid_glt], 
                                          self.glt_relative[0, self.valid_glt]]
            
        return GeoTensor(values=outdat, transform=self.transform, crs=self.crs,
                         fill_value_default=fill_value_default)

        
    @property
    def values(self) -> np.array:
        # return np.zeros(self.shape, dtype=self.dtype)
        raise self.load(boundless=True).values
    
    def __repr__(self)->str:
        return f""" 
         File: {self.filename}
         Transform: {self.transform}
         Shape: {self.shape}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         units: {self.units}
        """