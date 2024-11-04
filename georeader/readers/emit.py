"""
Module to read EMIT images.

Requires: netCDF4:
pip install netCDF4 

Some of the functions of this module are based on the official EMIT repo: https://github.com/emit-sds/emit-utils/

"""
import os
import json
from typing import Tuple, Optional, Any, Union, Dict
from georeader.readers.download_utils import download_product as download_product_base
from georeader.abstract_reader import AbstractGeoData
import rasterio
import rasterio.windows
import numpy as np
from georeader import window_utils
from shapely.geometry import Polygon
from georeader.geotensor import GeoTensor
from georeader import reflectance
from shapely.ops import unary_union
from georeader import read
import rasterio.warp
from numpy.typing import NDArray
from datetime import datetime, timezone
from georeader.griddata import georreference
from georeader import get_utm_epsg

try:
    import netCDF4
except ImportError:
    raise ImportError("netCDF4 is required to read EMIT images. Please install it with: pip install netCDF4")

AUTH_METHOD = "auth" # "auth" or "token"
TOKEN = None
WAVELENGTHS_RGB = np.array([640, 550, 460])

def _bounds_indexes_raw(glt:NDArray, valid_glt:NDArray) -> Tuple[int, int, int, int]:
        """ Return the bounds of the raw data: (min_x, min_y, max_x, max_y) """
        min_x = np.min(glt[0, valid_glt])
        max_x = np.max(glt[0, valid_glt])
        min_y = np.min(glt[1, valid_glt])
        max_y = np.max(glt[1, valid_glt])
        return min_x, min_y, max_x, max_y

def get_auth() -> Tuple[str, str]:
    home_dir = os.path.join(os.path.expanduser('~'),".georeader")
    json_file = os.path.join(home_dir, "auth_emit.json")
    if not os.path.exists(json_file):
        os.makedirs(home_dir, exist_ok=True)
        with open(json_file, "w") as fh:
            json.dump({"user": "SET-USER", "password": "SET-PASSWORD"}, fh)

        raise FileNotFoundError(f"In order to download EMIT images add user and password to file : {json_file}")

    with open(json_file, "r") as fh:
        data = json.load(fh)
    
    if data["user"] == "SET-USER":
        raise FileNotFoundError(f"In order to download EMIT images add user and password to file : {json_file}")

    return (data["user"], data["password"])


def get_headers() -> Optional[Dict[str, str]]:
    if TOKEN is None:
        return
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    return headers


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
                     display_progress_bar:bool=True,
                     auth:Optional[Tuple[str, str]] = None) -> str:
    """
    Download a product from the EMIT website (https://search.earthdata.nasa.gov/search). 
    It requires that you have an account in the NASA Earthdata portal. 

    This code is based on this example: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        link_down: link to the product
        filename: filename to save the product
        display_progress_bar: display tqdm progress bar
        auth: tuple with user and password to download the product. If None, it will try to read the user and password from ~/.georeader/auth_emit.json 

    Example:
        >>> link_down = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220828T051941_2224004_006/EMIT_L1B_RAD_001_20220828T051941_2224004_006.nc'
        >>> filename = download_product(link_down)
    """
    headers = None
    if auth is None:
        if AUTH_METHOD == "auth":
            auth = get_auth()
        elif AUTH_METHOD == "token":
            assert TOKEN is not None, "You need to set the TOKEN variable to download EMIT images"
            headers = get_headers()
    
    return download_product_base(link_down, filename=filename, auth=auth,
                                 headers=headers,
                                 display_progress_bar=display_progress_bar, 
                                 verify=False)


def get_radiance_link(product_path:str) -> str:
    """
    Get the link to download a product from the EMIT website.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        product_path: path to the product or filename of the product or product name with or without extension.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Example:
        >>> product_path = 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        >>> link = get_radiance_link(product_path)
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220827T060753_2223904_013/EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
    """
    "EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc"
    namefile = os.path.splitext(os.path.basename(product_path))[0]
    product_id = os.path.splitext(namefile)[0]
    content_id = product_id.split("_")
    content_id[1] = "L1B"
    content_id[2] = "RAD"
    content_id[3] = content_id[3].replace("V", "")
    product_id = "_".join(content_id)
    link = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{product_id}/{product_id}.nc"
    return link


def get_obs_link(product_path:str) -> str:
    """
    Get the link to download a product from the EMIT website.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        product_path: path to the product or filename of the product with or without extension.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Example:
        >>> product_path = 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        >>> link = get_radiance_link(product_path)
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220827T060753_2223904_013/EMIT_L1B_OBS_001_20220827T060753_2223904_013.nc'
    """
    namefile = os.path.splitext(os.path.basename(product_path))[0]

    product_id = os.path.splitext(namefile)[0]
    content_id = product_id.split("_")
    content_id[1] = "L1B"
    content_id[2] = "RAD"
    content_id[3] = content_id[3].replace("V", "")
    product_id = "_".join(content_id)

    content_id[2] = "OBS"
    namefile = "_".join(content_id)

    link = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/{product_id}/{namefile}.nc"
    return link


def get_ch4enhancement_link(tile:str) -> str:
    """
    Get the link to download a product from the EMIT website.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        product_path: path to the product or filename of the product with or without extension.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Example:
        >>> product_path = 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        >>> link = get_radiance_link(product_path)
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.001/EMIT_L2B_CH4ENH_001_20220810T064957_2222205_033/EMIT_L2B_CH4ENH_001_20220810T064957_2222205_033.tif'
    """
    namefile = os.path.splitext(os.path.basename(tile))[0]

    product_id = os.path.splitext(namefile)[0]
    content_id = product_id.split("_")
    content_id[1] = "L2B"
    content_id[2] = "CH4ENH"
    content_id[3] = content_id[3].replace("V", "")
    product_id = "_".join(content_id)
    link = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.001/{product_id}/{product_id}.tif"
    return link


def get_l2amask_link(tile:str) -> str:
    """
    Get the link to download a product from the EMIT website (https://search.earthdata.nasa.gov/search)


    Args:
        tile (str): path to the product or filename of the L1B product with or without extension.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        
    Returns:
        str: link to the L2A mask product
    
    Example:
        >>> tile = 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        >>> link = get_l2amask_link(tile)
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20220827T060753_2223904_013/EMIT_L2A_MASK_001_20220827T060753_2223904_013.nc'
    """
    namefile = os.path.splitext(os.path.basename(tile))[0]
    namefile = namefile + ".nc"

    product_id = os.path.splitext(namefile)[0]
    content_id = product_id.split("_")
    content_id[1] = "L2A"
    content_id[2] = "RFL"
    content_id[3] = content_id[3].replace("V", "")
    product_id = "_".join(content_id)
    
    content_id[2] = "MASK"
    namefilenew = "_".join(content_id) + ".nc"
    link = f"https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/{product_id}/{namefilenew}"
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
    attributes_set_if_exists = ["_nc_ds_obs", "_mean_sza", "_mean_vza", 
                                "_observation_bands", "_nc_ds_l2amask", "_mask_bands", 
                                "_nc_ds", "obs_file",
                                "l2amaskfile"]
    def __init__(self, filename:str, glt:Optional[GeoTensor]=None, 
                 band_selection:Optional[Union[int, Tuple[int, ...],slice]]=slice(None)):
        self.filename = filename
        self.nc_ds = netCDF4.Dataset(self.filename, 'r', format='NETCDF4')
        self._nc_ds_obs:Optional[netCDF4.Dataset] = None
        self._nc_ds_l2amask:Optional[netCDF4.Dataset] = None
        self._observation_bands = None
        self._mask_bands = None
        self.nc_ds.set_auto_mask(False) # disable automatic masking when reading data
        # self.real_shape = (self.nc_ds['radiance'].shape[-1],) + self.nc_ds['radiance'].shape[:-1]

        self._mean_sza = None
        self._mean_vza = None
        self.obs_file:Optional[str] = None
        self.l2amaskfile:Optional[str] = None

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
        self.glt_relative = self.glt.copy()
        self.glt_relative.values[0, self.valid_glt] -= xmin
        self.glt_relative.values[1, self.valid_glt] -= ymin

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
        self._observation_date_correction_factor:Optional[float] = None

    @property
    def observation_date_correction_factor(self) -> float:
        if self._observation_date_correction_factor is None:
            self._observation_date_correction_factor = reflectance.observation_date_correction_factor(date_of_acquisition=self.time_coverage_start,
                                                                                                      center_coords=self.footprint("EPSG:4326").centroid.coords[0])
        return self._observation_date_correction_factor
    
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
    def width(self) -> int:
        return self.shape[-1]
    
    @property
    def height(self) -> int:
        return self.shape[-2]

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
        
        self.obs_file = obs_file
        self._nc_ds_obs = netCDF4.Dataset(obs_file)
        self._nc_ds_obs.set_auto_mask(False)
        self._observation_bands = self._nc_ds_obs['sensor_band_parameters']['observation_bands'][:]
        return self._nc_ds_obs
    
    @property
    def nc_ds_l2amask(self, l2amaskfile:Optional[str]=None) -> netCDF4.Dataset:
        """
        Loads the L2A mask file. In this file we have information about the cloud mask.

        This function downloads the L2A mask file if it does not exist from the JPL portal.

        It caches the L2A mask file in the object. (self.nc_ds_l2amask)

        See https://lpdaac.usgs.gov/products/emitl2arflv001/ for info about the L2A mask file.

        Args:
            l2amaskfile (Optional[str], optional): Path to the L2A mask file. 
                Defaults to None. If none it will download the L2A mask file 
                from the EMIT server.
        """
        if self._nc_ds_l2amask is not None:
            return self._nc_ds_l2amask
        
        if l2amaskfile is None:
            link_l2amaskfile = get_l2amask_link(self.filename)
            l2amaskfile = os.path.join(os.path.dirname(self.filename), os.path.basename(link_l2amaskfile))
            if not os.path.exists(l2amaskfile):
                download_product(link_l2amaskfile, l2amaskfile)
        
        self.l2amaskfile = l2amaskfile
        self._nc_ds_l2amask = netCDF4.Dataset(l2amaskfile)
        self._nc_ds_l2amask.set_auto_mask(False)
        self._mask_bands = self._nc_ds_l2amask["sensor_band_parameters"]["mask_bands"][:]
        return self._nc_ds_l2amask
    
    @property
    def mask_bands(self) -> np.array:
        """ Returns the mask bands -> ['Cloud flag', 'Cirrus flag', 'Water flag', 'Spacecraft Flag',
       'Dilated Cloud Flag', 'AOD550', 'H2O (g cm-2)', 'Aggregate Flag'] """
        self.nc_ds_l2amask
        return self._mask_bands
    
    def validmask(self, with_buffer:bool=True) -> GeoTensor:
        """
        Return the validmask mask

    
        Returns:
            GeoTensor: bool mask. True means that the pixel is valid.
        """

        validmask = ~self.invalid_mask_raw(with_buffer=with_buffer)

        return self.georreference(validmask,
                                  fill_value_default=False)
    
    def invalid_mask_raw(self, with_buffer:bool=True) -> NDArray:
        """
        Returns the non georreferenced quality mask. True means that the pixel is not valid.

        This mask is computed as the sum of the Cloud flag, Cirrus flag, Spacecraft flag and Dilated Cloud Flag.
        True means that the pixel is not valid.

        From: https://github.com/nasa/EMIT-Data-Resources/blob/main/python/how-tos/How_to_use_EMIT_Quality_data.ipynb
        and https://github.com/nasa/EMIT-Data-Resources/blob/main/python/modules/emit_tools.py#L277


        """
        band_index =  [0,1,3]
        if with_buffer:
            band_index.append(4)
        
        slice_y, slice_x = self.window_raw.toslices()
        mask_arr = self.nc_ds_l2amask['mask'][slice_y, slice_x, band_index]
        mask_arr = np.sum(mask_arr, axis=-1)
        mask_arr = (mask_arr >= 1)
        return mask_arr
    
    @property
    def percentage_clear(self) -> float:
        """
        Return the percentage of clear pixels in the image

        Returns:
            float: percentage of clear pixels
        """
        
        invalids = self.invalid_mask_raw(with_buffer=False)
        return 100 * (1 - np.sum(invalids) / np.prod(invalids.shape))


    def mask(self, mask_name:str="cloud_mask") -> GeoTensor:
        """
        Return the mask layer with the given name.
        Mask shall be one of self.mask_bands -> ['Cloud flag', 'Cirrus flag', 'Water flag', 'Spacecraft Flag',
       'Dilated Cloud Flag', 'AOD550', 'H2O (g cm-2)', 'Aggregate Flag']

        Args:
            mask_name (str, optional): Name of the mask. Defaults to "cloud_mask".

        Returns:
            GeoTensor: mask
        """
        band_index = self.mask_bands.tolist().index(mask_name)
        slice_y, slice_x = self.window_raw.toslices()
        mask_arr = self.nc_ds_l2amask['mask'][slice_y, slice_x, band_index]
        return self.georreference(mask_arr,
                                  fill_value_default=self.nc_ds_l2amask['mask']._FillValue)
    
    def water_mask(self) -> GeoTensor:
        """ Returns the water mask """
        return self.mask("Water flag")
    
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
        self._mean_sza = float(np.mean(sza_arr[sza_arr != self.nc_ds_obs['obs']._FillValue]))
        return self._mean_sza
    
    @property
    def mean_vza(self) -> float:
        """ Return the mean view zenith angle """
        if self._mean_vza is not None:
            return self._mean_vza
        band_index = self.observation_bands.tolist().index('To-sensor zenith (0 to 90 degrees from zenith)')
        vza_arr = self.nc_ds_obs['obs'][..., band_index]
        self._mean_vza = float(np.mean(vza_arr[vza_arr != self.nc_ds_obs['obs']._FillValue]))
        return self._mean_vza
        
    def __copy__(self) -> '__class__':
        out = EMITImage(self.filename, glt=self.glt.copy(), band_selection=self.band_selection)
        
        # copy nc_ds_obs if it exists
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, getattr(self, attrname))

        return out
    def copy(self) -> '__class__':
        return self.__copy__()
    
    def to_crs(self, crs:Any="UTM", 
               resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=60) -> '__class__':
        """
        Reproject the image to a new crs

        Args:
            crs (Any): CRS. 

        Returns:
            EmitImage: EMIT image in the new CRS
        
        Example:
            >>> emit_image = EMITImage("path/to/emit_image.nc")
            >>> emit_image_utm = emit_image.to_crs(crs="UTM")
        """
        if crs == "UTM":
            footprint = self.glt.footprint("EPSG:4326")
            crs = get_utm_epsg(footprint)

        glt = read.read_to_crs(self.glt, crs, resampling=rasterio.warp.Resampling.nearest, 
                               resolution_dst_crs=resolution_dst_crs)

        out = EMITImage(self.filename, glt=glt, band_selection=self.band_selection)
        # Copy _pol attribute if it exists
        if hasattr(self, '_pol'):
            setattr(out, '_pol', window_utils.polygon_to_crs(self._pol, self.crs, crs))
        
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
  
    def load(self, boundless:bool=True, as_reflectance:bool=False)-> GeoTensor:
        data = self.load_raw() # (C, H, W) or (H, W)
        if as_reflectance:
            thuiller = reflectance.load_thuillier_irradiance()
            response = reflectance.srf(self.wavelengths, self.fwhm, thuiller["Nanometer"].values)
            solar_irradiance_norm = thuiller["Radiance(mW/m2/nm)"].values.dot(response) / 1_000
            data = reflectance.radiance_to_reflectance(data, solar_irradiance_norm,
                                                       units=self.units,
                                                       observation_date_corr_factor=self.observation_date_correction_factor)

        return self.georreference(data)
    
    def load_rgb(self, as_reflectance:bool=True) -> GeoTensor:
        bands_read = np.argmin(np.abs(WAVELENGTHS_RGB[:, np.newaxis] - self.wavelengths), axis=1).tolist()
        ei_rgb = self.read_from_bands(bands_read)
        return ei_rgb.load(boundless=True, as_reflectance=as_reflectance)

    @property
    def shape_raw(self) -> Tuple[int, int, int]:
        """ Return the shape of the raw data in (C, H, W) format """
        return (len(self.wavelengths),) + rasterio.windows.shape(self.window_raw)

    def _bounds_indexes_raw(self) -> Tuple[int, int, int, int]:
        """ Return the bounds of the raw data: (min_x, min_y, max_x, max_y) """
        return _bounds_indexes_raw(self.glt.values, self.valid_glt)


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
        return georreference(self.glt_relative, data, self.valid_glt, 
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


def valid_mask(filename:str, with_buffer:bool=False, 
               dst_crs:Optional[Any]="UTM", 
               resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=60) -> Tuple[GeoTensor, float]:
    """
    Loads the valid mask from the EMIT L2AMASK file.

    Args:
        filename (str): path to the L2AMASK file. e.g. EMIT_L2A_MASK_001_20220827T060753_2223904_013.nc
        with_buffer (bool, optional): If True, the buffer band is used to compute the valid mask. Defaults to False.

    Returns:
        GeoTensor: valid mask
    """
    
    nc_ds = netCDF4.Dataset(filename, 'r', format='NETCDF4')
    nc_ds.set_auto_mask(False)

    real_transform = rasterio.Affine(nc_ds.geotransform[1], nc_ds.geotransform[2], nc_ds.geotransform[0],
                                     nc_ds.geotransform[4], nc_ds.geotransform[5], nc_ds.geotransform[3])
    
    glt_arr = np.zeros((2,) + nc_ds.groups['location']['glt_x'].shape, dtype=np.int32)
    glt_arr[0] = np.array(nc_ds.groups['location']['glt_x'])
    glt_arr[1] = np.array(nc_ds.groups['location']['glt_y'])
    # glt_arr -= 1 # account for 1-based indexing

    # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html
    glt = GeoTensor(glt_arr, transform=real_transform, 
                    crs=rasterio.crs.CRS.from_wkt(nc_ds.spatial_ref),
                    fill_value_default=0)
    
    if dst_crs is not None:
        if dst_crs == "UTM":
            footprint = glt.footprint("EPSG:4326")
            dst_crs = get_utm_epsg(footprint)

        glt = read.read_to_crs(glt, dst_crs=dst_crs, 
                               resampling=rasterio.warp.Resampling.nearest, 
                               resolution_dst_crs=resolution_dst_crs)
    
    valid_glt = np.all(glt.values != glt.fill_value_default, axis=0)
    xmin = np.min(glt.values[0, valid_glt])
    ymin = np.min(glt.values[1, valid_glt])

    glt_relative = glt.copy()
    glt_relative.values[0, valid_glt] -= xmin
    glt_relative.values[1, valid_glt] -= ymin
    # mask_bands = nc_ds["sensor_band_parameters"]["mask_bands"][:]

    band_index =  [0,1,3]
    if with_buffer:
        band_index.append(4)
    
    mask_arr = nc_ds['mask'][:, :, band_index]
    invalidmask_raw = np.sum(mask_arr, axis=-1)
    invalidmask_raw = (invalidmask_raw >= 1)

    validmask = ~invalidmask_raw

    percentage_clear = 100 * (np.sum(validmask) / np.prod(validmask.shape))

    return georreference(glt_relative, validmask, valid_glt,
                         fill_value_default=False), percentage_clear