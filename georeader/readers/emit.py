"""
Module to read EMIT images.

Requires: netCDF4:
pip install netCDF4 

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
                                  display_progress_bar=display_progress_bar)



class EMITImage:
    """
    Class to read L1B EMIT images.

    See: https://github.com/emit-sds/emit-utils/
    

    """
    def __init__(self, filename:str, glt:Optional[GeoTensor]=None, 
                 band_selection:Optional[Union[int, Tuple[int, ...],slice]]=slice(None)):
        self.filename = filename
        self.nc_ds = netCDF4.Dataset(self.filename, 'r', format='NETCDF4')
        self.nc_ds.set_auto_mask(False)

        
        # self.real_shape = (self.nc_ds['radiance'].shape[-1],) + self.nc_ds['radiance'].shape[:-1]

        self.real_transform = rasterio.Affine(self.nc_ds.geotransform[1], self.nc_ds.geotransform[2], self.nc_ds.geotransform[0],
                                              self.nc_ds.geotransform[4], self.nc_ds.geotransform[5], self.nc_ds.geotransform[3])
        
        # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html

        self.dtype = self.nc_ds['radiance'].dtype
        self.dims = ("band", "y", "x")
        self.fill_value_default = self.nc_ds['radiance']._FillValue
        self.nodata = self.nc_ds['radiance']._FillValue
        self.units = self.nc_ds["radiance"].units

        if glt is None:
            glt = np.zeros((2,) + self.nc_ds.groups['location']['glt_x'].shape, dtype=np.int32)
            glt[0] = np.array(self.nc_ds.groups['location']['glt_x'])
            glt[1] = np.array(self.nc_ds.groups['location']['glt_y'])
            self.glt = GeoTensor(glt, transform=self.real_transform, 
                                 crs=rasterio.crs.CRS.from_wkt(self.nc_ds.spatial_ref),
                                 fill_value_default=0)
        else:
            self.glt = glt
        
        self.valid_glt = np.all(self.glt.values != self.glt.fill_value_default, axis=0)
        self.glt_zero_based = self.glt.values.copy()
        self.glt_zero_based[:, self.valid_glt] -= 1 # account for 1-based indexing

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
    
    def __copy__(self) -> '__class__':
        return EMITImage(self.filename, glt=self.glt.copy(), band_selection=self.band_selection)
    
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

        return EMITImage(self.filename, glt=glt, band_selection=self.band_selection)


    def read_from_window(self, window:Optional[rasterio.windows.Window]=None, boundless:bool=True) -> '__class__':
        glt_window = self.glt.read_from_window(window, boundless=boundless)
        return EMITImage(self.filename, glt=glt_window, band_selection=self.band_selection)
    
    def read_from_bands(self, bands:Union[int, Tuple[int, ...], slice]) -> '__class__':
        copy = self.__copy__()
        copy.set_band_selection(bands)
        return copy
  
    def load(self, boundless:bool=True)-> GeoTensor:
        data = self.load_raw() # (C, H, W) or (H, W)
        return self.orthorectify(data)
        
    
    def load_raw(self) -> np.array:
        """
        Load the raw data, without orthorectification

        Returns:
            np.array: raw data (C, H, W) or (H, W)
        """
        min_x = np.min(self.glt_zero_based[0, self.valid_glt])
        max_x = np.max(self.glt_zero_based[0, self.valid_glt])
        min_y = np.min(self.glt_zero_based[1, self.valid_glt])
        max_y = np.max(self.glt_zero_based[1, self.valid_glt])

        slice_x = slice(min_x, max_x + 1)
        slice_y = slice(min_y, max_y + 1)

        if isinstance(self.band_selection, slice):
            data = np.array(self.nc_ds['radiance'][slice_y, slice_x, self.band_selection])
        else:
            data = np.array(self.nc_ds['radiance'][slice_y, slice_x][..., self.band_selection])
        
        # transpose to (C, H, W)
        if len(data.shape) == 3:
            data = np.transpose(data, axes=(2, 0, 1))
        
        return data


    def orthorectify(self, data:np.array, 
                     fill_value_default:Optional[Union[int,float]]=None) -> GeoTensor:
        """
        Orthorectify an image in sensor coordinates to coordinates of the current 
        orthorectified object.

        Args:
            data (np.array): raw data (C, H, W) or (H, W). 

        Returns:
            array like: orthorectified version of data (C, H, W) or (H, W)
        """
        spatial_shape = self.shape[1:]
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
            outdat[:, self.valid_glt] = data[:, self.glt_zero_based[1, self.valid_glt], 
                                             self.glt_zero_based[0, self.valid_glt]]
        else:
            outdat[self.valid_glt] = data[self.glt_zero_based[1, self.valid_glt], 
                                          self.glt_zero_based[0, self.valid_glt]]
            
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