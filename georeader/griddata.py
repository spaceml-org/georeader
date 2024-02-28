import georeader
from shapely.geometry import Polygon
from georeader.abstract_reader import GeoData
from scipy.interpolate import CloughTocher2DInterpolator
from georeader.window_utils import polygon_to_crs, res, transform_to_resolution_dst
from typing import Tuple, Union, Optional, Any
import rasterio
import rasterio.transform
import rasterio.warp
from georeader.geotensor import GeoTensor
import numbers
import numpy as np
from numpy.typing import NDArray
import math


def footprint(lons:NDArray, lats:NDArray) -> Polygon:
    """
    Returns the Polygon surrounding the given longitudes and latitudes

    Args:
        lons (np.array): 2D array of longitudes
        lats (np.array): 2D array of latitudes

    Returns:
        Polygon: Polygon surrounding the given longitudes and latitudes
    """
    lonsrav = lons.ravel()
    latsrav = lats.ravel()
    idxminlon = np.argmin(lonsrav)
    idxminlat = np.argmin(latsrav)
    idxmaxlon = np.argmax(lonsrav)
    idxmaxlat = np.argmax(latsrav)

    return Polygon([(lonsrav[idx],latsrav[idx]) for idx in [idxminlon, idxminlat, idxmaxlon, idxmaxlat]])


# def bounds(lons:np.array, lats:np.array) -> Tuple[float, float, float, float]:
#     minx = np.min(lons)
#     maxx = np.max(lons)
#     miny = np.min(lats)
#     maxy = np.max(lats)
#     return minx, miny, maxx, maxy

def read_reproject_like(data:NDArray, lons: NDArray, lats:NDArray, 
                        data_like:GeoData, resolution_dst:Optional[Union[float, Tuple[float,float]]]=None,
                        fill_value_default:Optional[float]=None,
                        crs:Optional[Any]="EPSG:4326") -> GeoTensor:
    """
    Reprojects data to the same crs, transform and shape as data_like

    Args:
        data (Array): input data 2D or 3D in the form (height, width, bands)
        lons (Array): 2D array of longitudes
        lats (Array): 2D array of latitudes
        data_like (GeoData): GeoData to reproject to
        resolution_dst (Optional[Union[float, Tuple[float,float]]], optional): If provided, the output resolution will be set to this value.
         Otherwise, the output resolution will be the same as data_like. Defaults to None.
        fill_value_default (Optional[float], optional): fill value. Defaults to None.
        crs (Optional[Any], optional): Input crs. Defaults to "EPSG:4326".

    Returns:
        GeoTensor: with reprojected data
    """
    width = data_like.shape[-1]
    height = data_like.shape[-2]
    transform = data_like.transform
    dst_crs = data_like.crs
    if resolution_dst is not None:
        transform = transform_to_resolution_dst(transform, resolution_dst)

    fill_value_default = fill_value_default or data_like.fill_value_default
    return reproject(data, lons, lats, width, height, transform, dst_crs, 
                     fill_value_default=fill_value_default, crs=crs)


def read_to_crs(data:NDArray, lons: NDArray, lats:NDArray, 
                resolution_dst:Union[float, Tuple[float,float]], 
                dst_crs:Optional[Any]=None,fill_value_default:float=-1,
                crs:Optional[Any]="EPSG:4326") -> GeoTensor:
    """
    Reprojects data to the given dst_crs figuring out the transform and shape.

    Args:
        data (Array): 2D or 3D in the form (H, W, bands)
        lons (Array): 2D array of longitudes (H, W).
        lats (Array): 2D array of latitudes (H, W).
        resolution_dst (Union[float, Tuple[float,float]]): Output resolution
        dst_crs (Optional[Any], optional): Output crs. If None, 
            the dst_crs will be the UTM crs of the center of the data. Defaults to None.
        fill_value_default (float, optional): fill value. Defaults to -1.
        crs (_type_, optional): Input crs. Defaults to "EPSG:4326".

    Returns:
        GeoTensor: with reprojected data
    """
    
    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))
    
    # Figure out UTM crs
    if dst_crs is None:
        mean_lat = np.nanmean(lats)
        mean_lon = np.nanmean(lons)
        dst_crs = georeader.get_utm_epsg((mean_lon, mean_lat), 
                                         crs_point_or_geom=crs)

    # Figure out transform
    pol = footprint(lons, lats)
    pol_dst_crs = polygon_to_crs(pol, crs_polygon=crs, dst_crs=dst_crs)
    minx, miny, maxx, maxy = pol_dst_crs.bounds

    # Add the resolution to the max values to get the correct shape.
    maxx = maxx + resolution_dst[0]
    miny = miny - resolution_dst[1]
    transform = rasterio.transform.from_origin(minx, maxy, resolution_dst[0], resolution_dst[1])

    # resolution_dst= res(transform)
    width = math.ceil(abs((maxx -minx) / resolution_dst[0]))
    height = math.ceil(abs((maxy - miny) / resolution_dst[1]))

    return reproject(data, lons, lats, width, height, transform, dst_crs, 
                     fill_value_default=fill_value_default,
                     crs=crs)


def reproject(data:NDArray, lons: NDArray, lats: NDArray,
              width:int, height:int, transform:rasterio.transform.Affine,
              dst_crs:Any, crs:Optional[Any]="EPSG:4326", fill_value_default=-1) -> GeoTensor:
    """
    Reprojects data to  given crs, transform and shape

    Args:
        data (Array): input data 2D or 3D in the form (height, width, bands)
        lons (Array): 2D array of longitudes
        lats (Array): 2D array of latitudes
        width (int): Output width
        height (int): Output height
        transform (rasterio.transform.Affine): Output transform
        dst_crs (Any): Output crs
        crs (Any, optional): Input crs. Defaults to "EPSG:4326".
        fill_value_default (int, optional): fill value. Defaults to -1.

    Raises:
        ValueError: if data is not 2D or 3D

    Returns:
        GeoTensor: with reprojected data
    
    """
    data = data.squeeze()
    if len(data.shape) == 3:
        data_ravel = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    elif len(data.shape) == 2:
        data_ravel = data.ravel()
    else:
        raise ValueError("Data shape not supported")
    
    # Generate the meshgrid of lons and lats to interpolate the data
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.array(xs)
    ys = np.array(ys)
    lonsdst, latssdst = rasterio.warp.transform(dst_crs, crs, xs.ravel(),ys.ravel())
    lonsdst = np.array(lonsdst).reshape(height, width)
    latssdst = np.array(latssdst).reshape(height, width)

    interpfun = CloughTocher2DInterpolator(list(zip(lons.ravel(), lats.ravel())), 
                                           data_ravel)
    
    dataout = interpfun(lonsdst, latssdst) # (H, W) or (H, W, C)
    nanvals = np.isnan(dataout)
    if np.any(nanvals):
        dataout[nanvals] = fill_value_default
    
    # transpose if 3D to (C, H, W) format
    if len(data.shape) == 3:
        dataout = np.transpose(dataout, (2, 0, 1))

    return GeoTensor(dataout, transform=transform, 
                     crs=dst_crs, fill_value_default=fill_value_default)


def georreference(glt:GeoTensor, data:NDArray, valid_glt:Optional[NDArray] = None,
                  fill_value_default:Optional[Union[int,float]]=None) -> GeoTensor:
    """
        Georreference an image in sensor coordinates to coordinates of the current 
        georreferenced object. If you do some processing with the raw data, you can 
        georreference the raw output with this function.

        Args:
            data (np.array): raw data (C, H, W) or (H, W). 
            glt (GeoTensor): (2, H', W') GeoTensor with the geolocation data. This array tells
                the coordinates of each pixel in the raw data. That is:
                    output_data[:, i, j] = data[:, glt.values[1, i, j], glt.values[0, i, j]]
            valid_glt (Optional[np.array], optional): 2D array of valid pixels. Defaults to None.
                If None, the valid pixels are the ones that are not equal to the fill_value_default
                in the glt array.
            fill_value_default (Optional[Union[int,float]], optional): fill value for the output array.

        Returns:
            GeoTensor: georreferenced version of data (C, H', W') or (H', W')
        
    """
    spatial_shape = glt.shape[-2:]
    if len(data.shape) == 3:
        shape = data.shape[:-2] + spatial_shape
    elif len(data.shape) == 2:
        shape = spatial_shape
    else:
        raise ValueError(f"Data shape {data.shape} not supported")

    if fill_value_default is None:
        fill_value_default = 0
    outdat = np.full(shape, dtype=data.dtype, 
                      fill_value=fill_value_default)

    if valid_glt is None:
        valid_glt = np.all(glt.values != glt.fill_value_default, axis=0)
    
    if len(data.shape) == 3:
        outdat[:, valid_glt] = data[:, glt.values[1, valid_glt], 
                                            glt.values[0, valid_glt]]
    else:
        outdat[valid_glt] = data[glt.values[1, valid_glt], 
                                 glt.values[0, valid_glt]]
        
    return GeoTensor(values=outdat, transform=glt.transform, crs=glt.crs,
                     fill_value_default=fill_value_default)