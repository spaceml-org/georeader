import geopandas as gpd
from typing import Union, Tuple, Any, Optional
from georeader.geotensor import GeoTensor
import numpy as np
import rasterio
import rasterio.windows
import rasterio.features
import numbers
from georeader.window_utils import PIXEL_PRECISION
from shapely.geometry import Polygon, MultiPolygon, LineString
from numbers import Number
from georeader import window_utils
from georeader.abstract_reader import GeoData


def rasterize_geometry_like(geometry:Union[Polygon, MultiPolygon, LineString], data_like: GeoData, value:Number=1,
                            dtype:Any=np.uint8,
                            crs_geom_bounds:Optional[Any]=None, fill:Union[int, float]=0, all_touched:bool=False,
                            return_only_data:bool=False)-> Union[GeoTensor, np.ndarray]:
    shape_out = data_like.shape
    return rasterize_from_geometry(geometry, crs_geom_bounds=crs_geom_bounds,
                                   transform=data_like.transform,
                                   window_out=rasterio.windows.Window(0, 0, width=shape_out[-1], height=shape_out[-2]),
                                   return_only_data=return_only_data,dtype=dtype, value=value,
                                   fill=fill, all_touched=all_touched)


def rasterize_from_geometry(geometry:Union[Polygon, MultiPolygon, LineString],
                            bounds:Optional[Tuple[float, float, float, float]]=None,
                            transform:Optional[rasterio.Affine]=None,
                            resolution:Optional[Union[float, Tuple[float, float]]]=None,
                            window_out:Optional[rasterio.windows.Window]=None,
                            value:Number=1,
                            dtype:Any=np.uint8,
                            crs_geom_bounds:Optional[Any]=None, fill:Union[int, float]=0, all_touched:bool=False,
                            return_only_data:bool=False)-> Union[GeoTensor, np.ndarray]:
    """
    Rasterise the provided geometry over the bounds with the specified resolution.

    Args:
        geometry: geometry to rasterise
        bounds: bounds where the polygons will be rasterised.
        transform: if transform is provided it will use this instead of resolution
        resolution: spatial resolution of the rasterised array. It won't be used if transform is provided
        window_out: Window out in dst_crs. If not provided it is computed from the bounds.
        value: column to take the values for rasterisation.
        dtype: dtype of the rasterise raster.
        crs_geom_bounds: CRS of geometry and bounds
        fill: fill option for rasterio.features.rasterize
        all_touched: all_touched option for rasterio.features.rasterize
        return_only_data: if `True` returns only the np.ndarray without georref info.

    Returns:
        GeoTensor or np.ndarray with shape (H, W) with the rasterised polygons
    """

    transform = window_utils.figure_out_transform(transform=transform, bounds=bounds,
                                                  resolution_dst=resolution)
    if window_out is None:
        window_out = rasterio.windows.from_bounds(*bounds,
                                                  transform=transform).round_lengths(op="ceil",
                                                                                     pixel_precision=PIXEL_PRECISION)


    chip_label = rasterio.features.rasterize(shapes=[(geometry, value)],
                                             out_shape=(window_out.height, window_out.width),
                                             transform=transform,
                                             dtype=dtype,
                                             fill=fill,
                                             all_touched=all_touched)
    if return_only_data:
        return chip_label

    return GeoTensor(chip_label, transform=transform, crs=crs_geom_bounds, fill_value_default=fill)


def rasterize_from_geopandas(dataframe:gpd.GeoDataFrame, bounds:Tuple[float, float, float, float],
                             column:str,
                             transform:Optional[rasterio.Affine]=None,
                             resolution:Optional[Union[float, Tuple[float, float]]]=None,
                             crs_out:Optional[Any]=None, fill:Number=0, all_touched:bool=False,
                             return_only_data:bool=False) -> Union[GeoTensor, np.ndarray]:
    """
    Rasterise the provided geodataframe over the bounds with the specified resolution.

    Args:
        dataframe: geodataframe with columns geometry and `column`. The 'geometry' column is expected to have shapely geometries
        bounds: bounds where the polygons will be rasterised with CRS `crs_out`.
        transform: if transform is provided if will use this for the resolution.
        resolution: spatial resolution of the rasterised array
        column: column to take the values for rasterisation.
        crs_out: defaults to dataframe.crs. This function will transform the geometries from dataframe.crs to this crs
        before rasterisation. `bounds` are in this crs.
        fill: fill option for rasterio.features.rasterize
        all_touched: all_touched option for rasterio.features.rasterize
        return_only_data: if `True` returns only the np.ndarray without georref info.

    Returns:
        GeoTensor or np.ndarray with shape (H, W) with the rasterised polygons
    """

    if crs_out is None:
        crs_out = str(dataframe.crs).lower()
    else:
        data_crs = str(dataframe.crs).lower()
        crs_out = str(crs_out).lower().replace("+init=","")
        if data_crs != crs_out:
            dataframe = dataframe.to_crs(crs=crs_out)

    if transform is None:
        if isinstance(resolution, numbers.Number):
            resolution = (abs(resolution), abs(resolution))
        transform = rasterio.transform.from_origin(min(bounds[0], bounds[2]),
                                                   max(bounds[1], bounds[3]),
                                                   resolution[0], resolution[1])
    else:
        # Shift transform to current window
        window_current_transform = rasterio.windows.from_bounds(*bounds,
                                                                transform=transform)
        transform = rasterio.windows.transform(window_current_transform, transform)


    window_out = rasterio.windows.from_bounds(*bounds,
                                              transform=transform).round_lengths(op="ceil",
                                                                                 pixel_precision=PIXEL_PRECISION)
    dtype = dataframe[column].dtype
    chip_label = rasterio.features.rasterize(shapes=zip(dataframe.geometry, dataframe[column]),
                                             out_shape=(window_out.height, window_out.width),
                                             transform=transform,
                                             dtype=dtype,
                                             fill=fill,
                                             all_touched=all_touched)
    if return_only_data:
        return chip_label

    return GeoTensor(chip_label, transform=transform, crs=crs_out,fill_value_default=fill)