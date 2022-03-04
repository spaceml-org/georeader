import geopandas as gpd
from typing import Union, Tuple, Any
from georeader.geotensor import GeoTensor
import numpy as np
import rasterio
import rasterio.windows
import rasterio.features
import numbers
from georeader.window_utils import PIXEL_PRECISION
from shapely.geometry import Polygon, MultiPolygon, LineString
from numbers import Number


def rasterize_from_geometry(geometry:Union[Polygon, MultiPolygon, LineString],
                            bounds:Tuple[float, float, float, float],
                            resolution:Union[float, Tuple[float, float]], value:Number=1,
                            dtype:Any=np.uint8,
                            crs_out=None, fill=0, all_touched:bool=False,
                            return_only_data:bool=False)-> Union[GeoTensor, np.ndarray]:
    """
    Rasterise the provided geometry over the bounds with the specified resolution.

    Args:
        geometry: geometry to rasterise
        bounds: bounds where the polygons will be rasterised.
        resolution: spatial resolution of the rasterised array
        value: column to take the values for rasterisation.
        dtype: dtype of the rasterise raster.
        crs_out: if provided it will transform the geometries from dataframe.crs to this crs before rasterisation.
            It expects that `bounds` are in this crs.
        fill: fill option for rasterio.features.rasterize
        all_touched: all_touched option for rasterio.features.rasterize
        return_only_data: if `True` returns only the np.ndarray without georref info.

    Returns:
        GeoTensor or np.ndarray with shape (H, W) with the rasterised polygons
    """

    transform = rasterio.transform.from_origin(min(bounds[0], bounds[2]),
                                               max(bounds[1], bounds[3]),
                                               resolution[0], resolution[1])

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

    return GeoTensor(chip_label, transform=transform, crs=crs_out, fill_value_default=fill)


def rasterize_from_geopandas(dataframe:gpd.GeoDataFrame, bounds:Tuple[float, float, float, float],
                             resolution:Union[float, Tuple[float, float]], column:str,
                             crs_out=None, fill=0, all_touched:bool=False,
                             return_only_data:bool=False) -> Union[GeoTensor, np.ndarray]:
    """
    Rasterise the provided geodataframe over the bounds with the specified resolution.

    Args:
        dataframe: geodataframe with columns geometry and `column`. The 'geometry' column is expected to have shapely geometries
        bounds: bounds where the polygons will be rasterised.
        resolution: spatial resolution of the rasterised array
        column: column to take the values for rasterisation.
        crs_out: if provided it will transform the geometries from dataframe.crs to this crs before rasterisation.
            It expects that `bounds` are in this crs.
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

    if isinstance(resolution, numbers.Number):
        resolution = (abs(resolution), abs(resolution))

    transform = rasterio.transform.from_origin(min(bounds[0], bounds[2]),
                                               max(bounds[1], bounds[3]),
                                               resolution[0], resolution[1])

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