import xarray as xr
import rasterio
import rasterio.windows
import rasterio.transform
from georeader.geotensor import GeoTensor
from typing import Tuple, Any, Optional, Dict, Union
from numpy.typing import NDArray
import numpy as np
from collections import OrderedDict


def coords_to_transform(coords: xr.Coordinates, x_axis_name:str="x", y_axis_name:str="y") -> rasterio.Affine:
    """
    Compute the bounds and the geotransform from the coordinates of a xr.DataArray object.

    This function is an inverse of the coordinates computation that is in `xr.open_rasterio`

    Args:
        coords: if `data` is an xr.DataArray object data.coords has its coordinates.
        x_axis_name: name of the x axis. Defaults to "x".
        y_axis_name: name of the y axis. Defaults to "y".

    Returns:
        Bounds and geotransform of the coordinates.
    """

    resx = float(coords[x_axis_name][1] - coords[x_axis_name][0])
    resy = float(coords[y_axis_name][1] - coords[y_axis_name][0])
    resx2 = resx / 2.
    resy2 = resy / 2.

    x_min = float(coords[x_axis_name][0] - resx2)
    # x_max = float(coords[x_axis_name][-1] + resx2)  # why not + resx2
    y_min = float(coords[y_axis_name][0] - resy2)
    # y_max = float(coords[y_axis_name][-1] + resy2)  # why not - resy2

    # We add in the y coordinate because images are referenced from top coordinate,
    # see xr.open_rasterio or getcoords_from_transform_shape
    # bounds = (x_min, y_min, x_max, y_max)

    # Compute affine transform for a given bounding box and resolution.
    # transform = rasterio.transform.from_origin(bounds[0], bounds[3], resx, resy)

    return rasterio.transform.Affine.translation(x_min, y_min) * rasterio.transform.Affine.scale(resx, resy)


def getcoords_from_transform_shape(transform:rasterio.Affine, 
                                   shape:Tuple[int, int],
                                   x_axis_name:str="x", y_axis_name:str="y") -> Dict[str, NDArray]:
    """
     This function creates the coordinates for an xr.DataArray object from a transform and a shape tuple.
     This code is taken from xr.open_rasterio.

    Args:
        transform: Affine transform of the raster.
        shape: Shape of the raster.
        x_axis_name: name to the x axis. Defaults to "x".
        y_axis_name: name to the y axis. Defaults to "y".

    Returns:
        Dict with the coordinates for a xr.DataArray object
    """
    assert transform.is_rectilinear, "Expected rectilinear transform (i.e. transform.b and transform.d equal to zero)"

    nx, ny = shape[1], shape[0]
    x, _ = transform * (np.arange(nx) + 0.5, np.zeros(nx) + 0.5)
    _, y = transform * (np.zeros(ny) + 0.5, np.arange(ny) + 0.5)

    return {x_axis_name: x, y_axis_name: y}


def toDataArray(x:GeoTensor, x_axis_name:str="x", y_axis_name:str="y", extra_coords:Optional[Dict[str, Any]]=None) -> xr.DataArray:
    """
    Convert a GeoTensor to a xr.DataArray object.

    Args:
        x (GeoTensor): Input GeoTensor
        x_axis_name (str, optional): name to the x axis. Defaults to "x".
        y_axis_name (str, optional): name to the y axis. Defaults to "y".
        extra_coords (Optional[Dict[str, Any]], optional): Extra coordinates. Defaults to None.

    Returns:
        xr.DataArray: Output xr.DataArray
    """
    coords = getcoords_from_transform_shape(x.transform, x.shape[-2:], 
                                            x_axis_name=x_axis_name, y_axis_name=y_axis_name)
    cords_ordered = OrderedDict()
    for d in x.dims:
        if (extra_coords is not None) and (d in extra_coords):
            cords_ordered[d] = extra_coords[d]
        elif d in coords:
            cords_ordered[d] = coords[d]
        else:
            cords_ordered[d] = np.arange(x.shape[x.dims.index(d)])
    
    return xr.DataArray(x.values, coords=cords_ordered, 
                        dims=x.dims,
                        attrs={"crs":x.crs , 
                               "fill_value_default":x.fill_value_default})


def fromDataArray(x: xr.DataArray, crs:Optional[Any]=None, 
                  fill_value_default:Optional[Union[float, int]]=None,
                  x_axis_name:Optional[str]=None, y_axis_name:Optional[str]=None) -> GeoTensor:
    """
    Convert a xr.DataArray to a GeoTensor object.

    Args:
        x (xr.DataArray): Input xr.DataArray
        crs (Optional[Any], optional): crs. Defaults to None.
        fill_value_default (Optional[Union[float, int]], optional): fill value. Defaults to None.

    Returns:
        GeoTensor: Output GeoTensor
    """
    if crs is None:
        crs = x.attrs.get("crs", None)
    
    if fill_value_default is None:
        fill_value_default = x.attrs.get("fill_value_default", None)
    
    coords_names = list(x.coords.dims)
    if x_axis_name is None:
        if "x" in coords_names:
            x_axis_name = "x"
        elif "lon" in coords_names:
            x_axis_name = "lon"
        else:
            x_axis_name = coords_names[-2]
    if y_axis_name is None:
        if "y" in coords_names:
            y_axis_name = "y"
        elif "lat" in coords_names:
            y_axis_name = "lat"
        else:
            y_axis_name = coords_names[-1]
    
    return GeoTensor(x.values, transform=coords_to_transform(x.coords, x_axis_name=x_axis_name, y_axis_name=y_axis_name), 
                     crs=crs, fill_value_default=fill_value_default)
