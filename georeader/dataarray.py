import xarray as xr
import rasterio
import rasterio.windows
from georeader.geotensor import GeoTensor
from typing import Tuple, Any, Optional, Dict
from numpy.typing import NDArray
import numpy as np
from collections import OrderedDict


def coords_to_transform(coords: xr.Coordinates) -> rasterio.Affine:
    """
    Compute the bounds and the geotransform from the coordinates of a xr.DataArray object.

    This function is an inverse of the coordinates computation that is in `xr.open_rasterio`

    Args:
        coords: if `data` is an xr.DataArray object data.coords has its coordinates

    Returns:
        Bounds and geotransform of the coordinates.
    """

    resx = float(abs(coords["x"][1] - coords["x"][0]))
    resy = float(abs(coords["y"][1] - coords["y"][0]))
    resx2 = resx / 2.
    resy2 = resy / 2.

    x_min = float(coords["x"][0] - resx2)
    x_max = float(coords["x"][-1] - resx2 + resx)  # why not + resx2
    y_max = float(coords["y"][0] + resy2)
    y_min = float(coords["y"][-1] + resy2 - resy)  # why not - resy2

    # We add in the y coordinate because images are referenced from top coordinate,
    # see xr.open_rasterio or getcoords_from_transform_shape
    bounds = (x_min, y_min, x_max, y_max)

    # Compute affine transform for a given bounding box and resolution.
    transform = rasterio.transform.from_origin(bounds[0], bounds[3], resx, resy)

    return transform


def getcoords_from_transform_shape(transform:rasterio.Affine, 
                                   shape:Tuple[int, int]) -> Dict[str, NDArray]:
    """
     This function creates the coordinates for an xr.DataArray object from a transform and a shape tuple.
     This code is taken from xr.open_rasterio.

    Args:
        transform:
        shape:

    Returns:
        Dict with the coordinates for a xr.DataArray object
    """
    assert transform.is_rectilinear, "Expected rectilinear transform (i.e. transform.b and transform.d equal to zero)"

    nx, ny = shape[1], shape[0]
    x, _ = transform * (np.arange(nx) + 0.5, np.zeros(nx) + 0.5)
    _, y = transform * (np.zeros(ny) + 0.5, np.arange(ny) + 0.5)

    return {"x": x, "y": y}


def toDataArray(x:GeoTensor) -> xr.DataArray:
    """
    Convert a GeoTensor to a xr.DataArray object.

    Args:
        x (GeoTensor): Input GeoTensor

    Returns:
        xr.DataArray: Output xr.DataArray
    """
    coords = getcoords_from_transform_shape(x.transform, x.shape[-2:])
    cords_ordered = OrderedDict()
    for d in x.dims:
        if d not in coords:
            cords_ordered[d] = np.arange(x.shape[x.dims.index(d)])
        else:
            cords_ordered[d] = coords[d]
    
    return xr.DataArray(x.values, coords=coords, 
                        attrs={"crs":x.crs})


def fromDataArray(x: xr.DataArray, crs:Optional[Any]=None) -> GeoTensor:
    """
    Convert a xr.DataArray to a GeoTensor object.

    Args:
        x (xr.DataArray): Input xr.DataArray
        crs (Optional[Any], optional): crs. Defaults to None.

    Returns:
        GeoTensor: Output GeoTensor
    """
    if crs is None:
        crs = x.attrs.get("crs", None)
    
    return GeoTensor(x.values, transform=coords_to_transform(x.coords), 
                     crs=crs)
