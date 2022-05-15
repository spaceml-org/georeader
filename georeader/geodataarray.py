import xarray as xr
import rasterio
import rasterio.windows
from georeader.geotensor import GeoTensor
from typing import Tuple, Any, Optional, Union
import numpy as np
import warnings


def coords_to_transform(coords: xr.Coordinate) -> rasterio.Affine:
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


class GeoDataArray(xr.DataArray):
    def __init__(self, data:xr.DataArray, fill_value_default:int=0, crs:Optional[Any]=None):
        # TODO Assert x, y in dims
        self.fill_value_default = fill_value_default

        self.transform = coords_to_transform(data.coords)
        if crs is None:
            if "crs" in data.attrs:
                self.crs = data.attrs
            else:
                warnings.warn("CRS not provided")
                self.crs = None
        else:
            self.crs = crs

    @property
    def res(self) -> Tuple[float, float]:
        transform = self.transform
        #  compute resolution for non-rectilinear transforms!
        z0_0 = np.array(transform * (0, 0))
        z0_1 = np.array(transform * (0, 1))
        z1_0 = np.array(transform * (1, 0))

        return np.sqrt(np.sum((z0_0 - z1_0) ** 2)), np.sqrt(np.sum((z0_0 - z0_1) ** 2))

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return rasterio.windows.bounds(
            rasterio.windows.Window(col_off=0, row_off=0, width=self.shape[-1], height=self.shape[-2]),
            self.transform)

    def read_from_window(self, window: rasterio.windows.Window, boundless: bool) -> Union['__class__', GeoTensor]:
        # return GeoTensor(values=self.values, transform=self.transform, crs=self.crs)
        # TODO adjust coords
        raise NotImplementedError("Not implemented")