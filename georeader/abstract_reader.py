import numpy as np
from georeader.geotensor import GeoTensor
from georeader import window_utils
from typing import Tuple, Any, Union, Optional
from shapely.geometry import Polygon
import rasterio
import rasterio.windows

class AbstractGeoData:
    def __init__(self):
        self.dtype = np.float32
        self.dims = ("y", "x")
        # TODO replace fill_value_default with nodata
        self.fill_value_default = 0

    @property
    def shape(self) -> Tuple:
        # return 1000, 3000
        raise NotImplementedError("Not implemented")

    @property
    def transform(self) -> rasterio.Affine:
        # return rasterio.Affine(10, 0, 200, 0, 10, 400)
        raise NotImplementedError("Not implemented")

    @property
    def res(self) -> Tuple[float, float]:
        return window_utils.res(self.transform)

    @property
    def crs(self) -> Any:
        # return "EPSG:4326"
        raise NotImplementedError("Not implemented")

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return window_utils.window_bounds(rasterio.windows.Window(row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]),
                                          self.transform)

    def load(self, boundless:bool=True)-> GeoTensor:
        # return GeoTensor(values=self.values, transform=self.transform, crs=self.crs)
        raise NotImplementedError("Not implemented")

    def read_from_window(self, window:rasterio.windows.Window, boundless:bool) -> Union['__class__', GeoTensor]:
        # return GeoTensor(values=self.values, transform=self.transform, crs=self.crs)
        return self.load(boundless=True).read_from_window(window=window, boundless=boundless)

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        pol = window_utils.window_polygon(rasterio.windows.Window(row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]),
                                          self.transform)
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)

    @property
    def values(self) -> np.ndarray:
        # return np.zeros(self.shape, dtype=self.dtype)
        raise self.load(boundless=True).values

GeoData = Union[GeoTensor, AbstractGeoData]

def same_extent(geo1:GeoData, geo2:GeoData, precision:float=1e-3) -> bool:
    return geo1.transform.almost_equals(geo2.transform, precision=precision) and window_utils.compare_crs(geo1.crs, geo2.crs) and (geo1.shape[-2:] == geo2.shape[-2:])