import numpy as np
from georeader import window_utils
from georeader.geotensor import GeoTensor
from typing import Tuple, Any, Union, Optional
from shapely.geometry import Polygon
import rasterio
import rasterio.windows
from collections import namedtuple
from dataclasses import dataclass

# Import Protocol
from typing import Protocol


class GeoDataBase(Protocol):
    @property
    def transform(self) -> rasterio.Affine:
        pass

    @property
    def crs(self) -> Any:
        pass

    @property
    def shape(self) -> Tuple:
        pass

    @property
    def width(self) -> int:
        return self.shape[-1]

    @property
    def height(self) -> int:
        return self.shape[-2]


@dataclass
class FakeGeoData:
    crs: Any
    transform: rasterio.Affine
    shape: Optional[Tuple[int, ...]] = None

    @property
    def width(self) -> int:
        if self.shape is None:
            raise ValueError("Shape is not defined")
        return self.shape[-1]

    @property
    def height(self) -> int:
        if self.shape is None:
            raise ValueError("Shape is not defined")
        return self.shape[-2]


class GeoData(GeoDataBase):
    def load(self, boundless: bool = True) -> GeoTensor:
        raise NotImplementedError(
            "load method must be implemented in the subclass"
        )

    def read_from_window(
        self, window: rasterio.windows.Window, boundless: bool
    ) -> Union["__class__", GeoTensor]:
        raise NotImplementedError(
            "read_from_window method must be implemented in the subclass"
        )
    
    @property
    def values(self) -> np.ndarray:
        # return np.zeros(self.shape, dtype=self.dtype)
        raise self.load(boundless=True).values
    
    @property
    def res(self) -> Tuple[float, float]:
        return window_utils.res(self.transform)
    
    @property
    def dtype(self) -> Any:
        raise NotImplementedError(
            "dtype property must be implemented in the subclass"
        )
    
    @property
    def dims(self) -> list[str]:
        raise NotImplementedError(
            "dims property must be implemented in the subclass"
        )
    
    @property
    def fill_value_default(self) -> Any:
        raise NotImplementedError(
            "fill_value_default property must be implemented in the subclass"
        )

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return window_utils.window_bounds(
            rasterio.windows.Window(
                row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]
            ),
            self.transform,
        )
    
    def footprint(self, crs: Optional[str] = None) -> Polygon:
        pol = window_utils.window_polygon(
            rasterio.windows.Window(
                row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]
            ),
            self.transform,
        )
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)

AbstractGeoData = GeoData


def same_extent(geo1: GeoData, geo2: GeoData, precision: float = 1e-3) -> bool:
    return (
        geo1.transform.almost_equals(geo2.transform, precision=precision)
        and window_utils.compare_crs(geo1.crs, geo2.crs)
        and (geo1.shape[-2:] == geo2.shape[-2:])
    )
