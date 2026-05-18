"""
Abstract Reader: Base protocols and interfaces for geospatial data readers.

This module defines the abstract interfaces (Protocols) that all georeader
data sources must implement. These interfaces enable polymorphic processing
of raster data from diverse sources (files, cloud storage, web services).

Type Hierarchy
--------------

The module defines a hierarchy of data types::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    GEOREADER TYPE HIERARCHY                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  GeoDataBase (Protocol)           Minimal interface for geospatial data │
    │  ├── transform: Affine            Pixel → coordinate mapping            │
    │  ├── crs: Any                     Coordinate reference system           │
    │  └── shape: Tuple                 (C, H, W) or (H, W) dimensions        │
    │       │                                                                  │
    │       ▼                                                                  │
    │  AbstractGeoData (Protocol)       Adds read capabilities                │
    │  ├── values: ndarray              Array data                            │
    │  ├── fill_value_default           Nodata value                          │
    │  └── load(): GeoTensor            Read all data                         │
    │       │                                                                  │
    │       ├──────────────────────┬──────────────────────┐                   │
    │       ▼                      ▼                      ▼                   │
    │  RasterioReader         GeoTensor              Custom Readers           │
    │  (Lazy file access)     (In-memory)           (User-defined)           │
    │                                                                          │
    │  GeoData = Union[AbstractGeoData, GeoTensor]  ← Common type alias       │
    └─────────────────────────────────────────────────────────────────────────┘

Protocol Requirements
---------------------

To implement a custom reader, fulfill these interfaces::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    REQUIRED PROPERTIES                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Property              Type                  Description                 │
    │  ──────────           ──────                ───────────                 │
    │  transform            rasterio.Affine       6-element affine matrix     │
    │  crs                  Any (CRS-like)        EPSG code, WKT, or CRS obj  │
    │  shape                Tuple[int, ...]       (C, H, W) or (H, W)         │
    │  values               ndarray               Raster data array           │
    │  fill_value_default   number                Nodata/fill value           │
    │                                                                          │
    │  Required Methods:                                                       │
    │  ─────────────────                                                       │
    │  load() → GeoTensor   Read all data into memory                         │
    │                                                                          │
    │  Derived Properties (computed from above):                              │
    │  ──────────────────────────────────────────                             │
    │  width                shape[-1]             Number of columns           │
    │  height               shape[-2]             Number of rows              │
    │  bounds               From transform+shape  (minx, miny, maxx, maxy)    │
    │  res                  From transform        (xres, yres) pixel size     │
    │  footprint            Polygon               Bounding polygon in CRS     │
    └─────────────────────────────────────────────────────────────────────────┘

FakeGeoData Utility
-------------------

Lightweight placeholder for georeferencing without actual data::

    # Create placeholder for transform/CRS operations
    fake = FakeGeoData(
        crs="EPSG:4326",
        transform=Affine.translation(-122.5, 37.5) * Affine.scale(0.001, -0.001),
        shape=(3, 1000, 1000)
    )

    # Use for window calculations, bounds checking, etc.
    window = window_from_bounds(fake, bounds, crs_bounds)

Module Contents
---------------

Protocols:
    - :class:`GeoDataBase`: Minimal geospatial interface (transform, crs, shape)
    - :class:`AbstractGeoData`: Full reader interface (adds values, load)

Type Aliases:
    - :data:`GeoData`: Union[AbstractGeoData, GeoTensor]
    - :data:`GeoDataBase`: Protocol for any geospatial object

Utilities:
    - :class:`FakeGeoData`: Lightweight georeferencing placeholder
    - :func:`res`: Get pixel resolution from GeoData
    - :func:`bounds`: Get geographic bounds from GeoData
    - :func:`footprint`: Get bounding polygon from GeoData

Quick Start
-----------

Check if an object implements the GeoData interface::

    from georeader.abstract_reader import GeoDataBase

    def process_geodata(data: GeoDataBase):
        '''Works with any GeoData-compatible object'''
        print(f"CRS: {data.crs}")
        print(f"Shape: {data.shape}")
        print(f"Bounds: {window_utils.window_bounds(data)}")

Create a fake geodata for testing::

    from georeader.abstract_reader import FakeGeoData
    from rasterio.transform import from_bounds

    fake = FakeGeoData(
        crs="EPSG:4326",
        transform=from_bounds(-122.5, 37.0, -122.0, 37.5, 500, 500),
        shape=(3, 500, 500)
    )

See Also
--------
georeader.geotensor : Concrete in-memory implementation
georeader.rasterio_reader : Lazy file-backed implementation
typing.Protocol : Python protocol documentation

References
----------
- Python Protocols: https://peps.python.org/pep-0544/
- Rasterio DatasetReader: https://rasterio.readthedocs.io/en/latest/api/rasterio.html
"""
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

    @property
    def res(self) -> Tuple[float, float]:
        return window_utils.res(self.transform)

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
        return self.load(boundless=True).values

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


AbstractGeoData = GeoData


class AsyncGeoData(GeoDataBase):
    """Async mirror of :class:`GeoData`.

    Concrete async readers (e.g. ``AsyncGeoTIFFReader``) satisfy this
    interface. User code typed against ``AsyncGeoData`` accepts any
    conforming async reader without isinstance checks.

    Inherits the metadata surface and derived properties (``transform``,
    ``crs``, ``shape``, ``width``, ``height``, ``bounds``, ``res``,
    ``footprint``) from :class:`GeoDataBase`. Adds an ``async`` ``load``
    method, a **sync** ``read_from_window`` that returns a windowed view
    (mirroring :class:`~georeader.rasterio_reader.RasterioReader`), and
    the read-tier metadata properties (``dtype``, ``dims``,
    ``fill_value_default``).

    Notes
    -----
    There is no ``values`` property here (unlike :class:`GeoData`, where it
    materialises via a sync ``self.load()``). Properties cannot be ``async``,
    so callers materialise via ``await reader.load()`` and read
    ``.values`` on the returned :class:`~georeader.geotensor.GeoTensor`.

    ``read_from_window`` is **sync** by design: like
    :meth:`RasterioReader.read_from_window`, it only constructs a windowed
    view of the reader and performs no I/O. This means
    :func:`georeader.read.read_from_window` (and other ``read.*``
    functions) work polymorphically with both sync and async readers —
    the only difference is that the returned async view must be
    materialised via ``await view.load()``.
    """

    async def load(self, boundless: bool = True) -> GeoTensor:
        raise NotImplementedError(
            "load method must be implemented in the subclass"
        )

    def read_from_window(
        self, window: rasterio.windows.Window, boundless: bool = True
    ) -> Union["AsyncGeoData", GeoTensor]:
        raise NotImplementedError(
            "read_from_window method must be implemented in the subclass"
        )

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


def same_extent(geo1: GeoData, geo2: GeoData, precision: float = 1e-3) -> bool:
    return (
        geo1.transform.almost_equals(geo2.transform, precision=precision)
        and window_utils.compare_crs(geo1.crs, geo2.crs)
        and (geo1.shape[-2:] == geo2.shape[-2:])
    )
