# Abstract Reader Module

The `georeader.abstract_reader` module defines the core protocols and base classes
that all georeader data sources must implement. It establishes a common interface
for georeferenced raster data, enabling consistent behavior across different
data sources (GeoTIFF files, cloud storage, in-memory arrays, Earth Engine).

## Overview

The module provides a type hierarchy for georeferenced data:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     GEODATA TYPE HIERARCHY                                   │
│                                                                              │
│   GeoDataBase (Protocol)                                                    │
│   ├── Required: transform, crs, shape                                       │
│   ├── Derived: width, height                                                │
│   │                                                                          │
│   └──► GeoData (Protocol, extends GeoDataBase)                              │
│        ├── Required: load(), read_from_window()                             │
│        ├── Required: dtype, dims, fill_value_default                        │
│        ├── Derived: values, res, bounds, footprint()                        │
│        │                                                                     │
│        └──► Implementations:                                                │
│             ├── GeoTensor      (in-memory)                                  │
│             ├── RasterioReader (file-based, lazy)                           │
│             ├── S2Image        (Sentinel-2 SAFE)                            │
│             ├── EMITImage      (EMIT hyperspectral)                         │
│             └── ... other readers                                           │
│                                                                              │
│   FakeGeoData (dataclass)                                                   │
│   └── Minimal implementation for coordinate calculations                    │
│       (no actual data, just CRS + transform)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## GeoData Protocol

The `GeoData` protocol defines what any georeferenced data source must provide:

### Required Properties

| Property | Type | Description |
|----------|------|-------------|
| `transform` | `rasterio.Affine` | Pixel-to-coordinate transformation matrix |
| `crs` | `Any` | Coordinate reference system |
| `shape` | `Tuple[int, ...]` | Array shape `(C, H, W)` or `(H, W)` |
| `dtype` | `Any` | Data type (e.g., `np.float32`) |
| `dims` | `List[str]` | Dimension names `["band", "y", "x"]` |
| `fill_value_default` | `Any` | NoData value |

### Required Methods

| Method | Description |
|--------|-------------|
| `load(boundless=True)` | Load all data as GeoTensor |
| `read_from_window(window, boundless)` | Read subset by pixel window |

### Derived Properties

These are computed from the required properties:

| Property | Description |
|----------|-------------|
| `width` | `shape[-1]` - number of columns |
| `height` | `shape[-2]` - number of rows |
| `res` | `(abs(transform.a), abs(transform.e))` - pixel resolution |
| `bounds` | `(minx, miny, maxx, maxy)` in CRS units |
| `values` | Calls `load().values` - the actual array data |

### Derived Methods

| Method | Description |
|--------|-------------|
| `footprint(crs=None)` | Bounding polygon, optionally in different CRS |

## The Affine Transform

The `transform` property is a 2D affine transformation matrix that maps pixel
coordinates to geographic coordinates:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AFFINE TRANSFORM EXPLAINED                              │
│                                                                              │
│   Transform: Affine(a, b, c, d, e, f)                                       │
│                                                                              │
│   ┌     ┐   ┌         ┐   ┌     ┐                                          │
│   │  x  │   │  a  b c │   │ col │                                          │
│   │  y  │ = │  d  e f │ × │ row │                                          │
│   │  1  │   │  0  0 1 │   │  1  │                                          │
│   └     ┘   └         ┘   └     ┘                                          │
│                                                                              │
│   For typical north-up imagery:                                             │
│   ┌────────────────────────────────────────────────────────────────────┐   │
│   │  a = pixel width (positive, e.g., 10.0 for 10m pixels)             │   │
│   │  b = 0 (no rotation)                                               │   │
│   │  c = x-coordinate of upper-left corner                             │   │
│   │  d = 0 (no rotation)                                               │   │
│   │  e = -pixel height (NEGATIVE for north-up!)                        │   │
│   │  f = y-coordinate of upper-left corner                             │   │
│   └────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Example (10m UTM):                                                        │
│   transform = Affine(10, 0, 550000, 0, -10, 4200000)                       │
│                                                                              │
│   Pixel (0, 0) → (550000, 4200000)  # upper-left                           │
│   Pixel (1, 0) → (550010, 4200000)  # one pixel right                      │
│   Pixel (0, 1) → (550000, 4199990)  # one pixel down                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Using the Protocol

### Type Hints

Use `GeoData` for functions that work with any georeferenced data:

```python
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor

def compute_ndvi(data: GeoData) -> GeoTensor:
    """Works with any GeoData implementation."""
    arr = data.load()  # Load as GeoTensor
    red = arr.values[3]   # Assuming band 4 is red
    nir = arr.values[7]   # Assuming band 8 is NIR
    ndvi = (nir - red) / (nir + red + 1e-10)
    return GeoTensor(ndvi, transform=data.transform, crs=data.crs)
```

### FakeGeoData for Coordinate Operations

When you only need coordinate transformations (no actual data):

```python
from georeader.abstract_reader import FakeGeoData
from georeader import read
from rasterio import Affine

# Create minimal GeoData for coordinate calculations
fake = FakeGeoData(
    crs="EPSG:32610",
    transform=Affine(10, 0, 550000, 0, -10, 4200000)
)

# Use in window calculations
from shapely.geometry import box
aoi = box(-122.5, 37.7, -122.3, 37.9)
window = read.window_from_polygon(fake, aoi, crs_polygon="EPSG:4326")
```

### Implementing a Custom Reader

To create a new data source, implement the `GeoData` protocol:

```python
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
import rasterio
import numpy as np

class MyCustomReader(GeoData):
    def __init__(self, path: str):
        self._path = path
        # Read metadata only, don't load data yet
        with rasterio.open(path) as src:
            self._transform = src.transform
            self._crs = src.crs
            self._shape = (src.count, src.height, src.width)
            self._dtype = src.dtypes[0]
            self._nodata = src.nodata
    
    @property
    def transform(self):
        return self._transform
    
    @property
    def crs(self):
        return self._crs
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def dims(self):
        return ["band", "y", "x"]
    
    @property
    def fill_value_default(self):
        return self._nodata or 0
    
    def load(self, boundless=True):
        with rasterio.open(self._path) as src:
            data = src.read()
        return GeoTensor(data, self.transform, self.crs, self.fill_value_default)
    
    def read_from_window(self, window, boundless=True):
        with rasterio.open(self._path) as src:
            data = src.read(window=window, boundless=boundless)
        new_transform = rasterio.windows.transform(window, self.transform)
        return GeoTensor(data, new_transform, self.crs, self.fill_value_default)
```

## Utility Functions

### `same_extent`

Check if two GeoData objects have the same spatial extent:

```python
from georeader.abstract_reader import same_extent

if same_extent(geo1, geo2):
    # Can do element-wise operations
    result = geo1.values + geo2.values
else:
    # Need to reproject first
    geo2_aligned = read.read_reproject_like(geo2, geo1)
```

## API Reference

::: georeader.abstract_reader
    options:
      show_root_heading: true
      show_source: true
      members:
        - GeoDataBase
        - GeoData
        - FakeGeoData
        - same_extent

## See Also

- [GeoTensor](geotensor_module.md) - In-memory GeoData implementation
- [RasterioReader](rasterio_reader.md) - File-based lazy reader
- [Read Module](read_module.md) - Functions that operate on GeoData
