# georeader.rasterio_reader

The `RasterioReader` class is the primary interface for reading raster data from files. It wraps rasterio's dataset reader with additional functionality for windowed reading, on-the-fly reprojection, and lazy loading.

## Overview

`RasterioReader` provides:

- **Lazy loading**: Data is only read when accessed
- **Windowed reading**: Read specific regions without loading the entire file
- **Cloud support**: Read from S3, GCS, Azure Blob, and HTTP URLs
- **Reprojection**: On-the-fly coordinate system transformation
- **GeoTensor integration**: Returns GeoTensor objects with full geospatial metadata

## Quick Start

```python
from georeader.rasterio_reader import RasterioReader

# Open a local file
reader = RasterioReader("path/to/raster.tif")

# Open from cloud storage
reader = RasterioReader("s3://bucket/path/to/raster.tif")

# Read the entire raster as GeoTensor
gt = reader.load()

# Read a specific window (row_off, col_off, height, width)
window = rasterio.windows.Window(0, 0, 512, 512)
gt_window = reader.read_from_window(window)
```

## Key Properties

| Property | Description |
|----------|-------------|
| `shape` | Shape of the raster (bands, height, width) |
| `transform` | Affine transform for georeferencing |
| `crs` | Coordinate reference system |
| `bounds` | Geographic bounds (minx, miny, maxx, maxy) |
| `res` | Pixel resolution (x_res, y_res) |
| `dtype` | Data type of the raster |

## Reading Methods

| Method | Description |
|--------|-------------|
| `load()` | Load entire raster as GeoTensor |
| `read_from_window()` | Read a specific window |
| `isel()` | Select bands by index |

---

::: georeader.rasterio_reader
    options:
      members:
        - RasterioReader