# georeader.vectorize

This module provides functions to convert raster data (binary masks, segmentation outputs) into vector geometries (polygons).

## Overview

Vectorization is essential for:

- Converting segmentation masks to GeoJSON/Shapefile
- Extracting object boundaries from classification results
- Creating vector features from raster analysis

## Quick Start

```python
from georeader import vectorize
from georeader.geotensor import GeoTensor
import numpy as np
import rasterio

# Example 1: Get polygons from GeoTensor (automatically in geographic coordinates)
mask_data = np.zeros((100, 100), dtype=np.uint8)
mask_data[20:80, 20:80] = 1  # A square region
transform = rasterio.Affine(10.0, 0, 500000, 0, -10.0, 4500000)
gt_mask = GeoTensor(mask_data, transform, crs="EPSG:32610")

# Polygons are automatically in geographic coordinates (EPSG:32610)
polygons = vectorize.get_polygons(gt_mask, min_area=100)

# Example 2: Get polygons from numpy array (in pixel coordinates), then transform
mask_array = np.zeros((100, 100), dtype=np.uint8)
mask_array[20:80, 20:80] = 1
polygons_px = vectorize.get_polygons(mask_array, min_area=100)  # Pixel coordinates
polygon_geo = vectorize.transform_polygon(polygons_px[0], transform)  # Apply affine transform

# For CRS reprojection, use window_utils.polygon_to_crs
from georeader import window_utils
polygon_wgs84 = window_utils.polygon_to_crs(polygons[0], 
                                             crs_polygon="EPSG:32610",
                                             dst_crs="EPSG:4326")
```

## Key Functions

| Function | Description |
|----------|-------------|
| `get_polygons` | Extract polygons from binary mask with optional area filtering |
| `transform_polygon` | Apply affine transformation to polygon coordinates (e.g., pixel to geographic) |

## Parameters

### `get_polygons`

- `binary_mask`: Input mask (GeoTensor or numpy array)
- `min_area`: Minimum polygon area in pixel units (default: 25.5), applied before affine transform
- Returns: List of shapely Polygon objects (in geographic coordinates if transform provided, else pixel coordinates)

### `transform_polygon`

- `polygon`: Input shapely Polygon or MultiPolygon
- `transform`: Rasterio Affine transformation matrix
- `relative`: If True, output normalized coordinates in [0, 1] range (default: False)
- `shape_raster`: Raster dimensions (height, width), required if relative=True

**Note:** For CRS reprojection, use `georeader.window_utils.polygon_to_crs` instead.

---

::: georeader.vectorize