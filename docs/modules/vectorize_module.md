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

# Example 1: Get polygons from GeoTensor (automatically in the GeoTensor's CRS)
mask_data = np.zeros((100, 100), dtype=np.uint8)
mask_data[20:80, 20:80] = 1  # A square region
transform = rasterio.Affine(10.0, 0, 500000, 0, -10.0, 4500000)
gt_mask = GeoTensor(mask_data, transform, crs="EPSG:32610")

# Polygons are automatically in the GeoTensor's CRS (EPSG:32610)
polygons = vectorize.get_polygons(gt_mask, min_area=100)

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

## Parameters

### `get_polygons`

- `binary_mask`: Input mask (GeoTensor or numpy array)
- `min_area`: Minimum polygon area in pixel units (default: 25.5), applied before affine transform
- Returns: List of shapely Polygon objects (in CRS coordinates if transform provided, else pixel coordinates)

---

::: georeader.vectorize