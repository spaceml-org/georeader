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

# Create a binary mask GeoTensor
mask_data = np.zeros((100, 100), dtype=np.uint8)
mask_data[20:80, 20:80] = 1  # A square region
gt_mask = GeoTensor(mask_data, transform, crs="EPSG:4326")

# Convert to polygons (returns list of shapely Polygons in raster CRS)
polygons = vectorize.get_polygons(gt_mask, min_area=100)

# Transform polygon to different CRS
polygon_wgs84 = vectorize.transform_polygon(polygons[0], 
                                             crs_polygon="EPSG:32610",
                                             dst_crs="EPSG:4326")
```

## Key Functions

| Function | Description |
|----------|-------------|
| `get_polygons` | Extract polygons from binary mask with optional area filtering |
| `transform_polygon` | Reproject polygon between coordinate reference systems |

## Parameters

### `get_polygons`

- `binary_mask`: Input mask (GeoTensor or numpy array)
- `min_area`: Minimum polygon area in pixel units (mask coordinate units), applied before any affine transform (default: 25.5)
- Returns: List of shapely Polygon objects

### `transform_polygon`

- `polygon`: Input shapely Polygon or MultiPolygon
- `crs_polygon`: CRS of the input polygon
- `dst_crs`: Target CRS for the output polygon

---

::: georeader.vectorize