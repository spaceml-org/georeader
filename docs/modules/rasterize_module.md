# georeader.rasterize

This module provides functions to convert vector geometries (polygons, lines) into raster format, aligned to an existing GeoTensor or with custom georeferencing.

## Overview

Rasterization is essential for:

- Converting shapefile/GeoJSON boundaries to masks
- Creating training labels for machine learning from vector annotations
- Burning attribute values from GeoDataFrames into raster grids

## Quick Start

```python
from georeader import rasterize
from georeader.geotensor import GeoTensor
from shapely.geometry import Polygon
import geopandas as gpd

# Create a reference GeoTensor
gt = GeoTensor(np.zeros((100, 100)), transform, crs="EPSG:4326")

# Rasterize a single polygon to match a GeoTensor
polygon = Polygon([(-122.5, 37.5), (-122.0, 37.5), (-122.0, 38.0), (-122.5, 38.0)])
mask = rasterize.rasterize_geometry_like(polygon, gt)

# Rasterize a GeoDataFrame with attribute values
gdf = gpd.read_file("boundaries.geojson")
raster = rasterize.rasterize_geopandas_like(gdf, gt, column="class_id")
```

## Key Functions

| Function | Description |
|----------|-------------|
| `rasterize_geometry_like` | Rasterize a single geometry to match a GeoTensor |
| `rasterize_from_geometry` | Rasterize with custom transform and shape |
| `rasterize_geopandas_like` | Rasterize GeoDataFrame column to match a GeoTensor |
| `rasterize_from_geopandas` | Rasterize GeoDataFrame with custom georeferencing |

---

::: georeader.rasterize