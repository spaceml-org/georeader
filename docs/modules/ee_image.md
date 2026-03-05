# Google Earth Engine Integration

This module provides functions to query and export arbitrarily large images from Google Earth Engine (GEE). It handles the complexity of tiling large exports and provides convenient query functions for common satellite datasets.

## Overview

The GEE integration includes:

- **Export functions** (`georeader.readers.ee_image`): Export single images or time-series cubes
- **Query functions** (`georeader.readers.ee_query`): Search for Sentinel-1, Sentinel-2, and Landsat imagery

## Prerequisites

```python
import ee
ee.Authenticate()
ee.Initialize()
```

## Quick Start

```python
from georeader.readers import ee_image, ee_query
from shapely.geometry import box
from datetime import datetime
import ee

# Define area of interest
aoi = box(-122.5, 37.5, -122.0, 38.0)

# Query available Sentinel-2 images
images = ee_query.query(aoi, datetime(2023, 1, 1), datetime(2023, 12, 31), 
                        producttype="S2_SR")

# Export a single image
gt = ee_image.export_image(images[0], aoi, scale=10)

# Export a time-series cube
cube = ee_image.export_cube(images[:5], aoi, scale=10)
```

## Key Functions

### ee_image module

| Function | Description |
|----------|-------------|
| `export_image` | Export a single GEE image to GeoTensor |
| `export_cube` | Export multiple images as a 4D GeoTensor (time, bands, y, x) |

### ee_query module

| Function | Description |
|----------|-------------|
| `query` | Query any GEE image collection |
| `query_s1` | Query Sentinel-1 SAR imagery |
| `query_landsat_457` | Query Landsat 4, 5, 7 imagery |

---

::: georeader.readers.ee_image
    options:
      members:
        - export_image
        - export_cube

::: georeader.readers.ee_query
    options:
      members:
        - query
        - query_s1
        - query_landsat_457
