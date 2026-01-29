# georeader.save

This module provides functions to save GeoTensor and other geospatial data to GeoTIFF files, including Cloud Optimized GeoTIFF (COG) format.

## Overview

The save module supports:

- **Tiled GeoTIFF**: Save large rasters with internal tiling for efficient access
- **Cloud Optimized GeoTIFF (COG)**: Industry-standard format optimized for cloud storage and HTTP range requests
- **Automatic overviews**: Generate pyramid overviews for fast visualization at different zoom levels

## Quick Start

```python
from georeader import save, read
from georeader.geotensor import GeoTensor
import numpy as np

# Load or create a GeoTensor
gt = GeoTensor(np.random.rand(3, 1000, 1000), 
               transform=rasterio.Affine(10, 0, 0, 0, -10, 0),
               crs="EPSG:4326")

# Save as tiled GeoTIFF
save.save_tiled_geotiff(gt, "output.tif", descriptions=["Red", "Green", "Blue"])

# Save as Cloud Optimized GeoTIFF
save.save_cog(gt, "output_cog.tif", descriptions=["Red", "Green", "Blue"])
```

## Key Functions

| Function | Description |
|----------|-------------|
| `save_tiled_geotiff` | Save data as internally tiled GeoTIFF |
| `save_cog` | Save data as Cloud Optimized GeoTIFF with overviews |

---

::: georeader.save
    options:
      members:
        - save_tiled_geotiff
        - save_cog