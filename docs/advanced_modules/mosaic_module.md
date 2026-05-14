# Mosaic Module

The `mosaic` module provides functionality for creating spatial mosaics from multiple overlapping raster images. This is essential when working with satellite imagery that spans multiple tiles or when combining data from different acquisition times.

## Overview

Mosaicking is the process of combining multiple raster images into a single, seamless composite. The module handles:

- **Spatial alignment**: Reprojecting images to a common coordinate reference system
- **No-data handling**: Filling gaps in one image with data from overlapping images
- **Masking**: Custom masking functions to exclude invalid pixels (clouds, shadows, etc.)
- **Memory efficiency**: Window-based processing for large datasets

```
    ┌─────────┐   ┌─────────┐
    │ Image 1 │   │ Image 2 │     Input: Multiple overlapping rasters
    │  ████   │   │   ████  │
    │  ████   │   │   ████  │
    └─────────┘   └─────────┘
         \           /
          \         /
           ↓       ↓
         ┌───────────┐
         │  Mosaic   │          Output: Single seamless composite
         │  ████████ │
         │  ████████ │
         └───────────┘
```

## Quick Start

```python
from georeader import mosaic
from georeader.rasterio_reader import RasterioReader

# Load multiple overlapping images
images = [
    RasterioReader("image1.tif"),
    RasterioReader("image2.tif"),
    RasterioReader("image3.tif"),
]

# Create mosaic over the union of all footprints
result = mosaic.spatial_mosaic(images)

# Create mosaic over a specific polygon
from shapely.geometry import box
aoi = box(minx, miny, maxx, maxy)
result = mosaic.spatial_mosaic(images, polygon=aoi, crs_polygon="EPSG:4326")
```

## Key Functions

### spatial_mosaic

The main function for creating mosaics from a list of raster images.

::: georeader.mosaic.spatial_mosaic

## Algorithm Details

The `spatial_mosaic` function uses an iterative fill algorithm:

1. **Initialize output**: Load the first image, reprojecting to target CRS/bounds
2. **Track invalid pixels**: Identify pixels with no-data values
3. **Iterative fill**: For each subsequent image:
   - Skip if footprint doesn't intersect remaining invalid regions
   - Load only the overlapping region
   - Fill invalid pixels with valid data from the new image
   - Update the invalid pixel mask
4. **Early termination**: Stop when all pixels are valid

```
    Iteration 1              Iteration 2              Final Result
    ┌───────────┐            ┌───────────┐            ┌───────────┐
    │ ████ ░░░░ │     +      │ ░░░░ ████ │     =      │ ████ ████ │
    │ ████ ░░░░ │            │ ░░░░ ████ │            │ ████ ████ │
    └───────────┘            └───────────┘            └───────────┘
    (from image 1)           (from image 2)           (complete)
    
    ████ = valid data        ░░░░ = no-data
```

## Working with Masks

The module supports custom masking functions to exclude invalid pixels beyond simple no-data values:

```python
def cloud_mask_function(geotensor):
    """Custom mask function that returns True for invalid pixels."""
    # Example: mask pixels where any band exceeds threshold
    return geotensor.values.max(axis=0) > 10000

# Use with spatial_mosaic
result = mosaic.spatial_mosaic(
    images,
    masking_function=cloud_mask_function
)

# Or provide explicit masks as tuples
images_with_masks = [
    (image1, mask1),
    (image2, mask2),
]
result = mosaic.spatial_mosaic(images_with_masks)
```

## Memory-Efficient Processing

For large mosaics, use the `window_size` parameter to process in tiles:

```python
# Process in 512x512 windows
result = mosaic.spatial_mosaic(
    images,
    window_size=(512, 512),
    polygon=large_aoi,
    crs_polygon="EPSG:4326"
)
```

## Resampling Methods

The `resampling` parameter controls interpolation during reprojection:

| Method | Use Case |
|--------|----------|
| `nearest` | Categorical data (land cover, masks) |
| `bilinear` | Continuous data, faster |
| `cubic` | Continuous data, smoother |
| `cubic_spline` | Default, best quality for most imagery |

```python
import rasterio.warp

result = mosaic.spatial_mosaic(
    images,
    resampling=rasterio.warp.Resampling.bilinear
)
```

## See Also

- [`read.read_reproject`](../modules/read_module.md): Underlying reprojection function
- [`slices`](slices_module.md): Tiling utilities for windowed processing
