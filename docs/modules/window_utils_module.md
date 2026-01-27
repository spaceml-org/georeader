# Window Utilities Module

The `window_utils` module provides essential functions for working with rasterio windows, coordinate transformations, and spatial operations. Windows are fundamental to efficient raster I/O, allowing you to read and write specific regions of large datasets without loading everything into memory.

## Overview

A **window** in rasterio defines a rectangular region of a raster in pixel coordinates:

```
    Image (3000 × 4000 pixels)
    ┌──────────────────────────────────────────────┐
    │                                              │
    │      ┌────────────────┐                      │
    │      │    Window      │ row_off = 500        │
    │      │  width=800     │ col_off = 300        │
    │      │  height=600    │                      │
    │      └────────────────┘                      │
    │                                              │
    └──────────────────────────────────────────────┘
    
    Window(col_off=300, row_off=500, width=800, height=600)
```

Key concepts:
- **Window**: Pixel coordinates (col_off, row_off, width, height)
- **Bounds**: Geographic coordinates (xmin, ymin, xmax, ymax)
- **Transform**: Affine matrix mapping pixel ↔ geographic coordinates

## Quick Start

```python
import rasterio.windows
from georeader import window_utils
from shapely.geometry import box

# Create a window around a polygon
polygon = box(-3.71, 40.41, -3.69, 40.43)  # WGS84 coordinates
window = window_utils.window_from_polygon(reader, polygon, crs_polygon="EPSG:4326")

# Round to integer pixels (for reading)
window_int = window_utils.round_outer_window(window)

# Convert window back to geographic bounds
bounds = window_utils.window_bounds(window, reader.transform)
```

## Core Functions

### Window Creation and Manipulation

#### pad_window

Add padding to a window (useful for CNN inference context):

::: georeader.window_utils.pad_window

#### pad_window_to_size

Center-pad a window to a specific size:

::: georeader.window_utils.pad_window_to_size

#### round_outer_window

Round floating-point window coordinates to integers:

::: georeader.window_utils.round_outer_window

### Coordinate Transformations

#### transform_to_resolution_dst

Rescale a transform to a different resolution:

::: georeader.window_utils.transform_to_resolution_dst

#### figure_out_transform

Compute output transform from bounds and resolution:

::: georeader.window_utils.figure_out_transform

### Window-Bounds Conversion

#### window_bounds

Get geographic bounds from a window:

::: georeader.window_utils.window_bounds

#### window_polygon

Get a Shapely polygon from a window:

::: georeader.window_utils.window_polygon

#### normalize_bounds

Ensure bounds are valid (min < max):

::: georeader.window_utils.normalize_bounds

### Polygon Operations

#### polygon_to_crs

Reproject a polygon between coordinate systems:

::: georeader.window_utils.polygon_to_crs

#### exterior_pixel_coords

Convert polygon vertices to pixel coordinates:

::: georeader.window_utils.exterior_pixel_coords

### Boundless Reading Support

#### get_slice_pad

Compute slice and padding for reading beyond image boundaries:

::: georeader.window_utils.get_slice_pad

## Advanced Usage

### Tiling and Stitching for ML Inference

When running CNN models on large rasters, you typically need to:
1. Split the image into overlapping tiles
2. Add context padding for each tile
3. Run inference
4. Extract the valid (unpadded) region
5. Stitch results together

```
    Input: Large Raster (e.g., 10000 × 10000 pixels)
    ┌───────────────────────────────────────────┐
    │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
    │ │tile1│ │tile2│ │tile3│ │tile4│   ...    │
    │ └─────┘ └─────┘ └─────┘ └─────┘          │
    │ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
    │ │tile5│ │tile6│ │tile7│ │tile8│   ...    │
    │ └─────┘ └─────┘ └─────┘ └─────┘          │
    │   ...     ...     ...     ...            │
    └───────────────────────────────────────────┘
    
    Each tile: read with padding → inference → extract center → write
```

#### slice_save_for_pred

Compute slices to extract valid region from padded prediction:

::: georeader.window_utils.slice_save_for_pred

**Example workflow:**

```python
from georeader import window_utils, read

# Configuration
tile_size = 256       # Output tile size
cnn_input_size = 512  # CNN expects this size (with context)

# Generate write windows (non-overlapping)
windows_write = [
    rasterio.windows.Window(col, row, tile_size, tile_size)
    for row in range(0, image_height, tile_size)
    for col in range(0, image_width, tile_size)
]

# Process each tile
for w_write in windows_write:
    # Add padding for CNN context
    w_read = window_utils.pad_window_to_size(w_write, (cnn_input_size, cnn_input_size))
    
    # Compute extraction slice
    slice_save = window_utils.slice_save_for_pred(w_read, w_write)
    
    # Read with padding (boundless=True allows reading beyond edges)
    data = read.read_from_window(reader, window=w_read, boundless=True)
    
    # Run CNN (output shape matches input: 512×512)
    prediction = model(data)
    
    # Extract valid center region (256×256)
    valid_prediction = prediction[slice_save]
    
    # Write to output
    output.write(valid_prediction, window=w_write)
```

### Handling Edge Cases

#### Reading Beyond Image Boundaries

When reading tiles near edges, you may request pixels outside the image:

```python
# Window extends 50 pixels left and 30 pixels above image
window_data = rasterio.windows.Window(0, 0, 1000, 1000)  # Full image
window_read = rasterio.windows.Window(-50, -30, 200, 200)  # Extends beyond

# Get slice and padding info
slice_dict, pad_width = window_utils.get_slice_pad(window_data, window_read)

# slice_dict: {'x': slice(0, 150), 'y': slice(0, 170)}
# pad_width: {'x': (50, 0), 'y': (30, 0)}

# Read what exists, then pad
data = reader.read(window=...)
data_padded = np.pad(data, window_utils.pad_list_numpy(pad_width))
```

#### pad_list_numpy

Convert padding dict to numpy format:

::: georeader.window_utils.pad_list_numpy

## Coordinate Systems Reference

```
    Pixel Coordinates              Geographic Coordinates
    ┌───────────────┐              ┌───────────────┐
    │ (0,0)         │              │ (xmin, ymax)  │
    │   ┌─────────┐ │   Transform  │   ┌─────────┐ │
    │   │ Window  │ │  ──────────→ │   │ Bounds  │ │
    │   │(col,row)│ │              │   │ (x, y)  │ │
    │   └─────────┘ │              │   └─────────┘ │
    │               │              │               │
    │ (width,height)│              │ (xmax, ymin)  │
    └───────────────┘              └───────────────┘
    
    Transform T: (col, row) → (x, y)
    Inverse ~T:  (x, y) → (col, row)
```

## See Also

- [`read`](read_module.md): Functions using windows to read raster data
- [`mosaic`](mosaic_module.md): Mosaicking using windowed operations
- [`slices`](slices_module.md): Create window grids for tiled processing
- [`GeoTensor`](geotensor_module.md): Data structure with transform/CRS
