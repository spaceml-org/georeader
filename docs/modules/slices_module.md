# Slices Module

The `georeader.slices` module provides utilities for dividing large raster datasets
into smaller tiles (chips/windows) for batch processing, machine learning inference,
and memory-efficient workflows.

## Overview

When working with large satellite images that don't fit in memory, or when running
ML models that require fixed-size inputs, you need to tile the data into smaller
chunks. This module handles the complexity of:

- **Generating tile coordinates** without loading data
- **Overlap handling** for seamless predictions
- **Edge cases** at image boundaries
- **Multi-dimensional support** (time, bands, spatial)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TILING WORKFLOW OVERVIEW                              │
│                                                                              │
│    Large Raster (10000 × 10000)                                             │
│   ┌───────────────────────────────────────────────────────────┐             │
│   │                                                           │             │
│   │   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬───┐        │             │
│   │   │  1  │  2  │  3  │  4  │  5  │  6  │  7  │ 8 │        │             │
│   │   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼───┤        │             │
│   │   │  9  │ 10  │ 11  │ 12  │ 13  │ 14  │ 15  │16 │        │             │
│   │   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼───┤        │             │
│   │   │ 17  │ 18  │ ... │     │     │     │     │   │        │             │
│   │   ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼───┤        │             │
│   │   │     │     │     │     │     │     │     │   │        │  tiles =    │
│   │   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴───┘        │  create_    │
│   │                                                           │  windows()  │
│   │   Each tile: 256 × 256 pixels                            │             │
│   │   Edge tiles may be smaller (trim_incomplete=True)       │             │
│   │                                                           │             │
│   └───────────────────────────────────────────────────────────┘             │
│                                                                              │
│   Process each tile independently → combine results                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Functions

### `create_windows`

Generates a list of `rasterio.windows.Window` objects for tiling a raster:

```python
from georeader.slices import create_windows

# Create 256×256 tiles with 32px overlap
windows = create_windows(
    geodata_shape=(4000, 5600),  # (height, width)
    window_size=(256, 256),      # (height, width)
    overlap=(32, 32),            # (row_overlap, col_overlap)
    include_incomplete=True,     # Include edge tiles
    trim_incomplete=True         # Edge tiles have actual size, not padded
)

print(f"Created {len(windows)} tiles")
# Each window has: window.row_off, window.col_off, window.height, window.width
```

### `create_slices`

Lower-level function using dictionaries for named dimensions:

```python
from georeader.slices import create_slices

# For multi-dimensional data (e.g., time series)
named_shape = {"time": 12, "y": 4000, "x": 5600}
dims = {"y": 256, "x": 256}
overlap = {"y": 32, "x": 32}

slices = create_slices(named_shape, dims, overlap=overlap)
# Returns: [{"y": slice(0, 256), "x": slice(0, 256)}, ...]
```

## Overlap Strategies

Overlap is crucial for ML inference to avoid edge artifacts:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OVERLAP HANDLING                                   │
│                                                                              │
│   Without Overlap (artifacts at tile edges)                                 │
│   ┌────────┬────────┬────────┐                                              │
│   │        │        │        │   Each tile processed independently          │
│   │  Tile  │  Tile  │  Tile  │   → visible seams in output                  │
│   │   1    │   2    │   3    │                                              │
│   └────────┴────────┴────────┘                                              │
│            ↓        ↓                                                       │
│          seams   seams                                                      │
│                                                                              │
│   With Overlap (seamless predictions)                                       │
│   ┌──────────────────────────┐                                              │
│   │        ╔══════╗          │   Tiles overlap by N pixels                  │
│   │  Tile  ║Overlap║  Tile   │   → blend or crop overlap region             │
│   │   1    ║Region ║   2     │   → seamless output                          │
│   │        ╚══════╝          │                                              │
│   └──────────────────────────┘                                              │
│            ←─ overlap ─→                                                    │
│                                                                              │
│   Overlap = kernel_size / 2 is a good rule of thumb for CNNs               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Typical Overlap Values

| Model Type | Recommended Overlap | Reason |
|------------|---------------------|--------|
| Simple CNN | 16-32 px | Receptive field edge effects |
| U-Net | 32-64 px | Multi-scale features |
| Vision Transformer | patch_size / 2 | Patch boundary artifacts |
| Segmentation | class_boundary_width | Smooth boundaries |

## Complete ML Inference Example

```python
import numpy as np
from georeader import read
from georeader.geotensor import GeoTensor
from georeader.slices import create_windows
from georeader.window_utils import pad_window_to_size, slice_save_for_pred

def run_inference_tiled(geotensor, model, tile_size=256, window_size=512):
    """
    Run ML inference on a large image using tiled processing with padding.
    
    This example demonstrates the "tiling and stitching" strategy for large images:
    1. Divide image into non-overlapping output tiles
    2. Read each tile with padding for context (reduces edge artifacts)
    3. Run model on padded tile
    4. Extract only the center region (remove padding)
    5. Write to output mosaic
    
    Args:
        geotensor: Input GeoTensor (C, H, W)
        model: ML model with .predict(tile) method expecting (1, C, H, W)
        tile_size: Size of output tiles (without padding)
        window_size: Size of input to model (with padding)
    
    Returns:
        GeoTensor with predictions
    """
    # 1. Create output GeoTensor with same spatial dims as input
    # Assuming single-band output (e.g., segmentation mask)
    output_shape = (1,) + geotensor.shape[-2:]  # (1, H, W)
    output = GeoTensor(
        np.zeros(output_shape, dtype=np.float32),
        transform=geotensor.transform,
        crs=geotensor.crs,
        fill_value_default=0
    )
    
    # 2. Generate non-overlapping write windows
    windows_write = create_windows(
        geodata_shape=geotensor.shape[-2:],
        window_size=(tile_size, tile_size),
        overlap=(0, 0),  # No overlap - tiles are adjacent
        trim_incomplete=True
    )
    
    # 3. Process each tile with padding
    for w_write in windows_write:
        # Create larger read window with padding for context
        w_read = pad_window_to_size(w_write, size=(window_size, window_size))
        
        # Compute slice to extract valid region after inference
        slice_save = slice_save_for_pred(w_read, w_write)
        
        # Read padded tile (boundless=True handles edges)
        tile = read.read_from_window(
            geotensor, window=w_read, 
            boundless=True, trigger_load=True
        )
        
        # Run model inference
        pred = model.predict(tile.values[None, ...])[0]  # Add/remove batch dim
        
        # Extract valid region (remove padding)
        pred_center = pred[slice_save]
        
        # Write to output mosaic
        output.write_from_window(pred_center, window=w_write)
    
    return output
```

## Handling Edge Cases

### Incomplete Tiles at Edges

```python
# Option 1: Include smaller edge tiles (default)
windows = create_windows(shape, tile_size, 
                        include_incomplete=True,
                        trim_incomplete=True)
# Edge tiles have actual size: 256×200 for a 200px remainder

# Option 2: Pad edge tiles to full size
windows = create_windows(shape, tile_size,
                        include_incomplete=True, 
                        trim_incomplete=False)
# Edge tiles request 256×256 even if image is smaller → needs padding

# Option 3: Exclude edge tiles entirely
windows = create_windows(shape, tile_size,
                        include_incomplete=False)
# Only tiles that fit completely are included
```

### Negative Offsets for Symmetric Padding

```python
# For predictions where you want overlap on ALL edges (including first/last tile)
windows = create_windows(shape, tile_size,
                        overlap=(64, 64),
                        start_negative_if_padding=True)
# First window starts at row=-32, col=-32
# Requires handling out-of-bounds reads with padding
```

## Performance Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PERFORMANCE TRADEOFFS                                 │
│                                                                              │
│   Tile Size vs Overhead                                                      │
│                                                                              │
│   Small tiles (64×64)         Large tiles (512×512)                         │
│   ───────────────────         ─────────────────────                         │
│   ✓ Low memory per tile       ✓ Less overhead per tile                      │
│   ✗ Many tiles = overhead     ✓ Better GPU utilization                      │
│   ✗ More edge artifacts       ✗ Higher memory requirement                   │
│                                                                              │
│   Overlap Impact                                                             │
│   ──────────────                                                             │
│   Small overlap (16px)        Large overlap (128px)                         │
│   ───────────────────         ─────────────────────                         │
│   ✓ Fewer total tiles         ✓ Cleaner predictions                         │
│   ✗ Possible edge artifacts   ✗ Significant overhead                        │
│                                                                              │
│   Rule of thumb:                                                             │
│   - tile_size = 256-512 for most GPU workflows                              │
│   - overlap = 10-20% of tile_size                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## API Reference

::: georeader.slices
    options:
      show_root_heading: true
      show_source: true
      members:
        - create_windows
        - create_slices

## See Also

- [Window Utils](window_utils_module.md) - Coordinate transformations and stitching
- [Mosaic Module](mosaic_module.md) - Combining multiple rasters
- `georeader.read.read_from_window` - Reading tiles from GeoData
