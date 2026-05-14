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

## Examples

For complete examples of using tiling for ML inference workflows, including handling edge cases and stitching predictions, see the [Tiling and Stitching tutorial](../advanced/tiling_and_stitching.ipynb).

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

- [Read Module](../modules/read_module.md) - Image reading and reprojection
- [Mosaic Module](mosaic_module.md) - Combining multiple rasters
- `georeader.read.read_from_window` - Reading tiles from GeoData
