# Griddata Module

The `georeader.griddata` module provides functions for interpolating scattered
geographic data (irregularly-sampled points with per-pixel coordinates) onto
regular grids. This is essential for orthorectifying swath-based satellite data
like hyperspectral sensors.

## Overview

Many satellite sensors, particularly pushbroom and whiskbroom scanners, produce
data where each pixel has its own geographic coordinates rather than following
a regular grid. To analyze this data in GIS software or combine with other
datasets, you need to resample it to a regular grid.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              IRREGULAR vs REGULAR GRID REPRESENTATION                        │
│                                                                              │
│   Sensor Swath (Irregular)              Orthorectified (Regular)            │
│   ────────────────────────              ────────────────────────            │
│                                                                              │
│       ●  ●   ●  ●                        ┌──┬──┬──┬──┬──┐                   │
│     ●    ●  ●    ●                       ├──┼──┼──┼──┼──┤                   │
│      ●   ● ●  ●                          ├──┼──┼──┼──┼──┤                   │
│    ●   ●    ●   ●                        ├──┼──┼──┼──┼──┤                   │
│                                          └──┴──┴──┴──┴──┘                   │
│                                                                              │
│   Each pixel has (lon, lat)              Fixed affine transform             │
│   from attitude/ephemeris data           pixel (i,j) → (x,y) = T × (i,j)   │
│                                                                              │
│   Causes:                                Benefits:                           │
│   - Sensor scan geometry                 - GIS compatible                   │
│   - Platform motion                      - Easy reprojection                │
│   - Terrain relief                       - Stack multiple images            │
│   - Earth curvature                      - Standard analysis tools          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## When to Use This Module

| Scenario | Use griddata module? | Alternative |
|----------|---------------------|-------------|
| Hyperspectral with per-pixel coords | ✅ Yes | - |
| Swath data with lat/lon arrays | ✅ Yes | - |
| Point observations (weather stations) | ✅ Yes | - |
| Regular grid → different CRS | ❌ No | `georeader.read.read_reproject` |
| EMIT with GLT file | ⚠️ Use GLT | `georreference()` is faster |

## Interpolation Methods

The module uses `scipy.interpolate.griddata` internally, which supports three
interpolation methods:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INTERPOLATION METHODS                                 │
│                                                                              │
│   "nearest"                  "linear"                   "cubic" (default)   │
│   ──────────                 ────────                   ─────────────────   │
│                                                                              │
│   ■ ■ ■ ■ ■ ■               ╱╲  ╱╲  ╱╲                   ∿∿∿∿∿∿∿∿∿∿       │
│   ■ ■ ■ ■ ■ ■              ╱  ╲╱  ╲╱  ╲                                     │
│   ■ ■ ■ ■ ■ ■                                                              │
│                                                                              │
│   Voronoi cells            Barycentric on              Clough-Tocher       │
│   Nearest neighbor         Delaunay triangles          C² smooth surface   │
│                                                                              │
│   Continuity: C⁰           Continuity: C⁰              Continuity: C²      │
│   Speed: Fast              Speed: Medium               Speed: Slow         │
│                                                                              │
│   Use for:                 Use for:                    Use for:             │
│   - Classification maps    - Quick previews            - Radiance/Refl     │
│   - Masks                  - When speed matters        - Smooth data       │
│   - Categorical data       - Large datasets            - Final products    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Functions

### `read_to_crs` - Simple Orthorectification

The easiest way to orthorectify data when you just want a specific resolution:

```python
from georeader.griddata import read_to_crs
import numpy as np

# Hyperspectral radiance with per-pixel coordinates
radiance = np.random.rand(1000, 1000, 285)  # (H, W, bands)
lons = np.load("pixel_longitudes.npy")       # (1000, 1000)
lats = np.load("pixel_latitudes.npy")        # (1000, 1000)

# Orthorectify to 30m UTM grid (auto-detects UTM zone)
ortho = read_to_crs(
    radiance, lons, lats,
    resolution_dst=30.0,      # 30 meters
    method="cubic"            # Smooth interpolation
)

print(f"Input shape: {radiance.shape}")   # (1000, 1000, 285) - HWC
print(f"Output shape: {ortho.shape}")     # (285, H_out, W_out) - CHW
print(f"Output CRS: {ortho.crs}")         # e.g., EPSG:32610 (UTM Zone 10N)
```

### `read_reproject_like` - Match Existing Grid

Orthorectify to match an existing dataset's grid exactly:

```python
from georeader.griddata import read_reproject_like

# Load reference dataset (e.g., Sentinel-2)
reference = GeoTensor.load_file("sentinel2_tile.tif")

# Orthorectify hyperspectral to match Sentinel-2 grid
ortho = read_reproject_like(
    radiance, lons, lats,
    data_like=reference,     # Match this grid
    method="cubic"
)

# ortho now has same CRS, resolution, and extent as reference
```

### `reproject` - Full Control

When you need complete control over output parameters:

```python
import rasterio
from georeader.griddata import reproject

# Define exact output grid
transform = rasterio.transform.from_origin(
    west=550000,    # UTM easting
    north=4200000,  # UTM northing  
    xsize=30,       # 30m pixel width
    ysize=30        # 30m pixel height
)

ortho = reproject(
    radiance, lons, lats,
    width=1000,
    height=1000,
    transform=transform,
    dst_crs="EPSG:32610",
    method="cubic",
    fill_value_default=-9999
)
```

### `georreference` - GLT-Based (Fast)

For sensors that provide Geolocation Lookup Tables (like EMIT), use this for
exact pixel mapping without interpolation:

```python
from georeader.griddata import georreference

# GLT maps output pixel → sensor pixel
# glt[0, i, j] = source column
# glt[1, i, j] = source row  
glt = GeoTensor(glt_array, transform=output_transform, crs="EPSG:32610")

# Fast exact orthorectification (no interpolation)
ortho = georreference(glt, radiance)
```

## GLT vs Interpolation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              GLT ORTHORECTIFICATION vs INTERPOLATION                         │
│                                                                              │
│   Geolocation Lookup Table (GLT)         Interpolation (griddata)           │
│   ──────────────────────────────         ────────────────────────           │
│                                                                              │
│   ┌───────────────┐                      ┌───────────────┐                  │
│   │ Sensor Array  │                      │ Sensor Array  │                  │
│   │  ┌─┬─┬─┐      │                      │  ●  ●  ●      │                  │
│   │  │A│B│C│      │                      │   ●  ●  ●     │                  │
│   │  └─┴─┴─┘      │                      │  ●  ●  ●      │                  │
│   └───────┬───────┘                      └───────┬───────┘                  │
│           │ GLT lookup                           │ Interpolate              │
│           ▼                                      ▼                          │
│   ┌───────────────┐                      ┌───────────────┐                  │
│   │ Output Grid   │                      │ Output Grid   │                  │
│   │  ┌─┬─┬─┐      │                      │  ┌─┬─┬─┐      │                  │
│   │  │ │A│B│      │                      │  │≈│≈│≈│      │                  │
│   │  └─┴─┴─┘      │                      │  └─┴─┴─┘      │                  │
│   └───────────────┘                      └───────────────┘                  │
│                                                                              │
│   Pros:                                  Pros:                              │
│   ✓ Exact pixel values preserved         ✓ Works without GLT               │
│   ✓ Very fast (array indexing)           ✓ Smooth output                   │
│   ✓ No resampling artifacts              ✓ Any output resolution           │
│                                                                              │
│   Cons:                                  Cons:                              │
│   ✗ Requires GLT from data provider      ✗ Changes pixel values            │
│   ✗ Fixed output resolution              ✗ Slower (O(n log n) Delaunay)    │
│   ✗ May have gaps                        ✗ Edge artifacts possible         │
│                                                                              │
│   Use when:                              Use when:                          │
│   - GLT available (EMIT, etc.)           - No GLT available                │
│   - Exact values needed                  - Custom output resolution        │
│   - Processing derived products          - Point data sources              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Complete Example: EMIT-style Workflow

```python
import numpy as np
from georeader.griddata import read_to_crs, georreference, get_shape_transform_crs
from georeader.geotensor import GeoTensor

def process_hyperspectral_swath(radiance, lons, lats, glt=None, 
                                 resolution=30.0, method="cubic"):
    """
    Orthorectify hyperspectral swath data.
    
    Args:
        radiance: Raw radiance (H, W, C) or (H, W)
        lons: Per-pixel longitudes (H, W)
        lats: Per-pixel latitudes (H, W)
        glt: Optional GLT array (2, H_out, W_out) 
        resolution: Output resolution in meters
        method: Interpolation method if no GLT
    
    Returns:
        Orthorectified GeoTensor
    """
    if glt is not None:
        # Fast path: use GLT
        print("Using GLT-based orthorectification (exact)")
        
        # Transpose to (C, H, W) if needed
        if len(radiance.shape) == 3 and radiance.shape[-1] < radiance.shape[0]:
            radiance = np.transpose(radiance, (2, 0, 1))
        
        return georreference(glt, radiance)
    
    else:
        # Slow path: interpolation
        print(f"Using {method} interpolation")
        return read_to_crs(
            radiance, lons, lats,
            resolution_dst=resolution,
            method=method,
            fill_value_default=np.nan
        )


# Example usage
radiance = np.random.rand(1000, 1000, 100)
lons = np.linspace(-122.5, -122.0, 1000)[None, :].repeat(1000, axis=0)
lats = np.linspace(37.5, 38.0, 1000)[:, None].repeat(1000, axis=1)

# Add some irregularity (simulating real sensor geometry)
lons += np.random.normal(0, 0.001, lons.shape)
lats += np.random.normal(0, 0.001, lats.shape)

ortho = process_hyperspectral_swath(radiance, lons, lats, resolution=30.0)
print(f"Output: {ortho.shape}, CRS: {ortho.crs}")
```

## Performance Tips

1. **Use cubic interpolation sparingly**: It's O(n log n) for Delaunay + O(n) per query.
   For large arrays (>10M points), consider downsampling first.

2. **GLT is always faster**: If available, `georreference()` is a simple array lookup.

3. **Match grids efficiently**: Use `read_reproject_like` instead of computing
   output parameters manually.

4. **Handle fill values**: Areas outside the convex hull of input points will be
   filled with `fill_value_default`. Consider using NaN for easy masking.

## API Reference

::: georeader.griddata
    options:
      show_root_heading: true
      show_source: true
      members:
        - reproject
        - read_to_crs
        - read_reproject_like
        - georreference
        - meshgrid
        - footprint
        - get_shape_transform_crs

## See Also

- [EMIT Tutorial](../emit_explore.ipynb) - Reading EMIT hyperspectral data
- [Read Module](read_module.md) - Regular grid reprojection
- [Window Utils](window_utils_module.md) - Coordinate transformations
