# Plot Module

The `georeader.plot` module provides visualization utilities for geospatial data, built on top
of matplotlib with native support for `GeoData` objects. It handles coordinate reference system
transformations, geographic extent display, and common geospatial visualization patterns.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            georeader.plot                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Main Functions:                                                            │
│  ├── show()                    Plot GeoData with geographic extent          │
│  ├── add_shape_to_plot()       Overlay vector geometries                    │
│  ├── plot_segmentation_mask()  Categorical mask with legend                 │
│  └── colorbar_next_to()        Add colorbar alongside plot                  │
│                                                                             │
│  Key Features:                                                              │
│  ├── Automatic extent calculation from GeoData.bounds                       │
│  ├── CRS-aware axis labels (lat/lng display)                                │
│  ├── Optional scalebar with geographic units                                │
│  └── Vector overlay with automatic CRS reprojection                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Raster Visualization

```python
import matplotlib.pyplot as plt
from georeader import read, plot

# Load and display a GeoTIFF
gt = read.read_from_tif("path/to/raster.tif")

# Simple display with geographic axes
fig, ax = plt.subplots(figsize=(10, 8))
plot.show(gt, ax=ax, add_colorbar_next_to=True, add_scalebar=True)
plt.show()
```

### RGB Composite Display

```python
from georeader import plot
from georeader.readers import S2_SAFE_reader

# Load Sentinel-2 RGB bands
s2 = S2_SAFE_reader.s2loader("path/to/S2.SAFE", out_res=20)
rgb = s2.load(bands=["B04", "B03", "B02"])

# Normalize for display (0-1 range)
rgb_norm = rgb / 3000  # Adjust based on your data range
rgb_norm.values = rgb_norm.values.clip(0, 1)

plot.show(rgb_norm, title="Sentinel-2 RGB Composite")
```

### Overlay Vector Data

```python
import geopandas as gpd
from georeader import plot

# Load raster and vector
gt = read.read_from_tif("raster.tif")
aoi = gpd.read_file("aoi.geojson")

# Plot raster then overlay vector
fig, ax = plt.subplots()
plot.show(gt, ax=ax)
plot.add_shape_to_plot(
    aoi, 
    ax=ax, 
    crs_plot=gt.crs,
    polygon_no_fill=True,
    kwargs_geopandas_plot={"edgecolor": "red", "linewidth": 2}
)
```

## Function Reference

### show()

The primary function for displaying `GeoData` objects with proper geographic extent.

```python
plot.show(
    data,                          # GeoData object to plot
    add_colorbar_next_to=False,    # Add colorbar to the side
    add_scalebar=False,            # Add geographic scalebar
    kwargs_scalebar=None,          # Options for matplotlib-scalebar
    mask=False,                    # Mask invalid values (True uses fill_value_default)
    bounds_in_latlng=True,         # Display axes in lat/lng coordinates
    ax=None,                       # Target axes (uses gca() if None)
    title=None,                    # Plot title
    **kwargs                       # Additional imshow arguments
)
```

**Automatic Data Handling:**

```
Input Shape Handling:
┌─────────────────────────────────────────────────────────────────┐
│ (C, H, W)  →  Transpose to (H, W, C) for matplotlib            │
│ (1, H, W)  →  Squeeze to (H, W)                                 │
│ (H, W)     →  Use directly                                      │
└─────────────────────────────────────────────────────────────────┘

Masking with Alpha Channel:
┌─────────────────────────────────────────────────────────────────┐
│ If mask provided for RGB data:                                  │
│   (H, W, 3) + mask → (H, W, 4) with alpha channel               │
│   Masked pixels become transparent                              │
└─────────────────────────────────────────────────────────────────┘
```

**Coordinate Display:**

When `bounds_in_latlng=True`, axis ticks show lat/lng coordinates regardless of the 
data's native CRS. This is achieved via on-the-fly reprojection of tick positions.

### add_shape_to_plot()

Overlay vector geometries on an existing plot with automatic CRS handling.

```python
plot.add_shape_to_plot(
    shape,                     # GeoDataFrame, list of geometries, or single geometry
    ax=None,                   # Target axes
    crs_plot=None,             # CRS of the plot (for reprojection)
    crs_shape=None,            # CRS of input shape (if not GeoDataFrame)
    polygon_no_fill=False,     # Plot only polygon boundaries
    kwargs_geopandas_plot=None, # Pass-through to geopandas.plot()
    title=None                 # Plot title
)
```

**CRS Handling:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Shape Input                    → CRS Determination              │
├─────────────────────────────────────────────────────────────────┤
│ GeoDataFrame with .crs         → Use native CRS                 │
│ List[Geometry] + crs_shape     → Use crs_shape                  │
│ List[Geometry] no crs_shape    → Use crs_plot                   │
├─────────────────────────────────────────────────────────────────┤
│ If crs_plot specified → Reproject shape to match plot CRS       │
└─────────────────────────────────────────────────────────────────┘
```

**Example: Multi-layer Visualization**

```python
from georeader import plot

fig, ax = plt.subplots(figsize=(12, 10))

# Background raster
plot.show(raster, ax=ax)

# Roads layer
plot.add_shape_to_plot(
    roads_gdf,
    ax=ax,
    crs_plot=raster.crs,
    kwargs_geopandas_plot={"color": "yellow", "linewidth": 1}
)

# Buildings layer
plot.add_shape_to_plot(
    buildings_gdf,
    ax=ax,
    crs_plot=raster.crs,
    polygon_no_fill=True,
    kwargs_geopandas_plot={"edgecolor": "cyan"}
)

# Points of interest
plot.add_shape_to_plot(
    poi_gdf,
    ax=ax,
    crs_plot=raster.crs,
    kwargs_geopandas_plot={"color": "red", "markersize": 50}
)
```

### plot_segmentation_mask()

Specialized function for plotting categorical/classification maps with automatic legend generation.

```python
plot.plot_segmentation_mask(
    mask,                      # GeoData with integer class values
    color_array=None,          # Colors for each class (auto-generates if None)
    interpretation_array=None, # Class labels for legend
    legend=True,               # Show legend
    ax=None,                   # Target axes
    add_scalebar=False,        # Add geographic scalebar
    kwargs_scalebar=None,      # Scalebar options
    bounds_in_latlng=True      # Display axes in lat/lng
)
```

**Example: Land Cover Classification**

```python
from georeader import plot
import numpy as np

# Define class colors and names
colors = [
    [0.2, 0.6, 0.2],   # Forest - green
    [0.8, 0.8, 0.2],   # Cropland - yellow
    [0.5, 0.5, 0.5],   # Urban - gray
    [0.2, 0.4, 0.8],   # Water - blue
    [0.9, 0.9, 0.9],   # Snow - white
]
classes = ["Forest", "Cropland", "Urban", "Water", "Snow"]

# Plot classification result
fig, ax = plt.subplots(figsize=(10, 8))
plot.plot_segmentation_mask(
    classification_result,
    color_array=np.array(colors),
    interpretation_array=classes,
    legend=True,
    add_scalebar=True,
    ax=ax
)
ax.set_title("Land Cover Classification")
```

### colorbar_next_to()

Add a colorbar that doesn't steal space from the main plot.

```python
plot.colorbar_next_to(
    im,                    # AxesImage from imshow()
    ax,                    # Axes containing the image
    fig=None,              # Figure (uses gcf() if None)
    location='right',      # 'left', 'right', 'top', 'bottom'
    pad=0.05,              # Padding between plot and colorbar
    orientation='vertical', # 'vertical' or 'horizontal'
    label_colorbar=None    # Label for colorbar
)
```

**Example: Custom Colorbar**

```python
fig, ax = plt.subplots()
im = ax.imshow(gt.values, extent=(*gt.bounds[:2], *gt.bounds[2:]))
cbar = plot.colorbar_next_to(
    im, ax,
    location='bottom',
    orientation='horizontal',
    label_colorbar='Temperature (°C)'
)
```

## Scalebar Support

The scalebar feature requires the `matplotlib-scalebar` package:

```bash
pip install matplotlib-scalebar
```

**Automatic Unit Handling:**

```
┌─────────────────────────────────────────────────────────────────┐
│ CRS Type              │ dx Calculation                          │
├───────────────────────┼─────────────────────────────────────────┤
│ Projected (UTM, etc.) │ dx = 1 (coordinates already in meters)  │
│ Geographic (WGS84)    │ dx = great circle distance for 1°       │
│                       │ at image center latitude                │
└───────────────────────┴─────────────────────────────────────────┘
```

**Custom Scalebar Options:**

```python
plot.show(
    gt,
    add_scalebar=True,
    kwargs_scalebar={
        "dx": 1,
        "units": "m",
        "length_fraction": 0.25,
        "location": "lower right",
        "box_alpha": 0.8,
        "font_properties": {"size": 12}
    }
)
```

## Common Patterns

### Side-by-Side Comparison

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot.show(before_image, ax=ax1, title="Before")
plot.show(after_image, ax=ax2, title="After")

# Share the same extent for visual comparison
for ax in [ax1, ax2]:
    ax.set_xlim(before_image.bounds[0], before_image.bounds[2])
    ax.set_ylim(before_image.bounds[1], before_image.bounds[3])

plt.tight_layout()
```

### Masking NoData Values

```python
# Method 1: Use fill_value_default
gt = read.read_from_tif("raster.tif")
plot.show(gt, mask=True)  # Automatically masks fill_value_default

# Method 2: Custom mask
invalid_mask = (gt.values < 0) | (gt.values > 10000)
plot.show(gt, mask=invalid_mask)
```

### Publication-Quality Figure

```python
import matplotlib.pyplot as plt
from georeader import plot

# Set up figure with specific DPI
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

# Plot with all enhancements
plot.show(
    gt,
    ax=ax,
    add_colorbar_next_to=True,
    add_scalebar=True,
    kwargs_scalebar={"location": "lower left"},
    cmap="viridis",
    vmin=0,
    vmax=1
)

ax.set_title("Analysis Results", fontsize=14, fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.savefig("figure.png", bbox_inches="tight", dpi=300)
```

## Warnings and Limitations

**Non-Rectilinear Transforms:**

If the raster has a non-rectilinear transform (rotation, skew), a warning is issued:

```
The transform is not rectilinear. The x and y ticks and the scale bar 
are not going to be correct.
```

To suppress this warning:

```python
import warnings
warnings.filterwarnings('ignore', message='The transform is not rectilinear.')
```

**Large Rasters:**

For large rasters, consider downsampling before plotting:

```python
from georeader import window_utils

# Read at reduced resolution
gt_overview = read.read_from_tif(
    "large_raster.tif",
    overview_level=2  # Use pyramid level
)
plot.show(gt_overview)
```

## API Reference

::: georeader.plot
    options:
      show_root_heading: true
      members:
        - show
        - add_shape_to_plot
        - plot_segmentation_mask
        - colorbar_next_to
