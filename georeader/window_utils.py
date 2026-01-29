"""
Window Utilities: Coordinate transformation between pixel and geographic space.

This module provides utilities for working with rasterio Windows - the fundamental
data structure for specifying regions of interest in pixel coordinates. Windows
enable efficient reading of subsets from large raster files.

Window Coordinate System
------------------------

A Window represents a rectangular region in pixel space::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RASTERIO WINDOW ANATOMY                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │    Full Raster (0,0 at top-left)                                        │
    │    ↓                                                                     │
    │    ┌────────────────────────────────────────────────────────┐           │
    │    │ (0,0)                                        cols →    │           │
    │    │     ┌────────────────────┐                             │           │
    │    │     │← col_off →│        │                             │           │
    │    │     │    (row_off, col_off) ← Window origin            │           │
    │    │     │           ·─────────────────┐                    │           │
    │  r │     │           │    WINDOW       │                    │           │
    │  o │     │           │                 │ height             │           │
    │  w │     │           │    width        │                    │           │
    │  s │     │           └─────────────────┘                    │           │
    │    │     │                                                  │           │
    │  ↓ │     │                                                  │           │
    │    └────────────────────────────────────────────────────────┘           │
    │                                                                          │
    │    Window = rasterio.windows.Window(col_off, row_off, width, height)    │
    │                                     ───────  ───────  ─────  ──────     │
    │                                     column   row      cols   rows       │
    │                                     offset   offset                     │
    └─────────────────────────────────────────────────────────────────────────┘

    NOTE: Window constructor order is (col_off, row_off) but most geospatial
          operations use (row, col) or (y, x) order. Be careful!

Window ↔ Bounds Conversion
--------------------------

Converting between pixel windows and geographic bounds requires a transform::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │              WINDOW ↔ BOUNDS TRANSFORMATION                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │    WINDOW (pixels)              AFFINE TRANSFORM           BOUNDS       │
    │    ┌─────────────┐                    ║                ┌─────────────┐  │
    │    │ col_off=100 │                    ║                │ minx=-122.5 │  │
    │    │ row_off=200 │   ──────────────►  ║  ──────────►   │ miny=37.0   │  │
    │    │ width=256   │   window_bounds()  ║                │ maxx=-122.0 │  │
    │    │ height=256  │                    ║                │ maxy=37.5   │  │
    │    └─────────────┘                    ║                └─────────────┘  │
    │                                       ║                                  │
    │                                       ║   Affine(a, b, c,               │
    │                      ◄────────────────║          d, e, f)               │
    │                      bounds_to_windows()                                │
    │                                       ║                                  │
    │    Affine Transform encodes:          ║                                  │
    │    • Pixel resolution (a, e)          ║                                  │
    │    • Origin coordinates (c, f)        ║                                  │
    │    • Rotation/shear (b, d)            ║                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Window Rounding Strategies
--------------------------

When bounds don't align exactly with pixel boundaries::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WINDOW ROUNDING STRATEGIES                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Exact bounds (before rounding):                                       │
    │   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐                                │
    │   │           Desired area             │                                │
    │   └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘                                │
    │                                                                          │
    │   round_outer_window():                 round_inner_window():           │
    │   ┌─────────────────────────────┐      ┌─────────────────────┐         │
    │   │ ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │      │  ┌ ─ ─ ─ ─ ─ ─ ─ ┐ │         │
    │   │ │                       │  │      │  │               │  │         │
    │   │ │   Expands outward     │  │      │  │ Shrinks inward │  │         │
    │   │ │   to include all      │  │      │  │ to only fully  │  │         │
    │   │ │   partial pixels      │  │      │  │ covered pixels │  │         │
    │   │ └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │      │  └ ─ ─ ─ ─ ─ ─ ─ ┘ │         │
    │   └─────────────────────────────┘      └─────────────────────┘         │
    │                                                                          │
    │   Use outer when: You need all data that intersects the bounds          │
    │   Use inner when: You need only data fully within the bounds            │
    └─────────────────────────────────────────────────────────────────────────┘

Precision Handling
------------------

The module uses `PIXEL_PRECISION = 3` decimal places to handle floating-point
precision issues in coordinate transformations::

    Example: A window at col_off=99.9997 should be col_off=100
             A window at col_off=100.0003 should be col_off=100
             A window at col_off=100.5 should NOT be rounded

Module Functions Overview
-------------------------

Window Padding:
    - :func:`pad_window`: Add symmetric padding to window
    - :func:`pad_window_to_size`: Pad window to specific size
    - :func:`remove_pad`: Remove padding (slice array)

Window Rounding:
    - :func:`round_outer_window`: Round to include all partial pixels
    - :func:`round_inner_window`: Round to exclude partial pixels

Bounds Conversion:
    - :func:`window_bounds`: Window → geographic bounds
    - :func:`bounds_to_windows`: Geographic bounds → list of windows (handles antimeridian)
    - :func:`polygon_to_crs`: Transform polygon to target CRS

Grid Operations:
    - :func:`apply_transform_to_pol`: Apply affine transform to polygon
    - :func:`get_valid_mask`: Get mask of valid data region
    - :func:`figure_out_transform`: Create transform from bounds and shape

Quick Start
-----------

Convert bounds to a window::

    from georeader import window_utils
    import rasterio

    # Open raster and get transform
    with rasterio.open("data.tif") as src:
        transform = src.transform
        crs = src.crs

    # Convert geographic bounds to pixel window
    bounds = (-122.5, 37.0, -122.0, 37.5)  # (minx, miny, maxx, maxy)
    windows = window_utils.bounds_to_windows(
        data=src, bounds_dst=bounds, crs_dst="EPSG:4326"
    )
    # Returns list of Windows (usually 1, but 2 if crosses antimeridian)

Get geographic bounds from a window::

    window = rasterio.windows.Window(col_off=100, row_off=200, width=256, height=256)
    bounds = window_utils.window_bounds(window, transform)
    # Returns (minx, miny, maxx, maxy)

See Also
--------
georeader.read : Higher-level reading functions using windows
georeader.geotensor : GeoTensor with transform/bounds properties
rasterio.windows : Base Window class documentation

References
----------
- Rasterio windows: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
- Affine transforms: https://github.com/rasterio/affine
"""
import math
import numbers
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio.transform
import rasterio.warp
import rasterio.windows
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

from georeader import compare_crs, res

PIXEL_PRECISION = 3


def pad_window(window: rasterio.windows.Window, pad_size: Tuple[int, int]) -> rasterio.windows.Window:
    """
    Expand a window symmetrically by adding padding on all sides.

    Creates a larger window by adding the specified padding to each edge.
    The center of the output window remains at the center of the input window.
    Essential for CNN inference where context around tiles is needed.

    Window expansion diagram::

        Original window:              Padded window (pad_size=(2, 3)):
        ┌─────────────┐               ┌─────────────────────┐
        │             │               │← 3 cols → ← 3 cols →│
        │   100×50    │     ───►      │↑         ↑          │
        │   window    │               │2   106×54 window    │
        │             │               │↓         ↓          │
        └─────────────┘               │← 3 cols → ← 3 cols →│
                                      └─────────────────────┘

        Output: width = 100 + 2×3 = 106
                height = 50 + 2×2 = 54
                col_off = original - 3
                row_off = original - 2

    Args:
        window (rasterio.windows.Window): Input window to expand.
            Format: Window(col_off, row_off, width, height).
        pad_size (Tuple[int, int]): Padding amounts as (pad_rows, pad_cols).
            - pad_size[0]: Pixels to add above AND below (total 2× added to height)
            - pad_size[1]: Pixels to add left AND right (total 2× added to width)
            Note: Order is (rows, cols) matching numpy array convention.

    Returns:
        rasterio.windows.Window: Expanded window with:
            - col_off = original.col_off - pad_size[1]
            - row_off = original.row_off - pad_size[0]
            - width = original.width + 2 * pad_size[1]
            - height = original.height + 2 * pad_size[0]

    Examples:
        >>> import rasterio.windows
        >>>
        >>> # Original 100×50 window at (10, 20)
        >>> window = rasterio.windows.Window(10, 20, 100, 50)
        >>>
        >>> # Add 5 rows and 10 columns of padding
        >>> padded = pad_window(window, pad_size=(5, 10))
        >>> print(f"Original: {window}")
        Window(col_off=10, row_off=20, width=100, height=50)
        >>> print(f"Padded: {padded}")
        Window(col_off=0, row_off=15, width=120, height=60)
        >>> # New dimensions: 100+20=120, 50+10=60
        >>> # New offset: (10-10, 20-5) = (0, 15)
        >>>
        >>> # CNN context: add 32 pixels around 256×256 tile
        >>> tile = rasterio.windows.Window(128, 256, 256, 256)
        >>> tile_with_context = pad_window(tile, pad_size=(32, 32))
        >>> print(f"With context: {tile_with_context}")
        Window(col_off=96, row_off=224, width=320, height=320)

    Note:
        - Output window may have negative offsets (boundless reading)
        - Use with `read_from_window(..., boundless=True)` for edge tiles
        - For asymmetric padding, use `get_slice_pad` instead

    See Also:
        - `pad_window_to_size`: Pad to reach specific dimensions
        - `get_slice_pad`: Handle out-of-bounds windows with slicing
    """

    return rasterio.windows.Window(
        window.col_off - pad_size[1],
        window.row_off - pad_size[0],
        width=window.width + 2 * pad_size[1],
        height=window.height + 2 * pad_size[0],
    )


def pad_window_to_size(window: rasterio.windows.Window, size: Tuple[int, int]) -> rasterio.windows.Window:
    """
    Adjust window dimensions to exactly match a target size while maintaining center.

    Expands (or shrinks) a window symmetrically to reach the specified dimensions.
    The center pixel of the output window aligns with the center of the input window.
    Essential for ensuring fixed-size inputs for neural networks.

    This function handles both:
    - Expansion: When target size > window size (adds padding)
    - Contraction: When target size < window size (extracts center crop)

    Center-alignment behavior::

        Expansion (size > window):          Contraction (size < window):

        ┌───────────────────────┐          ┌───────────────────────┐
        │     padded area       │          │                       │
        │   ┌─────────────┐     │          │   ┌─────────────┐     │
        │   │             │     │          │   │┌───────────┐│     │
        │   │   original  │     │    ◄──   │   ││  center   ││     │
        │   │    window   │     │          │   ││   crop    ││     │
        │   └─────────────┘     │          │   │└───────────┘│     │
        │     padded area       │          │   └─────────────┘     │
        └───────────────────────┘          └───────────────────────┘

        Symmetric expansion             Symmetric contraction

    Args:
        window (rasterio.windows.Window): Input window to resize.
            Format: Window(col_off, row_off, width, height).
        size (Tuple[int, int]): Target dimensions as (height, width).
            Note: Order is (height, width) matching numpy array convention,
            NOT (width, height) like Window constructor.

    Returns:
        rasterio.windows.Window: Resized window with:
            - width = size[1] (target width)
            - height = size[0] (target height)
            - Center aligned with original window center

    Examples:
        >>> import rasterio.windows
        >>>
        >>> # Expand 100×100 window to 256×256 (for CNN)
        >>> window = rasterio.windows.Window(500, 500, 100, 100)
        >>> expanded = pad_window_to_size(window, size=(256, 256))
        >>> print(expanded)
        Window(col_off=422, row_off=422, width=256, height=256)
        >>> # Offset moved: (500 - 78, 500 - 78) = (422, 422)
        >>> # 78 = (256 - 100) / 2
        >>>
        >>> # Shrink large window to fixed size (center crop)
        >>> large_window = rasterio.windows.Window(0, 0, 1000, 800)
        >>> cropped = pad_window_to_size(large_window, size=(512, 512))
        >>> print(cropped)
        Window(col_off=244, row_off=144, width=512, height=512)
        >>> # Center crop: offset = (1000-512)/2, (800-512)/2
        >>>
        >>> # Asymmetric adjustment
        >>> window = rasterio.windows.Window(100, 100, 80, 120)
        >>> resized = pad_window_to_size(window, size=(100, 100))
        >>> print(resized)
        Window(col_off=90, row_off=110, width=100, height=100)
        >>> # Height shrinks by 20 (10 each side)
        >>> # Width expands by 20 (10 each side)

    Note:
        - Padding is distributed equally when possible; odd differences favor bottom/right
        - Output window may have negative offsets (use boundless reading)
        - For truly symmetric padding, use `pad_window` with explicit pad_size

    See Also:
        - `pad_window`: Add fixed padding amounts
        - `get_slice_pad`: Handle out-of-bounds with slicing + padding
    """
    pad_add_rows = size[0] - window.height
    pad_add_cols = size[1] - window.width

    pad_rows_half = pad_add_rows // 2
    pad_cols_half = pad_add_cols // 2

    return rasterio.windows.Window(
        window.col_off - pad_cols_half,
        window.row_off - pad_rows_half,
        width=window.width + pad_add_cols,
        height=window.height + pad_add_rows,
    )


def figure_out_transform(
    transform: Optional[rasterio.Affine] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resolution_dst: Optional[Union[float, Tuple[float, float]]] = None,
) -> rasterio.Affine:
    """
    Compute an output transform from combinations of transform, bounds, and resolution.

    Flexible factory function that creates an affine geotransform from various input
    combinations. Useful when you know some parameters but need to derive others,
    such as creating output transforms for reprojection or resampling operations.

    The function handles three main scenarios:

    1. **transform + resolution_dst**: Rescale existing transform to new resolution
    2. **bounds + resolution_dst**: Create new transform from scratch (rectilinear)
    3. **transform + bounds + resolution_dst**: Rescale and shift to new origin

    Input Combinations::

        ┌────────────┬────────┬──────────────┬─────────────────────────────┐
        │ transform  │ bounds │ resolution   │ Result                      │
        ├────────────┼────────┼──────────────┼─────────────────────────────┤
        │ ✓          │ ✗      │ ✗            │ Return unchanged            │
        │ ✓          │ ✗      │ ✓            │ Rescale resolution          │
        │ ✓          │ ✓      │ ✗            │ Shift origin to bounds      │
        │ ✓          │ ✓      │ ✓            │ Rescale + shift             │
        │ ✗          │ ✓      │ ✓            │ Create new rectilinear      │
        │ ✗          │ ✗      │ any          │ ERROR (need bounds)         │
        │ ✗          │ ✓      │ ✗            │ ERROR (need resolution)     │
        └────────────┴────────┴──────────────┴─────────────────────────────┘

    Args:
        transform (Optional[rasterio.Affine]): Base transform to modify. If None,
            creates a new rectilinear (north-up) transform from bounds and resolution.
        bounds (Optional[Tuple[float, float, float, float]]): Geographic extent as
            (minx, miny, maxx, maxy). Used to set the transform origin and/or compute
            window offset. Required if transform is None.
        resolution_dst (Optional[Union[float, Tuple[float, float]]]): Target pixel
            resolution. If float, same resolution in x and y. If tuple, (res_x, res_y).
            Units match CRS (meters for projected, degrees for geographic).
            Required if transform is None.

    Returns:
        rasterio.Affine: Output geotransform with requested resolution and origin.

    Raises:
        AssertionError: If transform is None and bounds is not provided.
        AssertionError: If transform is None and resolution_dst is not provided.

    Examples:
        >>> import rasterio
        >>>
        >>> # Create transform from bounds + resolution (new raster)
        >>> bounds = (-122.5, 37.0, -122.0, 37.5)  # WGS84 degrees
        >>> transform = figure_out_transform(bounds=bounds, resolution_dst=0.001)
        >>> print(f"Origin: ({transform.c}, {transform.f})")
        Origin: (-122.5, 37.5)  # Top-left corner
        >>> print(f"Resolution: {res(transform)}")
        (0.001, 0.001)
        >>>
        >>> # Rescale existing transform to new resolution
        >>> transform_10m = rasterio.Affine(10, 0, 500000, 0, -10, 4500000)
        >>> transform_30m = figure_out_transform(transform=transform_10m, resolution_dst=30)
        >>> print(f"New resolution: {res(transform_30m)}")
        (30.0, 30.0)
        >>>
        >>> # Shift transform to new origin with same resolution
        >>> new_bounds = (600000, 4400000, 700000, 4500000)
        >>> transform_shifted = figure_out_transform(
        ...     transform=transform_10m, bounds=new_bounds
        ... )
        >>> # Origin now at new_bounds top-left
        >>>
        >>> # Full specification: rescale and shift
        >>> transform_out = figure_out_transform(
        ...     transform=transform_10m,
        ...     bounds=(600000, 4400000, 700000, 4500000),
        ...     resolution_dst=20
        ... )

    Note:
        - Resolution is always positive (absolute value taken)
        - For rotated transforms, only resolution is modified (not rotation angle)
        - Origin is placed at (min_x, max_y) following north-up convention
        - Use `from_bounds` directly if you also need to compute shape from bounds

    See Also:
        - `transform_to_resolution_dst`: Rescale resolution only
        - `rasterio.transform.from_bounds`: Create transform with known shape
        - `rasterio.transform.from_origin`: Create transform from origin point
    """
    if resolution_dst is not None:
        if isinstance(resolution_dst, numbers.Number):
            resolution_dst = (abs(resolution_dst), abs(resolution_dst))

    if transform is None:
        assert bounds is not None, "Transform and bounds not provided"
        assert resolution_dst is not None, "Transform and resolution not provided"
        return rasterio.transform.from_origin(
            min(bounds[0], bounds[2]), max(bounds[1], bounds[3]), resolution_dst[0], resolution_dst[1]
        )

    if resolution_dst is None:
        dst_transform = transform
    else:
        dst_transform = transform_to_resolution_dst(transform, resolution_dst)

    if bounds is not None:
        # Shift the transform to start in the bounds
        window_current_transform = rasterio.windows.from_bounds(*bounds, transform=transform)
        dst_transform = rasterio.windows.transform(window_current_transform, dst_transform)

    return dst_transform


def transform_to_resolution_dst(
    transform: rasterio.Affine, resolution_dst: Union[float, Tuple[float, float]]
) -> rasterio.Affine:
    """
    Rescale an affine transform to a new spatial resolution.

    This function modifies the pixel size (ground sampling distance) of a geotransform
    while preserving the origin point. Useful for resampling operations where pixel
    resolution changes but geographic origin stays constant.

    The transformation is applied by computing a scale factor between original and
    destination resolutions, then multiplying the transform by a scaling matrix:
        T_dst = T_orig × S
    where S = diag(res_dst/res_orig)

    Args:
        transform (rasterio.Affine): Original affine geotransform matrix.
            Standard form: [a, b, c, d, e, f] where
            - a, e: pixel width and height (may be negative for north-up images)
            - b, d: rotation terms (usually 0 for rectilinear grids)
            - c, f: top-left corner coordinates (x₀, y₀)
        resolution_dst (Union[float, Tuple[float, float]]): Target resolution.
            If float: same resolution in x and y directions (in CRS units).
            If tuple: (res_x, res_y) for anisotropic resolution (in CRS units).
            Units: meters for projected CRS, degrees for geographic CRS.

    Returns:
        rasterio.Affine: New transform with scaled resolution. Origin point (c, f)
            remains unchanged, but pixel dimensions are scaled.

    Examples:
        >>> import rasterio
        >>> # Original 10m resolution transform
        >>> transform_10m = rasterio.Affine(10.0, 0.0, 500000,
        ...                                   0.0, -10.0, 4500000)
        >>> print(f"Original resolution: {res(transform_10m)}")  # (10.0, 10.0)

        >>> # Downsample to 30m resolution
        >>> transform_30m = transform_to_resolution_dst(transform_10m, 30.0)
        >>> print(f"New resolution: {res(transform_30m)}")  # (30.0, 30.0)

        >>> # Anisotropic resolution: 20m × 40m pixels
        >>> transform_aniso = transform_to_resolution_dst(transform_10m, (20.0, 40.0))
        >>> print(f"Anisotropic resolution: {res(transform_aniso)}")  # (20.0, 40.0)

    Note:
        - The scale factor is resolution_dst / resolution_original
        - For upsampling (higher resolution), scale < 1.0
        - For downsampling (lower resolution), scale > 1.0
        - Rotation terms (b, d) are also scaled if present in the original transform
    """
    # Normalize resolution to tuple format for consistent processing
    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))

    # Extract current resolution from transform: (res_x, res_y)
    resolution_or = res(transform)

    # Compute scale factors: scale_x = res_dst_x / res_orig_x
    # This determines how much to stretch/shrink each pixel dimension
    transform_scale = rasterio.Affine.scale(resolution_dst[0] / resolution_or[0], resolution_dst[1] / resolution_or[1])

    # Apply scaling: T_new = T_orig × S
    # Matrix multiplication scales pixel dimensions while preserving origin
    return transform * transform_scale


def round_outer_window(window: rasterio.windows.Window, precision=PIXEL_PRECISION) -> rasterio.windows.Window:
    """
    Round a window to the nearest outer (larger) integer pixel coordinates.

    This function expands floating-point window coordinates outward to integer values,
    ensuring the rounded window completely contains the original window. Essential for
    windowed reading operations where partial pixels cannot be read.

    The rounding strategy:
    - Floor the offsets (row_off, col_off) → smaller values, shifts window up/left
    - Ceiling the endpoints → larger values, shifts window down/right
    - Result: expanded window that fully encompasses the original

    Precision parameter handles floating-point arithmetic errors (e.g., 3.0000001 → 3.0).

    Args:
        window (rasterio.windows.Window): Input window with potentially floating-point
            coordinates. Window format: (col_off, row_off, width, height).
        precision (int, optional): Number of decimal places for pre-rounding to handle
            floating-point errors. Defaults to PIXEL_PRECISION (3).
            Example: 3.9999999 rounded to 3 decimals → 4.0 before ceiling.

    Returns:
        rasterio.windows.Window: Window with integer pixel coordinates, guaranteed to
            be equal or larger than the input window.
            - Offsets: floor(row_off), floor(col_off)
            - Dimensions: calculated to reach ceiling endpoints

    Examples:
        >>> # Window with floating-point coordinates from geometric calculation
        >>> window_float = rasterio.windows.Window(10.3, 20.7, 100.5, 50.2)
        >>> window_int = round_outer_window(window_float)
        >>> print(window_int)
        Window(col_off=10, row_off=20, width=101, height=51)
        >>> # Note: width = ceil(10.3 + 100.5) - floor(10.3) = 111 - 10 = 101

        >>> # Handle floating-point precision errors
        >>> window_precise = rasterio.windows.Window(5.0000001, 10.9999999, 20.0, 30.0)
        >>> window_rounded = round_outer_window(window_precise, precision=3)
        >>> print(window_rounded)
        Window(col_off=5, row_off=11, width=20, height=30)

    Note:
        - Always produces integer coordinates suitable for raster I/O
        - Guarantees: rounded_window.area >= original_window.area
        - Pre-rounding to `precision` decimals prevents 3.9999→3 but keeps 3.001→4
        - Critical for coordinate system transformations that produce near-integer values
    """
    # Calculate the endpoints (right and bottom edges) in floating-point
    # These mark where the window ends before rounding
    row_dst = math.ceil(round(window.row_off + window.height, ndigits=precision))
    col_dst = math.ceil(round(window.col_off + window.width, ndigits=precision))

    # Floor the starting coordinates (top-left corner) to expand outward
    col_off = math.floor(round(window.col_off, ndigits=precision))
    row_off = math.floor(round(window.row_off, ndigits=precision))

    # Construct window with integer coordinates
    # Width/height computed as: ceiling(end) - floor(start)
    return rasterio.windows.Window(col_off, row_off, col_dst - col_off, row_dst - row_off)


# Precision to round the windows before applying ceiling/floor. e.g. 3.0001 will be rounded to 3 but 3.001 will not
def _is_exact_round(x, precision=PIXEL_PRECISION):
    return abs(round(x, ndigits=precision) - x) < 1e-6


def get_slice_pad(
    window_data: rasterio.windows.Window, window_read: rasterio.windows.Window
) -> Tuple[Dict[str, slice], Dict[str, Tuple[int, int]]]:
    """
    Compute slice and padding parameters for boundless reading from a window.

    This function solves the problem of reading a window that extends beyond the
    available data boundaries. It decomposes the request into:
    1. A slice to extract existing data within bounds
    2. Padding amounts to fill out-of-bounds regions

    Common use case: Reading chips near image edges for CNN inference, where you
    need fixed-size inputs (e.g., 512×512) but the requested window extends beyond
    the raster. This function tells you which part to read and how much to pad.

    The algorithm handles four edge cases:
    - Left edge: window_read.col_off < 0
    - Top edge: window_read.row_off < 0
    - Right edge: window_read.col_off + width > data.width
    - Bottom edge: window_read.row_off + height > data.height

    Coordinate system:
    - window_data starts at (0, 0) covering the entire raster extent
    - window_read may have negative offsets or extend beyond data dimensions
    - Slices are relative to window_data (starting at 0)

    Args:
        window_data (rasterio.windows.Window): Window representing the full raster extent.
            Typically: Window(col_off=0, row_off=0, width=raster_width, height=raster_height).
        window_read (rasterio.windows.Window): Desired read window, potentially extending
            beyond window_data boundaries. May have negative offsets or exceed dimensions.

    Returns:
        Tuple[Dict[str, slice], Dict[str, Tuple[int, int]]]: Two dictionaries for slicing
            and padding operations:

            slice_dict: Defines which portion of the data to extract.
                Format: {"x": slice(col_start, col_end), "y": slice(row_start, row_end)}
                Compatible with xarray.DataArray.isel() or numpy array slicing.
                Indices are relative to window_data (0-based).

            pad_width: Defines how much padding to add on each side.
                Format: {"x": (pad_left, pad_right), "y": (pad_top, pad_bottom)}
                Compatible with xarray.DataArray.pad() or numpy.pad().
                Units: pixels.

    Raises:
        rasterio.windows.WindowError: If window_data and window_read do not intersect
            at all (completely disjoint).

    Examples:
        >>> # Case 1: Window fully within data (no padding needed)
        >>> window_data = rasterio.windows.Window(0, 0, width=1000, height=1000)
        >>> window_read = rasterio.windows.Window(100, 100, width=200, height=200)
        >>> slice_dict, pad_width = get_slice_pad(window_data, window_read)
        >>> print(slice_dict)
        {'x': slice(100, 300), 'y': slice(100, 300)}
        >>> print(pad_width)
        {'x': (0, 0), 'y': (0, 0)}
        >>> # Read data[100:300, 100:300], no padding needed

        >>> # Case 2: Window extends beyond left and top edges
        >>> window_read = rasterio.windows.Window(-50, -30, width=200, height=200)
        >>> slice_dict, pad_width = get_slice_pad(window_data, window_read)
        >>> print(slice_dict)
        {'x': slice(0, 150), 'y': slice(0, 170)}
        >>> print(pad_width)
        {'x': (50, 0), 'y': (30, 0)}
        >>> # Read data[0:170, 0:150], then pad 30 rows top, 50 cols left
        >>> # Shape check: 150 + 50 = 200 ✓, 170 + 30 = 200 ✓

        >>> # Case 3: Window extends beyond right and bottom edges
        >>> window_read = rasterio.windows.Window(900, 850, width=200, height=200)
        >>> slice_dict, pad_width = get_slice_pad(window_data, window_read)
        >>> print(slice_dict)
        {'x': slice(900, 1000), 'y': slice(850, 1000)}
        >>> print(pad_width)
        {'x': (0, 100), 'y': (0, 50)}
        >>> # Read data[850:1000, 900:1000], then pad 50 rows bottom, 100 cols right
        >>> # Shape check: 100 + 100 = 200 ✓, 150 + 50 = 200 ✓

        >>> # Case 4: Window extends on all sides
        >>> window_read = rasterio.windows.Window(-10, -20, width=1050, height=1080)
        >>> slice_dict, pad_width = get_slice_pad(window_data, window_read)
        >>> print(slice_dict)
        {'x': slice(0, 1000), 'y': slice(0, 1000)}
        >>> print(pad_width)
        {'x': (10, 40), 'y': (20, 60)}
        >>> # Read entire data[0:1000, 0:1000], pad 20 top, 60 bottom, 10 left, 40 right

        >>> # Usage with xarray
        >>> import xarray as xr
        >>> data_array = xr.DataArray(np.random.rand(1000, 1000), dims=['y', 'x'])
        >>> slice_dict, pad_width = get_slice_pad(window_data, window_read)
        >>> sliced = data_array.isel(slice_dict)  # Extract intersecting portion
        >>> padded = sliced.pad(pad_width, constant_values=0)  # Fill out-of-bounds

    Note:
        - Ensures output shape matches window_read dimensions after slicing + padding
        - Slice indices are always non-negative and within window_data bounds
        - Padding values are always non-negative
        - Left/top padding: window_read starts before window_data
        - Right/bottom padding: window_read ends after window_data
        - Works with any dimension names that have "x" and "y" axes
        - Compatible with numpy arrays, xarray DataArrays, and torch tensors
    """
    # Verify windows intersect (at least partially overlap)
    if not rasterio.windows.intersect([window_data, window_read]):
        raise rasterio.windows.WindowError(
            f"Window data: {window_data} and window read: {window_read} do not intersect"
        )

    # Compute vertical (row/y) slice and padding
    # Case: window_read starts above window_data (negative row offset relative to data)
    if window_read.row_off < window_data.row_off:
        # Need top padding to fill gap before data starts
        pad_y_0 = window_data.row_off - window_read.row_off
        # Slice starts at data's first row
        window_row_start = window_data.row_off
    else:
        pad_y_0 = 0  # No top padding needed
        # Slice starts at window_read position (relative to window_data)
        window_row_start = window_read.row_off - window_data.row_off

    # Compute horizontal (col/x) slice and padding
    # Case: window_read starts left of window_data (negative col offset relative to data)
    if window_read.col_off < window_data.col_off:
        # Need left padding to fill gap before data starts
        pad_x_0 = window_data.col_off - window_read.col_off
        # Slice starts at data's first column
        window_col_start = window_data.col_off
    else:
        pad_x_0 = 0  # No left padding needed
        # Slice starts at window_read position (relative to window_data)
        window_col_start = window_read.col_off - window_data.col_off

    # Compute right edge: does window_read extend beyond data width?
    # window_read right edge = col_off + width
    # window_data right edge = col_off + width
    if (window_read.width + window_read.col_off) > (window_data.width + window_data.col_off):
        # Need right padding to fill gap beyond data
        pad_x_1 = (window_read.width + window_read.col_off) - (window_data.width + window_data.col_off)
        # Slice ends at data's last column
        window_col_end = window_data.width + window_data.col_off
    else:
        pad_x_1 = 0  # No right padding needed
        # Slice ends at window_read's right edge
        window_col_end = window_read.width + window_read.col_off

    # Compute bottom edge: does window_read extend beyond data height?
    if (window_read.height + window_read.row_off) > (window_data.height + window_data.row_off):
        # Need bottom padding to fill gap beyond data
        pad_y_1 = (window_read.height + window_read.row_off) - (window_data.height + window_data.row_off)
        # Slice ends at data's last row
        window_row_end = window_data.height + window_data.row_off
    else:
        pad_y_1 = 0  # No bottom padding needed
        # Slice ends at window_read's bottom edge
        window_row_end = window_read.height + window_read.row_off

    # Construct slice objects for array indexing
    # Note: window_row/col_start and _end are relative to window_data origin (0, 0)
    row_slice = slice(window_row_start, window_row_end)
    col_slice = slice(window_col_start, window_col_end)

    # Package results in named dictionaries
    # Keys match dimension names in xarray ("x" for columns, "y" for rows)
    slice_dict = {"x": col_slice, "y": row_slice}
    # Padding format: (before, after) for each dimension
    pad_width = {"x": (pad_x_0, pad_x_1), "y": (pad_y_0, pad_y_1)}

    return slice_dict, pad_width


def window_polygon(
    window: rasterio.windows.Window, transform: rasterio.Affine, window_surrounding: bool = False
) -> Polygon:
    """
    Convert a pixel window to a geographic polygon.

    Creates a Shapely Polygon representing the spatial extent of a window by
    transforming the four corner pixels to geographic coordinates. This is
    useful for spatial queries, visualization, and intersection operations.

    Similar to `rasterio.windows.bounds`, but returns a full Polygon object
    rather than a bounding box tuple, which is essential for:
    - Intersection tests with other geometries
    - Visualization on maps
    - Spatial joins with vector data
    - Handling rotated/skewed transforms (where bounds != rectangle)

    The `window_surrounding` parameter controls whether the polygon surrounds
    pixel centers or pixel boundaries::

        window_surrounding=False (default):     window_surrounding=True:
        Polygon includes full pixels             Polygon passes through pixel centers

        ┌───┬───┬───┬───┐                       ┌───┬───┬───┬───┐
        │ P │ P │ P │ P │ ◄─ Polygon            │ · │ · │ · │ · │
        ├───┼───┼───┼───┤    edges              ├───○───○───○───┤
        │ P │ P │ P │ P │    touch              │ · │ · │ · │ · │ ◄─ Polygon
        ├───┼───┼───┼───┤    pixel              ├───○───○───○───┤    passes
        │ P │ P │ P │ P │    boundaries         │ · │ · │ · │ · │    through ○
        └───┴───┴───┴───┘                       └───┴───┴───┴───┘

    Args:
        window (rasterio.windows.Window): Pixel window to convert.
            Format: Window(col_off, row_off, width, height).
        transform (rasterio.Affine): Affine geotransform matrix mapping
            pixel coordinates to geographic coordinates.
        window_surrounding (bool, optional): If True, polygon vertices are
            at pixel centers rather than pixel corners. This shrinks the
            polygon by 1 pixel on the right and bottom edges.
            Defaults to False (polygon at pixel boundaries).

    Returns:
        Polygon: Shapely Polygon with 5 vertices (4 corners + closing vertex)
            in geographic coordinates. Coordinates are in the CRS defined
            by the transform.

    Examples:
        >>> import rasterio
        >>> import rasterio.windows
        >>>
        >>> # 10m resolution UTM transform
        >>> transform = rasterio.Affine(10.0, 0.0, 500000.0,
        ...                              0.0, -10.0, 4500000.0)
        >>> # Window: 100x50 pixels starting at (10, 20)
        >>> window = rasterio.windows.Window(10, 20, 100, 50)
        >>>
        >>> # Get geographic polygon
        >>> polygon = window_polygon(window, transform)
        >>> print(polygon.bounds)  # (xmin, ymin, xmax, ymax)
        (500100.0, 4499500.0, 501100.0, 4499800.0)
        >>> print(polygon.area)  # 1000m × 500m = 500000 m²
        500000.0
        >>>
        >>> # Use for intersection test
        >>> from shapely.geometry import Point
        >>> point = Point(500500, 4499600)  # Point within window
        >>> print(polygon.contains(point))
        True
        >>>
        >>> # Surrounding mode (vertices at pixel centers)
        >>> polygon_surr = window_polygon(window, transform, window_surrounding=True)
        >>> # 1 pixel smaller on right/bottom: 99×49 pixels worth of area

    Note:
        - For rectilinear transforms: polygon is a rectangle
        - For rotated transforms: polygon is a parallelogram
        - Use `window_bounds` if you only need (xmin, ymin, xmax, ymax)
        - Polygon is closed: first vertex == last vertex
    """
    row_off = window.row_off
    col_off = window.col_off
    row_max = row_off + window.height
    col_max = col_off + window.width
    if window_surrounding:
        row_max -= 1
        col_max -= 1

    polygon_idx = [(col_off, row_off), (col_off, row_max), (col_max, row_max), (col_max, row_off), (col_off, row_off)]

    return Polygon([transform * coord for coord in polygon_idx])


def window_bounds(window: rasterio.windows.Window, transform: rasterio.Affine) -> Tuple[float, float, float, float]:
    """
    Compute the spatial bounding box of a window in geographic coordinates.

    This function calculates the geographic extent of a raster window by transforming
    its pixel corners to geographic coordinates and finding the axis-aligned bounding
    box. Unlike rasterio's built-in bounds function, this implementation correctly
    handles non-rectilinear (rotated/skewed) transforms by checking all four corners.

    For rectilinear transforms (most common case), the corners define a rectangle in
    geographic space. For rotated/skewed transforms, the window becomes a parallelogram,
    and this function returns its axis-aligned bounding box (AABB) that fully contains it.

    The transformation process:
    1. Identify window corners in pixel space: (col_off, row_off), (col_off+width, row_off+height)
    2. Transform all four corners to geographic space: (x, y) = T × (col, row)
    3. Find min/max across all corners to create axis-aligned bounds

    Args:
        window (rasterio.windows.Window): Window defining the pixel region.
            Format: Window(col_off, row_off, width, height).
        transform (rasterio.Affine): Affine geotransform matrix mapping pixel to
            geographic coordinates. Standard form: [a, b, c, d, e, f] where:
            - (a, e): pixel width and height (may be negative)
            - (b, d): rotation/shear terms
            - (c, f): top-left corner coordinates

    Returns:
        Tuple[float, float, float, float]: Geographic bounding box as (xmin, ymin, xmax, ymax).
            - For projected CRS: typically (easting_min, northing_min, easting_max, northing_max)
            - For geographic CRS: (longitude_min, latitude_min, longitude_max, latitude_max)
            - Units match the CRS units (meters for UTM, degrees for WGS84, etc.)

    Examples:
        >>> import rasterio.windows
        >>> import rasterio.transform
        >>>
        >>> # Rectilinear transform: 10m resolution, origin at (500000, 4500000) UTM
        >>> transform = rasterio.Affine(10.0, 0.0, 500000.0,
        ...                              0.0, -10.0, 4500000.0)
        >>> window = rasterio.windows.Window(10, 20, 100, 50)
        >>> bounds = window_bounds(window, transform)
        >>> print(f"Bounds: {bounds}")
        (500100.0, 4499500.0, 501100.0, 4499800.0)
        >>> # Width: 100 pixels × 10m = 1000m; Height: 50 pixels × 10m = 500m

        >>> # Rotated transform (15° counterclockwise rotation)
        >>> import math
        >>> angle = math.radians(15)
        >>> transform_rotated = rasterio.Affine(
        ...     10 * math.cos(angle), -10 * math.sin(angle), 500000,
        ...     10 * math.sin(angle), 10 * math.cos(angle), 4500000
        ... )
        >>> window = rasterio.windows.Window(0, 0, 100, 100)
        >>> bounds = window_bounds(window, transform_rotated)
        >>> # Bounds will be slightly larger than 100×100 due to rotation

        >>> # Compare with rasterio's built-in (only works for rectilinear)
        >>> bounds_rasterio = rasterio.windows.bounds(window, transform)
        >>> bounds_custom = window_bounds(window, transform)
        >>> assert bounds_rasterio == bounds_custom  # Should match for rectilinear

    Note:
        - For rectilinear transforms: produces exact bounds
        - For rotated/skewed transforms: produces axis-aligned bounding box (may be
          slightly larger than the actual window extent)
        - Critical for reprojection operations where windows may become rotated
        - Handles both north-up (e < 0) and south-up (e > 0) orientations
    """
    # Pixel coordinates of window corners
    # Upper-left corner
    row_min = window.row_off
    col_min = window.col_off
    # Lower-right corner
    row_max = row_min + window.height
    col_max = col_min + window.width

    # Transform all four corners from pixel to geographic coordinates
    # T × (col, row) → (x, y)
    corner_00 = transform * (col_min, row_min)  # Top-left
    corner_01 = transform * (col_min, row_max)  # Bottom-left
    corner_10 = transform * (col_max, row_min)  # Top-right
    corner_11 = transform * (col_max, row_max)  # Bottom-right
    all_corners = [corner_00, corner_01, corner_10, corner_11]

    # Find axis-aligned bounding box containing all transformed corners
    # For rotated transforms, this creates AABB around parallelogram
    return (
        min(c[0] for c in all_corners),  # xmin
        min(c[1] for c in all_corners),  # ymin
        max(c[0] for c in all_corners),  # xmax
        max(c[1] for c in all_corners),  # ymax
    )


def normalize_bounds(
    bounds: Tuple[float, float, float, float], margin_add_if_equal: float = 0.0005
) -> Tuple[float, float, float, float]:
    """
    Normalize and validate bounding box coordinates to ensure proper geometry.

    This function ensures that bounds define a valid rectangle by:
    1. Correcting inverted coordinates (swapping if min > max)
    2. Adding a small margin for degenerate cases (point or line geometries)

    Bounding boxes can become degenerate in several scenarios:
    - Single-pixel windows where width or height is effectively zero
    - Coordinate transformations that collapse a dimension
    - Floating-point precision issues causing min == max

    The function prevents invalid geometries by ensuring xmin < xmax and ymin < ymax,
    which is required by most GIS operations (reprojection, intersection, etc.).

    Args:
        bounds (Tuple[float, float, float, float]): Input bounding box as
            (left, bottom, right, top) or (xmin, ymin, xmax, ymax).
            Coordinates may be in any order (will be corrected).
        margin_add_if_equal (float, optional): Margin to add when coordinates are
            equal or inverted, creating a small valid rectangle. Units match the
            CRS units (typically meters or degrees). Defaults to 0.0005.
            - For WGS84: ~50m at equator
            - For UTM: 0.5mm (effectively zero)

    Returns:
        Tuple[float, float, float, float]: Normalized bounds as (xmin, ymin, xmax, ymax)
            guaranteed to satisfy xmin < xmax and ymin < ymax.

    Examples:
        >>> # Normal bounds (no correction needed)
        >>> bounds = (10.0, 20.0, 30.0, 40.0)
        >>> normalized = normalize_bounds(bounds)
        >>> print(normalized)
        (10.0, 20.0, 30.0, 40.0)

        >>> # Inverted bounds (min/max swapped)
        >>> bounds_inverted = (30.0, 40.0, 10.0, 20.0)
        >>> normalized = normalize_bounds(bounds_inverted)
        >>> print(normalized)
        (10.0, 20.0, 30.0, 40.0)

        >>> # Degenerate bounds (point geometry) - adds margin
        >>> bounds_point = (100.0, 200.0, 100.0, 200.0)
        >>> normalized = normalize_bounds(bounds_point, margin_add_if_equal=1.0)
        >>> print(normalized)
        (99.0, 199.0, 101.0, 201.0)
        >>> # Creates 2m × 2m rectangle around point

        >>> # Single-pixel window in geographic coordinates
        >>> bounds_pixel = (-3.7038, 40.4168, -3.7038, 40.4168)
        >>> normalized = normalize_bounds(bounds_pixel)  # Uses default margin
        >>> # Creates ~50m × 50m rectangle at equator

    Note:
        - Essential for preventing errors in reprojection and spatial operations
        - The margin is symmetric: ±margin_add_if_equal from the degenerate coordinate
        - Does not validate CRS or coordinate ranges (allows any numeric values)
        - Useful after coordinate transformations that may collapse geometries
    """
    # Ensure min <= max by taking min/max of potentially inverted coordinates
    xmin = min(bounds[0], bounds[2])
    ymin = min(bounds[1], bounds[3])
    xmax = max(bounds[0], bounds[2])
    ymax = max(bounds[1], bounds[3])

    # Handle degenerate x-dimension (point or vertical line)
    # Comparison uses >= to catch floating-point equality
    if xmin >= xmax:
        xmin -= margin_add_if_equal  # Expand left
        xmax += margin_add_if_equal  # Expand right

    # Handle degenerate y-dimension (point or horizontal line)
    if ymin >= ymax:
        ymin -= margin_add_if_equal  # Expand bottom
        ymax += margin_add_if_equal  # Expand top

    return xmin, ymin, xmax, ymax


def polygon_to_crs(
    polygon: Union[Polygon, MultiPolygon], crs_polygon: Any, dst_crs: Any
) -> Union[Polygon, MultiPolygon]:
    """
    Reproject a Shapely geometry from one coordinate reference system to another.

    This function transforms polygon geometries between different coordinate systems,
    handling both simple Polygons and MultiPolygons. It's essential for spatial
    operations involving data in different CRSs, such as:
    - Reading raster data within a polygon defined in a different CRS
    - Overlaying vector and raster data with different projections
    - Converting between geographic (WGS84) and projected (UTM, Web Mercator) coordinates

    The transformation uses rasterio's warp.transform_geom, which leverages GDAL's
    reprojection engine for accurate coordinate transformations. The function preserves
    geometry topology (no simplification or densification is applied).

    Args:
        polygon (Union[Polygon, MultiPolygon]): Shapely geometry to transform.
            Can be a simple Polygon or MultiPolygon. Coordinates must be in the
            CRS specified by `crs_polygon`.
        crs_polygon (Any): Source coordinate reference system of the input polygon.
            Accepts various formats:
            - EPSG code as string: "EPSG:4326"
            - CRS object: rasterio.crs.CRS or pyproj.CRS
            - WKT string, PROJ string, or dictionary
        dst_crs (Any): Destination coordinate reference system for output.
            Same format options as `crs_polygon`.

    Returns:
        Union[Polygon, MultiPolygon]: Transformed geometry in the destination CRS.
            Returns the same type as input (Polygon → Polygon, MultiPolygon → MultiPolygon).
            If source and destination CRS are the same, returns the original polygon
            unchanged (no-op transformation).

    Examples:
        >>> from shapely.geometry import box
        >>> # WGS84 rectangle around Madrid
        >>> polygon_wgs84 = box(-3.71, 40.41, -3.69, 40.42)
        >>> print(f"WGS84 bounds: {polygon_wgs84.bounds}")
        (-3.71, 40.41, -3.69, 40.42)

        >>> # Transform to Web Mercator (EPSG:3857)
        >>> polygon_mercator = polygon_to_crs(polygon_wgs84, "EPSG:4326", "EPSG:3857")
        >>> print(f"Mercator bounds: {polygon_mercator.bounds}")
        # Output in meters: (~-413000, ~4895000, ~-411000, ~4897000)

        >>> # Transform to UTM Zone 30N (appropriate for Madrid)
        >>> polygon_utm = polygon_to_crs(polygon_wgs84, "EPSG:4326", "EPSG:32630")
        >>> print(f"UTM bounds: {polygon_utm.bounds}")
        # Output in meters: (~437000, ~4474000, ~439000, ~4476000)

        >>> # MultiPolygon transformation
        >>> from shapely.geometry import MultiPolygon
        >>> multi = MultiPolygon([polygon_wgs84, box(-3.75, 40.45, -3.73, 40.46)])
        >>> multi_utm = polygon_to_crs(multi, "EPSG:4326", "EPSG:32630")
        >>> print(len(multi_utm.geoms))  # Still has 2 polygons
        2

        >>> # No-op for same CRS (returns original)
        >>> polygon_same = polygon_to_crs(polygon_wgs84, "EPSG:4326", "EPSG:4326")
        >>> assert polygon_same is polygon_wgs84

    Note:
        - Uses GDAL transformation under the hood via rasterio
        - Preserves vertex count (no densification along edges)
        - For large transformations (e.g., polar regions), edges may not be geodesic
        - Coordinate order follows CRS convention (lon/lat for WGS84, x/y for projected)
        - Empty geometries are preserved (empty → empty)
        - Automatically handles CRS aliases (e.g., "EPSG:4326" == "WGS84")
    """
    # Early return if CRSs are equivalent (compare handles aliases)
    if compare_crs(crs_polygon, dst_crs):
        return polygon

    # Shapely → GeoJSON → transform → GeoJSON → Shapely pipeline
    # mapping() converts Shapely to GeoJSON dict
    # transform_geom() reprojects coordinates in GeoJSON format
    # shape() converts GeoJSON dict back to Shapely
    return shape(rasterio.warp.transform_geom(crs_polygon, dst_crs, mapping(polygon)))


def exterior_pixel_coords(
    transform: rasterio.Affine, crs: Any, polygon: Union[Polygon, MultiPolygon], crs_polygon: Optional[str] = None
) -> List[List[Tuple[float, float]]]:
    """
    Convert polygon exterior vertices from geographic to pixel coordinates.

    This function transforms the boundary vertices of one or more polygons from
    geographic (real-world) coordinates to pixel (array index) coordinates. It's
    essential for rasterization, masking, and window extraction operations where
    you need to know which pixels fall within a polygon.

    The transformation pipeline:
    1. Reproject polygon to raster's CRS (if needed)
    2. For each polygon, extract exterior ring vertices (excluding holes)
    3. Apply inverse transform: (x, y) = T⁻¹ × (geo_x, geo_y) → (col, row)
    4. Return as nested lists: one list per polygon

    Handles both simple Polygons and MultiPolygons (collections of disjoint polygons).
    Only exterior boundaries are returned; interior holes are ignored.

    Args:
        transform (rasterio.Affine): Affine geotransform mapping pixel to geographic
            coordinates. The inverse (~transform) maps geographic to pixel coordinates.
        crs (Any): Coordinate reference system of the transform/raster.
            Format: EPSG string ("EPSG:32630"), CRS object, or WKT.
        polygon (Union[Polygon, MultiPolygon]): Shapely geometry whose vertices
            will be transformed to pixel coordinates.
        crs_polygon (Optional[str], optional): CRS of the input polygon. If None,
            assumes polygon is already in the same CRS as the raster. Defaults to None.

    Returns:
        List[List[Tuple[float, float]]]: Nested list of pixel coordinates.
            - Outer list: one entry per polygon (length 1 for Polygon, N for MultiPolygon)
            - Inner list: vertices of that polygon's exterior ring
            - Tuple: (col, row) pixel coordinates as floats (may be fractional)
            Format: [[(col₀, row₀), (col₁, row₁), ...], ...]

    Raises:
        NotImplementedError: If polygon is neither Polygon nor MultiPolygon
            (e.g., LineString, Point).

    Examples:
        >>> import rasterio
        >>> from shapely.geometry import box, MultiPolygon
        >>>
        >>> # Define raster geotransform: 10m resolution UTM
        >>> transform = rasterio.Affine(10.0, 0.0, 500000.0,
        ...                              0.0, -10.0, 4500000.0)
        >>> crs = "EPSG:32630"  # UTM Zone 30N
        >>>
        >>> # Simple rectangle polygon in same CRS
        >>> polygon = box(500100, 4499500, 500300, 4499700)  # 200m × 200m
        >>> coords = exterior_pixel_coords(transform, crs, polygon)
        >>> print(len(coords))  # One polygon
        1
        >>> print(coords[0])  # Five vertices (closed ring)
        [(10.0, 30.0), (10.0, 50.0), (30.0, 50.0), (30.0, 30.0), (10.0, 30.0)]
        >>> # Top-left at pixel (10, 30), size 20×20 pixels

        >>> # Polygon in different CRS (WGS84)
        >>> polygon_wgs84 = box(-3.71, 40.41, -3.69, 40.42)
        >>> coords = exterior_pixel_coords(transform, crs, polygon_wgs84,
        ...                                 crs_polygon="EPSG:4326")
        >>> # Automatically reprojects WGS84 → UTM before converting to pixels

        >>> # MultiPolygon: two disjoint rectangles
        >>> poly1 = box(500100, 4499500, 500200, 4499600)
        >>> poly2 = box(500300, 4499700, 500400, 4499800)
        >>> multi_poly = MultiPolygon([poly1, poly2])
        >>> coords = exterior_pixel_coords(transform, crs, multi_poly)
        >>> print(len(coords))  # Two polygons
        2
        >>> print(len(coords[0]), len(coords[1]))  # Each has 5 vertices (closed)
        5 5

        >>> # Use for masking operation
        >>> from rasterio.features import rasterize
        >>> # Convert pixel coordinates to closed polygon for rasterization
        >>> # Note: Need to swap (col, row) → (x, y) for some operations

    Note:
        - Returns floating-point pixel coordinates (not rounded to integers)
        - Coordinates may be negative or exceed raster dimensions (not clipped)
        - Only exterior rings are returned; holes are ignored
        - For closed rings, first and last vertices are identical
        - Coordinate order: (col, row) matching array indexing convention
        - Use with `window_from_polygon` to get bounding window
    """
    # Reproject polygon to raster's CRS if necessary
    if (crs_polygon is not None) and not compare_crs(crs_polygon, crs):
        # Transform polygon coordinates to match raster CRS
        # See: https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.transform_geom
        polygon_crs_data = polygon_to_crs(polygon, crs_polygon, crs)
    else:
        polygon_crs_data = polygon

    # Normalize to list of polygons (handle both Polygon and MultiPolygon)
    if isinstance(polygon_crs_data, MultiPolygon):
        polygons = polygon_crs_data.geoms  # Extract individual polygons
    elif isinstance(polygon_crs_data, Polygon):
        polygons = [polygon_crs_data]  # Wrap single polygon in list
    else:
        raise NotImplementedError(
            f"Received shape of type {type(polygon_crs_data)} different from {Polygon} or {MultiPolygon}"
        )

    # Compute inverse transform for geographic → pixel conversion
    # ~transform inverts the affine matrix: (x, y) → (col, row)
    transform_inv = ~transform

    # Collect pixel coordinates for each polygon's exterior ring
    coords = []
    for pol in polygons:
        coords_iter = []
        # Iterate over vertices of exterior boundary (excludes interior holes)
        for pcoord in pol.exterior.coords:
            # Transform: (geo_x, geo_y) → (col, row) via T⁻¹ × (x, y)
            coords_iter.append(transform_inv * pcoord)
        coords.append(coords_iter)

    return coords


def row_end(wv: rasterio.windows.Window) -> int:
    return wv.row_off + wv.height


def col_end(wv: rasterio.windows.Window) -> int:
    return wv.col_off + wv.width


def slice_save_for_pred(w_read: rasterio.windows.Window, w_write: rasterio.windows.Window) -> tuple[slice, slice]:
    """
    Compute slice indices to extract valid region from padded CNN prediction.

    This function implements the "tiling and stitching" strategy for large image
    inference, as described in Huang et al. (2018). When applying CNNs to large
    rasters, you typically:
    1. Read overlapping tiles with padding to avoid edge artifacts
    2. Run CNN inference on padded tiles
    3. Extract only the center region (removing padding) for final output
    4. Stitch extracted regions together to form complete prediction

    The function calculates slice indices that extract the valid (unpadded) region
    from a CNN prediction, accounting for asymmetric padding when tiles are near
    image edges.

    Workflow:
    - w_read: Large window with padding (e.g., 512×512 for CNN input)
    - w_write: Smaller window without padding (e.g., 256×256 for output)
    - Result: Slices to extract w_write region from w_read-sized prediction

    Args:
        w_read (rasterio.windows.Window): Window used to read input data (with padding).
            This is the size passed to the CNN for inference.
        w_write (rasterio.windows.Window): Window where output should be written (no padding).
            This is the valid region to extract from the prediction.

    Returns:
        tuple[slice, slice]: Tuple of (row_slice, col_slice) for extracting valid region.
            Format: (slice(row_start, row_end), slice(col_start, col_end))
            Apply as: prediction[row_slice, col_slice] or prediction[:, row_slice, col_slice]

            Slice endpoints may be None (meaning "to the end") when w_write
            extends to the edge of w_read.

    Examples:
        >>> # Example 1: Centered tile with symmetric padding
        >>> w_write = rasterio.windows.Window(100, 100, 256, 256)  # Core region
        >>> w_read = window_utils.pad_window_to_size(w_write, (512, 512))  # Add 128px padding
        >>> # w_read = Window(col_off=-28, row_off=-28, width=512, height=512)
        >>> slice_save = slice_save_for_pred(w_read, w_write)
        >>> print(slice_save)
        (slice(128, -128), slice(128, -128))
        >>> # Extract center 256×256 from 512×512 prediction
        >>> # prediction shape: (512, 512) → extracted: (256, 256)

        >>> # Example 2: Edge tile with asymmetric padding (near left/top edge)
        >>> w_write = rasterio.windows.Window(0, 0, 256, 256)  # Top-left corner
        >>> w_read = window_utils.pad_window_to_size(w_write, (512, 512))
        >>> # w_read = Window(col_off=-128, row_off=-128, width=512, height=512)
        >>> slice_save = slice_save_for_pred(w_read, w_write)
        >>> print(slice_save)
        (slice(128, 384), slice(128, 384))
        >>> # Extract first 256×256 region (offset by padding)

        >>> # Example 3: Complete inference workflow
        >>> from tqdm import tqdm
        >>> from georeader import window_utils, read
        >>>
        >>> # Define tiling strategy: 256×256 tiles with 512×512 inference window
        >>> window_size_predict_nn = 512
        >>> tile_size = 256
        >>>
        >>> # Generate non-overlapping write windows
        >>> windows_write = [
        ...     rasterio.windows.Window(col, row, tile_size, tile_size)
        ...     for row in range(0, image_height, tile_size)
        ...     for col in range(0, image_width, tile_size)
        ... ]
        >>>
        >>> # Process each tile
        >>> for w_write in tqdm(windows_write):
        ...     # Read with padding for context
        ...     w_read = window_utils.pad_window_to_size(
        ...         w_write,
        ...         size=(window_size_predict_nn, window_size_predict_nn)
        ...     )
        ...
        ...     # Get slice to extract valid region after inference
        ...     slice_save = window_utils.slice_save_for_pred(w_read, w_write)
        ...
        ...     # Read data (shape: 512×512)
        ...     data = read.read_from_window(input_tensor, window=w_read,
        ...                                   boundless=True, trigger_load=True)
        ...
        ...     # Run CNN inference (output shape: 512×512)
        ...     prediction = model.predict(data)
        ...
        ...     # Extract valid region (shape: 256×256)
        ...     out = prediction[slice_save]
        ...
        ...     # Write to output mosaic
        ...     output_tensor.write_from_window(out, window=w_write)

    Note:
        - Handles asymmetric padding near image boundaries automatically
        - Uses None for slice endpoints when region extends to edge (pythonic slicing)
        - Negative slice indices indicate "from the end" (e.g., -128 means last 128)
        - Compatible with numpy, PyTorch, and TensorFlow array slicing
        - Ensures seamless mosaicking with no gaps or overlaps
        - Padding strategy avoids edge artifacts in CNN predictions

    References:
        Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2018).
        "Densely connected convolutional networks." CVPR 2017.
        Describes tiling and stitching for large image inference.
    """
    # Compute row offset: where w_write starts relative to w_read
    # Positive value = padding at top
    row_off_slice_pred = w_write.row_off - w_read.row_off

    # Compute column offset: where w_write starts relative to w_read
    # Positive value = padding at left
    col_off_slice_pred = w_write.col_off - w_read.col_off

    # Compute row end offset: where w_write ends relative to w_read
    # Negative value = padding at bottom; 0 = no padding at bottom
    row_end_slice_pred = row_end(w_write) - row_end(w_read)

    # Compute column end offset: where w_write ends relative to w_read
    # Negative value = padding at right; 0 = no padding at right
    col_end_slice_pred = col_end(w_write) - col_end(w_read)

    # Construct slice objects
    # If end offset is 0, use None to mean "to the end" (more pythonic)
    # If end offset is negative, use it to slice from the end (e.g., -128)
    slice_save = (
        slice(row_off_slice_pred, None if row_end_slice_pred == 0 else row_end_slice_pred),
        slice(col_off_slice_pred, None if col_end_slice_pred == 0 else col_end_slice_pred),
    )
    return slice_save


def pad_list_numpy(pad_width: Dict[str, Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Convert geospatial padding dictionary to numpy/scipy-compatible padding list.

    This utility function bridges the gap between geospatial dimension naming
    (using "x" and "y" keys) and numpy's positional padding format. It transforms
    padding specifications from a named dictionary to an ordered list suitable for
    numpy.pad() or scipy.ndimage operations.

    The conversion follows the standard raster dimension order:
    - Index 0 (rows, y-axis, height): first dimension in numpy arrays
    - Index 1 (columns, x-axis, width): second dimension in numpy arrays

    Missing dimensions in the input dictionary are filled with (0, 0) padding.

    Args:
        pad_width (Dict[str, Tuple[int, int]]): Padding specification with dimension
            names as keys. Format: {"x": (pad_left, pad_right), "y": (pad_top, pad_bottom)}
            Each tuple specifies (before, after) padding in pixels.
            Keys "x" and "y" are optional; missing keys result in (0, 0) padding.

    Returns:
        List[Tuple[int, int]]: Padding list in numpy dimension order [y, x].
            Format: [(pad_top, pad_bottom), (pad_left, pad_right)]
            Compatible with numpy.pad() as the `pad_width` parameter.

    Examples:
        >>> # Example 1: Full padding specification
        >>> pad_dict = {"x": (10, 15), "y": (5, 8)}
        >>> pad_list = pad_list_numpy(pad_dict)
        >>> print(pad_list)
        [(5, 8), (10, 15)]
        >>> # Order: [y, x] → [(top, bottom), (left, right)]

        >>> # Use with numpy.pad
        >>> import numpy as np
        >>> arr = np.random.rand(100, 200)  # Shape: (height=100, width=200)
        >>> padded = np.pad(arr, pad_list, mode='constant', constant_values=0)
        >>> print(padded.shape)
        (113, 225)  # (100+5+8, 200+10+15)

        >>> # Example 2: Partial padding (only x dimension)
        >>> pad_dict = {"x": (20, 20)}
        >>> pad_list = pad_list_numpy(pad_dict)
        >>> print(pad_list)
        [(0, 0), (20, 20)]
        >>> # y gets (0, 0) padding by default

        >>> # Example 3: Partial padding (only y dimension)
        >>> pad_dict = {"y": (30, 30)}
        >>> pad_list = pad_list_numpy(pad_dict)
        >>> print(pad_list)
        [(30, 30), (0, 0)]

        >>> # Example 4: Integration with get_slice_pad output
        >>> window_data = rasterio.windows.Window(0, 0, 1000, 1000)
        >>> window_read = rasterio.windows.Window(-50, -30, 200, 200)
        >>> _, pad_width = window_utils.get_slice_pad(window_data, window_read)
        >>> print(pad_width)
        {'x': (50, 0), 'y': (30, 0)}
        >>> pad_list = pad_list_numpy(pad_width)
        >>> print(pad_list)
        [(30, 0), (50, 0)]
        >>> # Ready for np.pad(array, pad_list)

    Note:
        - Output order [y, x] matches numpy array shape (height, width)
        - Input keys are case-sensitive: must be "x" and "y"
        - Missing keys default to (0, 0) - no padding
        - Compatible with numpy.pad(), scipy.ndimage.filters, and similar functions
        - For multi-dimensional arrays (e.g., bands), extend the list:
          [(0, 0), (pad_y_0, pad_y_1), (pad_x_0, pad_x_1)] for (C, H, W) arrays
    """
    pad_list_np = []
    # Iterate in y, x order to match numpy array dimension order
    for k in ["y", "x"]:
        if k in pad_width:
            # Use specified padding for this dimension
            pad_list_np.append(pad_width[k])
        else:
            # Default to no padding if dimension not specified
            pad_list_np.append((0, 0))
    return pad_list_np
