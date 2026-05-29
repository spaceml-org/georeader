"""
Read Module: Window-based raster reading with reprojection and resampling.

This module provides functions to read raster data from various sources using
window-based access patterns. It handles coordinate transformations, reprojection,
and resampling - the core I/O operations for geospatial raster processing.

Reading Workflow Overview
-------------------------

The module supports multiple ways to specify the area of interest::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    READING WORKFLOW: AREA SPECIFICATION                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Input Specification          Function                     Output       │
    │  ────────────────────         ─────────────────────        ──────────   │
    │                                                                          │
    │  Polygon (geometry)     ───►  read_from_polygon()    ───►  GeoTensor   │
    │                                                                          │
    │  Bounds (minx,miny,     ───►  read_from_bounds()     ───►  GeoTensor   │
    │          maxx,maxy)                                                      │
    │                                                                          │
    │  Center + Shape         ───►  read_from_center_coords() ─► GeoTensor   │
    │  (x, y) + (H, W)                                                         │
    │                                                                          │
    │  Window (row_off,       ───►  read_from_window()     ───►  GeoTensor   │
    │          col_off, H, W)                                                  │
    │                                                                          │
    │  Web Tile (x, y, z)     ───►  read_from_tile()       ───►  GeoTensor   │
    │                                                                          │
    │  Match another raster   ───►  read_reproject_like()  ───►  GeoTensor   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Window vs Bounds Coordinates
----------------------------

Understanding the difference between pixel windows and geographic bounds::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │              WINDOW (PIXELS) vs BOUNDS (GEOGRAPHIC COORDINATES)          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  WINDOW (pixel space)                BOUNDS (CRS units)                 │
    │  ─────────────────────               ──────────────────                 │
    │                                                                          │
    │  (col_off, row_off)                  (minx, maxy)  ← upper-left         │
    │       ↓                                   ↓                              │
    │    ┌──────────────┐                  ┌──────────────┐                   │
    │    │ width pixels │                  │              │ geographic        │
    │    │              │   ◄═══════►      │              │ extent in         │
    │    │ height pixels│    transform     │              │ CRS units         │
    │    └──────────────┘                  └──────────────┘                   │
    │                                           ↑                              │
    │                                      (maxx, miny)  ← lower-right        │
    │                                                                          │
    │  Window: rasterio.windows.Window(col_off, row_off, width, height)       │
    │  Bounds: (minx, miny, maxx, maxy) - order matches shapely/rasterio      │
    │                                                                          │
    │  Conversion:                                                             │
    │    bounds = window_utils.window_bounds(window, transform)               │
    │    window = window_from_bounds(data, bounds, crs_bounds)                │
    └─────────────────────────────────────────────────────────────────────────┘

Reprojection & Resampling
-------------------------

When reading data into a different CRS or resolution::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     REPROJECTION WORKFLOW                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Source CRS (e.g., EPSG:4326)         Target CRS (e.g., EPSG:32633)    │
    │  ┌─────────────────────┐              ┌─────────────────────┐           │
    │  │  ╱╲    ╱╲    ╱╲    │              │ □ □ □ □ □ □ □ □ □ │           │
    │  │ ╱  ╲  ╱  ╲  ╱  ╲   │    ═════►    │ □ □ □ □ □ □ □ □ □ │           │
    │  │╱    ╲╱    ╲╱    ╲  │   Reproject  │ □ □ □ □ □ □ □ □ □ │           │
    │  │ Irregular grid     │   + Resample │ Regular UTM grid   │           │
    │  └─────────────────────┘              └─────────────────────┘           │
    │                                                                          │
    │  Resampling Methods (rasterio.warp.Resampling):                         │
    │  ┌────────────────┬────────────────────────────────────────────────┐    │
    │  │ Method         │ Best for                                       │    │
    │  ├────────────────┼────────────────────────────────────────────────┤    │
    │  │ nearest        │ Categorical data, masks, classification        │    │
    │  │ bilinear       │ Continuous data, fast                          │    │
    │  │ cubic          │ Continuous data, smooth                        │    │
    │  │ cubic_spline   │ Continuous data, very smooth (DEFAULT)         │    │
    │  │ lanczos        │ Downsampling, sharp edges                      │    │
    │  │ average        │ Downsampling, area-weighted mean               │    │
    │  │ mode           │ Downsampling categorical data                  │    │
    │  └────────────────┴────────────────────────────────────────────────┘    │
    │                                                                          │
    │  Anti-aliasing: Automatic Gaussian blur before downsampling to          │
    │                 prevent aliasing artifacts. Controlled by:              │
    │                 - anti_aliasing=True (default in resize)                │
    │                 - anti_aliasing_sigma (auto-calculated or manual)       │
    └─────────────────────────────────────────────────────────────────────────┘

Boundless Reading
-----------------

Reading outside raster bounds returns fill values::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    BOUNDLESS READING (boundless=True)                    │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Requested Window              Result with boundless=True               │
    │  ─────────────────              ─────────────────────────               │
    │                                                                          │
    │       ┌─────────────┐           ┌─────────────┐                         │
    │       │ fill │ data │           │  0  │ data │   fill_value_default    │
    │       │ ─────┼───── │           │ ────┼───── │   fills out-of-bounds   │
    │       │ fill │ data │           │  0  │ data │   pixels                │
    │       └─────────────┘           └─────────────┘                         │
    │            ↑                                                             │
    │     Request extends                                                      │
    │     beyond raster bounds                                                 │
    │                                                                          │
    │  boundless=False: Raises error or clips to valid region                 │
    │  boundless=True:  Pads with fill_value_default (default behavior)       │
    └─────────────────────────────────────────────────────────────────────────┘

Module Functions Overview
-------------------------

Window Creation:
    - :func:`window_from_polygon`: Polygon geometry → pixel window
    - :func:`window_from_bounds`: Geographic bounds → pixel window
    - :func:`window_from_center_coords`: Center point + shape → pixel window
    - :func:`window_from_tile`: Web mercator tile (x,y,z) → pixel window

Reading Functions:
    - :func:`read_from_window`: Read using pixel window
    - :func:`read_from_polygon`: Read area within polygon
    - :func:`read_from_bounds`: Read area within bounds
    - :func:`read_from_center_coords`: Read centered on point
    - :func:`read_from_tile`: Read web mercator tile

Reprojection:
    - :func:`read_reproject`: Read with CRS transformation
    - :func:`read_reproject_like`: Match another raster's grid
    - :func:`read_to_crs`: Simple CRS conversion
    - :func:`resize`: Change resolution with anti-aliasing

Quick Start
-----------

Read a region by polygon::

    from georeader import read
    from shapely.geometry import box

    # Define area of interest in WGS84
    aoi = box(-122.5, 37.5, -122.0, 38.0)

    # Read from raster (auto-transforms polygon to raster CRS)
    gt = read.read_from_polygon(reader, aoi, crs_polygon="EPSG:4326")

Read and reproject to match another raster::

    # Make data_in match data_like's grid exactly
    gt_aligned = read.read_reproject_like(data_in, data_like)

Read a web map tile::

    # Read tile at zoom 15, coordinates (x=5242, y=12661)
    gt_tile = read.read_from_tile(reader, x=5242, y=12661, z=15)

See Also
--------
georeader.geotensor : GeoTensor class returned by read functions
georeader.window_utils : Lower-level window manipulation utilities
georeader.rasterio_reader : RasterioReader for lazy file access

References
----------
- Rasterio windowed reading: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
- Rasterio reprojection: https://rasterio.readthedocs.io/en/latest/topics/reproject.html
- Web Mercator tiles: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
"""
import itertools
import numbers
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import product
from math import ceil, copysign
from typing import Any, Dict, Optional, Tuple, Union

import mercantile
import numpy as np
import rasterio
import rasterio.crs
import rasterio.features
import rasterio.rpc
import rasterio.transform
import rasterio.warp
import rasterio.windows
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon, box

from georeader import window_utils
from georeader.abstract_reader import GeoData, GeoDataBase
from georeader.geotensor import GeoTensor
from georeader.window_utils import PIXEL_PRECISION, _is_exact_round, pad_window, round_outer_window

SIZE_DEFAULT = 256
WEB_MERCATOR_CRS = "EPSG:3857"


def _round_all(x):
    """
    Internal helper to round all elements in a sequence to nearest integers.

    Args:
        x: Sequence of numeric values to round.

    Returns:
        tuple: Tuple of rounded integer values.
    """
    x = tuple([int(round(xi)) for xi in x])
    return x


def _transform_from_crs(
    center_coords: Tuple[float, float], crs_input: Union[Dict[str, str], str], crs_output: Union[Dict[str, str], str]
) -> Tuple[float, float]:
    """
    Transform a coordinate tuple from one CRS to another.

    Internal helper function for coordinate transformation using rasterio.warp.

    Args:
        center_coords (Tuple[float, float]): Coordinates to transform as (x, y) tuple.
        crs_input (Union[Dict[str, str], str]): Source coordinate reference system.
        crs_output (Union[Dict[str, str], str]): Target coordinate reference system.

    Returns:
        Tuple[float, float]: Transformed coordinates as (x, y) tuple in target CRS.
    """
    coords_transformed = rasterio.warp.transform(crs_input, crs_output, [center_coords[0]], [center_coords[1]])
    return coords_transformed[0][0], coords_transformed[1][0]


def window_from_polygon(
    data_in: Union[GeoDataBase, rasterio.DatasetReader],
    polygon: Union[Polygon, MultiPolygon],
    crs_polygon: Optional[str] = None,
    window_surrounding: bool = False,
) -> rasterio.windows.Window:
    """
    Calculate the raster window that contains a polygon in pixel coordinates.

    This function converts polygon vertices from geographic coordinates to pixel coordinates,
    then creates a window that encompasses all vertices. Useful for extracting raster data
    within a specific geographic area.

    Args:
        data_in (Union[GeoDataBase, rasterio.DatasetReader]): Raster data source with crs
            and transform attributes defining the spatial reference.
        polygon (Union[Polygon, MultiPolygon]): Shapely geometry defining the area of interest.
            Can be a simple Polygon or MultiPolygon for complex areas.
        crs_polygon (Optional[str], optional): Coordinate reference system of the polygon.
            If None, assumes polygon is in the same CRS as `data_in`. Defaults to None.
        window_surrounding (bool, optional): If True, adds a 1-pixel buffer around the polygon
            to ensure complete coverage (i.e., window.row_off + window.height will not be a vertex).
            Defaults to False.

    Returns:
        rasterio.windows.Window: Window object with pixel coordinates (row_off, col_off, height, width)
            relative to `data_in` that encompasses the polygon.

    Examples:
        >>> from shapely.geometry import box
        >>> import rasterio
        >>> # Create a polygon in WGS84
        >>> polygon = box(-3.71, 40.41, -3.69, 40.42)  # Madrid area
        >>> with rasterio.open('image.tif') as src:
        ...     window = window_from_polygon(src, polygon, crs_polygon='EPSG:4326')
        ...     print(f"Window: {window.width}x{window.height} at ({window.col_off}, {window.row_off})")

    Note:
        The window coordinates are in pixel space, not geographic coordinates.
        Use with `read_from_window` to extract the actual data.
    """
    data_in_crs = data_in.crs
    data_in_transform = data_in.transform

    # Convert polygon vertices to pixel coordinates in the raster's CRS
    # This handles CRS transformation if polygon is in a different CRS
    coords_multipol = window_utils.exterior_pixel_coords(
        polygon=polygon, crs_polygon=crs_polygon, crs=data_in_crs, transform=data_in_transform
    )

    # Calculate bounding box in pixel coordinates
    # Find minimum row/col (upper-left corner)
    row_off = min(c[1] for coords in coords_multipol for c in coords)
    col_off = min(c[0] for coords in coords_multipol for c in coords)

    # Find maximum row/col (lower-right corner)
    row_max = max(c[1] for coords in coords_multipol for c in coords)
    col_max = max(c[0] for coords in coords_multipol for c in coords)

    # Add 1-pixel buffer if requested for complete surrounding coverage
    if window_surrounding:
        row_max += 1
        col_max += 1

    # Create window: (col_off, row_off, width, height)
    return rasterio.windows.Window(row_off=row_off, col_off=col_off, width=col_max - col_off, height=row_max - row_off)


def window_from_bounds(
    data_in: Union[GeoDataBase, rasterio.DatasetReader],
    bounds: Tuple[float, float, float, float],
    crs_bounds: Optional[str] = None,
) -> rasterio.windows.Window:
    """
    Calculate the raster window corresponding to geographic bounds.

    This function converts a bounding box from geographic coordinates to pixel coordinates,
    handling CRS transformation if needed. The bounds format follows the standard GIS convention.

    Args:
        data_in (Union[GeoDataBase, rasterio.DatasetReader]): Raster data source with crs
            and transform attributes defining the spatial reference.
        bounds (Tuple[float, float, float, float]): Bounding box as (left, bottom, right, top)
            or (min_x, min_y, max_x, max_y) in the CRS specified by `crs_bounds`.
        crs_bounds (Optional[str], optional): Coordinate reference system of the bounds.
            If None, assumes bounds are in the same CRS as `data_in`. Defaults to None.

    Returns:
        rasterio.windows.Window: Window object with pixel coordinates (row_off, col_off, height, width)
            relative to `data_in` that corresponds to the geographic bounds.

    Examples:
        >>> import rasterio
        >>> # Read a window from UTM bounds
        >>> bounds_utm = (500000, 4649000, 501000, 4650000)  # 1km x 1km area
        >>> with rasterio.open('utm_image.tif') as src:
        ...     window = window_from_bounds(src, bounds_utm)
        ...     data = src.read(window=window)

        >>> # Read with CRS transformation
        >>> bounds_wgs84 = (-3.71, 40.41, -3.69, 40.42)  # (lon_min, lat_min, lon_max, lat_max)
        >>> with rasterio.open('utm_image.tif') as src:  # UTM image
        ...     window = window_from_bounds(src, bounds_wgs84, crs_bounds='EPSG:4326')
        ...     data = src.read(window=window)

    Note:
        The returned window may extend beyond the raster boundaries. Use boundless reading
        or clip the window to raster extent as needed.
    """
    # Transform bounds to raster's CRS if they're in a different CRS
    if (crs_bounds is not None) and not window_utils.compare_crs(crs_bounds, data_in.crs):
        # Reproject bounds: (left, bottom, right, top) → same format in data_in.crs
        bounds_in = rasterio.warp.transform_bounds(crs_bounds, data_in.crs, *bounds)
    else:
        bounds_in = bounds

    # Convert geographic bounds to pixel window using raster's transform
    window_in = rasterio.windows.from_bounds(*bounds_in, transform=data_in.transform)

    return window_in


def window_from_center_coords(
    data_in: Union[GeoDataBase, rasterio.DatasetReader],
    center_coords: Tuple[float, float],
    shape: Tuple[int, int],
    crs_center_coords: Optional[Any] = None,
) -> rasterio.windows.Window:
    """
    Calculate a raster window of specified size centered on geographic coordinates.

    This function creates a window by converting the center point from geographic to pixel
    coordinates, then calculating the upper-left corner based on the desired shape. Handles
    both rectilinear and rotated/skewed transforms.

    Args:
        data_in (Union[GeoDataBase, rasterio.DatasetReader]): Raster data source with crs
            and transform attributes defining the spatial reference.
        center_coords (Tuple[float, float]): Center point as (x, y) in geographic coordinates.
            For WGS84, this would be (longitude, latitude).
        shape (Tuple[int, int]): Desired window size as (height, width) in pixels.
            Shape format: (n_rows, n_cols).
        crs_center_coords (Optional[Any], optional): Coordinate reference system of center_coords.
            If None, assumes coords are in the same CRS as `data_in`. Defaults to None.

    Returns:
        rasterio.windows.Window: Window object centered on the specified coordinates with
            the requested shape: (row_off, col_off, height, width) in pixel coordinates.

    Examples:
        >>> import rasterio
        >>> # Extract 256x256 window centered on a point
        >>> center = (-3.7038, 40.4168)  # Madrid (lon, lat)
        >>> window_shape = (256, 256)  # (height, width)
        >>> with rasterio.open('image.tif') as src:
        ...     window = window_from_center_coords(src, center, window_shape,
        ...                                          crs_center_coords='EPSG:4326')
        ...     data = src.read(window=window)  # Shape: (bands, 256, 256)

        >>> # For square chips, can use same value
        >>> window = window_from_center_coords(src, center, (512, 512))

    Note:
        The window may extend beyond raster boundaries if centered near edges.
        Use boundless reading to handle this case.
    """
    # Transform center coordinates to raster's CRS if needed
    if (crs_center_coords is not None) and not window_utils.compare_crs(crs_center_coords, data_in.crs):
        center_coords = _transform_from_crs(center_coords, crs_center_coords, data_in.crs)

    transform = data_in.transform

    # Convert geographic center to pixel coordinates
    # ~transform is the inverse: geo → pixel
    pixel_center_coords = ~transform * tuple(center_coords)

    # Calculate upper-left corner in pixel coordinates
    # For a window of shape (H, W), center is at (W/2, H/2) from upper-left
    # pixel_upper_left = pixel_center - (W/2, H/2)
    pixel_upper_left = _round_all((pixel_center_coords[0] - shape[1] / 2, pixel_center_coords[1] - shape[0] / 2))

    # Create window with calculated upper-left corner and requested shape
    # Window format: (col_off, row_off, width, height)
    window = rasterio.windows.Window(
        row_off=pixel_upper_left[1], col_off=pixel_upper_left[0], width=shape[1], height=shape[0]
    )
    return window


def window_from_tile(
    data_in: Union[GeoDataBase, rasterio.DatasetReader], x: int, y: int, z: int
) -> rasterio.windows.Window:
    """
    Calculate the raster window corresponding to a Web Mercator (XYZ) tile.

    This function converts XYZ tile coordinates (as used by web mapping services like
    OpenStreetMap, Google Maps) to a raster window. Tiles follow the TMS/Slippy Map
    convention where the world is divided into 2^z × 2^z tiles at zoom level z.

    At zoom z:
    - Tile (0, 0) is the top-left
    - x ranges from 0 to 2^z - 1 (west to east)
    - y ranges from 0 to 2^z - 1 (north to south)

    Args:
        data_in (Union[GeoDataBase, rasterio.DatasetReader]): Raster data source with crs
            and transform attributes. Can be in any CRS; tile bounds will be transformed.
        x (int): Tile column index (0 to 2^z - 1). Increases eastward.
        y (int): Tile row index (0 to 2^z - 1). Increases southward.
        z (int): Zoom level (0-22 typically). At z=0, the entire world is one tile.

    Returns:
        rasterio.windows.Window: Window object in pixel coordinates that corresponds
            to the geographic extent of the XYZ tile.

    Examples:
        >>> import rasterio
        >>> # Get window for a tile covering Madrid area at zoom 12
        >>> with rasterio.open('spain.tif') as src:
        ...     window = window_from_tile(src, x=2046, y=1537, z=12)
        ...     tile_data = src.read(window=window)

        >>> # Tile coordinates for lower zoom (more area coverage)
        >>> window_z8 = window_from_tile(src, x=127, y=96, z=8)  # Larger area

        >>> # Higher zoom = smaller area, more detail
        >>> window_z15 = window_from_tile(src, x=16374, y=12297, z=15)

    References:
        - OSM Slippy map tilenames: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        - XYZ tiles: https://en.wikipedia.org/wiki/Tiled_web_map

    Note:
        Tiles are in Web Mercator projection (EPSG:3857). The function handles
        transformation to the raster's native CRS automatically.
    """
    # Get tile bounds in Web Mercator (EPSG:3857) coordinates
    bounds_wgs = mercantile.xy_bounds(int(x), int(y), int(z))

    # Create polygon from tile bounds
    polygon_crs_webmercator = box(bounds_wgs.left, bounds_wgs.bottom, bounds_wgs.right, bounds_wgs.top)

    # Convert to window with surrounding buffer for complete tile coverage
    return window_from_polygon(data_in, polygon_crs_webmercator, WEB_MERCATOR_CRS, window_surrounding=True)


def _window_intersects_data(data_in: GeoData, window: rasterio.windows.Window) -> bool:
    """Return True iff `window` overlaps the data extent."""
    named_shape = OrderedDict(zip(data_in.dims, data_in.shape))
    window_data = rasterio.windows.Window(
        col_off=0, row_off=0, width=named_shape["x"], height=named_shape["y"]
    )
    return bool(rasterio.windows.intersect([window_data, window]))


def _build_no_intersect_result(
    data_in: GeoData,
    window: rasterio.windows.Window,
    boundless: bool,
    return_only_data: bool,
) -> Union[GeoData, np.ndarray, None]:
    """Build the result when `window` does not intersect `data_in`.

    Returns None when boundless is False (caller signals "no data"); otherwise
    returns a fill-valued array/GeoTensor matching the window shape. Pure CPU.
    Shared between the sync `read_from_window` and `asyncread.read_from_window`
    so the no-I/O fallback path stays identical.
    """
    if not boundless:
        return None

    named_shape = OrderedDict(zip(data_in.dims, data_in.shape))
    expected_shapes = {"x": window.width, "y": window.height}
    shape = tuple(
        [named_shape[s] if s not in ["x", "y"] else expected_shapes[s] for s in data_in.dims]
    )
    data = np.zeros(shape, dtype=data_in.dtype)
    fill_value_default = getattr(data_in, "fill_value_default", 0)
    if fill_value_default != 0:
        data += fill_value_default
    if return_only_data:
        return data
    return GeoTensor(
        data,
        crs=data_in.crs,
        transform=rasterio.windows.transform(window, transform=data_in.transform),
        fill_value_default=fill_value_default,
    )


def read_from_window(
    data_in: GeoData,
    window: rasterio.windows.Window,
    return_only_data: bool = False,
    trigger_load: bool = False,
    boundless: bool = True,
) -> Union[GeoData, np.ndarray, None]:
    """
    Read raster data from a specified window, with optional padding for out-of-bounds areas.

    This function extracts data from a raster using pixel window coordinates. When the window
    extends beyond raster boundaries, it can pad with fill values (boundless=True) or return
    None/clipped data (boundless=False).

    Args:
        data_in (GeoData): Input raster data with spatial reference (crs, transform).
            Must implement the GeoData protocol with "x" and "y" dimensions.
        window (rasterio.windows.Window): Window defining the area to read in pixel coordinates.
            Format: Window(col_off, row_off, width, height).
        return_only_data (bool, optional): If True, returns numpy array without georeferencing.
            If False, returns GeoData object with spatial metadata. Defaults to False.
        trigger_load (bool, optional): If True, forces loading data into memory (for lazy readers).
            Defaults to False.
        boundless (bool, optional): If True, output always matches window shape, padding with
            fill_value_default for out-of-bounds areas. If False, only reads intersecting area
            or returns None. Defaults to True.

    Returns:
        Union[GeoData, np.ndarray, None]:
            - If return_only_data=True: numpy array with shape matching the data dimensions
            - If return_only_data=False: GeoData object with updated transform for the window
            - If boundless=False and no intersection: None

    Examples:
        >>> import rasterio
        >>> from georeader import GeoTensor
        >>> # Read a 256x256 window starting at pixel (100, 200)
        >>> window = rasterio.windows.Window(col_off=100, row_off=200, width=256, height=256)
        >>> with rasterio.open('image.tif') as src:
        ...     data = GeoTensor.load_from_window(src, window)
        ...     result = read_from_window(data, window)
        ...     print(result.shape)  # Shape: (bands, 256, 256)

        >>> # Read without padding (only intersecting area)
        >>> window_large = rasterio.windows.Window(0, 0, 10000, 10000)  # Beyond bounds
        >>> result = read_from_window(data, window_large, boundless=False)
        >>> # result will be clipped to actual data extent

    Note:
        The output transform is adjusted to correspond to the window's geographic location.
        For windows partially outside bounds, boundless=True pads with fill_value_default.
    """
    # Handle case where window doesn't intersect data at all (pure CPU; no I/O).
    # Shared with asyncread.read_from_window via the private helpers.
    if not _window_intersects_data(data_in, window):
        return _build_no_intersect_result(data_in, window, boundless, return_only_data)

    # Read data from window using the reader's method (handles padding automatically)
    data_sel = data_in.read_from_window(window=window, boundless=boundless)

    if return_only_data:
        return data_sel.values

    # Load into memory if requested (useful for lazy readers)
    if trigger_load:
        data_sel = data_sel.load()

    return data_sel


def read_from_center_coords(
    data_in: GeoData,
    center_coords: Tuple[float, float],
    shape: Tuple[int, int],
    crs_center_coords: Optional[Any] = None,
    return_only_data: bool = False,
    trigger_load: bool = False,
    boundless: bool = True,
) -> Union[GeoData, np.ndarray]:
    """
    Extract a rectangular chip from raster data centered on geographic coordinates.

    This function combines window calculation and data reading in one step. It's particularly
    useful for creating training chips for machine learning, extracting regions around points
    of interest, or generating thumbnails centered on specific locations.

    Args:
        data_in (GeoData): Input raster data with spatial reference (crs, transform).
            Must implement the GeoData protocol.
        center_coords (Tuple[float, float]): Center point as (x, y) in geographic coordinates.
            For WGS84, this would be (longitude, latitude). For projected CRS, (easting, northing).
        shape (Tuple[int, int]): Desired output size as (height, width) in pixels.
            The chip will have exactly this shape if boundless=True.
        crs_center_coords (Optional[Any], optional): Coordinate reference system of center_coords.
            If None, assumes coords are in the same CRS as `data_in`. Can be EPSG code string,
            CRS object, or WKT. Defaults to None.
        return_only_data (bool, optional): If True, returns numpy array without georeferencing.
            If False, returns GeoData object with spatial metadata. Defaults to False.
        trigger_load (bool, optional): If True, forces loading data into memory (for lazy readers).
            Defaults to False.
        boundless (bool, optional): If True, output always matches shape, padding with
            fill_value_default for out-of-bounds areas. If False, clips to actual data extent.
            Defaults to True.

    Returns:
        Union[GeoData, np.ndarray]:
            - If return_only_data=True: numpy array with shape (bands, height, width) or (height, width)
            - If return_only_data=False: GeoData object with transform adjusted to chip location

    Examples:
        >>> import rasterio
        >>> from georeader import RasterioReader
        >>>
        >>> # Extract 512x512 chip centered on a location
        >>> with rasterio.open('sentinel2.tif') as src:
        ...     reader = RasterioReader(src)
        ...     center = (-3.7038, 40.4168)  # Madrid (lon, lat)
        ...     chip = read_from_center_coords(reader, center, (512, 512),
        ...                                     crs_center_coords='EPSG:4326')
        ...     print(chip.shape)  # (bands, 512, 512)
        ...     print(chip.bounds)  # Geographic bounds of the chip

        >>> # Get just the numpy array without georeference
        >>> data_array = read_from_center_coords(reader, center, (256, 256),
        ...                                       crs_center_coords='EPSG:4326',
        ...                                       return_only_data=True)

        >>> # Extract chip with different aspect ratio
        >>> chip_rect = read_from_center_coords(reader, center, (256, 512))  # height=256, width=512

    Note:
        - The center coordinate refers to the geographic center, which maps to the pixel at
          (height/2, width/2) in the output chip.
        - For chips near image boundaries, boundless=True pads with fill_value_default.
        - The output transform is adjusted so the chip maintains correct georeferencing.
    """
    # Calculate the window that encompasses the desired chip area
    window = window_from_center_coords(data_in, center_coords, shape, crs_center_coords)

    # Read data from the calculated window
    return read_from_window(
        data_in, window=window, return_only_data=return_only_data, trigger_load=trigger_load, boundless=boundless
    )


def read_from_bounds(
    data_in: GeoData,
    bounds: Tuple[float, float, float, float],
    crs_bounds: Optional[str] = None,
    pad_add: Tuple[int, int] = (0, 0),
    return_only_data: bool = False,
    trigger_load: bool = False,
    boundless: bool = True,
) -> Union[GeoData, np.ndarray]:
    """
    Extract raster data within a geographic bounding box, with optional CRS transformation.

    This function is the primary interface for reading raster data by geographic extent. It's
    particularly useful for:
    - Extracting specific geographic regions from large rasters
    - Reading data in a different CRS than the source (e.g., WGS84 bounds from UTM raster)
    - Creating training chips for machine learning with consistent geographic extents
    - Subsetting satellite imagery to areas of interest
    - Co-registration workflows requiring precise spatial alignment

    The function handles the complete workflow: converts bounds to pixel window, optionally
    adds padding (useful for edge-aware processing like CNNs or interpolation), and returns
    the data with correct georeferencing. When bounds are in a different CRS, it automatically
    transforms them to match the raster's coordinate system.

    Algorithm:
    1. Transform bounds from crs_bounds to data_in.crs (if needed)
    2. Calculate pixel window corresponding to geographic bounds
    3. Add padding to window if requested (for algorithms needing context)
    4. Round window to integer pixel coordinates (ceil for outer bounds)
    5. Read data using read_from_window with boundless support

    Args:
        data_in (GeoData): Input georeferenced data with spatial reference (crs, transform).
            Must implement the GeoData protocol with "x" and "y" dimensions.
        bounds (Tuple[float, float, float, float]): Geographic bounding box as
            (left, bottom, right, top) or (xmin, ymin, xmax, ymax) in the CRS specified
            by crs_bounds. For WGS84, this would be (lon_min, lat_min, lon_max, lat_max).
            For UTM, (easting_min, northing_min, easting_max, northing_max).
        crs_bounds (Optional[str], optional): Coordinate reference system of the bounds.
            If None, assumes bounds are in the same CRS as data_in. Common formats:
            "EPSG:4326" (WGS84), "EPSG:32630" (UTM Zone 30N), CRS object, or WKT string.
            Defaults to None.
        pad_add (Tuple[int, int], optional): Additional padding in pixels to add around
            the bounding box as (pad_y, pad_x). Useful for:
            - CNN inference needing receptive field context
            - Interpolation algorithms requiring neighboring pixels
            - Co-registration workflows with geometric transformations
            - Edge-aware image processing
            Format: (rows_padding, cols_padding). Defaults to (0, 0).
        return_only_data (bool, optional): If True, returns numpy array without georeferencing.
            If False, returns GeoData object with spatial metadata (transform, crs).
            Defaults to False.
        trigger_load (bool, optional): If True, forces loading data into memory (for lazy readers
            like xarray or dask-backed arrays). Defaults to False.
        boundless (bool, optional): If True, output always matches window shape, padding with
            fill_value_default for out-of-bounds areas. If False, clips to actual data extent.
            Defaults to True.

    Returns:
        Union[GeoData, np.ndarray]:
            - If return_only_data=False: GeoData object with transform adjusted to the bounds
              and shape matching the geographic extent (plus padding if specified)
            - If return_only_data=True: numpy array with shape (bands, height, width) or
              (height, width) depending on input dimensions

    Examples:
        >>> import rasterio
        >>> from georeader import RasterioReader
        >>>
        >>> # Example 1: Read a 1km x 1km area from UTM raster
        >>> with rasterio.open('sentinel2_utm.tif') as src:
        ...     reader = RasterioReader(src)
        ...     # Bounds in UTM Zone 30N (meters)
        ...     bounds_utm = (500000, 4649000, 501000, 4650000)  # 1km x 1km square
        ...     data = read_from_bounds(reader, bounds_utm)
        ...     print(f"Shape: {data.shape}")  # e.g., (13, 100, 100) at 10m resolution
        ...     print(f"Bounds: {data.bounds}")  # Should match requested bounds

        >>> # Example 2: Read with CRS transformation (WGS84 → UTM)
        >>> with rasterio.open('landsat_utm.tif') as src:  # UTM Zone 33N raster
        ...     reader = RasterioReader(src)
        ...     # Specify bounds in WGS84 (degrees)
        ...     bounds_wgs84 = (13.37, 52.51, 13.38, 52.52)  # Small area in Berlin
        ...     data = read_from_bounds(reader, bounds_wgs84, crs_bounds='EPSG:4326')
        ...     print(f"CRS: {data.crs}")  # Still UTM (no reprojection, just subsetting)

        >>> # Example 3: Read with padding for CNN inference
        >>> # Padding ensures the CNN has context at edges
        >>> bounds = (-3.71, 40.41, -3.69, 40.42)  # Madrid area in WGS84
        >>> data_padded = read_from_bounds(reader, bounds,
        ...                                 crs_bounds='EPSG:4326',
        ...                                 pad_add=(16, 16))  # 16-pixel padding
        ...     # Output will be larger than actual bounds to include context

        >>> # Example 4: Extract training chips at consistent locations
        >>> # For machine learning, we often need chips at specific coordinates
        >>> training_areas = [
        ...     (-122.5, 37.7, -122.4, 37.8),   # San Francisco
        ...     (-118.3, 34.0, -118.2, 34.1),   # Los Angeles
        ...     (-73.9, 40.7, -73.8, 40.8),     # New York
        ... ]
        >>> chips = []
        >>> for bounds_wgs in training_areas:
        ...     chip = read_from_bounds(reader, bounds_wgs,
        ...                            crs_bounds='EPSG:4326',
        ...                            return_only_data=True)
        ...     chips.append(chip)
        >>> # All chips now have consistent geographic extent for training

        >>> # Example 5: Clip to actual extent (no padding)
        >>> # Useful when you don't want data outside raster boundaries
        >>> bounds_large = (-10, 30, 10, 50)  # Large area, may exceed raster
        >>> data_clipped = read_from_bounds(reader, bounds_large,
        ...                                  crs_bounds='EPSG:4326',
        ...                                  boundless=False)
        >>> # Output only contains pixels within both bounds AND raster extent

    Note:
        - Window coordinates are rounded outward (ceil) to ensure complete coverage
        - The output transform is adjusted to match the actual pixel boundaries
        - Padding is added in pixel space after CRS transformation
        - For interpolation/resampling needing edge context, use pad_add=(3, 3) minimum
        - Boundless reading uses fill_value_default from data_in for out-of-bounds pixels
    """
    window_in = window_from_bounds(data_in, bounds, crs_bounds)
    if any(p > 0 for p in pad_add):
        window_in = pad_window(window_in, pad_add)  # Add padding for bicubic int or for co-registration
    window_in = round_outer_window(window_in)

    return read_from_window(
        data_in, window_in, return_only_data=return_only_data, trigger_load=trigger_load, boundless=boundless
    )


def read_from_polygon(
    data_in: GeoData,
    polygon: Union[Polygon, MultiPolygon],
    crs_polygon: Optional[str] = None,
    pad_add: Tuple[int, int] = (0, 0),
    return_only_data: bool = False,
    trigger_load: bool = False,
    boundless: bool = True,
    window_surrounding: bool = False,
) -> Union[GeoData, np.ndarray]:
    """
    Extract raster data within a polygon boundary, supporting complex shapes and masking.

    This function reads the minimum bounding rectangle containing a polygon, making it ideal for:
    - Extracting irregular-shaped regions (e.g., administrative boundaries, watersheds)
    - Processing data within specific land parcels or management zones
    - Creating masks for pixel-wise operations within complex geometries
    - Reducing memory footprint by reading only the extent containing features of interest
    - Multi-polygon support for disconnected regions

    The function calculates the pixel window that encompasses all polygon vertices, reads that
    rectangular region, and preserves georeferencing for downstream processing. For actual
    polygon masking (setting pixels outside polygon to nodata), combine this with
    `rasterio.features.geometry_mask`.

    Common workflow:
    1. Read data within polygon bounds → get rectangular chip
    2. Create geometry mask → binary array (True inside polygon)
    3. Apply mask → set pixels outside polygon to nodata or 0

    Algorithm:
    1. Transform polygon vertices from crs_polygon to data_in.crs (if needed)
    2. Find minimum bounding rectangle in pixel coordinates
    3. Optionally add 1-pixel buffer (window_surrounding=True) for complete coverage
    4. Add user-specified padding (pad_add) for processing context
    5. Round window and read data using read_from_window

    Args:
        data_in (GeoData): Input georeferenced data with spatial reference (crs, transform).
            Must implement the GeoData protocol with "x" and "y" dimensions.
        polygon (Union[Polygon, MultiPolygon]): Shapely geometry defining the area of interest.
            Can be a simple Polygon for single region or MultiPolygon for disconnected areas.
            Polygon vertices define the boundary; function reads the bounding rectangle.
        crs_polygon (Optional[str], optional): Coordinate reference system of the polygon.
            If None, assumes polygon is in the same CRS as data_in. Common formats:
            "EPSG:4326" (WGS84), "EPSG:32630" (UTM), CRS object, or WKT string.
            Defaults to None.
        pad_add (Tuple[int, int], optional): Additional padding in pixels as (pad_y, pad_x).
            Useful for:
            - Ensuring complete polygon coverage at edges
            - CNN inference needing receptive field context
            - Interpolation requiring neighboring pixels
            - Edge-aware processing algorithms
            Format: (rows_padding, cols_padding). Defaults to (0, 0).
        return_only_data (bool, optional): If True, returns numpy array without georeferencing.
            If False, returns GeoData object with spatial metadata. Defaults to False.
        trigger_load (bool, optional): If True, forces loading data into memory (for lazy readers).
            Defaults to False.
        boundless (bool, optional): If True, output matches window shape, padding with
            fill_value_default for out-of-bounds areas. If False, clips to actual data extent.
            Defaults to True.
        window_surrounding (bool, optional): If True, adds 1-pixel buffer around polygon
            to ensure complete surrounding coverage (window edges won't align with vertices).
            Useful when polygon vertices align exactly with pixel boundaries and you need
            complete coverage. Defaults to False.

    Returns:
        Union[GeoData, np.ndarray]:
            - If return_only_data=False: GeoData object with transform adjusted to the
              minimum bounding rectangle and shape matching polygon extent (plus padding)
            - If return_only_data=True: numpy array with shape (bands, height, width) or
              (height, width) depending on input dimensions

    Examples:
        >>> from shapely.geometry import Polygon, box
        >>> import rasterio
        >>> import rasterio.features
        >>> from georeader import RasterioReader, read
        >>>
        >>> # Example 1: Read rectangular region using polygon
        >>> polygon = box(-3.71, 40.41, -3.69, 40.42)  # Madrid area (WGS84)
        >>> with rasterio.open('sentinel2.tif') as src:
        ...     reader = RasterioReader(src)
        ...     data = read.read_from_polygon(reader, polygon, crs_polygon='EPSG:4326')
        ...     print(f"Shape: {data.shape}")  # (13, H, W) - minimum rect containing polygon
        ...     print(f"Bounds: {data.bounds}")

        >>> # Example 2: Read irregular polygon with masking workflow
        >>> # Step 1: Define irregular polygon (e.g., agricultural field)
        >>> field_boundary = Polygon([
        ...     (-3.7050, 40.4150), (-3.7030, 40.4150),
        ...     (-3.7030, 40.4170), (-3.7040, 40.4180),
        ...     (-3.7050, 40.4170), (-3.7050, 40.4150)
        ... ])
        >>> # Step 2: Read bounding rectangle
        >>> data = read.read_from_polygon(reader, field_boundary, crs_polygon='EPSG:4326')
        >>> # Step 3: Create mask (True inside polygon, False outside)
        >>> from rasterio.features import geometry_mask
        >>> mask = geometry_mask(
        ...     [field_boundary],
        ...     transform=data.transform,
        ...     invert=True,  # True inside polygon
        ...     out_shape=data.shape[-2:]
        ... )
        >>> # Step 4: Apply mask
        >>> data.values[:, ~mask] = data.fill_value_default  # Mask outside polygon
        >>> # Now data only contains pixels within field_boundary

        >>> # Example 3: Multi-polygon (disconnected regions)
        >>> from shapely.geometry import MultiPolygon
        >>> # Read multiple farms in one operation
        >>> farm1 = box(-3.71, 40.41, -3.70, 40.42)
        >>> farm2 = box(-3.68, 40.41, -3.67, 40.42)
        >>> farms = MultiPolygon([farm1, farm2])
        >>> data = read.read_from_polygon(reader, farms, crs_polygon='EPSG:4326')
        >>> # Returns bounding rectangle containing all polygons

        >>> # Example 4: Read with padding for CNN inference
        >>> # Polygon defines ROI, padding provides context
        >>> roi = Polygon([(-3.70, 40.41), (-3.69, 40.41), (-3.69, 40.42), (-3.70, 40.42)])
        >>> data_padded = read.read_from_polygon(
        ...     reader, roi,
        ...     crs_polygon='EPSG:4326',
        ...     pad_add=(32, 32),  # 32-pixel padding for CNN receptive field
        ...     window_surrounding=True  # Ensure complete coverage
        ... )

        >>> # Example 5: Memory-efficient masking for large areas
        >>> # Read only the extent containing the polygon, not entire raster
        >>> watershed = Polygon([...])  # Complex watershed boundary
        >>> # This reads only the bounding box, not the full raster
        >>> data = read.read_from_polygon(reader, watershed, crs_polygon='EPSG:4326')
        >>> print(f"Read shape: {data.shape}")  # Much smaller than full raster
        >>> # Apply masking as in Example 2
        >>> mask = geometry_mask([watershed], transform=data.transform,
        ...                      invert=True, out_shape=data.shape[-2:])

        >>> # Example 6: Time series analysis within boundary
        >>> # Multi-temporal stack reading same spatial extent
        >>> boundary = box(-3.71, 40.41, -3.69, 40.42)
        >>> time_series_data = []
        >>> for date, raster_path in date_raster_pairs:
        ...     with rasterio.open(raster_path) as src:
        ...         reader = RasterioReader(src)
        ...         data = read.read_from_polygon(reader, boundary,
        ...                                       crs_polygon='EPSG:4326')
        ...         time_series_data.append(data)
        >>> # All chips have consistent extent for temporal analysis

    Note:
        - Function reads the minimum bounding RECTANGLE, not the exact polygon shape
        - For actual polygon masking, use rasterio.features.geometry_mask after reading
        - Window coordinates are rounded outward to ensure complete polygon coverage
        - MultiPolygon returns single rectangular chip containing all disconnected parts
        - window_surrounding=True adds 1-pixel buffer for edge cases
        - Padding is applied in pixel space after CRS transformation
    """
    window_in = window_from_polygon(data_in, polygon, crs_polygon, window_surrounding=window_surrounding)
    if any(p > 0 for p in pad_add):
        window_in = pad_window(window_in, pad_add)  # Add padding for bicubic int or for co-registration
    window_in = round_outer_window(window_in)

    return read_from_window(
        data_in, window_in, return_only_data=return_only_data, trigger_load=trigger_load, boundless=boundless
    )


def read_reproject_like(
    data_in: GeoData,
    data_like: GeoData,
    resolution_dst: Optional[Union[float, Tuple[float, float]]] = None,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    dtype_dst: Any = None,
    return_only_data: bool = False,
    dst_nodata: Optional[int] = None,
) -> Union[GeoTensor, np.ndarray]:
    """
    Reads from `data_in` and reprojects to have the same extent and resolution than `data_like`.

    Args:
        data_in: GeoData to read and reproject. Expected coords "x" and "y".
        data_like: GeoData to get the bounds and resolution to reproject `data_in`.
        resolution_dst: if not None it will overwrite the resolution of `data_like`.
        resampling: specifies how data is reprojected from `rasterio.warp.Resampling`.
        dtype_dst: if None it will be inferred
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoTensor object (georreferenced array).
        dst_nodata: dst_nodata value

    Returns:
        GeoTensor read from `data_in` with same transform, crs, shape and bounds than `data_like`.
    """

    shape_out = data_like.shape[-2:]
    if resolution_dst is not None:
        if isinstance(resolution_dst, float):
            resolution_dst = (resolution_dst, resolution_dst)

        resolution_data_like = data_like.res

        shape_out = (
            int(round(shape_out[0] / resolution_dst[0] * resolution_data_like[0])),
            int(round(shape_out[1] / resolution_dst[1] * resolution_data_like[1])),
        )

    return read_reproject(
        data_in,
        dst_crs=data_like.crs,
        dst_transform=data_like.transform,
        resolution_dst_crs=resolution_dst,
        window_out=rasterio.windows.Window(0, 0, width=shape_out[-1], height=shape_out[-2]),
        resampling=resampling,
        dtype_dst=dtype_dst,
        return_only_data=return_only_data,
        dst_nodata=dst_nodata,
    )


def apply_anti_aliasing(
    data_in: GeoData,
    anti_aliasing_sigma: Optional[Union[float, np.ndarray]] = None,
    resolution_dst: Optional[Union[float, Tuple[float, float]]] = None,
) -> GeoTensor:
    """
    Apply anti-aliasing to `data_in` assuming it will be downsampled to `resolution_dst`.

    Args:
        data_in (GeoData): GeoData to apply anti-aliasing
        anti_aliasing_sigma (Optional[Union[float,np.ndarray]], optional): Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the downsampling factor, where s > 1. Defaults to None.
        resolution_dst (Optional[Union[float, Tuple[float, float]]], optional): spatial resolution in data_in crs. Defaults
            to None.

    Returns:
        GeoTensor: GeoTensor with anti-aliasing applied
    """
    resolution_or = data_in.res
    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))

    scale = np.array([resolution_dst[0] / resolution_or[0], resolution_dst[1] / resolution_or[1]])

    if any(s1 < s2 for s1, s2 in zip(resolution_or, resolution_dst)):
        # If we are downscaling the image and requested anti_aliasing
        try:
            from scipy import ndimage as ndi
        except ImportError:
            raise ImportError("scipy is required to apply anti-aliasing")

        # Copy or load the tensor in memory
        if isinstance(data_in, GeoTensor):
            data_in = data_in.copy()
        else:
            data_in = data_in.load()

        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = np.mean(np.maximum(0, (scale - 1) / 2))

        # TODO if data_in.values is a torch.Tensor use kornia gaussian filter instead of ndi

        input_shape = data_in.shape
        if len(input_shape) == 4:
            for i, j in product(range(0, input_shape[0]), range(0, input_shape[1])):
                if isinstance(anti_aliasing_sigma, numbers.Number):
                    anti_aliasing_sigma_iter = anti_aliasing_sigma
                else:
                    anti_aliasing_sigma_iter = anti_aliasing_sigma[i, j]
                data_in.values[i, j] = ndi.gaussian_filter(
                    data_in.values[i, j], anti_aliasing_sigma_iter, cval=0, mode="reflect"
                )
        elif len(input_shape) == 3:
            for i in range(0, input_shape[0]):
                if isinstance(anti_aliasing_sigma, numbers.Number):
                    anti_aliasing_sigma_iter = anti_aliasing_sigma
                else:
                    anti_aliasing_sigma_iter = anti_aliasing_sigma[i]

                data_in.values[i] = ndi.gaussian_filter(
                    data_in.values[i], anti_aliasing_sigma_iter, cval=0, mode="reflect"
                )
        else:
            data_in.values[...] = ndi.gaussian_filter(data_in.values, anti_aliasing_sigma, cval=0, mode="reflect")

    return data_in


def resize(
    data_in: GeoData,
    resolution_dst: Union[float, Tuple[float, float]],
    window_out: Optional[rasterio.windows.Window] = None,
    anti_aliasing: bool = True,
    anti_aliasing_sigma: Optional[Union[float, np.ndarray]] = None,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    return_only_data: bool = False,
) -> Union[GeoTensor, np.ndarray]:
    """
    Resample raster data to a different spatial resolution with optional anti-aliasing.

    This function changes the pixel size (spatial resolution) of raster data while preserving
    geographic extent and CRS. It's essential for:
    - Downsampling high-resolution imagery to reduce file size or processing time
    - Upsampling low-resolution data for visualization or analysis
    - Matching resolution across multi-source datasets
    - Creating image pyramids for multi-scale processing
    - Preparing data at specific resolutions for machine learning models

    The function intelligently handles both upsampling (resolution_dst > resolution_src) and
    downsampling (resolution_dst < resolution_src). For downsampling, it applies Gaussian
    anti-aliasing by default to prevent aliasing artifacts (moiré patterns, jagged edges).
    This is critical for maintaining visual quality and preventing information loss when
    reducing resolution.

    Anti-aliasing workflow (for downsampling):
    1. Determine downsampling factor: scale = resolution_dst / resolution_src
    2. Calculate Gaussian sigma: σ = (scale - 1) / 2 for scale > 1
    3. Apply Gaussian filter to smooth high-frequency components
    4. Resample to target resolution using specified resampling algorithm

    The function preserves georeferencing, adjusting the transform to reflect the new
    pixel size while maintaining the same geographic extent (upper-left corner stays fixed).

    Algorithm:
    1. Compute scale factors: scale = (res_dst_y/res_src_y, res_dst_x/res_src_x)
    2. Calculate output shape: shape_out = shape_in / scale (rounded up)
    3. If downsampling (scale > 1) and anti_aliasing=True:
       - Apply Gaussian filter with sigma = (scale - 1) / 2
    4. Call read_reproject with same CRS but updated resolution
    5. Adjust transform: new pixel size = resolution_dst

    Args:
        data_in (GeoData): Input georeferenced data to resample. Expected to have "x" and "y"
            spatial dimensions. Can be 2D (H, W), 3D (C, H, W), or 4D (T, C, H, W).
        resolution_dst (Union[float, Tuple[float, float]]): Target spatial resolution in
            data_in's CRS units. If float, assumes same resolution in x and y directions.
            If tuple, (res_y, res_x). Units:
            - Meters for projected CRS (e.g., UTM: 10 = 10m/pixel)
            - Degrees for geographic CRS (e.g., WGS84: 0.0001 = ~11m at equator)
        window_out (Optional[rasterio.windows.Window], optional): Explicit output window
            dimensions. If None, automatically computed from input shape and scale factor
            (ceiling operation to ensure complete coverage). Format: Window(col_off, row_off,
            width, height). Defaults to None.
        anti_aliasing (bool, optional): Whether to apply Gaussian filter before downsampling
            to reduce aliasing artifacts. Highly recommended for downsampling (scale > 1) to:
            - Prevent moiré patterns and jagged edges
            - Reduce high-frequency noise
            - Improve visual quality of downsampled images
            - Preserve spatial structure at coarser resolutions
            Has no effect when upsampling (scale ≤ 1). Defaults to True.
        anti_aliasing_sigma (Optional[Union[float, np.ndarray]], optional): Standard deviation
            for Gaussian filtering. If None, automatically computed as (scale - 1) / 2 where
            scale is the downsampling factor. Can be:
            - float: Same sigma for all bands
            - np.ndarray: Per-band sigma values with shape matching non-spatial dims
            Larger sigma = more smoothing (blurrier but less aliasing). Defaults to None.
        resampling (rasterio.warp.Resampling, optional): Resampling algorithm for interpolation.
            Common options:
            - cubic_spline: Smooth, good for continuous data (DEFAULT)
            - bilinear: Faster, slight quality loss
            - nearest: Categorical data (land cover, labels)
            - lanczos: High quality, slower
            - average: Good for downsampling continuous data
            Defaults to rasterio.warp.Resampling.cubic_spline.
        return_only_data (bool, optional): If True, returns numpy array without georeferencing.
            If False, returns GeoTensor with updated transform. Defaults to False.

    Returns:
        Union[GeoTensor, np.ndarray]:
            - If return_only_data=False: GeoTensor with shape determined by resolution ratio,
              transform adjusted to reflect new pixel size, same CRS and bounds as input
            - If return_only_data=True: numpy array with resampled data

    Examples:
        >>> from georeader import GeoTensor, read
        >>> import rasterio
        >>> import numpy as np
        >>>
        >>> # Example 1: Downsample Sentinel-2 from 10m to 30m (Landsat resolution)
        >>> # Load Sentinel-2 data at 10m resolution
        >>> s2_data = GeoTensor.load_file('sentinel2_10m.tif')
        >>> print(f"Original: {s2_data.shape}, res: {s2_data.res}")  # (13, 1000, 1000), res: (10, 10)
        >>>
        >>> # Downsample to 30m (3x reduction)
        >>> s2_30m = read.resize(s2_data, resolution_dst=30.0)
        >>> print(f"Downsampled: {s2_30m.shape}, res: {s2_30m.res}")  # (13, 334, 334), res: (30, 30)
        >>> # Shape reduction: 1000 / 3 ≈ 334 pixels
        >>> # Anti-aliasing automatically applied to prevent artifacts

        >>> # Example 2: Upsample low-resolution data (2x magnification)
        >>> # Coarse data at 60m resolution
        >>> coarse_data = GeoTensor.load_file('data_60m.tif')
        >>> print(f"Original: {coarse_data.shape}")  # (4, 100, 100)
        >>>
        >>> # Upsample to 30m resolution
        >>> upsampled = read.resize(coarse_data, resolution_dst=30.0)
        >>> print(f"Upsampled: {upsampled.shape}")  # (4, 200, 200)
        >>> # Shape increase: 100 * 2 = 200 pixels
        >>> # Uses cubic_spline interpolation for smooth result

        >>> # Example 3: Downsample with custom anti-aliasing
        >>> # Strong smoothing before downsampling (reduce noise)
        >>> smoothed = read.resize(s2_data, resolution_dst=50.0,
        ...                       anti_aliasing=True,
        ...                       anti_aliasing_sigma=3.0)  # Custom sigma
        >>> # More aggressive smoothing than default

        >>> # Example 4: Disable anti-aliasing (faster but lower quality)
        >>> # For quick previews or when speed is critical
        >>> fast_downsample = read.resize(s2_data, resolution_dst=30.0,
        ...                              anti_aliasing=False)
        >>> # Faster but may show aliasing artifacts

        >>> # Example 5: Different resolutions in x and y
        >>> # Non-square pixels (uncommon but supported)
        >>> anisotropic = read.resize(s2_data,
        ...                          resolution_dst=(20.0, 30.0))  # (res_y, res_x)
        >>> print(f"Resolution: {anisotropic.res}")  # (20, 30)
        >>> # Different sampling rates in each dimension

        >>> # Example 6: Resampling for categorical data (land cover)
        >>> labels = GeoTensor.load_file('land_cover_10m.tif')
        >>> # Use nearest neighbor to preserve class values
        >>> labels_30m = read.resize(labels, resolution_dst=30.0,
        ...                         resampling=rasterio.warp.Resampling.nearest,
        ...                         anti_aliasing=False)  # No smoothing for discrete data
        >>> # Class labels preserved (no interpolation)

        >>> # Example 7: Create image pyramid (multi-resolution)
        >>> # Generate multiple resolution levels for fast visualization
        >>> pyramid = {}
        >>> base_res = 10.0
        >>> for level in range(5):  # 5 pyramid levels
        ...     resolution = base_res * (2 ** level)  # 10m, 20m, 40m, 80m, 160m
        ...     pyramid[level] = read.resize(s2_data, resolution_dst=resolution)
        ...     print(f"Level {level}: {pyramid[level].shape}, res: {resolution}m")

        >>> # Example 8: Match resolution to reference dataset
        >>> reference = GeoTensor.load_file('reference_30m.tif')
        >>> # Resample data to match reference resolution
        >>> matched = read.resize(s2_data, resolution_dst=reference.res[0])
        >>> assert matched.res == reference.res
        >>> # Now both datasets have same resolution for analysis

    Note:
        - Function preserves CRS (no projection change, only resolution change)
        - Geographic bounds remain constant (upper-left corner fixed)
        - Transform is updated: pixel size = resolution_dst
        - Output shape computed as: shape_out = ceil(shape_in * res_in / res_dst)
        - Anti-aliasing only applied when downsampling (scale > 1)
        - For upsampling, resampling algorithm determines interpolation quality
        - Uses scipy.ndimage.gaussian_filter for anti-aliasing (requires scipy)
        - Efficient: operates in-place when possible to minimize memory usage
    """
    resolution_or = data_in.res
    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))
    scale = np.array([resolution_dst[0] / resolution_or[0], resolution_dst[1] / resolution_or[1]])

    if window_out is None:
        spatial_shape = data_in.shape[-2:]

        # scale < 1 => make image smaller (resolution_or < resolution_dst)
        # scale > 1 => make image larger (resolution_or > resolution_dst)
        output_shape_exact = spatial_shape[0] / scale[0], spatial_shape[1] / scale[1]
        output_shape_rounded = round(output_shape_exact[0], ndigits=3), round(output_shape_exact[1], ndigits=3)
        output_shape = ceil(output_shape_rounded[0]), ceil(output_shape_rounded[1])
        window_out = rasterio.windows.Window(col_off=0, row_off=0, width=output_shape[1], height=output_shape[0])

    if anti_aliasing:
        data_in = apply_anti_aliasing(data_in, anti_aliasing_sigma=anti_aliasing_sigma, resolution_dst=resolution_dst)

    return read_reproject(
        data_in,
        dst_crs=data_in.crs,
        resolution_dst_crs=resolution_dst,
        dst_transform=data_in.transform,
        window_out=window_out,
        resampling=resampling,
        return_only_data=return_only_data,
    )


def read_to_crs(
    data_in: GeoData,
    dst_crs: Any,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None,
    return_only_data: bool = False,
) -> Union[GeoTensor, np.ndarray]:
    """
    Change the crs of data_in to dst_crs. This function is a wrapper of the `read_reproject` function
    to reproject data_in to dst_crs.

    Args:
        data_in (GeoData): GeoData to reproyect
        dst_crs (Any): dst crs. Examples: "EPSG:4326", "EPSG:3857"
        resampling (rasterio.warp.Resampling, optional):
            Defaults to `rasterio.warp.Resampling.cubic_spline`
        resolution_dst_crs (Optional[Union[float, Tuple[float, float]]], optional):
            spatial resolution of the output `GeoTensor` in `dst_crs` CRS. Defaults to None.
            If not provided it will compute the resolution to match the resolution of the input.
        return_only_data (bool, optional): Defaults to `False`.
            If `True` it returns a np.ndarray otherwise a `GeoTensor` object (georreferenced array).

    Returns:
        Union[GeoTensor, np.ndarray]: data in dst_crs
    """
    if window_utils.compare_crs(data_in.crs, dst_crs):
        return data_in

    window_data, dst_transform = calculate_transform_window(data_in, dst_crs, resolution_dst_crs)

    return read_reproject(
        data_in,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        window_out=window_data,
        resampling=resampling,
        return_only_data=return_only_data,
    )


def calculate_transform_window(
    data_in: GeoData, dst_crs: Any, resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None
) -> Tuple[rasterio.Affine, rasterio.windows.Window]:
    """
    Calculate the default transform to reproject data to dst_crs with resolution_dst_crs

    Args:
        data_in (GeoData): GeoData to reproyect
        dst_crs (Any): dst crs
        resolution_dst_crs (Optional[Union[float, Tuple[float, float]]], optional): Defaults to None.
    """

    if resolution_dst_crs is not None:
        if isinstance(resolution_dst_crs, numbers.Number):
            resolution_dst_crs = (abs(resolution_dst_crs), abs(resolution_dst_crs))

    in_height, in_width = data_in.shape[-2:]
    dst_transform, width, height = rasterio.warp.calculate_default_transform(
        data_in.crs, dst_crs, in_width, in_height, *data_in.bounds, resolution=resolution_dst_crs
    )
    window_data = rasterio.windows.Window(0, 0, width=width, height=height)

    return window_data, dst_transform


@dataclass
class _ReprojectPlan:
    """Resolved parameters for `read_reproject`.

    `_reproject_setup` produces this; the sync and async orchestrators decide
    whether to fast-path, return empty, or proceed to a windowed read +
    `_reproject_finalize`. Pure data — no I/O involved.
    """

    dst_transform: rasterio.Affine
    dst_crs: Any
    window_out: rasterio.windows.Window
    crs_data_in: Any
    polygon_dst_crs: Polygon
    destination: np.ndarray
    dst_nodata: Any
    dtype_dst: Any
    cast: bool
    isbool_dtypein: bool
    isbool_dtypedst: bool
    named_shape: "OrderedDict[str, int]"
    # Early-exit signals (mutually exclusive with the normal path):
    fast_path_window: Optional[rasterio.windows.Window] = None  # source-aligned no-op window
    nonintersecting: bool = False  # data does not overlap dst extent


def _reproject_setup(
    data_in: GeoData,
    dst_crs: Optional[Any],
    bounds: Optional[Tuple[float, float, float, float]],
    resolution_dst_crs: Optional[Union[float, Tuple[float, float]]],
    dst_transform: Optional[rasterio.Affine],
    window_out: Optional[rasterio.windows.Window],
    dtype_dst: Any,
    dst_nodata: Optional[int],
) -> _ReprojectPlan:
    """Compute the reproject plan: destination grid, dtypes, allocation, early-exit flags.

    Pure CPU; no I/O. Shared between `read.read_reproject` and `asyncread.read_reproject`.
    Mirrors steps 1–6 of the original `read_reproject` body.
    """
    named_shape = OrderedDict(zip(data_in.dims, data_in.shape))

    # STEP 1: destination transform
    dst_transform = window_utils.figure_out_transform(
        transform=dst_transform, bounds=bounds, resolution_dst=resolution_dst_crs
    )

    # STEP 2: destination window
    if window_out is None:
        assert bounds is not None, (
            "Both window_out and bounds are None. This is needed to figure out the size of the output array"
        )
        window_out = rasterio.windows.from_bounds(*bounds, transform=dst_transform).round_lengths(
            op="ceil", pixel_precision=PIXEL_PRECISION
        )

    crs_data_in = data_in.crs
    if dst_crs is None:
        dst_crs = crs_data_in

    # STEP 3: same-CRS/aligned-grid fast path → caller should `read_from_window` and skip warp.
    fast_path_window: Optional[rasterio.windows.Window] = None
    if window_utils.compare_crs(dst_crs, crs_data_in):
        transform_data = data_in.transform
        if (
            (dst_transform.a == transform_data.a)
            and (dst_transform.b == transform_data.b)
            and (dst_transform.d == transform_data.d)
            and (dst_transform.e == transform_data.e)
        ):
            x_dst, y_dst = dst_transform.c, dst_transform.f
            col_off, row_off = ~transform_data * (x_dst, y_dst)
            window_in_data = rasterio.windows.Window(col_off, row_off, window_out.width, window_out.height)
            if _is_exact_round(window_in_data.row_off) and _is_exact_round(window_in_data.col_off):
                fast_path_window = window_in_data.round_offsets(
                    op="floor", pixel_precision=PIXEL_PRECISION
                )

    # STEP 4: dtype handling
    isbool_dtypein = data_in.dtype == "bool"
    isbool_dtypedst = False
    cast = True
    if dtype_dst is None:
        cast = False
        dtype_dst = data_in.dtype
        if isbool_dtypein:
            isbool_dtypedst = True
    elif np.dtype(dtype_dst) == "bool":
        isbool_dtypedst = True

    # STEP 5: pre-allocate output
    dict_shape_window_out = {"x": window_out.width, "y": window_out.height}
    shape_out = tuple(
        [named_shape[s] if s not in ["x", "y"] else dict_shape_window_out[s] for s in named_shape]
    )
    dst_nodata_resolved = dst_nodata or data_in.fill_value_default
    if isbool_dtypedst:
        dst_nodata_resolved = bool(dst_nodata_resolved)
    destination = np.full(shape_out, fill_value=dst_nodata_resolved, dtype=dtype_dst)

    # STEP 6: intersection check
    polygon_dst_crs = window_utils.window_polygon(window_out, dst_transform)
    nonintersecting = not data_in.footprint(crs=dst_crs).intersects(polygon_dst_crs)

    return _ReprojectPlan(
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        window_out=window_out,
        crs_data_in=crs_data_in,
        polygon_dst_crs=polygon_dst_crs,
        destination=destination,
        dst_nodata=dst_nodata_resolved,
        dtype_dst=dtype_dst,
        cast=cast,
        isbool_dtypein=isbool_dtypein,
        isbool_dtypedst=isbool_dtypedst,
        named_shape=named_shape,
        fast_path_window=fast_path_window,
        nonintersecting=nonintersecting,
    )


def _reproject_finalize(
    geotensor_in: GeoTensor,
    plan: _ReprojectPlan,
    resampling: rasterio.warp.Resampling,
    return_only_data: bool,
) -> Union[GeoTensor, np.ndarray]:
    """Run the warp loop and pack the result. Pure CPU.

    Steps 7 (type casting on the already-loaded input) and 8 (per-slice warp)
    of the original `read_reproject` body. Shared with `asyncread.read_reproject`.
    """
    np_array_in = np.asanyarray(geotensor_in.values)

    if plan.cast:
        if plan.isbool_dtypedst:
            np_array_in = np_array_in.astype(np.float32)
        else:
            np_array_in = np_array_in.astype(plan.dtype_dst)
    elif plan.isbool_dtypein:
        np_array_in = np_array_in.astype(np.float32)

    index_iter = [
        [(ns, i) for i in range(s)] for ns, s in plan.named_shape.items() if ns not in ["x", "y"]
    ]
    destination = plan.destination

    for current_select_tuple in itertools.product(*index_iter):
        i_sel_tuple = tuple(t[1] for t in current_select_tuple)

        np_array_iter = np_array_in[i_sel_tuple]
        if plan.isbool_dtypedst:
            dst_iter_write = destination[i_sel_tuple].astype(np.float32)
            dst_nodata_iter = float(plan.dst_nodata)
        else:
            dst_iter_write = destination[i_sel_tuple]
            dst_nodata_iter = plan.dst_nodata

        rasterio.warp.reproject(
            np_array_iter,
            dst_iter_write,
            src_transform=geotensor_in.transform,
            src_crs=plan.crs_data_in,
            dst_transform=plan.dst_transform,
            dst_crs=plan.dst_crs,
            src_nodata=geotensor_in.fill_value_default,
            dst_nodata=dst_nodata_iter,
            resampling=resampling,
        )

        if plan.isbool_dtypedst:
            destination[i_sel_tuple] = dst_iter_write > 0.5

    if return_only_data:
        return destination
    return GeoTensor(
        destination,
        transform=plan.dst_transform,
        crs=plan.dst_crs,
        fill_value_default=plan.dst_nodata,
    )


def read_reproject(
    data_in: GeoData,
    dst_crs: Optional[str] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None,
    dst_transform: Optional[rasterio.Affine] = None,
    window_out: Optional[rasterio.windows.Window] = None,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    dtype_dst: Any = None,
    return_only_data: bool = False,
    dst_nodata: Optional[int] = None,
) -> Union[GeoTensor, np.ndarray]:
    """
    Reproject raster data to a different CRS, resolution, and/or extent.

    This is the core reprojection function in georeader, providing fine-grained control
    over the output coordinate system, spatial resolution, geographic extent, and
    resampling method. It handles complex transformations including:
    - CRS changes (e.g., WGS84 → UTM, UTM → Web Mercator)
    - Resolution changes (resampling/downsampling)
    - Geographic subsetting (reading only a portion in destination CRS)
    - Data type conversions

    The function uses rasterio's warp.reproject under the hood, which leverages GDAL's
    high-performance reprojection engine. It automatically handles:
    - Non-intersecting regions (returns nodata-filled array)
    - Multi-band and multi-temporal data (iterates over all bands/times)
    - Boolean arrays (converts to float32 for interpolation, then back)
    - Edge cases near poles or antimeridian

    Algorithm:
    1. Determine output transform from bounds/resolution or use provided transform
    2. Check if source data intersects destination extent
    3. Read input data with small buffer (3 pixels) for edge handling
    4. Iterate over each band/time slice and call rasterio.warp.reproject
    5. Package result as GeoTensor with destination CRS and transform

    Args:
        data_in (GeoData): Input georeferenced data to reproject. Must have "x" and "y"
            spatial dimensions. Can be 2D (H, W), 3D (C, H, W), or 4D (T, C, H, W).
        dst_crs (Optional[str], optional): Destination coordinate reference system.
            If None, uses the same CRS as data_in (useful for resolution change only).
            Format: "EPSG:4326", "EPSG:32630", CRS object, or WKT string. Defaults to None.
        bounds (Optional[Tuple[float, float, float, float]], optional): Output extent as
            (xmin, ymin, xmax, ymax) in dst_crs coordinates. If None, must provide window_out.
            Useful for reading a specific geographic region. Defaults to None.
        resolution_dst_crs (Optional[Union[float, Tuple[float, float]]], optional):
            Target resolution in dst_crs units. If float, same resolution in x and y.
            If tuple, (res_x, res_y). If None, uses resolution from dst_transform.
            Units: meters for projected CRS, degrees for geographic CRS. Defaults to None.
        dst_transform (Optional[rasterio.Affine], optional): Output affine transform.
            If None, computed automatically from bounds and resolution. Useful for
            aligning to an existing grid. Defaults to None.
        window_out (Optional[rasterio.windows.Window], optional): Output size as
            Window(col_off=0, row_off=0, width=W, height=H). If None, computed from bounds.
            Defines output array dimensions. Defaults to None.
        resampling (rasterio.warp.Resampling, optional): Resampling algorithm.
            Options: nearest, bilinear, cubic, cubic_spline, lanczos, average, mode, etc.
            Default: cubic_spline (smooth, accurate for continuous data).
        dtype_dst (Any, optional): Output data type. If None, uses data_in.dtype.
            Examples: np.float32, np.uint8, np.int16. Defaults to None.
        return_only_data (bool, optional): If True, returns numpy array without georeference.
            If False, returns GeoTensor with spatial metadata. Defaults to False.
        dst_nodata (Optional[int], optional): Fill value for out-of-bounds regions.
            If None, uses data_in.fill_value_default. Defaults to None.

    Returns:
        Union[GeoTensor, np.ndarray]: Reprojected data.
            - If return_only_data=False: GeoTensor with shape matching window_out,
              georeferenced to dst_crs with dst_transform
            - If return_only_data=True: numpy array with same shape

    Examples:
        >>> # Example 1: Simple CRS change (WGS84 → UTM Zone 30N)
        >>> from georeader import GeoTensor, read
        >>> import rasterio
        >>>
        >>> # Create sample data in WGS84
        >>> transform_wgs84 = rasterio.Affine(0.001, 0, -3.71, 0, -0.001, 40.42)
        >>> data_wgs84 = GeoTensor(np.random.rand(100, 100), transform_wgs84, "EPSG:4326")
        >>>
        >>> # Reproject to UTM (no bounds = full extent)
        >>> data_utm = read.read_reproject(data_wgs84, dst_crs="EPSG:32630")
        >>> print(f"Input shape: {data_wgs84.shape}, Output shape: {data_utm.shape}")
        >>> print(f"Output CRS: {data_utm.crs}, resolution: {data_utm.res}")

        >>> # Example 2: Reproject with specific resolution (10m pixels)
        >>> data_utm_10m = read.read_reproject(
        ...     data_wgs84,
        ...     dst_crs="EPSG:32630",
        ...     resolution_dst_crs=10.0  # 10 meters
        ... )
        >>> print(f"Resolution: {data_utm_10m.res}")  # (10.0, 10.0)

        >>> # Example 3: Reproject and subset by bounds
        >>> bounds_madrid = (437000, 4474000, 439000, 4476000)  # UTM coords (2km × 2km)
        >>> data_subset = read.read_reproject(
        ...     data_wgs84,
        ...     dst_crs="EPSG:32630",
        ...     bounds=bounds_madrid,
        ...     resolution_dst_crs=10.0
        ... )
        >>> print(f"Subset shape: {data_subset.shape}")  # ~(200, 200) at 10m resolution

        >>> # Example 4: Align to existing grid (match another raster)
        >>> reference = GeoTensor.load_file("reference_grid.tif")
        >>> aligned = read.read_reproject(
        ...     data_wgs84,
        ...     dst_crs=reference.crs,
        ...     dst_transform=reference.transform,
        ...     window_out=rasterio.windows.Window(0, 0, reference.width, reference.height)
        ... )
        >>> # Output exactly matches reference grid

        >>> # Example 5: Custom resampling for categorical data
        >>> labels = GeoTensor(np.random.randint(0, 10, (100, 100)), transform_wgs84, "EPSG:4326")
        >>> labels_reprojected = read.read_reproject(
        ...     labels,
        ...     dst_crs="EPSG:32630",
        ...     resampling=rasterio.warp.Resampling.nearest  # Preserve class labels
        ... )

    Note:
        - Performance: Reads input data with 3-pixel buffer to avoid edge artifacts
        - Optimization: Detects no-op cases (same CRS + resolution + alignment) and
          uses faster read_from_window instead
        - Boolean handling: Converts bool → float32 → interpolate → threshold > 0.5 → bool
        - Multi-dimensional: Processes each (time, band) slice independently
        - Memory: Output array allocated upfront and filled via rasterio.warp.reproject
        - Non-intersecting: Returns nodata-filled array if source doesn't overlap destination
    """

    # The setup (output grid + alloc + intersection check) and warp loop are
    # extracted into `_reproject_setup` / `_reproject_finalize` so the async
    # sibling in `georeader.asyncread` can share the non-I/O code paths.
    plan = _reproject_setup(
        data_in=data_in,
        dst_crs=dst_crs,
        bounds=bounds,
        resolution_dst_crs=resolution_dst_crs,
        dst_transform=dst_transform,
        window_out=window_out,
        dtype_dst=dtype_dst,
        dst_nodata=dst_nodata,
    )

    # Same-CRS aligned-grid fast path: skip warp, just window-read.
    if plan.fast_path_window is not None:
        return read_from_window(
            data_in, plan.fast_path_window, return_only_data=return_only_data, trigger_load=True
        )

    # Source doesn't overlap destination → return the pre-allocated nodata fill.
    if plan.nonintersecting:
        return GeoTensor(
            plan.destination,
            transform=plan.dst_transform,
            crs=plan.dst_crs,
            fill_value_default=plan.dst_nodata,
        )

    # Windowed read of the input region that will contribute to the output,
    # with a 3-pixel buffer for interpolation edge handling.
    if not isinstance(data_in, GeoTensor):
        geotensor_in = read_from_polygon(
            data_in,
            plan.polygon_dst_crs,
            crs_polygon=plan.dst_crs,
            pad_add=(3, 3),
            return_only_data=False,
            trigger_load=True,
        )
    else:
        geotensor_in = data_in

    return _reproject_finalize(geotensor_in, plan, resampling=resampling, return_only_data=return_only_data)


def read_from_tile(
    data: GeoData,
    x: int,
    y: int,
    z: int,
    dst_crs: Optional[Any] = WEB_MERCATOR_CRS,
    out_shape: Optional[Tuple[int, int]] = (SIZE_DEFAULT, SIZE_DEFAULT),
    resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None,
    assert_if_not_intersects: bool = False,
) -> Optional[GeoTensor]:
    """
    Read raster data corresponding to a Web Mercator (XYZ) tile for web mapping applications.

    This function extracts and optionally reprojects raster data to match XYZ tile coordinates
    used by web mapping services (OpenStreetMap, Google Maps, Mapbox, etc.). It's the primary
    interface for:
    - Creating tile servers from arbitrary raster data
    - Building custom web map overlays from satellite imagery
    - Generating tiles for Leaflet, OpenLayers, or Mapbox GL JS
    - Creating tile caches for faster web mapping performance
    - Converting between different tile schemas and CRS

    XYZ tiles follow the Slippy Map / TMS convention where:
    - The world is divided into 2^z × 2^z tiles at zoom level z
    - Tile (0, 0) is at the top-left (northwest corner)
    - x increases eastward (0 to 2^z - 1)
    - y increases southward (0 to 2^z - 1)
    - Each tile represents the same geographic area at different resolutions

    The function handles the complete tile workflow:
    1. Calculate tile bounds in Web Mercator (EPSG:3857)
    2. Check if tile intersects the raster footprint
    3. Extract data with optional reprojection to destination CRS
    4. Resize/resample to standard tile dimensions (typically 256×256)

    Algorithm:
    1. Convert (x, y, z) to geographic bounds using mercantile
    2. Check intersection with data footprint (skip if no overlap)
    3. If reader has read_from_tile method, delegate to it (optimized path)
    4. Otherwise, read polygon extent and reproject/resize as needed
    5. Return tile with correct georeferencing in dst_crs

    Args:
        data (GeoData): Input georeferenced raster data with spatial reference (crs, transform).
            Can be in any CRS; the function handles transformation to tile coordinates.
        x (int): Tile column index (0 to 2^z - 1). Increases eastward from the prime meridian.
            At z=0, x=0 covers the entire world. At z=1, x=0 is western hemisphere.
        y (int): Tile row index (0 to 2^z - 1). Increases southward from the north pole.
            At z=0, y=0 covers the entire world. At z=1, y=0 is northern hemisphere.
        z (int): Zoom level (typically 0-22). Determines tile resolution:
            - z=0: 1 tile for entire world (~40,075 km at equator)
            - z=1: 2×2 = 4 tiles
            - z=10: 1024×1024 = 1,048,576 tiles
            - z=15: ~2.4 meters/pixel at equator
            - z=20: ~7.5 cm/pixel at equator
        dst_crs (Optional[Any], optional): Output coordinate reference system.
            Defaults to WEB_MERCATOR_CRS (EPSG:3857) which is standard for web maps.
            Can be set to None to use data's native CRS (less common for web tiles).
        out_shape (Optional[Tuple[int, int]], optional): Output tile dimensions as (height, width).
            Defaults to (SIZE_DEFAULT, SIZE_DEFAULT) which is typically (256, 256).
            Standard tile sizes: 256×256 (most common), 512×512 (retina), 128×128 (rare).
            If None, output size matches the native resolution in the tile extent.
        resolution_dst_crs (Optional[Union[float, Tuple[float, float]]], optional):
            Target resolution in dst_crs units (meters for EPSG:3857, degrees for WGS84).
            Defaults to None. If both out_shape and resolution_dst_crs are None, uses
            native data resolution. If out_shape is provided, this parameter is ignored.
        assert_if_not_intersects (bool, optional): If True, raises AssertionError when
            tile doesn't intersect data footprint. If False, returns None for non-intersecting
            tiles (useful for tile servers that expect None for empty tiles). Defaults to False.

    Returns:
        Optional[GeoTensor]:
            - If tile intersects data: GeoTensor with shape (bands, height, width) or
              (height, width), georeferenced to dst_crs at the tile's location
            - If tile doesn't intersect: None (or raises AssertionError if assert_if_not_intersects=True)

    Examples:
        >>> from georeader import RasterioReader, read
        >>> import rasterio
        >>>
        >>> # Example 1: Generate standard 256×256 web tile
        >>> with rasterio.open('sentinel2_spain.tif') as src:
        ...     reader = RasterioReader(src)
        ...     # Tile covering Madrid at zoom 12
        ...     tile = read.read_from_tile(reader, x=2046, y=1537, z=12)
        ...     print(f"Tile shape: {tile.shape}")  # (13, 256, 256) - 13 Sentinel-2 bands
        ...     print(f"Tile CRS: {tile.crs}")  # EPSG:3857 (Web Mercator)
        ...     print(f"Tile bounds: {tile.bounds}")  # Bounds in Web Mercator meters

        >>> # Example 2: High-resolution retina tile (512×512)
        >>> tile_retina = read.read_from_tile(reader, x=2046, y=1537, z=12,
        ...                                    out_shape=(512, 512))
        >>> # Twice the resolution for high-DPI displays

        >>> # Example 3: Tile server implementation
        >>> def get_tile(z, x, y, raster_path):
        ...     '''Simple tile server endpoint'''
        ...     with rasterio.open(raster_path) as src:
        ...         reader = RasterioReader(src)
        ...         tile = read.read_from_tile(reader, x=x, y=y, z=z)
        ...         if tile is None:
        ...             return None  # Empty tile (outside data extent)
        ...         return tile.values  # Return as numpy array for rendering
        >>>
        >>> # Usage: tile = get_tile(12, 2046, 1537, 'sentinel2.tif')

        >>> # Example 4: Generate tile at native CRS (less common)
        >>> # Useful when serving tiles in non-Web Mercator projections
        >>> tile_utm = read.read_from_tile(reader, x=2046, y=1537, z=12,
        ...                                 dst_crs=None)  # Uses data's native CRS
        >>> print(f"Native CRS: {tile_utm.crs}")

        >>> # Example 5: Tile generation across zoom levels
        >>> # Generate tiles for a pyramid (zoom levels 8-14)
        >>> import mercantile
        >>> bounds_wgs84 = (-3.75, 40.35, -3.65, 40.50)  # Madrid area
        >>> for z in range(8, 15):
        ...     # Get tiles covering the area at this zoom
        ...     tiles = list(mercantile.tiles(*bounds_wgs84, z))
        ...     print(f"Zoom {z}: {len(tiles)} tiles")
        ...     for tile_coords in tiles:
        ...         tile_data = read.read_from_tile(reader,
        ...                                         x=tile_coords.x,
        ...                                         y=tile_coords.y,
        ...                                         z=tile_coords.z)
        ...         if tile_data is not None:
        ...             # Save tile to disk: tiles/{z}/{x}/{y}.png
        ...             # tile_data.save(f'tiles/{z}/{tile_coords.x}/{tile_coords.y}.tif')
        ...             pass

        >>> # Example 6: Check tile coverage before processing
        >>> tile_check = read.read_from_tile(reader, x=0, y=0, z=5,
        ...                                   assert_if_not_intersects=True)
        >>> # Raises AssertionError if tile doesn't intersect data
        >>> # Useful for validating tile requests

    References:
        - OSM Slippy Map: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
        - XYZ Tiles: https://en.wikipedia.org/wiki/Tiled_web_map
        - Mercantile library: https://github.com/mapbox/mercantile
        - Web Mercator: https://epsg.io/3857

    Note:
        - Tiles are in EPSG:3857 by default (required for most web mapping libraries)
        - The function uses mercantile to convert tile coordinates to geographic bounds
        - For non-intersecting tiles, returns None (standard behavior for tile servers)
        - Output size defaults to 256×256 (standard for web maps since Google Maps)
        - Optimized readers may implement read_from_tile for better performance
        - Tile coordinates follow TMS/XYZ convention (y increases southward)
    """
    bounds_wgs = mercantile.xy_bounds(int(x), int(y), int(z))
    polygon_crs_webmercator = box(bounds_wgs.left, bounds_wgs.bottom, bounds_wgs.right, bounds_wgs.top)

    intersects = polygon_crs_webmercator.intersects(data.footprint(crs=WEB_MERCATOR_CRS))

    if not intersects:
        assert not assert_if_not_intersects, "Tile does not intersect data"
        return  # Non-intersecting tile — return None (standard tile-server behaviour)

    if out_shape is not None and hasattr(data, "read_from_tile"):
        return data.read_from_tile(x, y, z, dst_crs=dst_crs, out_shape=out_shape)

    if dst_crs is None:
        dst_crs = data.crs

    if window_utils.compare_crs(data.crs, dst_crs) and (out_shape is None) and (resolution_dst_crs is None):
        # read from polygon handles the case where the data does not intersect the polygon
        return read_from_polygon(data, polygon_crs_webmercator, WEB_MERCATOR_CRS, window_surrounding=True).load()

    if out_shape is not None:
        polygon_crs_dst = window_utils.polygon_to_crs(polygon_crs_webmercator, WEB_MERCATOR_CRS, dst_crs)
        bounds_dst = polygon_crs_dst.bounds
        dst_transform = rasterio.transform.from_bounds(*bounds_dst, width=out_shape[1], height=out_shape[0])
        window_data = rasterio.windows.Window(0, 0, width=out_shape[1], height=out_shape[0])
    else:
        if resolution_dst_crs is not None:
            if isinstance(resolution_dst_crs, numbers.Number):
                resolution_dst_crs = (abs(resolution_dst_crs), abs(resolution_dst_crs))

        polygon_crs_data = window_utils.polygon_to_crs(polygon_crs_webmercator, WEB_MERCATOR_CRS, data.crs)
        bounds_crs_data = polygon_crs_data.bounds

        in_height, in_width = data.shape[-2:]
        dst_transform, width, height = rasterio.warp.calculate_default_transform(
            data.crs, dst_crs, in_width, in_height, *bounds_crs_data, resolution=resolution_dst_crs
        )
        window_data = rasterio.windows.Window(0, 0, width=width, height=height)
        dst_transform, window_data = calculate_transform_window(data, dst_crs, resolution_dst_crs)

    return read_reproject(data, dst_crs=dst_crs, dst_transform=dst_transform, window_out=window_data)


def read_rpcs(
    input_npy: NDArray,
    rpcs: rasterio.rpc.RPC,
    fill_value_default: int = 0,
    dst_crs: Optional[Any] = None,
    resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    return_only_data: bool = False,
) -> GeoTensor:
    """
    This function georreferences an array using the RPCs.
        The RPCs are used to compute the transform from the input array to the destination crs.

        This function assumes that the RPCs are in EPSG:4326.

    Args:
        input_npy (NDArray): Array to georeference. It must have 2, 3 or 4 dimensions.
        rpcs (rasterio.rpc.RPC): RPCs to compute the transform.
        fill_value_default (int, optional): how to encode the nodata value. Defaults to 0.
        dst_crs (Optional[Any], optional): Destination crs. Defaults to None.
            If None, the dst_crs is the same as in the RPC polynomial (EPSG:4326).
        resampling (rasterio.warp.Resampling, optional): Resampling method.
            Defaults to rasterio.warp.Resampling.cubic_spline.
        return_only_data (bool, optional): If True it returns only the data. Defaults to False.

    Returns:
        GeoTensor: GeoTensor with the georeferenced array based on the RPCs.
    """

    isbool_dtypedst = input_npy.dtype == "bool"
    if isbool_dtypedst:
        fill_value_default = bool(fill_value_default)

    assert input_npy.ndim >= 2 and input_npy.ndim <= 4, "Input array must have 2, 3 or 4 dimensions"

    named_shape = OrderedDict(reversed(list(zip(["y", "x", "band", "time"], reversed(input_npy.shape)))))

    index_iter = [[(ns, i) for i in range(s)] for ns, s in named_shape.items() if ns not in ["x", "y"]]
    # e.g. if named_shape = {'time': 4, 'band': 2, 'x':10, 'y': 10} index_iter ->
    # [[('time', 0), ('time', 1), ('time', 2), ('time', 3)],
    #  [('band', 0), ('band', 1)]]

    if dst_crs is None:
        dst_crs = rasterio.crs.CRS.from_epsg(4326)

    src_crs = rasterio.crs.CRS.from_epsg(4326)

    if resolution_dst_crs is not None:
        if isinstance(resolution_dst_crs, float):
            resolution_dst_crs = (resolution_dst_crs, resolution_dst_crs)

    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs=None,
        dst_crs=dst_crs,
        width=input_npy.shape[-1],
        height=input_npy.shape[-2],
        resolution=resolution_dst_crs,
        rpcs=rpcs,
        dst_width=None,
        dst_height=None,
    )

    destination = np.full(
        input_npy.shape[:-2] + (dst_height, dst_width), fill_value=fill_value_default, dtype=input_npy.dtype
    )

    for current_select_tuple in itertools.product(*index_iter):
        # current_select_tuple = (('time', 0), ('band', 0))
        i_sel_tuple = tuple(t[1] for t in current_select_tuple)

        np_array_iter = input_npy[i_sel_tuple]
        if isbool_dtypedst:
            dst_iter_write = destination[i_sel_tuple].astype(np.float32)
            fill_value_default_iter = float(fill_value_default)
        else:
            dst_iter_write = destination[i_sel_tuple]
            fill_value_default_iter = fill_value_default

        rasterio.warp.reproject(
            np_array_iter,
            dst_iter_write,
            src_transform=None,
            rpcs=rpcs,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=fill_value_default_iter,
            dst_nodata=fill_value_default_iter,
            resampling=resampling,
        )

        if isbool_dtypedst:
            destination[i_sel_tuple] = dst_iter_write > 0.5

    if return_only_data:
        return destination

    return GeoTensor(destination, transform=dst_transform, crs=dst_crs, fill_value_default=fill_value_default)
