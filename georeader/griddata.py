"""
Irregular Grid Interpolation and Georeferencing Module.

This module provides functions for interpolating scattered (non-gridded) geographic
data onto regular grids, and for applying geolocation lookup tables (GLT). 

Coordinate Systems & Grid Types
-------------------------------

::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │   IRREGULAR vs REGULAR GRIDS                                            │
    │                                                                         │
    │   Irregular (Swath/Sensor)           Regular (Orthorectified)           │
    │   ─────────────────────────           ──────────────────────            │
    │                                                                         │
    │       ●  ●   ●  ●                     ┌──┬──┬──┬──┐                     │
    │     ●    ●  ●    ●                    ├──┼──┼──┼──┤                     │
    │      ●   ● ●  ●                       ├──┼──┼──┼──┤                     │
    │    ●   ●    ●   ●                     ├──┼──┼──┼──┤                     │
    │                                       └──┴──┴──┴──┘                     │
    │                                                                         │
    │   Each pixel has unique (lon, lat)    Fixed transform: pixel → geo      │
    │   Spacing varies with scan angle      Uniform spacing, axis-aligned     │
    │   Common in: pushbroom sensors        Required for: GIS, web maps       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Interpolation Methods
--------------------

The module uses :func:`scipy.interpolate.griddata` for interpolation:

::

    ┌────────────────────────────────────────────────────────────────────────┐
    │  INTERPOLATION METHOD COMPARISON                                       │
    │                                                                        │
    │  Method      │ Continuity │ Speed  │ Best For                          │
    │  ────────────┼────────────┼────────┼─────────────────────────────────  │
    │  "nearest"   │ C⁰         │ Fast   │ Categorical data, masks           │
    │  "linear"    │ C⁰         │ Medium │ Simple surfaces, quick preview    │
    │  "cubic"     │ C²         │ Slow   │ Smooth continuous data (default)  │
    │                                                                        │
    │  C⁰ = continuous but not differentiable (may have sharp edges)         │
    │  C² = smooth, twice differentiable (recommended for radiance/refl)     │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

Geolocation Lookup Tables (GLT)
------------------------------

Some sensors (like NASA EMIT) provide a GLT array that maps output grid pixels
to input sensor pixels. This is faster than interpolation for orthorectification:

::

    ┌────────────────────────────────────────────────────────────────────────┐
    │  GLT-BASED ORTHORECTIFICATION                                          │
    │                                                                        │
    │  Sensor Array (irregular)              Output Grid (regular)           │
    │  ┌───────────────────────┐             ┌──┬──┬──┬──┬──┐                │
    │  │  0   1   2   3   ...  │             │  │  │  │  │  │                │
    │  │                       │             ├──┼──┼──┼──┼──┤                │
    │  │ [r,c] = radiance      │     GLT     │  │██│██│  │  │                │
    │  │                       │  ────────►  ├──┼──┼──┼──┼──┤                │
    │  │                       │             │  │██│██│██│  │                │
    │  └───────────────────────┘             └──┴──┴──┴──┴──┘                │
    │                                                                        │
    │  GLT[0, i, j] = column in sensor array                                 │
    │  GLT[1, i, j] = row in sensor array                                    │
    │  output[i, j] = sensor[GLT[1,i,j], GLT[0,i,j]]                         │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘

Module Functions
---------------

Grid Interpolation:
    - :func:`reproject`: Core interpolation from lon/lat arrays to regular grid
    - :func:`read_reproject_like`: Match grid to existing GeoData
    - :func:`read_to_crs`: Auto-compute grid for given resolution

Grid Utilities:
    - :func:`meshgrid`: Generate coordinate arrays from transform
    - :func:`get_shape_transform_crs`: Compute output grid parameters
    - :func:`footprint`: Bounding polygon from lon/lat arrays
    - :func:`polygon_to_image_coords`: Map a (lon, lat) polygon back to image (col, row) coordinates

GLT Operations:
    - :func:`georreference`: Apply GLT for fast orthorectification

Example Workflow
---------------

Orthorectify PRISMA-style data with per-pixel coordinates::

    import numpy as np
    from georeader.griddata import read_to_crs
    
    # Hyperspectral radiance with irregular coordinates
    radiance = np.random.rand(1000, 1000, 285)  # (H, W, bands)
    lons = np.random.uniform(-122.5, -122.3, (1000, 1000))  # irregular
    lats = np.random.uniform(37.7, 37.9, (1000, 1000))      # irregular
    
    # Interpolate to regular 30m UTM grid
    ortho = read_to_crs(
        radiance, lons, lats,
        resolution_dst=30.0,  # 30 meters
        method="cubic"        # smooth interpolation
    )
    # ortho.shape: (285, H_out, W_out) - regular grid
    # ortho.crs: auto-detected UTM zone
    # ortho.transform: proper affine transform

See Also
--------
georeader.readers.emit : EMIT reader with built-in GLT handling
georeader.readers.prisma : PRISMA reader with built-in interpolation handling
georeader.read : Regular grid reprojection (for already-gridded data)

References
----------
- SciPy griddata: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
- NASA EMIT L2A Products: https://lpdaac.usgs.gov/products/emitl2arflv001/
"""
import georeader
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from georeader.abstract_reader import GeoData
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from georeader.window_utils import polygon_to_crs, transform_to_resolution_dst
from typing import Tuple, Union, Optional, Any
import rasterio
import rasterio.transform
import rasterio.warp
from georeader.geotensor import GeoTensor
import numbers
import numpy as np
from numpy.typing import NDArray
import math

METHOD_DEFAULT = "cubic"

def footprint(lons:NDArray, lats:NDArray) -> Polygon:
    """
    Returns the Polygon surrounding the given longitudes and latitudes

    Args:
        lons (np.array): 2D array of longitudes
        lats (np.array): 2D array of latitudes

    Returns:
        Polygon: Polygon surrounding the given longitudes and latitudes
    """
    lonsrav = lons.ravel()
    latsrav = lats.ravel()
    idxminlon = np.argmin(lonsrav)
    idxminlat = np.argmin(latsrav)
    idxmaxlon = np.argmax(lonsrav)
    idxmaxlat = np.argmax(latsrav)

    return Polygon([(lonsrav[idx],latsrav[idx]) for idx in [idxminlon, idxminlat, idxmaxlon, idxmaxlat]])


def polygon_to_image_coords(polygon: Union[Polygon, MultiPolygon],
                            lons: NDArray, lats: NDArray,
                            method: str = "linear") -> Union[Polygon, MultiPolygon]:
    """
    Map a polygon defined in the same CRS as ``lons``/``lats`` back to image
    (column, row) coordinates of an irregularly-sampled image.

    Inverts the per-pixel lon/lat lookup table so each polygon vertex is
    expressed in pixel space: ``x = column``, ``y = row``. The resulting
    geometry can be overlaid directly on the original ``(H, W)`` image (e.g.
    via ``matplotlib.pyplot.imshow`` with ``origin="upper"``) without any
    further reprojection.

    Algorithm Overview
    ------------------

    ::

        ┌────────────────────────────────────────────────────────────────────┐
        │  POLYGON → IMAGE COORDINATE INVERSION                              │
        │                                                                    │
        │  Known per-pixel LUT:                                              │
        │    pixel (r, c)  ──►  (lons[r, c], lats[r, c])                     │
        │                                                                    │
        │  We invert it at the polygon's vertices only:                      │
        │    1. Build Delaunay triangulation of the H·W LUT points (once).   │
        │    2. For each vertex (lon, lat), barycentric-interpolate          │
        │       (col, row) within the enclosing triangle.                    │
        │    3. Vertices outside the convex hull fall back to nearest-       │
        │       neighbor lookup (linear method only).                        │
        │                                                                    │
        │  The triangulation is built once and reused across all rings of    │
        │  all (multi)polygon parts.                                         │
        │                                                                    │
        └────────────────────────────────────────────────────────────────────┘

    Args:
        polygon: Shapely ``Polygon`` or ``MultiPolygon`` whose coordinates
            share the CRS of ``lons``/``lats`` (typically EPSG:4326). Holes
            (interior rings) are preserved.
        lons: 2D array of pixel longitudes, shape ``(H, W)``.
        lats: 2D array of pixel latitudes, shape ``(H, W)``.
        method: Inversion method.

            - ``"linear"`` (default): barycentric interpolation on a Delaunay
              triangulation of the LUT. Sub-pixel accurate. Vertices outside
              the convex hull of the LUT fall back to nearest-neighbor.
            - ``"nearest"``: snap each vertex to the closest pixel index.
              Faster and tolerant of vertices outside the LUT extent, but
              yields integer-valued coordinates.

    Returns:
        Polygon or MultiPolygon (matching the input type) with vertices in
        image ``(col, row)`` coordinates. An empty input returns an empty
        geometry of the same type.

    Raises:
        ValueError: If ``method`` is not ``"linear"`` or ``"nearest"``, if
            ``lons``/``lats`` have mismatched shapes, or if they are not 2D.
        TypeError: If ``polygon`` is not a Polygon or MultiPolygon.

    Examples
    --------
    Map a polygon to image coordinates and overlay on an image::

        >>> import numpy as np
        >>> from shapely.geometry import box
        >>> from georeader.griddata import polygon_to_image_coords
        >>>
        >>> # Regular grid: lon spans [0, 1] over W=30, lat spans [45, 46] over H=20
        >>> lons, lats = np.meshgrid(np.linspace(0, 1, 30), np.linspace(45, 46, 20))
        >>> pol = box(0.0, 45.0, 0.5, 45.5)
        >>> pol_img = polygon_to_image_coords(pol, lons, lats)
        >>> # pol_img has x in [0, 14.5] (cols) and y in [0, 9.5] (rows)

    See Also
    --------
    reproject : Forward direction (sensor → ortho via the same LUT).
    footprint : Bounding polygon of the LUT in lon/lat space.
    """
    if method not in ("linear", "nearest"):
        raise ValueError(
            f"method must be 'linear' or 'nearest', got {method!r}"
        )
    if lons.shape != lats.shape:
        raise ValueError(
            f"lons and lats must share shape, got {lons.shape} vs {lats.shape}"
        )
    if lons.ndim != 2:
        raise ValueError(f"lons and lats must be 2D, got ndim={lons.ndim}")
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise TypeError(
            f"polygon must be a Polygon or MultiPolygon, got {type(polygon).__name__}"
        )

    if polygon.is_empty:
        return type(polygon)()

    H, W = lons.shape
    rows, cols = np.mgrid[0:H, 0:W]
    pts = np.column_stack([lons.ravel(), lats.ravel()])
    cols_flat = cols.ravel().astype(float)
    rows_flat = rows.ravel().astype(float)

    if method == "linear":
        col_interp = LinearNDInterpolator(pts, cols_flat)
        row_interp = LinearNDInterpolator(pts, rows_flat)
        col_fallback = NearestNDInterpolator(pts, cols_flat)
        row_fallback = NearestNDInterpolator(pts, rows_flat)
    else:
        col_interp = NearestNDInterpolator(pts, cols_flat)
        row_interp = NearestNDInterpolator(pts, rows_flat)
        col_fallback = None
        row_fallback = None

    def _ring_to_image(ring: LinearRing) -> LinearRing:
        coords = np.asarray(ring.coords)
        x = coords[:, 0]
        y = coords[:, 1]
        col = np.asarray(col_interp(x, y), dtype=float)
        row = np.asarray(row_interp(x, y), dtype=float)
        if col_fallback is not None:
            nan = np.isnan(col) | np.isnan(row)
            if nan.any():
                col[nan] = col_fallback(x[nan], y[nan])
                row[nan] = row_fallback(x[nan], y[nan])
        return LinearRing(np.column_stack([col, row]))

    def _polygon_to_image(poly: Polygon) -> Polygon:
        if poly.is_empty:
            return Polygon()
        shell = _ring_to_image(poly.exterior)
        holes = [_ring_to_image(interior) for interior in poly.interiors]
        return Polygon(shell, holes)

    if isinstance(polygon, Polygon):
        return _polygon_to_image(polygon)

    return MultiPolygon([_polygon_to_image(p) for p in polygon.geoms])


# def bounds(lons:np.array, lats:np.array) -> Tuple[float, float, float, float]:
#     minx = np.min(lons)
#     maxx = np.max(lons)
#     miny = np.min(lats)
#     maxy = np.max(lats)
#     return minx, miny, maxx, maxy

def read_reproject_like(data:NDArray, lons: NDArray, lats:NDArray, 
                        data_like:GeoData, resolution_dst:Optional[Union[float, Tuple[float,float]]]=None,
                        fill_value_default:Optional[float]=None,
                        crs:Optional[Any]="EPSG:4326",
                        method:str=METHOD_DEFAULT) -> GeoTensor:
    """
    Reprojects data to the same crs, transform and shape as data_like

    Args:
        data (Array): input data 2D or 3D in the form (height, width, bands)
        lons (Array): 2D array of longitudes
        lats (Array): 2D array of latitudes
        data_like (GeoData): GeoData to reproject to
        resolution_dst (Optional[Union[float, Tuple[float,float]]], optional): If provided, the output
            resolution will be set to this value. Otherwise, the output resolution will be the same
            as data_like. Defaults to None.
        fill_value_default (Optional[float], optional): fill value. Defaults to None.
        crs (Optional[Any], optional): Input crs. Defaults to "EPSG:4326".
        method (str, optional): Interpolation method. Defaults to "cubic". One of
            "nearest", "linear", "cubic".

    Returns:
        GeoTensor: with reprojected data
    """
    width = data_like.shape[-1]
    height = data_like.shape[-2]
    transform = data_like.transform
    dst_crs = data_like.crs
    if resolution_dst is not None:
        transform = transform_to_resolution_dst(transform, resolution_dst)

    fill_value_default = fill_value_default or data_like.fill_value_default
    return reproject(data, lons, lats, width, height, transform, dst_crs, 
                     fill_value_default=fill_value_default, crs=crs,
                     method=method)


def read_to_crs(data:NDArray, lons: NDArray, lats:NDArray, 
                resolution_dst:Union[float, Tuple[float,float]], 
                dst_crs:Optional[Any]=None,fill_value_default:float=-1,
                crs:Optional[Any]="EPSG:4326",
                method:str=METHOD_DEFAULT) -> GeoTensor:
    """
    Reprojects data to the given dst_crs figuring out the transform and shape.

    Args:
        data (Array): 2D or 3D in the form (H, W, bands)
        lons (Array): 2D array of longitudes (H, W).
        lats (Array): 2D array of latitudes (H, W).
        resolution_dst (Union[float, Tuple[float,float]]): Output resolution
        dst_crs (Optional[Any], optional): Output crs. If None, 
            the dst_crs will be the UTM crs of the center of the data. Defaults to None.
        fill_value_default (float, optional): fill value. Defaults to -1.
        crs (_type_, optional): Input crs. Defaults to "EPSG:4326".
        method (str, optional): Interpolation method. Defaults to "cubic". One of
            "nearest", "linear", "cubic". (See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata)

    Returns:
        GeoTensor: with reprojected data
    """
    width, height, transform, dst_crs = get_shape_transform_crs(lons, lats, 
                                                                resolution_dst=resolution_dst, 
                                                                dst_crs=dst_crs, crs=crs)

    return reproject(data, lons=lons, lats=lats, width=width, 
                     height=height, transform=transform, dst_crs=dst_crs,
                     fill_value_default=fill_value_default,
                     crs=crs, method=method)

def get_shape_transform_crs(lons: NDArray, lats:NDArray, 
                            resolution_dst:Union[float, Tuple[float,float]], 
                            dst_crs:Optional[Any]=None,
                            crs:Optional[Any]="EPSG:4326") -> Tuple[int, int, rasterio.transform.Affine, Any]:
    """
    Get the shape, transform and crs for the given lons and lats and resolution_dst.    

    Args:
        lons (NDArray): 2D array of longitudes (H, W).
        lats (NDArray): 2D array of latitudes (H, W).
        resolution_dst (Union[float, Tuple[float,float]]): Output resolution.
            If a single float is provided, the resolution will be (resolution_dst, resolution_dst).
        dst_crs (Optional[Any], optional): Output crs. If None,
            the dst_crs will be the UTM crs of the center of the data. Defaults to None.
        crs (Any, optional): Input crs. Defaults to "EPSG:4326".

    Returns:
        Tuple[int, int, rasterio.transform.Affine, Any]: width, height, transform and dst_crs.
    """
    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))
    
    # Figure out UTM crs
    if dst_crs is None:
        mean_lat = np.nanmean(lats)
        mean_lon = np.nanmean(lons)
        dst_crs = georeader.get_utm_epsg((mean_lon, mean_lat), 
                                         crs_point_or_geom=crs)

    # Figure out transform
    pol = footprint(lons, lats)
    pol_dst_crs = polygon_to_crs(pol, crs_polygon=crs, dst_crs=dst_crs)
    minx, miny, maxx, maxy = pol_dst_crs.bounds

    # Add the resolution to the max values to get the correct shape.
    maxx = maxx + resolution_dst[0]
    miny = miny - resolution_dst[1]
    transform = rasterio.transform.from_origin(minx, maxy, resolution_dst[0], resolution_dst[1])

    # resolution_dst= res(transform)
    width = math.ceil(abs((maxx -minx) / resolution_dst[0]))
    height = math.ceil(abs((maxy - miny) / resolution_dst[1]))
    return width, height, transform, dst_crs


def reproject(data:NDArray, lons: NDArray, lats: NDArray,
              width:int, height:int, transform:rasterio.transform.Affine,
              dst_crs:Any, crs:Optional[Any]="EPSG:4326", fill_value_default=-1,
              method:str=METHOD_DEFAULT) -> GeoTensor:
    """
    Interpolate scattered data to a regular georeferenced grid.

    This is the core function for converting irregularly-sampled geographic
    data (e.g., from pushbroom sensors or point observations) to a regular
    grid suitable for analysis and visualization.

    Algorithm Overview
    ------------------

    ::

        ┌────────────────────────────────────────────────────────────────────┐
        │  INTERPOLATION WORKFLOW                                            │
        │                                                                    │
        │  1. Flatten inputs                                                 │
        │     data: (H, W, C) → (H×W, C)  [or (H, W) → (H×W,)]               │
        │     lons/lats: (H, W) → (H×W,)                                     │
        │                                                                    │
        │  2. Generate output coordinate grid                                │
        │     meshgrid(transform, width, height) → (xs, ys)                  │
        │     Transform xs, ys from dst_crs to input crs if different        │
        │                                                                    │
        │  3. Call scipy.interpolate.griddata                                │
        │     points = (lons_flat, lats_flat)                                │
        │     values = data_flat                                             │
        │     xi = (xs_grid, ys_grid)                                        │
        │     result = griddata(points, values, xi, method=method)           │
        │                                                                    │
        │  4. Reshape and handle nodata                                      │
        │     Fill NaN regions with fill_value_default                       │
        │     Transpose to (C, H, W) if multi-band                           │
        │                                                                    │
        └────────────────────────────────────────────────────────────────────┘

    Interpolation Methods
    ---------------------

    - ``"nearest"``: Voronoi cell assignment. Fast but produces blocky output.
      Use for categorical data or masks.

    - ``"linear"``: Barycentric interpolation on Delaunay triangulation.
      Continuous but not smooth (C⁰ continuity).

    - ``"cubic"`` (default): Clough-Tocher scheme on Delaunay triangulation.
      Smooth and twice-differentiable (C² continuity). Best for continuous
      data like radiance or reflectance. Slower than linear.

    Args:
        data: Input array, either:
            - 2D: (H, W) single-band image
            - 3D: (H, W, C) multi-band image with C channels
            Note: This is **height × width × channels** order, not CHW!
        lons: Longitude coordinates for each pixel, shape (H, W).
            Must be same spatial shape as data.
        lats: Latitude coordinates for each pixel, shape (H, W).
            Must be same spatial shape as data.
        width: Output grid width in pixels.
        height: Output grid height in pixels.
        transform: Output affine transform mapping pixel to CRS coordinates.
        dst_crs: Output coordinate reference system (e.g., "EPSG:32610").
        crs: CRS of input lon/lat arrays. Default "EPSG:4326" (WGS84).
        fill_value_default (float): Value for pixels outside convex hull of input
            points. Default -1.
        method: Interpolation method: "nearest", "linear", or "cubic".
            Default "cubic".

    Returns:
        GeoTensor with shape (H, W) or (C, H, W), georeferenced with
        ``transform`` and ``dst_crs``.

    Raises:
        ValueError: If data is not 2D or 3D.

    Examples
    --------
    Orthorectify single-band thermal data::

        >>> import numpy as np
        >>> from georeader.griddata import reproject
        >>> import rasterio
        >>> 
        >>> # Simulated thermal data with irregular coords
        >>> temperature = np.random.uniform(280, 320, (100, 100))  # Kelvin
        >>> lons = np.linspace(-122.5, -122.3, 100)[None, :] + np.random.normal(0, 0.001, (100, 100))
        >>> lats = np.linspace(37.7, 37.9, 100)[:, None] + np.random.normal(0, 0.001, (100, 100))
        >>> 
        >>> # Define output grid (UTM Zone 10N, 100m resolution)
        >>> transform = rasterio.transform.from_origin(550000, 4200000, 100, 100)
        >>> ortho = reproject(temperature, lons, lats, 
        ...                   width=200, height=200, 
        ...                   transform=transform, 
        ...                   dst_crs="EPSG:32610")
        >>> print(ortho.shape, ortho.crs)
        (200, 200) EPSG:32610

    Multi-band hyperspectral orthorectification::

        >>> # Shape: (H, W, bands) - note the axis order!
        >>> radiance = np.random.rand(1000, 1000, 285)
        >>> lons = np.load("pixel_lons.npy")  # (1000, 1000)
        >>> lats = np.load("pixel_lats.npy")  # (1000, 1000)
        >>> 
        >>> ortho = reproject(radiance, lons, lats,
        ...                   width=500, height=500,
        ...                   transform=my_transform,
        ...                   dst_crs="EPSG:32610",
        ...                   method="cubic")
        >>> # Output shape: (285, 500, 500) - transposed to (C, H, W)

    Warning
    -------
    - Input data must be (H, W) or (H, W, C), NOT (C, H, W)
    - Cubic interpolation can be slow for large arrays (>1M points)
    - Output pixels outside the convex hull of input points get fill_value

    See Also
    --------
    read_reproject_like : Match output grid to existing GeoData
    read_to_crs : Auto-compute grid dimensions from resolution
    georreference : Fast orthorectification using GLT (no interpolation)
    """
    data = data.squeeze()
    if len(data.shape) == 3:
        data_ravel = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    elif len(data.shape) == 2:
        data_ravel = data.ravel()
    else:
        raise ValueError("Data shape not supported")
    
    # Generate the meshgrid of lons and lats to interpolate the data
    lonsdst, latssdst = meshgrid(transform, width, height, source_crs=dst_crs, dst_crs=crs)

    # interpfun = CloughTocher2DInterpolator(list(zip(lons.ravel(), lats.ravel())), 
    #                                        data_ravel)
    
    # dataout = interpfun(lonsdst, latssdst) # (H, W) or (H, W, C)

    dataout = griddata((lons.ravel(), lats.ravel()), data_ravel, 
                       (lonsdst, latssdst), method=method)

    nanvals = np.isnan(dataout)
    if np.any(nanvals):
        dataout[nanvals] = fill_value_default
    
    # transpose if 3D to (C, H, W) format
    if len(data.shape) == 3:
        dataout = np.transpose(dataout, (2, 0, 1))

    return GeoTensor(dataout, transform=transform, 
                     crs=dst_crs, fill_value_default=fill_value_default)


def meshgrid(transform:rasterio.transform.Affine, width:int, height:int, 
             source_crs:Optional[Any]=None, dst_crs:Optional[Any]=None) -> Tuple[NDArray, NDArray]:
    """
    Generate the meshgrid of geographic coordinates from the transform.
    If source_crs and dst_crs are provided, the meshgrid will be transformed to the dst_crs.

    Args:
        transform (rasterio.transform.Affine): transform
        width (int): width
        height (int): height
        source_crs (Optional[Any], optional): source crs. Defaults to None.
        dst_crs (Optional[Any], optional): destination crs. Defaults to None.

    Returns:
        Tuple[NDArray, NDArray]: 2D arrays of xs and ys coordinates
    """
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs= np.array(xs)
    ys = np.array(ys)

    if dst_crs is not None:
        assert source_crs is not None, "source_crs must be provided if dst_crs is provided"
        xs, ys = rasterio.warp.transform(source_crs, dst_crs, xs.ravel(),ys.ravel())
        xs = np.array(xs).reshape(height, width)
        ys = np.array(ys).reshape(height, width)

    return xs, ys


def georreference(glt:GeoTensor, data:NDArray, valid_glt:Optional[NDArray] = None,
                  fill_value_default:Optional[Union[int,float]]=None) -> GeoTensor:
    """
    Apply a Geolocation Lookup Table (GLT) to orthorectify sensor data.

    This function performs fast, exact orthorectification by using a pre-computed
    lookup table that maps output grid pixels to input sensor pixels. Unlike
    interpolation-based methods, GLT orthorectification preserves original
    pixel values without resampling artifacts.

    GLT Structure
    -------------

    ::

        ┌────────────────────────────────────────────────────────────────────┐
        │  GLT ARRAY STRUCTURE                                               │
        │                                                                    │
        │  glt.shape = (2, H_out, W_out)                                     │
        │                                                                    │
        │  glt[0, i, j] = source column (x-index in sensor array)            │
        │  glt[1, i, j] = source row    (y-index in sensor array)            │
        │                                                                    │
        │  For each output pixel (i, j):                                     │
        │    output[..., i, j] = data[..., glt[1,i,j], glt[0,i,j]]           │
        │                                                                    │
        │  ┌──────────────────────┐      ┌──────────────────────┐            │
        │  │    Sensor Array     │       │   Output Grid        │            │
        │  │    (raw data)       │       │   (orthorectified)   │            │
        │  │   ┌───┬───┬───┐     │       │  ┌──┬──┬──┬──┐       │            │
        │  │   │ A │ B │ C │     │       │  │  │ A│ B│  │       │            │
        │  │   ├───┼───┼───┤     │ GLT   │  ├──┼──┼──┼──┤       │            │
        │  │   │ D │ E │ F │  ───────►   │  │  │ D│ E│ F│       │            │
        │  │   ├───┼───┼───┤     │       │  ├──┼──┼──┼──┤       │            │
        │  │   │ G │ H │ I │     │       │  │ G│ H│ I│  │       │            │
        │  │   └───┴───┴───┘     │       │  └──┴──┴──┴──┘       │            │
        │  └──────────────────────┘      └──────────────────────┘            │
        │                                                                    │
        │  GLT handles: terrain distortion, sensor geometry, Earth curvature │
        │  Invalid pixels: glt values = fill_value_default (typically -1)    │
        │                                                                    │
        └────────────────────────────────────────────────────────────────────┘

    Common Use Cases
    ----------------

    1. **Post-processing orthorectified products**: If you compute spectral
       indices or run ML inference on sensor-geometry data, use this to
       orthorectify the results without re-processing the full cube.

    2. **Custom band math**: Calculate a derived product from raw bands,
       then apply GLT for geographic alignment.

    3. **Mask application**: Create masks in sensor space, then georeference
       to match the ortho grid for overlay.

    Args:
        glt: GLT GeoTensor with shape (2, H_out, W_out). Contains the
            (column, row) indices mapping each output pixel to the sensor
            array. Must have valid transform and CRS for the output grid.
        data: Sensor-space data array with shape (H_sensor, W_sensor) or
            (C, H_sensor, W_sensor). Will be indexed by GLT values.
        valid_glt: Optional boolean mask of shape (H_out, W_out) indicating
            valid GLT pixels. If None, auto-computed as pixels where both
            GLT channels differ from fill_value_default.
        fill_value_default: Fill value for output pixels with invalid GLT.
            If None, defaults to 0.

    Returns:
        GeoTensor with shape (H_out, W_out) or (C, H_out, W_out) matching
        the GLT's spatial dimensions, with transform and CRS from the GLT.

    Raises:
        ValueError: If data shape is not 2D or 3D.

    Examples
    --------
    Orthorectify a spectral index computed in sensor space::

        >>> import numpy as np
        >>> from georeader.griddata import georreference
        >>> 
        >>> # Load EMIT data (sensor geometry)
        >>> emit = EMITImage("EMIT_L2A_file.nc")
        >>> radiance = emit.load_raw()  # (285, 1242, 1280) - sensor space
        >>> 
        >>> # Compute NDVI in sensor space (faster than on ortho grid)
        >>> nir = radiance[120]  # ~850nm
        >>> red = radiance[60]   # ~665nm  
        >>> ndvi = (nir - red) / (nir + red + 1e-6)  # (1242, 1280)
        >>> 
        >>> # Get GLT and orthorectify
        >>> glt = emit.load_glt()  # (2, H_ortho, W_ortho)
        >>> ndvi_ortho = georreference(glt, ndvi)
        >>> 
        >>> print(f"Sensor: {ndvi.shape}, Ortho: {ndvi_ortho.shape}")
        Sensor: (1242, 1280), Ortho: (1500, 1600)

    Orthorectify multi-band processed data::

        >>> # ML model output in sensor space
        >>> class_probs = model.predict(radiance)  # (10, 1242, 1280)
        >>> 
        >>> # Orthorectify all probability bands at once
        >>> probs_ortho = georreference(glt, class_probs, fill_value_default=0)
        >>> print(probs_ortho.shape)  # (10, H_ortho, W_ortho)

    Notes
    -----
    - This function does NOT interpolate - it performs exact pixel lookup
    - Much faster than :func:`reproject` for sensor→ortho conversion
    - The GLT must be pre-computed (provided by data producer or computed
      from RPC/sensor model)
    - Invalid GLT values (outside valid sensor range) must be marked with
      fill_value_default in the GLT

    See Also
    --------
    reproject : Interpolation-based orthorectification (no GLT required)
    georeader.readers.emit : EMIT reader that provides GLT automatically
    """
    spatial_shape = glt.shape[-2:]
    if len(data.shape) == 3:
        shape = data.shape[:-2] + spatial_shape
    elif len(data.shape) == 2:
        shape = spatial_shape
    else:
        raise ValueError(f"Data shape {data.shape} not supported")

    if fill_value_default is None:
        fill_value_default = 0
    outdat = np.full(shape, dtype=data.dtype, 
                      fill_value=fill_value_default)

    if valid_glt is None:
        valid_glt = np.all(glt.values != glt.fill_value_default, axis=0)
    
    if len(data.shape) == 3:
        outdat[:, valid_glt] = data[:, glt.values[1, valid_glt], 
                                            glt.values[0, valid_glt]]
    else:
        outdat[valid_glt] = data[glt.values[1, valid_glt], 
                                 glt.values[0, valid_glt]]
        
    return GeoTensor(values=outdat, transform=glt.transform, crs=glt.crs,
                     fill_value_default=fill_value_default)