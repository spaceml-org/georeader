"""
Vectorize Module: Convert raster masks to vector polygons.

This module provides functions to extract polygon geometries from binary
raster masks. Essential for converting classification results, segmentation
outputs, and masks into GIS-compatible vector formats.

Vectorization Process
---------------------

Converting binary rasters to polygon geometries::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    VECTORIZATION PROCESS                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Raster (Binary Mask)                Vector (Polygons)                  │
    │  ────────────────────                ─────────────────                  │
    │                                                                          │
    │  ┌─┬─┬─┬─┬─┬─┬─┬─┐                       ╔═══════════╗                  │
    │  │0│0│0│1│1│1│0│0│                      ╔╝           ╚╗                 │
    │  ├─┼─┼─┼─┼─┼─┼─┼─┤                     ╔╝             ╚╗                │
    │  │0│0│1│1│1│1│1│0│   ═══════════►     ╔╝               ╚╗               │
    │  ├─┼─┼─┼─┼─┼─┼─┼─┤   Vectorize        ║    Polygon 1    ║               │
    │  │0│1│1│1│1│1│1│0│                    ╚╗               ╔╝               │
    │  ├─┼─┼─┼─┼─┼─┼─┼─┤                     ╚╗             ╔╝                │
    │  │0│0│1│1│1│1│0│0│                      ╚╗           ╔╝                 │
    │  └─┴─┴─┴─┴─┴─┴─┴─┘                       ╚═══════════╝                  │
    │                                                                          │
    │  1 = foreground (vectorized)                                            │
    │  0 = background (ignored)                                               │
    └─────────────────────────────────────────────────────────────────────────┘

Polygon Simplification
----------------------

Reducing vertex count while preserving shape::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │              POLYGON SIMPLIFICATION (tolerance parameter)                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Raw (pixelated)              Simplified (tolerance=1)                  │
    │  ────────────────              ──────────────────────                   │
    │                                                                          │
    │  ┌─┐                                ╭───────╮                            │
    │  │ └─┐                             ╱         ╲                           │
    │  │   └─┐                          ╱           ╲                          │
    │  │     └─┐   ────────────►       │             │    Fewer vertices,     │
    │  │       │   simplify            │             │    smoother edges      │
    │  │     ┌─┘                        ╲           ╱                          │
    │  │   ┌─┘                           ╲         ╱                           │
    │  └───┘                              ╰───────╯                            │
    │                                                                          │
    │  tolerance=0: Keep all vertices (staircase pattern)                     │
    │  tolerance=1: Simplify ~1 pixel tolerance (DEFAULT)                     │
    │  tolerance>1: More aggressive simplification                            │
    └─────────────────────────────────────────────────────────────────────────┘

Filtering Options
-----------------

Removing small or unwanted polygons::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    POLYGON FILTERING                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Parameters:                                                             │
    │  ───────────                                                             │
    │                                                                          │
    │  min_area=25.5 (default)    Remove polygons smaller than ~5x5 pixels    │
    │                             Helps filter noise and artifacts             │
    │                                                                          │
    │  polygon_buffer=0           Buffer/erode polygons by N pixels           │
    │                             Positive: expand                             │
    │                             Negative: shrink (erode)                     │
    │                                                                          │
    │  Before (min_area=0):                After (min_area=25):               │
    │  ┌────────────────────┐              ┌────────────────────┐             │
    │  │  ■   ┌───────┐     │              │      ┌───────┐     │             │
    │  │ ■ ■  │       │  ■  │   ═══════►   │      │       │     │             │
    │  │      │       │     │   Filter     │      │       │     │             │
    │  │ ■    └───────┘     │              │      └───────┘     │             │
    │  └────────────────────┘              └────────────────────┘             │
    │     ↑ small polygons removed                                            │
    └─────────────────────────────────────────────────────────────────────────┘

Module Functions Overview
-------------------------

Vectorization:
    - :func:`get_polygons`: Extract polygons from binary mask
    - :func:`vectorize_raster`: Vectorize with geographic coordinates

Utilities:
    - Automatic CRS handling from GeoData inputs
    - Integration with shapely for geometry operations

Quick Start
-----------

Extract polygons from a binary mask::

    from georeader import vectorize
    import numpy as np

    # Binary mask (e.g., from classification)
    mask = (classified_image == 1).astype(np.uint8)

    # Extract polygons
    polygons = vectorize.get_polygons(
        mask,
        min_area=100,        # Minimum 10x10 pixel area
        tolerance=1.5,       # Simplification tolerance
        transform=transform  # Affine transform for georeferencing
    )

    # Polygons are in CRS units (georeferenced)
    for poly in polygons:
        print(f"Area: {poly.area} sq. CRS units")

Vectorize a GeoTensor mask::

    # GeoTensor carries its own transform
    polygons = vectorize.get_polygons(
        water_mask_geotensor,  # GeoTensor with transform
        min_area=50,
        polygon_buffer=-1      # Erode by 1 pixel
    )

See Also
--------
georeader.rasterize : Inverse operation (vector → raster)
georeader.geotensor : Input format with transform
rasterio.features.shapes : Underlying implementation

References
----------
- Rasterio shapes: https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html
- Shapely simplify: https://shapely.readthedocs.io/en/latest/manual.html#object.simplify
"""
from rasterio import features
import rasterio
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import numpy as np
from typing import List, Optional, Union, Tuple
from georeader.abstract_reader import GeoData


def get_polygons(binary_mask: Union[np.ndarray, GeoData], min_area:float=25.5,
                 polygon_buffer:int=0, tolerance:float=1., transform: Optional[rasterio.Affine]=None) -> List[Polygon]:
    """
    Convert a binary raster mask to vector polygons.

    Extracts connected regions of True/nonzero pixels as polygon geometries.
    Includes options for filtering small polygons, buffering/eroding boundaries,
    and simplifying vertex counts.

    If `binary_mask` is a GeoTensor (has .transform attribute), polygons are
    returned in geographic coordinates. If it's a numpy array, polygons are in
    pixel coordinates unless a transform is provided.

    Args:
        binary_mask (Union[np.ndarray, GeoData]): 2D binary mask where nonzero
            pixels represent features to vectorize. Accepts GeoTensor (uses its
            transform) or numpy array.
        min_area (float): Minimum polygon area in square pixels to include.
            Polygons smaller than this are filtered out. Default 25.5 (~5x5 px).
            Set to 0 to keep all polygons.
        polygon_buffer (int): Buffer distance in pixels to apply to polygons.
            Default 0 (no buffering).

            - Positive: Expand/dilate polygons
            - Negative: Shrink/erode polygons (useful to remove edge noise)

        tolerance (float): Simplification tolerance in pixels. Higher values
            produce simpler polygons with fewer vertices. Default 1.0.
            Set to 0 for no simplification (keeps staircase pixel edges).
        transform (Optional[rasterio.Affine]): Affine transform to convert pixel
            coordinates to geographic coordinates. Only used if binary_mask is
            a numpy array. If binary_mask is GeoTensor, uses its transform.

    Returns:
        List[Polygon]: List of shapely Polygon objects. Coordinates are in:
            - Geographic CRS units if transform provided or binary_mask is GeoTensor
            - Pixel coordinates otherwise

    Examples:
        Vectorize a classification result:

        >>> import numpy as np
        >>> from georeader import vectorize
        >>>
        >>> # Binary mask from classification
        >>> water_mask = (classification == 1).astype(np.uint8)
        >>>
        >>> # Extract water body polygons
        >>> polygons = vectorize.get_polygons(
        ...     water_mask,
        ...     min_area=100,  # Filter out tiny artifacts
        ...     tolerance=2.0  # Simplify boundaries
        ... )
        >>> print(f"Found {len(polygons)} water bodies")

        Vectorize with erosion to remove edge noise:

        >>> polygons = vectorize.get_polygons(
        ...     noisy_mask,
        ...     min_area=50,
        ...     polygon_buffer=-2  # Erode 2 pixels
        ... )

        Vectorize GeoTensor (auto-uses transform):

        >>> from georeader.geotensor import GeoTensor
        >>> # mask_gt is a GeoTensor with transform
        >>> polygons = vectorize.get_polygons(mask_gt, min_area=200)
        >>> # Polygons are in geographic coordinates
        >>> for poly in polygons:
        ...     print(f"Area: {poly.area:.2f} sq. CRS units")

        Get pixel-coordinate polygons from numpy array:

        >>> polygons_px = vectorize.get_polygons(
        ...     mask_array,
        ...     transform=None  # No transform = pixel coordinates
        ... )

    Note:
        - Polygons are simplified AFTER buffering, so buffer then simplify
        - For very large masks, consider processing in tiles
        - MultiPolygon results are returned as separate Polygon objects

    See Also:
        transform_polygon: Transform polygon between coordinate systems.
        georeader.rasterize: Inverse operation (vector → raster).
    """

    if not hasattr(binary_mask, "transform"):
        binary_mask_np = binary_mask
    else:
        binary_mask_np = np.array(binary_mask)

        assert transform is None, "transform only must be used if input is np.ndarray"
        transform = binary_mask.transform

    shape_ = binary_mask_np.shape
    if len(shape_) != 2:
        binary_mask_np.squeeze()

    assert len(binary_mask_np.shape) == 2, f"Expected mask with 2 dim found {binary_mask_np.shape}"

    geoms_polygons = []
    polygon_generator = features.shapes(binary_mask_np.astype(np.int16),
                                        binary_mask_np)

    for polygon, _ in polygon_generator:
        p = shape(polygon)
        if polygon_buffer > 0:
            p = p.buffer(polygon_buffer)
        if p.area >= min_area:
            p = p.simplify(tolerance=tolerance)
            if transform is not None:
                p = transform_polygon(p, transform) # Convert polygon to raster coordinates
            geoms_polygons.append(p)

    return geoms_polygons


def transform_polygon(polygon:Union[Polygon, MultiPolygon], 
                      transform: rasterio.Affine, relative:bool=False,
                      shape_raster:Optional[Tuple[int,int]] = None) -> Union[Polygon, MultiPolygon]:
    """
    Transform polygon coordinates using an affine transformation.

    Applies a rasterio Affine transform to all vertices of a polygon,
    converting between pixel and geographic coordinate systems. Handles
    both simple Polygons and MultiPolygons, including holes.

    Common use cases:
    - Pixel coordinates → Geographic coordinates (using raster transform)
    - Geographic coordinates → Relative coordinates (0-1 range for ML)

    Args:
        polygon (Union[Polygon, MultiPolygon]): Shapely geometry to transform.
            Coordinates can be any numeric type.
        transform (rasterio.Affine): 2D affine transformation matrix.
            Common sources: raster.transform, rasterio.Affine.scale(), etc.
        relative (bool): If True, output normalized coordinates in [0, 1] range
            relative to the raster dimensions. Useful for ML model inputs.
            Requires shape_raster. Default False.
        shape_raster (Optional[Tuple[int, int]]): Raster dimensions as
            (height, width). Required if relative=True. Ignored otherwise.

    Returns:
        Union[Polygon, MultiPolygon]: Transformed geometry with same type as
            input. All coordinates are transformed by the affine matrix.

    Raises:
        AssertionError: If relative=True but shape_raster not provided.

    Examples:
        Convert pixel polygon to geographic coordinates:

        >>> from shapely.geometry import Polygon
        >>> import rasterio
        >>> from georeader.vectorize import transform_polygon
        >>>
        >>> # Polygon in pixel coordinates
        >>> poly_px = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        >>>
        >>> # Transform from a raster
        >>> transform = rasterio.Affine(10.0, 0, 500000, 0, -10.0, 4500000)
        >>>
        >>> # Convert to UTM coordinates
        >>> poly_geo = transform_polygon(poly_px, transform)
        >>> print(poly_geo.bounds)
        (500000.0, 4499000.0, 501000.0, 4500000.0)

        Get relative coordinates for ML input:

        >>> poly_rel = transform_polygon(
        ...     poly_px,
        ...     transform=rasterio.Affine.identity(),
        ...     relative=True,
        ...     shape_raster=(1000, 1000)  # 1000x1000 image
        ... )
        >>> print(poly_rel.bounds)  # Values in [0, 1]
        (0.0, 0.0, 0.1, 0.1)

        Transform MultiPolygon:

        >>> from shapely.geometry import MultiPolygon
        >>> multi = MultiPolygon([poly_px, poly_px.buffer(10)])
        >>> multi_geo = transform_polygon(multi, transform)
        >>> print(type(multi_geo))
        <class 'shapely.geometry.multipolygon.MultiPolygon'>

    Note:
        - The transform is applied as: (x_out, y_out) = transform * (x_in, y_in)
        - For pixel-to-geo conversion, use the raster's transform directly
        - For geo-to-pixel conversion, use ~transform (inverse)
        - Holes in polygons are preserved after transformation
    """
    if relative:
        assert shape_raster is not None, "shape_raster must be provided if relative is True"
        transform = rasterio.Affine.scale(1/shape_raster[1], 1/shape_raster[0]) * transform
    
    geojson_dict = mapping(polygon)
    if geojson_dict["type"] == "Polygon":
        geojson_dict["coordinates"] = [geojson_dict["coordinates"]]

    multipol_coords = []
    for pol in geojson_dict["coordinates"]:
        pol_coords = []
        for shell_or_holes in pol:
            pol_out = []
            for coords in shell_or_holes:
                pol_out.append(transform * coords)

            pol_coords.append(pol_out)
        
        multipol_coords.append(pol_coords)

    if geojson_dict["type"] == "Polygon":
        geojson_dict["coordinates"] = multipol_coords[0]
    else:
        geojson_dict["coordinates"] = multipol_coords

    return shape(geojson_dict)