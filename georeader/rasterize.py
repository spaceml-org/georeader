"""
Rasterize Module: Convert vector geometries to raster format.

This module provides functions to burn vector geometries (polygons, lines)
into raster grids aligned with existing GeoTensor objects. Essential for
creating masks, labels, and region-of-interest maps.

Rasterization Concepts
----------------------

Converting vector shapes to pixel grids::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    RASTERIZATION PROCESS                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Vector (Polygon)                    Raster (Grid)                      │
    │  ─────────────────                   ─────────────                      │
    │                                                                          │
    │       ╔═══════════╗                  ┌─┬─┬─┬─┬─┬─┬─┬─┐                  │
    │      ╔╝           ╚╗                 │░│░│░│▓│▓│▓│░│░│                  │
    │     ╔╝             ╚╗                ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
    │    ╔╝               ╚╗   ═══════►   │░│░│▓│▓│▓│▓│▓│░│                  │
    │    ║     Polygon     ║   Rasterize  ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
    │    ╚╗               ╔╝               │░│▓│▓│▓│▓│▓│▓│░│                  │
    │     ╚╗             ╔╝                ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
    │      ╚╗           ╔╝                 │░│░│▓│▓│▓│▓│░│░│                  │
    │       ╚═══════════╝                  └─┴─┴─┴─┴─┴─┴─┴─┘                  │
    │                                                                          │
    │  ░ = fill value (outside polygon)                                       │
    │  ▓ = burn value (inside polygon)                                        │
    └─────────────────────────────────────────────────────────────────────────┘

all_touched Option
------------------

Controls which pixels are included::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │              all_touched=False vs all_touched=True                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  all_touched=False (default)         all_touched=True                   │
    │  ───────────────────────────         ──────────────────                 │
    │                                                                          │
    │  Only pixels with CENTER inside      ALL pixels that TOUCH the polygon  │
    │                                                                          │
    │  ┌─┬─┬─┬─┬─┬─┐                       ┌─┬─┬─┬─┬─┬─┐                      │
    │  │░│░│░│░│░│░│  ╱ polygon edge       │░│▓│▓│▓│▓│░│                      │
    │  ├─┼─┼─┼─┼─┼─┤ ╱                     ├─┼─┼─┼─┼─┼─┤                      │
    │  │░│░│·│▓│▓│░│╱                      │░│▓│▓│▓│▓│▓│                      │
    │  ├─┼─┼─┼─┼─┼─┤                       ├─┼─┼─┼─┼─┼─┤                      │
    │  │░│░│▓│▓│▓│░│    · = center         │░│▓│▓│▓│▓│░│                      │
    │  └─┴─┴─┴─┴─┴─┘    ▓ = included       └─┴─┴─┴─┴─┴─┘                      │
    │                                                                          │
    │  Best for:                           Best for:                          │
    │  • Area calculations                 • Inclusive masks                   │
    │  • Avoiding edge pixels              • No gaps at boundaries             │
    │  • Conservative estimates            • Complete coverage                 │
    └─────────────────────────────────────────────────────────────────────────┘

GeoDataFrame Rasterization
--------------------------

Burn multiple geometries with attribute values::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │              GEODATAFRAME RASTERIZATION                                  │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  GeoDataFrame:                       Output Raster:                     │
    │  ┌──────────────────────┐            ┌─┬─┬─┬─┬─┬─┬─┬─┐                  │
    │  │ geometry │ class_id  │            │0│0│0│0│0│0│0│0│                  │
    │  ├──────────┼───────────┤            ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
    │  │ Poly A   │    1      │  ══════►   │0│1│1│1│0│2│2│0│                  │
    │  │ Poly B   │    2      │            ├─┼─┼─┼─┼─┼─┼─┼─┤                  │
    │  │ Poly C   │    3      │            │0│1│1│0│0│0│3│0│                  │
    │  └──────────┴───────────┘            └─┴─┴─┴─┴─┴─┴─┴─┘                  │
    │                                                                          │
    │  Usage:                                                                  │
    │    rasterize_geodataframe(gdf, data_like, attribute="class_id")         │
    └─────────────────────────────────────────────────────────────────────────┘

Module Functions Overview
-------------------------

Single Geometry:
    - :func:`rasterize_geometry_like`: Rasterize one geometry to match GeoData
    - :func:`rasterize_geometry`: Rasterize with explicit transform/shape

GeoDataFrame:
    - :func:`rasterize_geodataframe`: Burn multiple geometries with attributes

Quick Start
-----------

Create a mask from a polygon::

    from georeader import rasterize
    from shapely.geometry import box

    # Area of interest polygon
    aoi = box(-122.5, 37.0, -122.0, 37.5)

    # Create mask aligned with existing raster
    mask = rasterize.rasterize_geometry_like(
        aoi,
        data_like=my_raster,
        crs_geometry="EPSG:4326",
        value=1,
        fill=0
    )

Rasterize a GeoDataFrame with class labels::

    import geopandas as gpd

    # GeoDataFrame with land cover polygons
    gdf = gpd.read_file("landcover.geojson")

    # Burn class_id values into raster
    labels = rasterize.rasterize_geodataframe(
        gdf,
        data_like=my_raster,
        attribute="class_id",
        all_touched=True
    )

See Also
--------
georeader.vectorize : Inverse operation (raster → vector)
georeader.read : Reading raster data
rasterio.features.rasterize : Underlying implementation

References
----------
- Rasterio rasterize: https://rasterio.readthedocs.io/en/latest/api/rasterio.features.html
- GDAL rasterize: https://gdal.org/programs/gdal_rasterize.html
"""
import geopandas as gpd
from typing import Union, Tuple, Any, Optional
from georeader.geotensor import GeoTensor
import numpy as np
import rasterio
import rasterio.windows
import rasterio.features
from georeader.window_utils import PIXEL_PRECISION
from shapely.geometry import Polygon, MultiPolygon, LineString
from numbers import Number
from georeader import window_utils
from georeader.abstract_reader import GeoData


def rasterize_geometry_like(geometry:Union[Polygon, MultiPolygon, LineString], 
                            data_like: GeoData, value:Number=1,
                            dtype:Any=np.uint8,
                            crs_geometry:Optional[Any]=None, fill:Union[int, float]=0, all_touched:bool=False,
                            return_only_data:bool=False)-> Union[GeoTensor, np.ndarray]:
    """
    Rasterize a geometry to match an existing GeoData object's grid.

    Creates a raster mask from a vector geometry, aligned to the same extent,
    resolution, and CRS as the reference `data_like` object. This is the
    recommended function when you have an existing raster and want to create
    a corresponding mask or label layer.

    The function automatically reprojects the geometry if its CRS differs from
    the target raster.

    Args:
        geometry (Union[Polygon, MultiPolygon, LineString]): Shapely geometry to
            rasterize. Polygons burn filled areas, LineStrings burn line pixels.
        data_like (GeoData): Reference raster defining the output grid. The result
            will have matching shape[-2:], transform, and CRS.
        value (Number): Pixel value to burn inside the geometry. Default 1.
            Use different values for multi-class rasterization.
        dtype (Any): Output array data type. Default np.uint8.
            Use np.float32 for continuous values, np.int32 for large class IDs.
        crs_geometry (Optional[Any]): CRS of the input geometry. If provided and
            different from data_like.crs, geometry is reprojected automatically.
            Accepts EPSG codes, WKT strings, or pyproj CRS objects.
        fill (Union[int, float]): Background value for pixels outside geometry.
            Default 0.
        all_touched (bool): Pixel inclusion rule. Default False.

            - False: Only pixels whose center falls inside the geometry
            - True: All pixels that touch the geometry boundary

        return_only_data (bool): If True, return raw numpy array instead of
            GeoTensor. Default False.

    Returns:
        Union[GeoTensor, np.ndarray]: Rasterized geometry as 2D array (H, W).
            GeoTensor includes georeferencing; np.ndarray if return_only_data=True.

    Examples:
        Create a mask from an area of interest polygon:

        >>> from georeader import rasterize
        >>> from shapely.geometry import box
        >>>
        >>> # Define AOI in WGS84
        >>> aoi = box(-122.5, 37.0, -122.0, 37.5)
        >>>
        >>> # Create mask matching existing raster
        >>> mask = rasterize.rasterize_geometry_like(
        ...     aoi,
        ...     data_like=my_raster,
        ...     crs_geometry="EPSG:4326",
        ...     value=1,
        ...     fill=0
        ... )
        >>> mask.shape  # Matches my_raster spatial dims
        (1000, 1000)

        Rasterize with all_touched for inclusive mask:

        >>> mask_inclusive = rasterize.rasterize_geometry_like(
        ...     aoi,
        ...     data_like=my_raster,
        ...     crs_geometry="EPSG:4326",
        ...     all_touched=True  # Include all edge pixels
        ... )

        Rasterize a line feature:

        >>> from shapely.geometry import LineString
        >>> road = LineString([(-122.4, 37.2), (-122.3, 37.3), (-122.2, 37.25)])
        >>> road_mask = rasterize.rasterize_geometry_like(
        ...     road,
        ...     data_like=my_raster,
        ...     crs_geometry="EPSG:4326",
        ...     value=255,
        ...     dtype=np.uint8
        ... )

    See Also:
        rasterize_geopandas_like: Rasterize multiple geometries with attributes.
        rasterize_from_geometry: Rasterize with explicit transform/bounds.
        georeader.vectorize: Inverse operation (raster → vector).
    """
    shape_out = data_like.shape
    if crs_geometry and not window_utils.compare_crs(data_like.crs, crs_geometry):
        geometry = window_utils.polygon_to_crs(geometry, crs_geometry, data_like.crs)

    return rasterize_from_geometry(geometry, crs_geom_bounds=data_like.crs,
                                   transform=data_like.transform,
                                   window_out=rasterio.windows.Window(0, 0, width=shape_out[-1], height=shape_out[-2]),
                                   return_only_data=return_only_data,dtype=dtype, value=value,
                                   fill=fill, all_touched=all_touched)


def rasterize_from_geometry(geometry:Union[Polygon, MultiPolygon, LineString],
                            bounds:Optional[Tuple[float, float, float, float]]=None,
                            transform:Optional[rasterio.Affine]=None,
                            resolution:Optional[Union[float, Tuple[float, float]]]=None,
                            window_out:Optional[rasterio.windows.Window]=None,
                            value:Number=1,
                            dtype:Any=np.uint8,
                            crs_geom_bounds:Optional[Any]=None, fill:Union[int, float]=0, all_touched:bool=False,
                            return_only_data:bool=False)-> Union[GeoTensor, np.ndarray]:
    """
    Rasterise the provided geometry over the bounds with the specified resolution, transform, shape and crs.

    Args:
        geometry: geometry to rasterise (with crs `crs_geom_bounds`)
        bounds: bounds where the polygons will be rasterised. (with crs `crs_geom_bounds`)
        transform: if transform is provided it will use this instead of `resolution` (with crs `crs_geom_bounds`)
        resolution: spatial resolution of the rasterised array. It won't be used if transform is provided (with crs `crs_geom_bounds`)
        window_out: Window out in `crs_geom_bounds`. If not provided it is computed from the bounds.
        value: column to take the values for rasterisation.
        dtype: dtype of the rasterise raster.
        crs_geom_bounds: CRS of geometry and bounds
        fill: fill option for `rasterio.features.rasterize`. Value for pixels not covered by the geometries.
        all_touched: all_touched option for `rasterio.features.rasterize`. If True, all pixels touched 
            by geometries will be burned in.  If false, only pixels whose center is within the polygon or that
            are selected by Bresenham's line algorithm will be burned in.
        return_only_data: if `True` returns only the np.ndarray without georref info.

    Returns:
        `GeoTensor` or `np.ndarray` with shape `(H, W)` with the rasterised polygon
    """

    transform = window_utils.figure_out_transform(transform=transform, bounds=bounds,
                                                  resolution_dst=resolution)
    if window_out is None:
        window_out = rasterio.windows.from_bounds(*bounds,
                                                  transform=transform).round_lengths(op="ceil",
                                                                                     pixel_precision=PIXEL_PRECISION)


    chip_label = rasterio.features.rasterize(shapes=[(geometry, value)],
                                             out_shape=(window_out.height, window_out.width),
                                             transform=transform,
                                             dtype=dtype,
                                             fill=fill,
                                             all_touched=all_touched)
    if return_only_data:
        return chip_label

    return GeoTensor(chip_label, transform=transform, crs=crs_geom_bounds, fill_value_default=fill)

def rasterize_geopandas_like(dataframe:gpd.GeoDataFrame,data_like: GeoData, column:str,
                             fill:Union[int, float]=0, all_touched:bool=False,
                             return_only_data:bool=False)-> Union[GeoTensor, np.ndarray]:
    """
    Rasterize a GeoDataFrame to match an existing GeoData object's grid.

    Burns attribute values from the specified column into a raster aligned with
    the reference data_like object. Ideal for creating labeled training data
    from vector annotations or converting land cover polygons to raster format.

    The GeoDataFrame is automatically reprojected if its CRS differs from
    the target raster.

    Args:
        dataframe (gpd.GeoDataFrame): GeoDataFrame with geometry column and
            value column. Must have a valid CRS set.
        data_like (GeoData): Reference raster defining output grid (extent,
            resolution, CRS). Output will have matching shape[-2:] and transform.
        column (str): Column name containing values to burn. Values are cast
            to the output dtype. Example: 'class_id', 'land_cover', 'priority'.
        fill (Union[int, float]): Background value for pixels not covered by
            any geometry. Default 0.
        all_touched (bool): Pixel inclusion rule. Default False.

            - False: Burn only pixels whose center falls inside a geometry
            - True: Burn all pixels that touch any geometry boundary

        return_only_data (bool): Return raw numpy array instead of GeoTensor.
            Default False.

    Returns:
        Union[GeoTensor, np.ndarray]: Rasterized geometries as 2D array (H, W).
            Pixel values come from the specified column. Overlapping geometries
            use the last geometry's value (order matters).

    Examples:
        Rasterize land cover polygons:

        >>> import geopandas as gpd
        >>> from georeader import rasterize
        >>>
        >>> # Load land cover polygons with class_id column
        >>> gdf = gpd.read_file("landcover.geojson")
        >>> print(gdf[['class_id', 'geometry']].head())
           class_id                                           geometry
        0         1  POLYGON ((-122.5 37.0, -122.4 37.0, ...))
        1         2  POLYGON ((-122.3 37.1, -122.2 37.1, ...))
        >>>
        >>> # Create label raster matching satellite image
        >>> labels = rasterize.rasterize_geopandas_like(
        ...     gdf,
        ...     data_like=satellite_image,
        ...     column='class_id'
        ... )
        >>> np.unique(labels.values)
        array([0, 1, 2], dtype=uint8)  # 0=background, 1,2=classes

        All-touched for inclusive boundaries:

        >>> labels_inclusive = rasterize.rasterize_geopandas_like(
        ...     gdf,
        ...     data_like=satellite_image,
        ...     column='class_id',
        ...     all_touched=True  # Better for thin/small features
        ... )

    See Also:
        rasterize_geometry_like: Rasterize single geometry with constant value.
        rasterize_from_geopandas: Rasterize with explicit transform/bounds.
    """

    shape_out = data_like.shape
    return rasterize_from_geopandas(dataframe, column=column,
                                    crs_out=data_like.crs,
                                    transform=data_like.transform,
                                    window_out=rasterio.windows.Window(0, 0, width=shape_out[-1], height=shape_out[-2]),
                                    return_only_data=return_only_data,
                                    fill=fill, all_touched=all_touched)


def rasterize_from_geopandas(dataframe:gpd.GeoDataFrame,
                             column:str,
                             bounds:Optional[Tuple[float, float, float, float]]=None,
                             transform:Optional[rasterio.Affine]=None,
                             window_out:Optional[rasterio.windows.Window]=None,
                             resolution:Optional[Union[float, Tuple[float, float]]]=None,
                             crs_out:Optional[Any]=None, fill:Union[int, float]=0, all_touched:bool=False,
                             return_only_data:bool=False) -> Union[GeoTensor, np.ndarray]:
    """
    Rasterise the provided geodataframe over the bounds with the specified resolution.

    Args:
        dataframe: `GeoDataFrame` with columns `geometry` and `column`. 
            The 'geometry' column is expected to have shapely geometries.
        bounds: bounds where the polygons will be rasterised with CRS `crs_out`.
        transform: if transform is provided if will use this for the resolution.
        resolution: spatial resolution of the rasterised array
        window_out: Window out in `crs_geom_bounds`. If not provided it is computed from the bounds.
        column: column to take the values for rasterisation.
        crs_out: defaults to dataframe.crs. This function will transform the geometries from dataframe.crs to this crs
            before rasterisation. `bounds` are in this crs.
        fill: fill option for `rasterio.features.rasterize`. Value for pixels not covered by the geometries.
        all_touched: all_touched option for `rasterio.features.rasterize`. If True, all pixels touched 
            by geometries will be burned in.  If false, only pixels whose center is within the polygon or that
            are selected by Bresenham's line algorithm will be burned in.
        return_only_data: if `True` returns only the `np.ndarray`.

    Returns:
        `GeoTensor` or `np.ndarray` with shape `(H, W)` with the rasterised polygons of the dataframe
    """

    if crs_out is None:
        crs_out = str(dataframe.crs).lower()
    else:
        data_crs = str(dataframe.crs).lower()
        crs_out = str(crs_out).lower().replace("+init=","")
        if data_crs != crs_out:
            dataframe = dataframe.to_crs(crs=crs_out)

    transform = window_utils.figure_out_transform(transform=transform, bounds=bounds,
                                                  resolution_dst=resolution)
    if window_out is None:
        window_out = rasterio.windows.from_bounds(*bounds,
                                                  transform=transform).round_lengths(op="ceil",
                                                                                     pixel_precision=PIXEL_PRECISION)

    dtype = dataframe[column].dtype
    chip_label = rasterio.features.rasterize(shapes=zip(dataframe.geometry, dataframe[column]),
                                             out_shape=(window_out.height, window_out.width),
                                             transform=transform,
                                             dtype=dtype,
                                             fill=fill,
                                             all_touched=all_touched)
    if return_only_data:
        return chip_label

    return GeoTensor(chip_label, transform=transform, crs=crs_out, fill_value_default=fill)