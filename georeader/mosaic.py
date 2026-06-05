"""
Mosaic Module: Combine multiple rasters into seamless composite images.

This module provides functions to merge multiple overlapping rasters into
a single output, handling reprojection, resampling, and nodata filling.
Essential for creating cloud-free composites and gap-free mosaics.

Spatial Mosaic Overview
-----------------------

Combining multiple rasters with varying coverage::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    SPATIAL MOSAIC CONCEPT                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Input Rasters (with gaps)              Output Mosaic                   │
    │  ─────────────────────────              ─────────────                   │
    │                                                                          │
    │   Raster 1         Raster 2                                             │
    │  ┌─────────┐      ┌─────────┐           ┌─────────────────┐            │
    │  │▓▓▓░░░░░░│      │░░░░░▓▓▓▓│           │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │
    │  │▓▓▓▓░░░░░│  +   │░░░░▓▓▓▓▓│    ═══►   │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │
    │  │▓▓▓▓▓░░░░│      │░░░▓▓▓▓▓▓│           │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │
    │  │░░░░░░░░░│      │░░▓▓▓▓▓▓▓│           │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│            │
    │  └─────────┘      └─────────┘           └─────────────────┘            │
    │                                                                          │
    │   ░ = nodata/gaps                        Gaps filled from               │
    │   ▓ = valid data                         overlapping rasters            │
    │                                                                          │
    │  Processing Order:                                                       │
    │  • First raster fills as much as possible                               │
    │  • Each subsequent raster fills remaining gaps                          │
    │  • Continues until no nodata remains (or list exhausted)                │
    └─────────────────────────────────────────────────────────────────────────┘

Temporal Mosaic / Reduction
---------------------------

Combining rasters from multiple time steps::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    TEMPORAL REDUCTION CONCEPT                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Time Series Input                      Reduction Output                │
    │  ─────────────────                      ────────────────                │
    │                                                                          │
    │   t=1    t=2    t=3                                                     │
    │  ┌───┐  ┌───┐  ┌───┐                   ┌───────────────┐               │
    │  │ 5 │  │ 7 │  │ 6 │                   │               │               │
    │  │   │  │   │  │   │   ─────────────►  │  median = 6   │               │
    │  │   │  │   │  │   │   np.nanmedian    │  mean = 6.0   │               │
    │  └───┘  └───┘  └───┘   np.nanmean      │  max = 7      │               │
    │                                        └───────────────┘               │
    │                                                                          │
    │  Common Reduction Functions:                                            │
    │  • np.nanmedian: Robust to outliers (clouds, shadows)                   │
    │  • np.nanmean: Average value                                            │
    │  • np.nanmax: Maximum composite (e.g., max NDVI)                        │
    │  • np.nanmin: Minimum composite                                         │
    │  • np.nanstd: Temporal variability                                      │
    └─────────────────────────────────────────────────────────────────────────┘

Mosaic with Masks
-----------------

Using external validity masks to control which pixels are used::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MASKED MOSAIC WORKFLOW                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Input: (data, mask) tuples                                             │
    │  ─────────────────────────                                              │
    │                                                                          │
    │   Raster 1     Cloud Mask      │    Raster 2     Cloud Mask            │
    │  ┌─────────┐  ┌─────────┐      │   ┌─────────┐  ┌─────────┐            │
    │  │▓▓▓▓▓▓▓▓▓│  │░░░█████░│      │   │▓▓▓▓▓▓▓▓▓│  │░░░░░░░░░│            │
    │  │▓▓▓▓▓▓▓▓▓│  │░░█████░░│      │   │▓▓▓▓▓▓▓▓▓│  │░░░░░░░░░│            │
    │  │▓▓▓▓▓▓▓▓▓│  │░██████░░│   +  │   │▓▓▓▓▓▓▓▓▓│  │░░░░░░░░░│            │
    │  └─────────┘  └─────────┘      │   └─────────┘  └─────────┘            │
    │                   ↑            │                                        │
    │               █ = invalid      │   Uses Raster 2 where Raster 1        │
    │               (cloud/shadow)   │   is masked as invalid                 │
    │                                                                          │
    │  Usage:                                                                  │
    │    data_list = [(raster1, mask1), (raster2, mask2), ...]               │
    │    mosaic = spatial_mosaic(data_list, ...)                             │
    └─────────────────────────────────────────────────────────────────────────┘

Module Functions Overview
-------------------------

Spatial Mosaicking:
    - :func:`spatial_mosaic`: Merge rasters to fill nodata gaps
    - :func:`spatial_mosaic_chunked`: Memory-efficient chunked processing

Temporal Reduction:
    - :func:`rasters_reduction`: Apply reduction function across rasters
    - :func:`pad_add_rasters`: Align and stack rasters for reduction

Quick Start
-----------

Create a cloud-free mosaic from multiple images::

    from georeader import mosaic, read

    # List of overlapping raster readers
    rasters = [reader1, reader2, reader3]

    # Create mosaic (fills gaps with subsequent images)
    result = mosaic.spatial_mosaic(
        rasters,
        bounds=(-122.5, 37.0, -122.0, 37.5),
        dst_crs="EPSG:4326",
        dst_nodata=0
    )

Compute median composite from time series::

    from georeader import mosaic
    import numpy as np

    # Stack aligned rasters and compute median
    result = mosaic.rasters_reduction(
        raster_list,
        reducer=np.nanmedian,
        dst_crs="EPSG:32610"
    )

See Also
--------
georeader.read : Reading and reprojection functions
georeader.slices : Array slicing for chunked processing
georeader.geotensor : Output format

References
----------
- Rasterio merge: https://rasterio.readthedocs.io/en/latest/api/rasterio.merge.html
- Cloud masking strategies: See georeader.readers.cloudsen12
"""
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from typing import List, Optional, Tuple, Union, Callable
import rasterio.warp
from georeader import read
from georeader.read import read_reproject
import numpy as np
from georeader import window_utils
from georeader import slices
from shapely.geometry import Polygon, MultiPolygon, box
import rasterio.windows
from collections import namedtuple
import georeader


def spatial_mosaic(data_list:Union[List[GeoData], List[Tuple[GeoData,GeoData]]],
                   polygon:Optional[Polygon]=None,
                   crs_polygon:Optional[str]=None,
                   dst_transform:Optional[rasterio.transform.Affine]=None,
                   bounds:Optional[Tuple[float, float, float, float]]=None,
                   dst_crs:Optional[str]=None,
                   dtype_dst:Optional[str]=None,
                   window_size: Optional[Tuple[int, int]]= None,
                   resampling:rasterio.warp.Resampling=rasterio.warp.Resampling.cubic_spline,
                   masking_function:Optional[Callable[[GeoData], GeoData]]=None,
                   dst_nodata:Optional[int]=None) -> GeoTensor:
    """
    Create a spatial mosaic by filling gaps with data from overlapping rasters.

    Combines multiple rasters into a single output by iteratively filling nodata
    regions with valid data from subsequent rasters. The first raster is used as
    the base, and remaining rasters fill in only where the base has nodata values.

    This function is similar to `rasterio.merge.merge` but with support for:

    - Custom validity masks per raster
    - Masking functions (e.g., cloud masks)
    - Windowed processing for memory efficiency

    Args:
        data_list (Union[List[GeoData], List[Tuple[GeoData, GeoData]]]): Input
            rasters to mosaic. Can be:

            - List of GeoData objects: Each raster's nodata value determines validity
            - List of (data, mask) tuples: mask indicates invalid pixels to fill

            Rasters are processed in order; first valid pixel wins.
        polygon (Optional[Polygon]): Output extent as a shapely Polygon. If provided,
            mosaic is clipped to this polygon. CRS specified by crs_polygon.
        crs_polygon (Optional[str]): CRS of the polygon. If not provided, uses
            the CRS of the first raster.
        dst_transform (Optional[Affine]): Output transform. If not provided, computed
            from bounds or polygon.
        bounds (Optional[Tuple[float, float, float, float]]): Output extent as
            (minx, miny, maxx, maxy). Alternative to polygon.
        dst_crs (Optional[str]): Output CRS. If not provided, uses CRS of first raster.
        dtype_dst (Optional[str]): Output data type. If not provided, uses dtype
            of first raster.
        window_size (Optional[Tuple[int, int]]): Process in tiles of this size
            (height, width) for memory efficiency. Default None (process all at once).
        resampling (Resampling): Resampling method for reprojection.
            Default cubic_spline for continuous data.
        masking_function (Optional[Callable[[GeoData], GeoData]]): Function that
            takes a GeoData and returns a boolean mask of INVALID pixels.
            Applied to each raster before mosaicking.
        dst_nodata (Optional[int]): Output nodata value. If not provided, uses
            fill_value_default of first raster.

    Returns:
        GeoTensor: Mosaic covering the specified extent. Nodata regions are filled
            by iterating through data_list until all pixels are valid or list is
            exhausted.

    Examples:
        Basic mosaic of overlapping Sentinel-2 scenes:

        >>> from georeader import mosaic
        >>> from georeader.rasterio_reader import RasterioReader
        >>>
        >>> # Load overlapping scenes
        >>> scene1 = RasterioReader("scene1.tif")
        >>> scene2 = RasterioReader("scene2.tif")
        >>> scene3 = RasterioReader("scene3.tif")
        >>>
        >>> # Create seamless mosaic
        >>> result = mosaic.spatial_mosaic(
        ...     [scene1, scene2, scene3],
        ...     bounds=(-122.5, 37.0, -121.5, 38.0),
        ...     dst_crs="EPSG:4326"
        ... )

        Mosaic with cloud masks (tuple format):

        >>> # Each tuple is (data, cloud_mask) where cloud_mask=True means cloudy
        >>> result = mosaic.spatial_mosaic(
        ...     [(scene1, cloud1), (scene2, cloud2), (scene3, cloud3)],
        ...     bounds=(-122.5, 37.0, -121.5, 38.0),
        ...     dst_crs="EPSG:4326"
        ... )
        >>> # Cloud-covered pixels in scene1 are filled from scene2, etc.

        Memory-efficient tiled processing:

        >>> result = mosaic.spatial_mosaic(
        ...     large_scene_list,
        ...     bounds=extent,
        ...     window_size=(1024, 1024)  # Process in 1024x1024 tiles
        ... )

    See Also:
        georeader.read.read_reproject: Underlying reprojection function.
        rasterio.merge.merge: Similar functionality in rasterio.

    Note:
        - Processing order matters: earlier rasters have priority
        - Use window_size for large outputs to avoid memory issues
        - Set appropriate resampling for your data type (nearest for categorical)
    """

    assert len(data_list) > 0, f"Expected at least one product found 0 {data_list}"

    if isinstance(data_list[0], tuple):
        first_data_object =  data_list[0][0]
        first_mask_object = data_list[0][1]
    else:
        first_data_object = data_list[0]
        first_mask_object = None
    
    if dst_transform is None:
        dst_transform = first_data_object.transform

    if dst_crs is None:
        dst_crs = first_data_object.crs

    if polygon is None:
        if bounds is not None:
            polygon = box(*bounds)
        else:
            # Polygon is the Union of the polygons of all the data
            for data in data_list:
                if isinstance(data, tuple):
                    data = data[0]
                polygon_iter = data.footprint(crs=dst_crs)

                if polygon is None:
                    polygon = polygon_iter
                else:
                    polygon = polygon.union(polygon_iter)
    else:
        if crs_polygon is None:
            crs_polygon = dst_crs
        elif not georeader.compare_crs(crs_polygon, dst_crs):
            polygon = window_utils.polygon_to_crs(polygon, crs_polygon, dst_crs)

    GeoDataFake = namedtuple("GeoDataFake", ["transform", "crs"])
    window_polygon = read.window_from_polygon(GeoDataFake(transform=dst_transform, crs=dst_crs),
                                              polygon, crs_polygon=dst_crs)

    window_polygon = window_utils.round_outer_window(window_polygon)

    # Shift transform to window
    dst_transform = rasterio.windows.transform(window_polygon, transform=dst_transform)
    dst_nodata = dst_nodata or first_data_object.fill_value_default

    # Get object to save the results
    data_return = read_reproject(first_data_object,
                                 dst_crs=dst_crs, dst_transform=dst_transform,
                                 resampling=resampling,
                                 dtype_dst=dtype_dst,
                                 window_out=rasterio.windows.Window(row_off=0, col_off=0, width=window_polygon.width,
                                                                    height=window_polygon.height),
                                 dst_nodata=dst_nodata)

    # invalid_values of spatial locations only  -> any
    invalid_values = data_return.values == dst_nodata
    if len(data_return.shape) > 2:
        axis_any = tuple(i for i in range(len(data_return.shape)-2))
        invalid_values = np.any(invalid_values, axis=axis_any) # (H, W)
    else:
        axis_any = None

    if first_mask_object is not None:
        if (masking_function is None) and len(first_mask_object.shape) > 2:
            assert (len(first_mask_object.shape) == 3) and (first_mask_object.shape[0] == 1), f"Expected two dims, found {first_mask_object.shape}"

        invalid_geotensor = read_reproject(first_mask_object,
                                           dst_crs=dst_crs, dst_transform=dst_transform,
                                           resampling=rasterio.warp.Resampling.nearest,
                                           window_out=rasterio.windows.Window(row_off=0, col_off=0,
                                                                              width=window_polygon.width,
                                                                              height=window_polygon.height))
        if masking_function is not None:
            invalid_geotensor = masking_function(invalid_geotensor)

        invalid_geotensor = invalid_geotensor.astype(bool).squeeze()
        assert len(invalid_geotensor.shape) == 2, f"Invalid mask expected 2 dims found {invalid_geotensor.shape}"

        invalid_values|= invalid_geotensor.values
    elif masking_function is not None:
        # Apply masking funtion to the readed data
        invalid_geotensor = masking_function(data_return)

        invalid_geotensor = invalid_geotensor.astype(bool).squeeze()
        assert len(invalid_geotensor.shape) == 2, f"Invalid mask expected 2 dims found {invalid_geotensor.shape}"
        invalid_values |= invalid_geotensor.values

    # data_return.values[..., invalid_values] = data_return.fill_value_default

    if not np.any(invalid_values):
        return data_return

    if len(data_list) == 1:
        return data_return

    if window_size is not None:
        windows = slices.create_windows(data_return.shape[-2:], window_size)
    else:
        windows = [rasterio.windows.Window(row_off=0, col_off=0, width=data_return.shape[-1],
                                           height=data_return.shape[-2])]

    # Cache of the polygons geodata
    polygons_geodata = [None for _ in range(len(data_list)-1)]

    for window in windows:
        slice_spatial = window.toslices()
        invalid_values_window = invalid_values[slice_spatial]
        if not np.any(invalid_values_window):
            continue

        # Add dims to slice_obj
        slice_obj = tuple(slice(None) for _ in range(len(data_return.shape)-2)) + slice_spatial
        dst_transform_iter = rasterio.windows.transform(window, transform=dst_transform)
        window_reproject_iter = rasterio.windows.Window(row_off=0, col_off=0, width=window.width, height=window.height)
        polygon_iter = window_utils.window_polygon(window, dst_transform)

        for _i, data in enumerate(data_list[1:]):
            if isinstance(data, tuple):
                geodata = data[0]
                geomask = data[1]
            else:
                geodata = data
                geomask = None

            if polygons_geodata[_i] is None:
                polygons_geodata[_i] = geodata.footprint(crs=dst_crs)

            polygon_geodata = polygons_geodata[_i]

            if not polygon_geodata.intersects(polygon_iter):
                continue

            if geomask is not None:
                if (masking_function is None) and len(geomask.shape) > 2:
                    assert (len(geomask.shape) == 3) and (
                                geomask.shape[0] == 1), f"Expected two dims, found {geomask.shape}"

                invalid_geotensor = read_reproject(geomask,
                                                   dst_crs=dst_crs, dst_transform=dst_transform_iter,
                                                   resampling=rasterio.warp.Resampling.nearest,
                                                   window_out=window_reproject_iter)
                if masking_function is not None:
                    invalid_geotensor = masking_function(invalid_geotensor)

                invalid_geotensor = invalid_geotensor.astype(bool).squeeze()
                assert len(invalid_geotensor.shape) == 2, f"Invalid mask expected 2 dims found {invalid_geotensor.shape}"
                if np.all(invalid_geotensor.values):
                    continue
                invalid_values_iter = invalid_geotensor.values

            data_read = read_reproject(geodata, dst_crs=dst_crs, window_out=window_reproject_iter,
                                       dst_transform=dst_transform_iter, resampling=resampling,
                                       dtype_dst=dtype_dst,
                                       dst_nodata=dst_nodata)

            if (geomask is None) and (masking_function is not None):
                invalid_geotensor = masking_function(data_read)

                invalid_geotensor = invalid_geotensor.astype(bool).squeeze()
                assert len(invalid_geotensor.shape) == 2, f"Invalid mask expected 2 dims found {invalid_geotensor.shape}"
                if np.all(invalid_geotensor.values):
                    continue
                invalid_values_iter = invalid_geotensor.values

            # data_read could have more dims -> any
            masked_values_read = data_read.values == dst_nodata
            if axis_any is not None:
                masked_values_read = np.any(masked_values_read, axis=axis_any)  # (H, W)

            if (geomask is not None) or (masking_function is not None):
                invalid_values_iter |= masked_values_read
            else:
                invalid_values_iter = masked_values_read

            # Copy values invalids in window and valids in iter
            mask_values_copy_out = invalid_values_window & ~invalid_values_iter
            data_return.values[slice_obj][..., mask_values_copy_out] = data_read.values[...,mask_values_copy_out]

            invalid_values_window &= invalid_values_iter

            if not np.any(invalid_values_window):
                break


    return data_return