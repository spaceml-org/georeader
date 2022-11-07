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


def spatial_mosaic(data_list:Union[List[GeoData], List[Tuple[GeoData,GeoData]]],
                   polygon:Optional[Polygon]=None,
                   dst_transform:Optional[rasterio.transform.Affine]=None,
                   bounds:Optional[Tuple[float, float, float, float]]=None,
                   dst_crs:Optional[str]=None,
                   window_size: Optional[Tuple[int, int]]= None,
                   resampling:rasterio.warp.Resampling=rasterio.warp.Resampling.cubic_spline,
                   masking_function:Optional[Callable[[GeoData], GeoData]]=None,
                   dst_nodata:Optional[int]=None) -> GeoTensor:
    """
    Computes the spatial mosaic of all input products in `data_list`. It iteratively calls `read_reproject` with
    all the list of rasters while there is any `dst_nodata` value. This function m requires that the copy of the output
    fits in memory.

    This function is very similar to `rasterio.merge.merge`.

    Args:
        data_list: List of raster objects. each element could be a single geodata object or a tuple of an object and a
            mask (second item will be considered the invalid values mask).
        polygon: polygon to compute the mosaic in dst_crs
        bounds: bounds to compute the mosaic.
        dst_crs: CRS of the product. If not provided it will use the CRS of the first product of the list
        dst_transform: Optional dest transform. If not provided the dst_transform is a rectilinear transform computed
        window_size: The mosaic will be computed by windows of this size (for efficiency purposes)
        resampling:specifies how data is reprojected from `rasterio.warp.Resampling`.
        masking_function: function to call to the mask if provided or to the tensor (if not provided) should return a bool tensor
            with only spatial dimensions.
        dst_nodata: no data value. if None will use `data_list[0].fill_value_default`

    Returns:
        GeoTensor with mosaic over the given bounds

    """

    assert len(data_list) > 0, f"Expected at least one product found 0 {data_list}"

    if isinstance(data_list[0], tuple):
        first_data_object =  data_list[0][0]
        first_mask_object = data_list[0][1]
    else:
        first_data_object = data_list[0]
        first_mask_object = None

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

    if dst_transform is None:
        dst_transform = first_data_object.transform

    if dst_crs is None:
        dst_crs = first_data_object.crs

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

        invalid_geotensor.values = invalid_geotensor.values.astype(bool)
        invalid_geotensor.values =  invalid_geotensor.values.squeeze()
        assert len(invalid_geotensor.shape) == 2, f"Invalid mask expected 2 dims found {invalid_geotensor.shape}"

        invalid_values|= invalid_geotensor.values
    elif masking_function is not None:
        # Apply masking funtion to the readed data
        invalid_geotensor = masking_function(data_return)

        invalid_geotensor.values = invalid_geotensor.values.astype(bool)
        invalid_geotensor.values = invalid_geotensor.values.squeeze()
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

                invalid_geotensor.values = invalid_geotensor.values.astype(bool)
                invalid_geotensor.values = invalid_geotensor.values.squeeze()
                assert len(invalid_geotensor.shape) == 2, f"Invalid mask expected 2 dims found {invalid_geotensor.shape}"
                if np.all(invalid_geotensor.values):
                    continue
                invalid_values_iter = invalid_geotensor.values

            data_read = read_reproject(geodata, dst_crs=dst_crs, window_out=window_reproject_iter,
                                       dst_transform=dst_transform_iter, resampling=resampling,
                                       dst_nodata=dst_nodata)

            if (geomask is None) and (masking_function is not None):
                invalid_geotensor = masking_function(data_read)

                invalid_geotensor.values = invalid_geotensor.values.astype(bool)
                invalid_geotensor.values = invalid_geotensor.values.squeeze()
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