from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from typing import List, Optional, Tuple
import rasterio.warp
from georeader.read import read_reproject
import numpy as np

def spatial_mosaic(data_list:List[GeoData],
                   bounds:Optional[Tuple[float, float, float, float]]=None,
                   dst_crs:Optional[str]=None,
                   resampling:rasterio.warp.Resampling=rasterio.warp.Resampling.cubic_spline,
                   dst_nodata:Optional[int]=None) -> GeoTensor:
    """
    Computes the spatial mosaic of all input products in `data_list`. It iteratively calls `read_reproject` with
    all the list of rasters while there is any `dst_nodata` value. This function might consume a lot of memory, it requires
    that the copy of the output fits in memory.

    This function is very similar to `rasterio.merge.merge`.

    Args:
        data_list: List of raster objects
        bounds: bounds to compute the mosaic. If not provided it will use the union of the bounds of all the products.
        dst_crs: CRS of the product. If not provided it will use the CRS of the first product of the list
        resampling:s pecifies how data is reprojected from `rasterio.warp.Resampling`.
        dst_nodata: no data value. if None will use `data_list[0].fill_value_default`

    Returns:
        GeoTensor with mosaic over the given bounds

    """

    assert len(data_list) > 0, f"Expected at least one product found 0 {data_list}"

    if bounds is None:
        # Union of all bounds
        xs,ys = [], []
        for data in data_list:
            left, bottom, right, top = data.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
        bounds = min(xs), min(ys), max(xs), max(ys)

    if dst_crs is None:
        dst_crs = data_list[0].crs

    resolution_dst_crs = data_list[0].res

    dst_nodata = dst_nodata or data_list[0].fill_value_default
    data_return = read_reproject(data_list[0], bounds=bounds, dst_crs=dst_crs,
                                 resolution_dst_crs=resolution_dst_crs, resampling=resampling,
                                 dst_nodata=dst_nodata)
    invalid_values = data_return.values == dst_nodata
    if not np.any(invalid_values):
        return data_return

    for data in data_list[1:]:
        data_read = read_reproject(data, bounds=bounds, dst_crs=dst_crs,
                                   resolution_dst_crs=resolution_dst_crs, resampling=resampling,
                                   dst_nodata=dst_nodata)
        assert data_read.shape == data_return.shape, f"Different read shapes for product {data} {data_read.shape} and {data_list[0]} {data_return.shape}"
        data_return.values[invalid_values] = data_read.values[invalid_values]
        invalid_values = data_return.values == dst_nodata
        if not np.any(invalid_values):
            return data_return

    return data_return