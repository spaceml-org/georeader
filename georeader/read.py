import rasterio
import rasterio.windows
import rasterio.warp
import rasterio.features
import numbers
import numpy as np
from typing import Tuple, Union, Optional, Dict, Any, List
from collections import OrderedDict
import itertools
from georeader.geotensor import GeoTensor
from georeader import window_utils
from georeader.window_utils import PIXEL_PRECISION, pad_window, round_outer_window, _is_exact_round
from georeader.abstract_reader import GeoData


def _round_all(x):
    x = tuple([int(round(xi)) for xi in x])
    return x

def _normalize_crs(a_crs):
    a_crs = str(a_crs)
    if "+init=" in a_crs:
        a_crs = a_crs.replace("+init=","")
    return a_crs.lower()


def compare_crs(a_crs:str, b_crs:str) -> bool:
    return _normalize_crs(a_crs) == _normalize_crs(b_crs)


def _transform_from_crs(center_coords:Tuple[float, float], crs_input:Union[Dict[str,str],str],
                       crs_output:Union[Dict[str,str],str]) -> Tuple[float, float]:
    """ Transforms a coordinate tuple from crs_input to crs_output """

    coords_transformed = rasterio.warp.transform(crs_input, crs_output, [center_coords[0]], [center_coords[1]])
    return coords_transformed[0][0], coords_transformed[1][0]


def window_from_bounds(data_in: GeoData, bounds:Tuple[float, float, float, float],
                       crs_bounds:Optional[str]=None) -> rasterio.windows.Window:
    """
    Compute window to read in data_in from bounds in crs_bounds. If crs_bounds is None it assumes bounds are in the
    crs of data_in

    Args:
        data_in: Reader with crs and transform attributes
        bounds: tuple with bounds to find the corresponding window
        crs_bounds: Optional coordinate reference system of the bounds. If not provided assumes same crs as `data_in`

    Returns:
        Window object with location in pixel coordinates relative to `data_in` of the bounds

    """
    if (crs_bounds is not None) and not compare_crs(crs_bounds, data_in.crs):

        bounds_in = rasterio.warp.transform_bounds(crs_bounds,
                                                   data_in.crs, *bounds)
    else:
        bounds_in = bounds
    transform = data_in.transform
    window_in = rasterio.windows.from_bounds(*bounds_in, transform=transform)
    return window_in


def window_from_center_coords(data_in: GeoData, center_coords:Tuple[float, float],
                              shape:Tuple[int,int], crs_center_coords:Optional[Any]=None) -> rasterio.windows.Window:
    """
     Compute window to read in `data_in` from the coordinates of the center pixel. If `crs_center_coords` is None it assumes
     `center_coords` are in the crs of `data_in`.

     THIS FUNCTION ASSUMES data_in.transform.is_rectilinear IT WILL PRODUCE INCORRECT RESULTS OTHERWISE.

    Args:
        data_in: Reader with crs and transform attributes
        center_coords: Tuple with center coords (x, y) format
        shape: Tuple with shape to read (H, W) format
        crs_center_coords: Optional coordinate reference system of the bounds. If not provided assumes same crs as `data_in`

    Returns:
         Window object with location in pixel coordinates relative to `data_in` of the window centered on `center_coords`
    """

    if (crs_center_coords is not None) and not compare_crs(crs_center_coords, data_in.crs):
        center_coords = _transform_from_crs(center_coords, crs_center_coords, data_in.crs)

    # The compute of the corner coordinates from the center is the same as in utils.polygon_slices
    transform = data_in.transform

    pixel_center_coords = ~transform * tuple(center_coords)
    pixel_upper_left =  _round_all((pixel_center_coords[0] - shape[1] / 2, pixel_center_coords[1] - shape[0] / 2))

    # OLD CODE that didn't support non-rectilinear transforms
    # assert transform.is_rectilinear(), "Transform is not rectilear"
    #
    # upper_left_coords = (center_coords[0] - (transform.a * shape[1] / 2),
    #                      center_coords[1] - (transform.e * shape[0] / 2))
    # pixel_upper_left = _round_all(~transform * upper_left_coords)

    window = rasterio.windows.Window(row_off=pixel_upper_left[1], col_off=pixel_upper_left[0],
                                     width=shape[1], height=shape[0])
    return window


def read_from_window(data_in: GeoData,
                     window: rasterio.windows.Window, return_only_data: bool = False,
                     trigger_load: bool = False,
                     boundless: bool = True) -> Union[GeoData, np.ndarray, None]:
    """
    Reads a window from data_in padding with 0 if needed (output GeoData will have window.height, window.width shape
    if boundless is `True`).

    Args:
        data_in: GeoData with "x" and "y" coordinates
        window: window to slice the GeoData with.
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoData georreferenced object.
        trigger_load: defaults to `False`. Trigger loading the data to memory.
        boundless: if `True` data read will always have the shape of the provided window
            (padding with `fill_value_default`)

    Returns:
        GeoData object
    """

    named_shape = OrderedDict(zip(data_in.dims, data_in.shape))

    window_data = rasterio.windows.Window(col_off=0, row_off=0,
                                          width=named_shape["x"], height=named_shape["y"])

    # get transform of current window
    transform = data_in.transform

    # Case the window does not intersect the data
    if not rasterio.windows.intersect([window_data, window]):
        if not boundless:
            return None

        expected_shapes = {"x": window.width, "y": window.height}
        shape = tuple([named_shape[s] if s not in ["x", "y"] else expected_shapes[s] for s in data_in.dims])
        data = np.zeros(shape, dtype=data_in.dtype)
        fill_value_default = getattr(data_in, "fill_value_default", 0)
        if fill_value_default != 0:
            data += fill_value_default
        if return_only_data:
            return data

        return GeoTensor(data, crs=data_in.crs,
                         transform=rasterio.windows.transform(window, transform=transform),
                         fill_value_default=fill_value_default)

    # Read data directly with rasterio (handles automatically the padding)
    data_sel = data_in.read_from_window(window=window, boundless=boundless)

    if return_only_data:
        return data_sel.values

    if trigger_load:
        data_sel = data_sel.load()

    return data_sel


def read_from_center_coords(data_in: GeoData, center_coords:Tuple[float, float], shape:Tuple[int,int],
                            crs_center_coords:Optional[Any]=None,
                            return_only_data:bool=False, trigger_load:bool=False,
                            boundless:bool=True) -> Union[GeoData, np.ndarray]:
    """
    Returns a chip of `data_in` centered on `center_coords` of shape `shape`.

    Notes:
        This function assumes that the transform of data_in is rectilinear. (see `rasterio.Affine.is_rectilinear`).
        IT WILL PRODUCE INCORRECT RESULTS OTHERWISE.

    Args:
        data_in: GeoData object
        center_coords: x, y tuple of coords in `data_in` crs.
        shape: shape of the window to read
        crs_center_coords: CRS of center coords. If provided will check if it needs to reproject the coords before
            computing the reading window.
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoData georreferenced object.
        trigger_load: defaults to `False`. Trigger loading the data to memory.
        boundless: if `True` data read will always have the shape of the provided window
            (padding with `fill_value_default`)

    Returns:
        GeoData or np.array sliced from `data_in` of shape `shape`.

    """

    window = window_from_center_coords(data_in, center_coords, shape, crs_center_coords)

    return read_from_window(data_in, window=window, return_only_data=return_only_data,
                            trigger_load=trigger_load, boundless=boundless)


def read_from_bounds(data_in: GeoData, bounds: Tuple[float, float, float, float],
                     crs_bounds: Optional[str] = None, pad_add=(0, 0),
                     return_only_data: bool = False, trigger_load: bool = False,
                     boundless: bool = True) -> Union[GeoData, np.ndarray]:
    """
    Reads a slice of data_in covering the `bounds`.

    Args:
        data_in: GeoData with geographic info (crs and geotransform).
        bounds:  bounding box to read.
        crs_bounds: if not None will transform the bounds from that crs to the data.crs to read the chip.
        pad_add: pad in pixels to add to the `window` that is read.This is useful when this function is called for
         interpolation/CNN prediction.
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoData georreferenced object.
        trigger_load: defaults to `False`. Trigger loading the data to memory.
        boundless: if `True` data read will always have the shape of the provided window
            (padding with `fill_value_default`)

    Returns:
        sliced GeoData
    """
    window_in = window_from_bounds(data_in, bounds, crs_bounds)
    if any(p > 0 for p in pad_add):
        window_in = pad_window(window_in, pad_add)  # Add padding for bicubic int or for co-registration
    window_in = round_outer_window(window_in)

    return read_from_window(data_in, window_in, return_only_data=return_only_data, trigger_load=trigger_load,
                            boundless=boundless)


def read_reproject_like(data_in: GeoData, data_like: GeoData,
                        resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
                        dtpye_dst=None, return_only_data: bool = False,
                        dst_nodata: Optional[int] = None) -> Union[GeoTensor, np.ndarray]:
    """
    Reads from `data_in` and reprojects to have the same extent and resolution than `data_like`.

    Args:
        data_in: GeoData to read and reproject. Expected coords "x" and "y".
        data_like: GeoData to get the bounds and resolution to reproject `data_in`.
        resampling: specifies how data is reprojected
        dtpye_dst: if None it will be inferred
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoTensor object (georreferenced array).
        dst_nodata: dst_nodata value

    Returns:
        GeoTensor read from `data_in` with same transform, crs, shape and bounds than `data_like`.
    """

    shape_out = data_like.shape
    return read_reproject(data_in, dst_crs=data_like.crs, dst_transform=data_like.transform,
                          window_out=rasterio.windows.Window(0,0, width=shape_out[-1], height=shape_out[-2]),
                          resampling=resampling,dtpye_dst=dtpye_dst, return_only_data=return_only_data,
                          dst_nodata=dst_nodata)


def read_reproject(data_in: GeoData, dst_crs: Optional[str]=None,
                   bounds: Optional[Tuple[float, float, float, float]]=None,
                   resolution_dst_crs: Optional[Union[float, Tuple[float, float]]]=None,
                   dst_transform:Optional[rasterio.Affine]=None,
                   window_out:Optional[rasterio.windows.Window]=None,
                   resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
                   dtpye_dst=None, return_only_data: bool = False, dst_nodata: Optional[int] = None) -> Union[
    GeoTensor, np.ndarray]:
    """
    This function slices the data by the bounds and reprojects it to the dst_crs and resolution_dst_crs

    Args:
        data_in: GeoData to read and reproject. Expected coords "x" and "y".
        bounds: Bounds in CRS specified by dst_crs.
        dst_crs: CRS to reproject.
        resolution_dst_crs: resolution in the CRS specified by dst_crs
        dst_transform: Optional dest transform. If not provided the dst_transform is a rectilinear transform computed
        with the bounds and resolution_dst_crs.
        window_out: Window out in dst_crs. If not provided it is computed from the bounds.
        resampling: specifies how data is reprojected
        dtpye_dst: if None it will be inferred
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoTensor object (georreferenced array).
        dst_nodata: dst_nodata value

    Returns:
        GeoTensor reprojected to dst_crs with resolution_dst_crs

    """

    named_shape = OrderedDict(zip(data_in.dims, data_in.shape))

    # Compute output transform
    dst_transform = window_utils.figure_out_transform(transform=dst_transform, bounds=bounds,
                                                      resolution_dst=resolution_dst_crs)

    # Compute size of window in out crs
    if window_out is None:
        assert bounds is not None, "Both window_out and bounds are None. This is needed to figure out the size of the output array"
        window_out = rasterio.windows.from_bounds(*bounds,
                                                  transform=dst_transform).round_lengths(op="ceil",
                                                                                         pixel_precision=PIXEL_PRECISION)

    # Compute real bounds that are going to be read
    bounds = window_utils.normalize_bounds(rasterio.windows.bounds(window_out, dst_transform))

    crs_data_in = data_in.crs
    if dst_crs is None:
        dst_crs = crs_data_in

    #  if dst_crs == data_in.crs and the resolution is the same and window is exact return read_from_window
    if compare_crs(dst_crs, crs_data_in):
        transform_data = data_in.transform
        if (dst_transform.a == transform_data.a) and (dst_transform.b == transform_data.b) and (
                dst_transform.d == transform_data.d) and (dst_transform.e == transform_data.e):
            window_in_data = rasterio.windows.from_bounds(*bounds, transform=transform_data).round_lengths(op="ceil",
                                                                                                           pixel_precision=PIXEL_PRECISION)
            if _is_exact_round(window_in_data.row_off) and _is_exact_round(
                    window_in_data.col_off) and window_in_data.width == window_out.width \
                    and window_in_data.height == window_out.height:
                window_in_data = window_in_data.round_offsets(op="floor", pixel_precision=PIXEL_PRECISION)
                return read_from_window(data_in, window_in_data, return_only_data=return_only_data)

    cast = False
    if dtpye_dst is None:
        cast = True
        dtpye_dst = data_in.dtype

    # Create out array for reprojection
    dict_shape_window_out = {"x": window_out.width, "y": window_out.height}
    shape_out = tuple([named_shape[s] if s not in ["x", "y"] else dict_shape_window_out[s] for s in named_shape])
    destination = np.zeros(shape_out, dtype=dtpye_dst)

    # Read a padded window of the input data. This data will be then used for reprojection
    dataarray_in = read_from_bounds(data_in, bounds, dst_crs,
                                    pad_add=(3, 3), return_only_data=False,
                                    trigger_load=True)
    # Trigger load makes that fill_value_default goes to nodata

    np_array_in = np.asanyarray(dataarray_in.values)
    if cast:
        np_array_in = np_array_in.astype(dtpye_dst)

    dst_nodata = dst_nodata or dataarray_in.fill_value_default

    index_iter = [[(ns, i) for i in range(s)] for ns, s in named_shape.items() if ns not in ["x", "y"]]
    # e.g. if named_shape = {'time': 4, 'band': 2, 'x':10, 'y': 10} index_iter ->
    # [[('time', 0), ('time', 1), ('time', 2), ('time', 3)],
    #  [('band', 0), ('band', 1)]]

    for current_select_tuple in itertools.product(*index_iter):
        # current_select_tuple = (('time', 0), ('band', 0))
        i_sel_tuple = tuple(t[1] for t in current_select_tuple)

        np_array_iter = np_array_in[i_sel_tuple]
        dst_iter_write = destination[i_sel_tuple]

        rasterio.warp.reproject(
            np_array_iter,
            dst_iter_write,
            src_transform=dataarray_in.transform,
            src_crs=crs_data_in,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=dataarray_in.fill_value_default,
            dst_nodata=dst_nodata,
            resampling=resampling)

    if return_only_data:
        return destination

    return GeoTensor(destination, transform=dst_transform, crs=dst_crs,
                     fill_value_default=dst_nodata)