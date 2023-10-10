import rasterio.windows
import rasterio.transform
from typing import Tuple, Dict, Optional, Union, Any, List
import numbers
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
import rasterio.warp
from georeader import compare_crs

PIXEL_PRECISION = 3

def pad_window(window: rasterio.windows.Window, pad_size: Tuple[int, int]) -> rasterio.windows.Window:
    """
    Add the provided pad to a rasterio window object

    Args:
        window: input window
        pad_size: Tuple,
            `pad_size[0]` pad to add in rows (in both sides).
            `pad_size[1]` pad to add in columns (in both sides).

    Returns:
        window with width `window.width + 2*pad_size[1]` and height `window.height + 2 * pad_size[0]`
    """

    return rasterio.windows.Window(window.col_off - pad_size[1],
                                   window.row_off - pad_size[0],
                                   width=window.width + 2 * pad_size[1],
                                   height=window.height + 2 * pad_size[0])

def pad_window_to_size(window: rasterio.windows.Window, size: Tuple[int, int]) -> rasterio.windows.Window:
    """
    Center pad the given window to the given size. if `size` is smaller than the size of the window it returns
    the center pad of that window of the requested size.

    Args:
        window: input window
        size: Tuple. size of the output image

    Returns:
         window centered in `window` with width `size[1]` and height `size[0]`

    """
    pad_add_rows = size[0] - window.height
    pad_add_cols = size[1] - window.width

    pad_rows_half = pad_add_rows // 2
    pad_cols_half = pad_add_cols // 2

    return rasterio.windows.Window(window.col_off - pad_rows_half,
                                   window.row_off - pad_cols_half,
                                   width=window.width + pad_add_cols,
                                   height=window.height + pad_add_rows)


def figure_out_transform(transform: Optional[rasterio.Affine] = None,
                         bounds: Optional[Tuple[float, float, float, float]] = None,
                         resolution_dst: Optional[Union[float, Tuple[float, float]]] = None) -> rasterio.Affine:
    """
    Based on transform, bounds and resolution_dst computes the output transform.

    Args:
        transform: base transform used as reference. If not provided will return a rectilinear transform.
        bounds: bounds of the output transform
        resolution_dst: resolution of the output transform

    Returns:
        rasterio.Affine object with resolution `resolution_dst` and origin at the bounds
    """
    if resolution_dst is not None:
        if isinstance(resolution_dst, numbers.Number):
            resolution_dst = (abs(resolution_dst), abs(resolution_dst))

    if transform is None:
        assert bounds is not None, "Transform and bounds not provided"
        assert resolution_dst is not None, "Transform and resolution not provided"
        return rasterio.transform.from_origin(min(bounds[0], bounds[2]),
                                              max(bounds[1], bounds[3]),
                                              resolution_dst[0], resolution_dst[1])

    if resolution_dst is None:
        dst_transform = transform
    else:
        resolution_or = res(transform)
        transform_scale = rasterio.Affine.scale(resolution_dst[0] / resolution_or[0],
                                                resolution_dst[1] / resolution_or[1])
        dst_transform = transform * transform_scale

    if bounds is not None:
        # Shift the transform to start in the bounds
        window_current_transform = rasterio.windows.from_bounds(*bounds,
                                                                transform=transform)
        dst_transform = rasterio.windows.transform(window_current_transform, dst_transform)

    return dst_transform


def round_outer_window(window:rasterio.windows.Window)-> rasterio.windows.Window:
    """ Rounds a rasterio.windows.Window object to outer (larger) window """
    return window.round_lengths(op="ceil", pixel_precision=PIXEL_PRECISION).round_offsets(op="floor",
                                                                                          pixel_precision=PIXEL_PRECISION)

# Precision to round the windows before applying ceiling/floor. e.g. 3.0001 will be rounded to 3 but 3.001 will not
def _is_exact_round(x, precision=PIXEL_PRECISION):
    return abs(round(x,ndigits=precision)-x) < 1e-6


def res(transform:rasterio.Affine) -> Tuple[float, float]:
    """
    Computes the resolution from a given transform

    Args:
        transform:

    Returns:
        resolution (tuple of floats)
    """

    z0_0 = np.array(transform * (0, 0))
    z0_1 = np.array(transform * (0, 1))
    z1_0 = np.array(transform * (1, 0))

    return np.sqrt(np.sum((z0_0 - z1_0) ** 2)), np.sqrt(np.sum((z0_0 - z0_1) ** 2))


def get_slice_pad(window_data:rasterio.windows.Window,
                  window_read:rasterio.windows.Window) -> Tuple[Dict[str, slice], Dict[str, Tuple[int, int]]]:
    """
    It returns the slice w.r.t. `window_data` and the pad to add to read a window `window_read`.
    It returns two dictionaries to be used with `xr.DataArray.isel` and `xr.DataArray.pad` (resp.)

    Args:
        window_data: `rasterio.windows.Window(col_off=0, row_off=0, width=named_shape["x"], height=named_shape["y"])`
        window_read: window intersecting `window_data`.

    Returns: Tuple with two dictionaries
        slice_dict: `{"x": slice(a,b), "y": slice(c,d)}`
        pad_width: {"x": (pad_x_0, pad_x_1), "y": (pad_y_0, pad_y_1)}

    Raises:
        rasterio.windows.WindowError if `window_data` and `window_read` do not intersect

    """
    if not rasterio.windows.intersect([window_data, window_read]):
        raise rasterio.windows.WindowError(f"Window data: {window_data} and window read: {window_read} do not intersect")

    if window_read.row_off < window_data.row_off:
        pad_y_0 = window_data.row_off - window_read.row_off
        window_row_start = window_data.row_off
    else:
        pad_y_0 = 0
        window_row_start = window_read.row_off - window_data.row_off

    if window_read.col_off < window_data.col_off:
        pad_x_0 = window_data.col_off - window_read.col_off
        window_col_start = window_data.col_off
    else:
        pad_x_0 = 0
        window_col_start = window_read.col_off - window_data.col_off

    if (window_read.width + window_read.col_off) > (window_data.width + window_data.col_off):
        pad_x_1 = (window_read.width + window_read.col_off) - (window_data.width + window_data.col_off)
        window_col_end = window_data.width + window_data.col_off
    else:
        pad_x_1 = 0
        window_col_end = window_read.width + window_read.col_off

    if (window_read.height + window_read.row_off) > (window_data.height + window_data.row_off):
        pad_y_1 = (window_read.height + window_read.row_off) - (window_data.height + window_data.row_off)
        window_row_end = window_data.height + window_data.row_off
    else:
        pad_y_1 = 0
        window_row_end = window_read.height + window_read.row_off

    row_slice = slice(window_row_start, window_row_end)
    col_slice = slice(window_col_start, window_col_end)

    slice_dict = {"x": col_slice, "y": row_slice}
    pad_width = {"x": (pad_x_0, pad_x_1), "y": (pad_y_0, pad_y_1)}

    return slice_dict, pad_width

def window_polygon(window:rasterio.windows.Window, transform:rasterio.Affine, window_surrounding:bool=False) -> Polygon:
    """
    Computes the Polygon that contains the window

    works similarly as `rasterio.windows.bounds` function

    Args:
        window: Window object to compute the polygon
        transform: geotransform
        window_surrounding: The window surrounds the polygon. (i.e. window.row_off + window.height will not be a vertex)

    Returns:
        Polygon surrounding the window in geographical coordinates
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


def window_bounds(window:rasterio.windows.Window,
                  transform:rasterio.Affine) -> Tuple[float, float, float, float]:
    """Get the spatial bounds of a window.

    This is a re-implementation of rasterio.window.bounds that works with non-rectilinear transforms!

    Parameters
    ----------
    window: Window
        The input window.
    transform: Affine
        an affine transform matrix.

    Returns
    -------
    xmin, ymin, xmax, ymax: float
        A tuple of spatial coordinate bounding values.
    """

    row_min = window.row_off
    row_max = row_min + window.height
    col_min = window.col_off
    col_max = col_min + window.width
    corner_00 = transform * (col_min, row_min)
    corner_01 = transform * (col_min, row_max)
    corner_10 = transform * (col_max, row_min)
    corner_11 = transform * (col_max, row_max)
    all_corners = [corner_00, corner_01, corner_10, corner_11]

    return min(c[0] for c in all_corners), min(c[1] for c in all_corners), \
           max(c[0] for c in all_corners), max(c[1] for c in all_corners)

def normalize_bounds(bounds:Tuple[float, float, float, float], margin_add_if_equal:float=.0005) -> Tuple[float, float, float, float]:
    """ Return bounds with a small margin if it is not a rectangle """
    xmin = min(bounds[0], bounds[2])
    ymin = min(bounds[1], bounds[3])
    xmax = max(bounds[0], bounds[2])
    ymax = max(bounds[1], bounds[3])

    if xmin >= xmax:
        xmin-=margin_add_if_equal
        xmax+=margin_add_if_equal

    if ymin >= ymax:
        ymin-= margin_add_if_equal
        ymax+=margin_add_if_equal

    return xmin, ymin, xmax, ymax


def polygon_to_crs(polygon:Union[Polygon, MultiPolygon], crs_polygon:Any, dst_crs:Any) -> Union[Polygon, MultiPolygon]:
    return shape(rasterio.warp.transform_geom(crs_polygon, dst_crs, mapping(polygon)))


def pad_list_numpy(pad_width:Dict[str, Tuple[int, int]]) -> List[Tuple[int, int]]:
    pad_list_np = []
    for k in ["y", "x"]:
        if k in pad_width:
            pad_list_np.append(pad_width[k])
        else:
            pad_list_np.append((0, 0))
    return pad_list_np