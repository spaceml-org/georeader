import rasterio.windows
from typing import Tuple, Dict

PIXEL_PRECISION = 3

def pad_window(window: rasterio.windows.Window, pad_size: Tuple[int, int]) -> rasterio.windows.Window:
    """ Add the provided pad to a rasterio window object """
    return rasterio.windows.Window(window.col_off - pad_size[1],
                                   window.row_off - pad_size[0],
                                   width=window.width + 2 * pad_size[1],
                                   height=window.height + 2 * pad_size[1])


def round_outer_window(window:rasterio.windows.Window)-> rasterio.windows.Window:
    """ Rounds a rasterio.windows.Window object to outer (larger) window """
    return window.round_lengths(op="ceil", pixel_precision=PIXEL_PRECISION).round_offsets(op="floor",
                                                                                          pixel_precision=PIXEL_PRECISION)


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