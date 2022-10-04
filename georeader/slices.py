import rasterio.windows
import itertools
from typing import Dict, List, Tuple, Optional


def _slices(dimsize: int, size: int, overlap: int = 0, include_incomplete: bool = True,
            start_negative_if_padding: bool = False, trim_incomplete: bool = False) -> List[slice]:
    """
    Return a list of slices to chop up a single dimension.

    Args:
        dimsize: size of the dimension
        size: size of the window (e.g. 128 for slices of shape 128)
        overlap: number of pixels to read with overlap (e.g. 16 to read with 16 px overlap)
        include_incomplete: if `True` includes incomplete slices in the borders if `False` it discards the slices in the
        borders that will have size lower than dims
        start_negative_if_padding: if `True` starts in -overlap//2.
        trim_incomplete: for the end of the array, trim the value
            (e.g. if the end raster has dimsize=11, and size=5 the last slice will be slice(10,11) if True)

    Returns:
        List of slice objects.
    """
    slices = []
    if dimsize < size:
        end = dimsize if trim_incomplete else size
        return [slice(0, end)]

    stride = size - overlap
    assert stride > 0, f"{stride} less than 0"
    assert stride < dimsize, f"{stride} < {dimsize}"
    if start_negative_if_padding:
        start_value = -overlap // 2
    else:
        start_value = 0
    for start in range(start_value, dimsize, stride):
        end = start + size
        if include_incomplete or (end <= dimsize):
            if trim_incomplete and end > dimsize:
                end = dimsize
            slices.append(slice(start, end))
    return slices


def create_slices(named_shape: Dict[str, int],
                  dims: Dict[str, int], overlap: Optional[Dict[str, int]] = None,
                  include_incomplete: bool = True, start_negative_if_padding: bool = False,
                  trim_incomplete: bool = True) -> List[Dict[str, slice]]:
    """
    This function creates a list of slice objects to slice the dataset over the given dimensions

    Args:
        named_shape: {"x": 5600, "y": 4000} shape to split in slices
        dims: size of the slices {"x": 128, "y": 128}
        overlap: number of pixels to read with overlap e.g. {"x": 16, "y": 16} to read with 16 px overlap
        include_incomplete: if `True` includes incomplete slices in the borders if `False` it discards the slices in the
        borders that will have size lower than dims.
        start_negative_if_padding: if `True` starts in -overlap//2 each slice. Useful to create slices to write in predict
            mode.
        trim_incomplete: for the end of the array, trim the value
            (e.g. if the end raster has dimsize=11, and size=5 the last slice will be slice(10,11) if True and slice(10,15) if False)

    Returns:
        List of dictionaries of slice objects that can be used to chip the data.
    """
    if overlap is None:
        overlap = {}

    dim_slices = []
    for dim in dims:
        dimsize = named_shape[dim]
        size = dims[dim]
        olap = overlap.get(dim, 0)
        dim_slices.append(_slices(dimsize, size, olap, include_incomplete=include_incomplete,
                                  start_negative_if_padding=start_negative_if_padding,
                                  trim_incomplete=trim_incomplete))

    return [{key: slic for key, slic in zip(dims, tuple_slices)} for tuple_slices in itertools.product(*dim_slices)]


def create_windows(geodata_shape: Tuple[int, int],
                   window_size: Tuple[int, int], overlap: Optional[Tuple[str, int]] = None,
                   include_incomplete: bool = True, start_negative_if_padding: bool = False,
                   trim_incomplete: bool = True) -> List[rasterio.windows.Window]:
    """
    This function creates a list of window objects to slice the dataset in windows of shape `window_size`

    Args:
        geodata_shape: tuple with the spatial shape of hte geodata object `(n_rows, n_cols)` `(height, width)`
        window_size: shape of the windows to yield
        overlap: number of pixels to read with overlap (same concept as stride in neural networks)
        include_incomplete: if `True` includes incomplete slices in the borders if `False` it discards the slices in the
        borders that will have size lower than dims.
        start_negative_if_padding: if `True` starts in -overlap//2 each slice. Useful to create slices to write in predict
            mode.
        trim_incomplete: if `True` windows at the edge of the array will have smaller size if needed (i.e. those windows
            will have a smaller width or height)

    Returns:
        List of window objects covering the data

    """
    named_shape = {"x":geodata_shape[-1], "y":geodata_shape[-2]}

    if overlap is not None:
        overlap = {"x": overlap[1], "y": overlap[0]}

    list_of_dict_slices = create_slices(named_shape,
                                        {"x": window_size[1], "y":window_size[0]},
                                        overlap=overlap, include_incomplete=include_incomplete,
                                        start_negative_if_padding=start_negative_if_padding,
                                        trim_incomplete=trim_incomplete)

    return [rasterio.windows.Window.from_slices(dict_slices["y"], dict_slices["x"], boundless=True) for dict_slices in list_of_dict_slices]

