"""
Slices Module: Generate windows for tiled/chunked raster processing.

This module provides utilities to divide large rasters into overlapping or
non-overlapping tiles (slices) for memory-efficient processing. Essential
for processing datasets that don't fit in memory.

Tiling Strategies
-----------------

Choosing the right tiling approach for your use case::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    TILING STRATEGIES                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Non-overlapping (overlap=0)          Overlapping (overlap>0)           │
    │  ─────────────────────────            ──────────────────────            │
    │                                                                          │
    │  ┌────┬────┬────┬────┐              ┌────┬────┬────┬────┐               │
    │  │ 1  │ 2  │ 3  │ 4  │              │ 1 ─┼─ 2 ┼─ 3 ┼─ 4 │               │
    │  ├────┼────┼────┼────┤              ├───┬┼───┬┼───┬┼───┬┤               │
    │  │ 5  │ 6  │ 7  │ 8  │              │ 5 │├ 6 │├ 7 │├ 8 ││               │
    │  ├────┼────┼────┼────┤              ├───┼┼───┼┼───┼┼───┼┤               │
    │  │ 9  │ 10 │ 11 │ 12 │              │ 9 ││10 ││11 ││12 ││               │
    │  └────┴────┴────┴────┘              └───┴┴───┴┴───┴┴───┴┘               │
    │                                          └─ overlap region              │
    │  Best for:                          Best for:                           │
    │  • Independent tiles                • Edge-sensitive algorithms         │
    │  • Aggregation tasks                • Convolutions/filters              │
    │  • Simple mosaicking                • Seamline blending                 │
    └─────────────────────────────────────────────────────────────────────────┘

Slice Coordinates vs Windows
----------------------------

Understanding the relationship between slices and windows::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │              SLICES (Python) vs WINDOWS (Rasterio)                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  slice(start, stop)              Window(col_off, row_off, width, height)│
    │  ──────────────────              ──────────────────────────────────────│
    │                                                                          │
    │  • Python array indexing         • Rasterio file reading                │
    │  • 2D: (row_slice, col_slice)    • 2D: explicit offsets + sizes         │
    │  • End-exclusive                 • Width/height inclusive               │
    │                                                                          │
    │  Conversion:                                                             │
    │    slices_to_windows((row_slice, col_slice)) → Window                   │
    │    window_to_slices(window) → (row_slice, col_slice)                    │
    │                                                                          │
    │  Example:                                                                │
    │    slice(100, 356), slice(200, 456)  →  Window(200, 100, 256, 256)      │
    │    (rows 100-355, cols 200-455)          (col=200, row=100, 256x256)    │
    └─────────────────────────────────────────────────────────────────────────┘

Edge Handling Options
---------------------

What happens when tiles don't fit evenly::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    EDGE HANDLING OPTIONS                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Raster: 500px, Tile: 128px, Overlap: 0                                 │
    │  ──────────────────────────────────────                                 │
    │                                                                          │
    │  include_incomplete=True (default):                                     │
    │  ┌────────┬────────┬────────┬────────┬────┐                             │
    │  │  128   │  128   │  128   │  128   │ 12 │  ← 5 tiles, last is 12px   │
    │  └────────┴────────┴────────┴────────┴────┘                             │
    │                                                                          │
    │  include_incomplete=False:                                              │
    │  ┌────────┬────────┬────────┐                                           │
    │  │  128   │  128   │  128   │  ← 3 tiles, drops edge                    │
    │  └────────┴────────┴────────┘                                           │
    │                                                                          │
    │  trim_incomplete=True:                                                   │
    │  Same as include_incomplete=True but last slice is trimmed              │
    │  slice(384, 500) instead of slice(384, 512)                             │
    └─────────────────────────────────────────────────────────────────────────┘

Module Functions Overview
-------------------------

Slice Generation:
    - :func:`_slices`: Generate 1D slices for a single dimension
    - :func:`_slices_nd`: Generate N-dimensional slice tuples
    - :func:`_slices_2d`: Generate 2D (row, col) slice tuples

Window Conversion:
    - :func:`slices_to_windows`: Convert (row_slice, col_slice) → Window
    - :func:`window_to_slices`: Convert Window → (row_slice, col_slice)
    - :func:`windows_from_shape`: Generate Windows covering a shape

Quick Start
-----------

Generate tiles for processing a large raster::

    from georeader import slices
    import rasterio

    # Generate 256x256 tiles with 32px overlap
    with rasterio.open("large_raster.tif") as src:
        tiles = slices._slices_2d(
            shape=(src.height, src.width),
            size=(256, 256),
            overlap=(32, 32)
        )

        for row_slice, col_slice in tiles:
            # Read this tile
            data = src.read(window=slices.slices_to_windows((row_slice, col_slice)))
            # Process...

Convert between slices and windows::

    from georeader import slices

    # Slices to Window
    row_s, col_s = slice(100, 356), slice(200, 456)
    window = slices.slices_to_windows((row_s, col_s))
    # Window(col_off=200, row_off=100, width=256, height=256)

    # Window to slices
    row_s, col_s = slices.window_to_slices(window)
    # slice(100, 356), slice(200, 456)

See Also
--------
georeader.mosaic : Mosaic processing using tiles
georeader.read : Reading functions that accept Windows
rasterio.windows : Window class documentation
"""
import rasterio.windows
import itertools
from typing import Dict, List, Tuple, Optional


def _slices(dimsize: int, size: int, overlap: int = 0, include_incomplete: bool = True,
            start_negative_if_padding: bool = False, trim_incomplete: bool = False) -> List[slice]:
    """
    Generate a list of slice objects to divide a dimension into overlapping windows.

    This is the core 1D tiling function. It divides a dimension of length `dimsize`
    into windows of length `size`, with optional overlap between adjacent windows.
    The stride (step between window starts) is computed as: stride = size - overlap.

    Args:
        dimsize (int): Total size of the dimension to slice (e.g., image width=1000).
        size (int): Size of each window/tile (e.g., 256 for 256-pixel tiles).
        overlap (int): Pixels shared between adjacent windows. Default 0 (no overlap).
            Common values: 16, 32, 64 for CNN inference with edge artifacts.
        include_incomplete (bool): Whether to include final windows that are smaller
            than `size`. Default True. Set False to get only full-size windows.
        start_negative_if_padding (bool): Start first window at -overlap//2 instead
            of 0. Useful for prediction mode where you want symmetric padding at
            both edges. Default False.
        trim_incomplete (bool): For edge windows, trim the slice to actual data bounds
            instead of extending beyond. Default False.

            - False: slice(900, 1156) for 256-window at edge of 1000px image
            - True: slice(900, 1000) trimmed to actual bounds

    Returns:
        List[slice]: List of slice objects covering the dimension.
            Each slice has format slice(start, stop).

    Raises:
        AssertionError: If stride <= 0 (overlap >= size) or stride >= dimsize.

    Examples:
        Basic non-overlapping slices:

        >>> _slices(dimsize=1000, size=256, overlap=0)
        [slice(0, 256), slice(256, 512), slice(512, 768), slice(768, 1024)]
        # Note: last slice extends beyond dimsize (1024 > 1000)

        With overlap for CNN inference:

        >>> _slices(dimsize=1000, size=256, overlap=32)
        [slice(0, 256), slice(224, 480), slice(448, 704), slice(672, 928), slice(896, 1152)]
        # stride = 256 - 32 = 224

        Exclude incomplete edge windows:

        >>> _slices(dimsize=1000, size=256, overlap=0, include_incomplete=False)
        [slice(0, 256), slice(256, 512), slice(512, 768)]
        # Only 3 complete windows, drops slice(768, 1024)

        Trim edge windows to actual bounds:

        >>> _slices(dimsize=1000, size=256, overlap=0, trim_incomplete=True)
        [slice(0, 256), slice(256, 512), slice(512, 768), slice(768, 1000)]
        # Last slice trimmed: 1024 → 1000

        Start negative for symmetric padding:

        >>> _slices(dimsize=100, size=64, overlap=16, start_negative_if_padding=True)
        [slice(-8, 56), slice(40, 104)]
        # First slice starts at -overlap//2 = -8

    Note:
        - For ML inference, use overlap to handle edge artifacts, then crop predictions
        - stride = size - overlap, so overlap=0 means non-overlapping tiles
        - Negative slice indices require boundless=True when reading from rasterio
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
    Generate N-dimensional slice combinations for tiled array processing.

    This function extends 1D slicing (:func:`_slices`) to multiple named dimensions,
    returning all combinations of slices as a Cartesian product. Designed for
    xarray-style named dimensions but works with any string keys.

    The function is useful for processing large multidimensional datasets in
    chunks, such as temporal-spatial data cubes or multi-resolution pyramids.

    Args:
        named_shape (Dict[str, int]): Shape of the array with named dimensions.
            Example: ``{"x": 5600, "y": 4000}`` for a 5600×4000 raster.
        dims (Dict[str, int]): Size of tiles for each dimension to slice.
            Only dimensions in this dict will be sliced; others are left whole.
            Example: ``{"x": 128, "y": 128}`` for 128×128 tiles.
        overlap (Optional[Dict[str, int]]): Overlap pixels per dimension.
            Example: ``{"x": 16, "y": 16}`` for 16-pixel overlap.
            Defaults to 0 for any dimension not specified.
        include_incomplete (bool): Include edge tiles smaller than `dims`.
            Default True.
        start_negative_if_padding (bool): Start at -overlap//2 for symmetric
            edge handling. Default False.
        trim_incomplete (bool): Trim edge tiles to actual array bounds rather
            than extending beyond. Default True.

    Returns:
        List[Dict[str, slice]]: List of slice dictionaries, one per tile.
            Each dict maps dimension names to slice objects.
            Total count = product of tile counts per dimension.

    Examples:
        Simple 2D tiling:

        >>> named_shape = {"x": 500, "y": 400}
        >>> dims = {"x": 256, "y": 256}
        >>> tiles = create_slices(named_shape, dims)
        >>> len(tiles)  # 2 in x × 2 in y = 4 tiles
        4
        >>> tiles[0]
        {'x': slice(0, 256), 'y': slice(0, 256)}

        With overlap for ML inference:

        >>> tiles = create_slices(
        ...     named_shape={"x": 1000, "y": 1000},
        ...     dims={"x": 512, "y": 512},
        ...     overlap={"x": 64, "y": 64}
        ... )
        >>> # stride = 512 - 64 = 448
        >>> len(tiles)  # 3 in x × 3 in y = 9 tiles
        9

        Use with xarray:

        >>> import xarray as xr
        >>> data = xr.DataArray(np.random.rand(400, 500), dims=['y', 'x'])
        >>> tiles = create_slices(
        ...     named_shape={"x": data.sizes["x"], "y": data.sizes["y"]},
        ...     dims={"x": 128, "y": 128}
        ... )
        >>> for tile_slices in tiles:
        ...     chunk = data.isel(tile_slices)  # Extract tile
        ...     # Process chunk...

    See Also:
        create_windows: Returns rasterio.windows.Window objects instead of dicts.
        _slices: Underlying 1D slicing function.
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
    Generate rasterio Window objects covering a raster in tiles.

    This is the primary function for tiled raster processing with rasterio/georeader.
    It creates a list of Windows that can be passed directly to read functions.

    Args:
        geodata_shape (Tuple[int, int]): Spatial shape of the raster as (height, width)
            or equivalently (n_rows, n_cols). Note: height first, matching numpy convention.
        window_size (Tuple[int, int]): Size of each tile as (height, width).
            Example: (256, 256) for 256×256 pixel tiles.
        overlap (Optional[Tuple[int, int]]): Overlap in pixels as (row_overlap, col_overlap).
            Example: (32, 32) for 32-pixel overlap in both dimensions.
            Default None (no overlap, stride = window_size).
        include_incomplete (bool): Include edge tiles that are smaller than window_size.
            Default True.
        start_negative_if_padding (bool): Start tiling at -overlap//2 for symmetric
            edge padding. Requires boundless=True when reading. Default False.
        trim_incomplete (bool): Trim edge tiles to raster bounds rather than extending
            beyond. Default True.

    Returns:
        List[rasterio.windows.Window]: Windows covering the raster. Each window has
            integer col_off, row_off, width, height attributes.

    Examples:
        Generate tiles for a raster:

        >>> from georeader import slices
        >>> import rasterio
        >>>
        >>> # 1000×1200 raster into 256×256 tiles
        >>> windows = slices.create_windows(
        ...     geodata_shape=(1000, 1200),
        ...     window_size=(256, 256)
        ... )
        >>> len(windows)
        20  # 4 rows × 5 cols
        >>> windows[0]
        Window(col_off=0, row_off=0, width=256, height=256)

        Process raster in tiles:

        >>> with rasterio.open("large_image.tif") as src:
        ...     windows = slices.create_windows(
        ...         geodata_shape=(src.height, src.width),
        ...         window_size=(512, 512),
        ...         overlap=(64, 64)
        ...     )
        ...     for window in windows:
        ...         data = src.read(window=window)
        ...         # Process tile...

        With GeoTensor:

        >>> from georeader import read
        >>> windows = slices.create_windows(
        ...     geodata_shape=geotensor.shape[-2:],  # (H, W)
        ...     window_size=(256, 256)
        ... )
        >>> for window in windows:
        ...     tile = read.read_from_window(geotensor, window)

    See Also:
        create_slices: Returns dict of slices for xarray-style indexing.
        georeader.read.read_from_window: Read data using a Window.
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

