"""Async siblings of :mod:`georeader.read` for :class:`AsyncGeoData` inputs.

Mirror the read + reproject family in :mod:`georeader.read` for inputs that
satisfy :class:`~georeader.abstract_reader.AsyncGeoData` (e.g.
:class:`~georeader.async_geotiff_reader.AsyncGeoTIFFReader`). Each function
here has the same signature and semantics as its sync sibling, except it is
declared ``async def`` and uses ``await`` at the single I/O boundary —
materialising the windowed view via ``await view.load()``.

The reprojection helpers (``read_reproject``, ``read_reproject_like``,
``read_to_crs``, ``resize``, ``read_from_tile``) **stream only the input
window required for the destination grid** — they do **not** load the entire
raster first. The setup, intersection check, dtype handling and warp loop are
all shared with the sync path via the private helpers
:func:`georeader.read._reproject_setup` and
:func:`georeader.read._reproject_finalize`; the only difference between sync
and async is which reader-method is used to fetch the input bytes.

Usage::

    from obstore.store import S3Store
    from georeader.async_geotiff_reader import AsyncGeoTIFFReader
    from georeader import asyncread

    store = S3Store(bucket="my-bucket", region="us-east-1")
    reader = await AsyncGeoTIFFReader.open("scene.tif", store=store)

    # Same call shape as `read.read_to_crs`, but awaitable and streams only
    # the input window that contributes to the EPSG:4326 destination grid.
    gt = await asyncread.read_to_crs(reader, dst_crs="EPSG:4326")

See Also
--------
georeader.read : sync counterpart used by :class:`RasterioReader` and other
    :class:`~georeader.abstract_reader.GeoData` inputs. Refer to its
    docstrings for the full parameter/example treatment — the contracts here
    are identical apart from ``await``-ability.
"""
from __future__ import annotations

import inspect
import numbers
from math import ceil
from typing import Any, Optional, Tuple, Union

import mercantile
import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows
from shapely.geometry import MultiPolygon, Polygon, box

from georeader import window_utils
from georeader.abstract_reader import AsyncGeoData
from georeader.geotensor import GeoTensor
from georeader.read import (
    SIZE_DEFAULT,
    WEB_MERCATOR_CRS,
    _build_no_intersect_result,
    _reproject_finalize,
    _reproject_setup,
    _window_intersects_data,
    apply_anti_aliasing,
    calculate_transform_window,
    window_from_bounds,
    window_from_center_coords,
    window_from_polygon,
)
from georeader.window_utils import pad_window, round_outer_window


async def read_from_window(
    data_in: AsyncGeoData,
    window: rasterio.windows.Window,
    return_only_data: bool = False,
    trigger_load: bool = False,
    boundless: bool = True,
) -> Union[AsyncGeoData, GeoTensor, np.ndarray, None]:
    """Async sibling of :func:`georeader.read.read_from_window`.

    Returns:
        - When ``return_only_data=True``: ``np.ndarray`` (forces materialise).
        - When ``trigger_load=True``: :class:`GeoTensor`.
        - Otherwise: the unmaterialised windowed view (an :class:`AsyncGeoData`
          subclass). Call ``await view.load()`` to materialise.

    The no-intersection branch never awaits — it returns the synthetic padded
    array / GeoTensor directly. The single ``await`` is on
    ``view.load()`` when materialisation is requested.
    """
    if not _window_intersects_data(data_in, window):
        return _build_no_intersect_result(data_in, window, boundless, return_only_data)

    view = data_in.read_from_window(window=window, boundless=boundless)

    # If a concrete reader happens to return a GeoTensor here (e.g. a fully
    # eager async reader), accept it; otherwise materialise the view.
    if isinstance(view, GeoTensor):
        data_sel: Union[AsyncGeoData, GeoTensor] = view
    elif return_only_data or trigger_load:
        data_sel = await view.load()
    else:
        return view

    if return_only_data:
        return data_sel.values  # type: ignore[union-attr]
    return data_sel


async def read_from_bounds(
    data_in: AsyncGeoData,
    bounds: Tuple[float, float, float, float],
    crs_bounds: Optional[str] = None,
    pad_add: Tuple[int, int] = (0, 0),
    return_only_data: bool = False,
    trigger_load: bool = False,
    boundless: bool = True,
) -> Union[AsyncGeoData, GeoTensor, np.ndarray, None]:
    """Async sibling of :func:`georeader.read.read_from_bounds`."""
    window_in = window_from_bounds(data_in, bounds, crs_bounds)
    if any(p > 0 for p in pad_add):
        window_in = pad_window(window_in, pad_add)
    window_in = round_outer_window(window_in)
    return await read_from_window(
        data_in,
        window_in,
        return_only_data=return_only_data,
        trigger_load=trigger_load,
        boundless=boundless,
    )


async def read_from_polygon(
    data_in: AsyncGeoData,
    polygon: Union[Polygon, MultiPolygon],
    crs_polygon: Optional[str] = None,
    pad_add: Tuple[int, int] = (0, 0),
    return_only_data: bool = False,
    trigger_load: bool = False,
    boundless: bool = True,
    window_surrounding: bool = False,
) -> Union[AsyncGeoData, GeoTensor, np.ndarray, None]:
    """Async sibling of :func:`georeader.read.read_from_polygon`."""
    window_in = window_from_polygon(
        data_in, polygon, crs_polygon, window_surrounding=window_surrounding
    )
    if any(p > 0 for p in pad_add):
        window_in = pad_window(window_in, pad_add)
    window_in = round_outer_window(window_in)
    return await read_from_window(
        data_in,
        window_in,
        return_only_data=return_only_data,
        trigger_load=trigger_load,
        boundless=boundless,
    )


async def read_from_center_coords(
    data_in: AsyncGeoData,
    center_coords: Tuple[float, float],
    shape: Tuple[int, int],
    crs_center_coords: Optional[Any] = None,
    return_only_data: bool = False,
    trigger_load: bool = False,
    boundless: bool = True,
) -> Union[AsyncGeoData, GeoTensor, np.ndarray, None]:
    """Async sibling of :func:`georeader.read.read_from_center_coords`."""
    window = window_from_center_coords(data_in, center_coords, shape, crs_center_coords)
    return await read_from_window(
        data_in,
        window=window,
        return_only_data=return_only_data,
        trigger_load=trigger_load,
        boundless=boundless,
    )


async def read_reproject(
    data_in: AsyncGeoData,
    dst_crs: Optional[str] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None,
    dst_transform: Optional[rasterio.Affine] = None,
    window_out: Optional[rasterio.windows.Window] = None,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    dtype_dst: Any = None,
    return_only_data: bool = False,
    dst_nodata: Optional[int] = None,
) -> Union[GeoTensor, np.ndarray]:
    """Async sibling of :func:`georeader.read.read_reproject`.

    Streams only the input window that contributes to the destination grid —
    matches the sync path's "windowed read + per-slice warp" semantics
    exactly. The non-I/O setup and warp loop are shared with the sync path via
    :func:`georeader.read._reproject_setup` and
    :func:`georeader.read._reproject_finalize`.
    """
    plan = _reproject_setup(
        data_in=data_in,
        dst_crs=dst_crs,
        bounds=bounds,
        resolution_dst_crs=resolution_dst_crs,
        dst_transform=dst_transform,
        window_out=window_out,
        dtype_dst=dtype_dst,
        dst_nodata=dst_nodata,
    )

    if plan.fast_path_window is not None:
        gt = await read_from_window(
            data_in,
            plan.fast_path_window,
            return_only_data=return_only_data,
            trigger_load=True,
        )
        return gt  # type: ignore[return-value]

    if plan.nonintersecting:
        return GeoTensor(
            plan.destination,
            transform=plan.dst_transform,
            crs=plan.dst_crs,
            fill_value_default=plan.dst_nodata,
        )

    if isinstance(data_in, GeoTensor):
        geotensor_in: GeoTensor = data_in
    else:
        geotensor_in = await read_from_polygon(
            data_in,
            plan.polygon_dst_crs,
            crs_polygon=plan.dst_crs,
            pad_add=(3, 3),
            return_only_data=False,
            trigger_load=True,
        )  # type: ignore[assignment]

    return _reproject_finalize(
        geotensor_in, plan, resampling=resampling, return_only_data=return_only_data
    )


async def read_reproject_like(
    data_in: AsyncGeoData,
    data_like: Any,
    resolution_dst: Optional[Union[float, Tuple[float, float]]] = None,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    dtype_dst: Any = None,
    return_only_data: bool = False,
    dst_nodata: Optional[int] = None,
) -> Union[GeoTensor, np.ndarray]:
    """Async sibling of :func:`georeader.read.read_reproject_like`."""
    shape_out = data_like.shape[-2:]
    if resolution_dst is not None:
        if isinstance(resolution_dst, float):
            resolution_dst = (resolution_dst, resolution_dst)
        resolution_data_like = data_like.res
        shape_out = (
            int(round(shape_out[0] / resolution_dst[0] * resolution_data_like[0])),
            int(round(shape_out[1] / resolution_dst[1] * resolution_data_like[1])),
        )

    return await read_reproject(
        data_in,
        dst_crs=data_like.crs,
        dst_transform=data_like.transform,
        resolution_dst_crs=resolution_dst,
        window_out=rasterio.windows.Window(0, 0, width=shape_out[-1], height=shape_out[-2]),
        resampling=resampling,
        dtype_dst=dtype_dst,
        return_only_data=return_only_data,
        dst_nodata=dst_nodata,
    )


async def read_to_crs(
    data_in: AsyncGeoData,
    dst_crs: Any,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None,
    return_only_data: bool = False,
) -> Union[AsyncGeoData, GeoTensor, np.ndarray]:
    """Async sibling of :func:`georeader.read.read_to_crs`."""
    if window_utils.compare_crs(data_in.crs, dst_crs):
        return data_in

    window_data, dst_transform = calculate_transform_window(data_in, dst_crs, resolution_dst_crs)

    return await read_reproject(
        data_in,
        dst_crs=dst_crs,
        dst_transform=dst_transform,
        window_out=window_data,
        resampling=resampling,
        return_only_data=return_only_data,
    )


async def resize(
    data_in: AsyncGeoData,
    resolution_dst: Union[float, Tuple[float, float]],
    window_out: Optional[rasterio.windows.Window] = None,
    anti_aliasing: bool = True,
    anti_aliasing_sigma: Optional[Union[float, np.ndarray]] = None,
    resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
    return_only_data: bool = False,
) -> Union[GeoTensor, np.ndarray]:
    """Async sibling of :func:`georeader.read.resize`.

    Note: when ``anti_aliasing=True`` and the input is lazy, the anti-aliasing
    step materialises the input via the lazy view's ``.values``. Pass an
    already-materialised :class:`GeoTensor` if you need to avoid the eager
    load — this matches the sync path's contract.
    """
    resolution_or = data_in.res
    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))
    scale = np.array(
        [resolution_dst[0] / resolution_or[0], resolution_dst[1] / resolution_or[1]]
    )

    if window_out is None:
        spatial_shape = data_in.shape[-2:]
        output_shape_exact = spatial_shape[0] / scale[0], spatial_shape[1] / scale[1]
        output_shape_rounded = (
            round(output_shape_exact[0], ndigits=3),
            round(output_shape_exact[1], ndigits=3),
        )
        output_shape = ceil(output_shape_rounded[0]), ceil(output_shape_rounded[1])
        window_out = rasterio.windows.Window(
            col_off=0, row_off=0, width=output_shape[1], height=output_shape[0]
        )

    src: Any = data_in
    if anti_aliasing:
        src = apply_anti_aliasing(
            data_in, anti_aliasing_sigma=anti_aliasing_sigma, resolution_dst=resolution_dst
        )

    return await read_reproject(
        src,
        dst_crs=src.crs,
        resolution_dst_crs=resolution_dst,
        dst_transform=src.transform,
        window_out=window_out,
        resampling=resampling,
        return_only_data=return_only_data,
    )


async def read_from_tile(
    data: AsyncGeoData,
    x: int,
    y: int,
    z: int,
    dst_crs: Optional[Any] = WEB_MERCATOR_CRS,
    out_shape: Optional[Tuple[int, int]] = (SIZE_DEFAULT, SIZE_DEFAULT),
    resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None,
    assert_if_not_intersects: bool = False,
) -> Optional[GeoTensor]:
    """Async sibling of :func:`georeader.read.read_from_tile`."""
    bounds_wgs = mercantile.xy_bounds(int(x), int(y), int(z))
    polygon_crs_webmercator = box(
        bounds_wgs.left, bounds_wgs.bottom, bounds_wgs.right, bounds_wgs.top
    )

    intersects = polygon_crs_webmercator.intersects(data.footprint(crs=WEB_MERCATOR_CRS))

    if not intersects:
        assert not assert_if_not_intersects, "Tile does not intersect data"
        return None

    # Fast-path delegation when the reader implements its own tile reader.
    # Support both sync (rare) and async (likely) overrides.
    if out_shape is not None and hasattr(data, "read_from_tile"):
        reader_tile = data.read_from_tile  # type: ignore[attr-defined]
        if inspect.iscoroutinefunction(reader_tile):
            return await reader_tile(x, y, z, dst_crs=dst_crs, out_shape=out_shape)
        return reader_tile(x, y, z, dst_crs=dst_crs, out_shape=out_shape)

    if dst_crs is None:
        dst_crs = data.crs

    if (
        window_utils.compare_crs(data.crs, dst_crs)
        and (out_shape is None)
        and (resolution_dst_crs is None)
    ):
        return await read_from_polygon(  # type: ignore[return-value]
            data,
            polygon_crs_webmercator,
            WEB_MERCATOR_CRS,
            window_surrounding=True,
            trigger_load=True,
        )

    if out_shape is not None:
        polygon_crs_dst = window_utils.polygon_to_crs(
            polygon_crs_webmercator, WEB_MERCATOR_CRS, dst_crs
        )
        bounds_dst = polygon_crs_dst.bounds
        dst_transform = rasterio.transform.from_bounds(
            *bounds_dst, width=out_shape[1], height=out_shape[0]
        )
        window_data = rasterio.windows.Window(0, 0, width=out_shape[1], height=out_shape[0])
    else:
        if resolution_dst_crs is not None:
            if isinstance(resolution_dst_crs, numbers.Number):
                resolution_dst_crs = (abs(resolution_dst_crs), abs(resolution_dst_crs))
        dst_transform, window_data = calculate_transform_window(data, dst_crs, resolution_dst_crs)

    gt = await read_reproject(
        data, dst_crs=dst_crs, dst_transform=dst_transform, window_out=window_data
    )
    return gt  # type: ignore[return-value]
