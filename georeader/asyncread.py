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
    """Read a pixel window from an async reader.

    Mirrors :func:`georeader.read.read_from_window` for
    :class:`AsyncGeoData` inputs. The async path has two branches:

    1. **No-intersection branch (pure CPU, no ``await``):** if the requested
       window does not overlap the data extent, the function returns a
       synthetic padded array / :class:`GeoTensor` (when ``boundless=True``)
       or ``None`` (when ``boundless=False``) directly. No I/O happens.
    2. **Intersecting branch:** constructs the windowed view via the
       reader's sync ``read_from_window`` (no I/O — just a view object),
       then ``await``s ``view.load()`` only when materialisation is
       requested. This is the single I/O hop.

    Args:
        data_in: Async reader (e.g. :class:`AsyncGeoTIFFReader`).
        window: Pixel window to read (``col_off``, ``row_off``, ``width``,
            ``height``) in the reader's coordinates.
        return_only_data: If True, materialise and return only the numpy
            array (drops the :class:`GeoTensor` wrapper). Forces an
            ``await``.
        trigger_load: If True, materialise and return a :class:`GeoTensor`.
            Forces an ``await``. Ignored when ``return_only_data=True``.
        boundless: If True, windows extending past the raster are padded
            with ``data_in.fill_value_default``; if False, returns ``None``
            on disjoint windows.

    Returns:
        - ``np.ndarray`` when ``return_only_data=True``;
        - :class:`GeoTensor` when ``trigger_load=True``;
        - the lazy windowed view (an :class:`AsyncGeoData` instance) when
          neither flag is set — call ``await view.load()`` yourself to
          materialise at your own cadence;
        - ``None`` when ``boundless=False`` and the window misses the data.

    Example:
        >>> import rasterio.windows
        >>> from georeader import asyncread
        >>>
        >>> window = rasterio.windows.Window(
        ...     col_off=10, row_off=10, width=64, height=64
        ... )
        >>> # Materialise immediately
        >>> gt = await asyncread.read_from_window(reader, window, trigger_load=True)
        >>> print(gt.shape)  # (bands, 64, 64)
        >>>
        >>> # Or compose more before materialising — useful for fan-out
        >>> view = await asyncread.read_from_window(reader, window)
        >>> gt = await view.load()

    See Also:
        :func:`georeader.read.read_from_window`: sync counterpart with the
        full parameter treatment.
    """
    # No-intersection branch: handled entirely in CPU via shared helpers.
    # No await — we never reach the reader's I/O.
    if not _window_intersects_data(data_in, window):
        return _build_no_intersect_result(data_in, window, boundless, return_only_data)

    # `AsyncGeoData.read_from_window` is **sync** by contract — it returns
    # a windowed VIEW (another AsyncGeoData), not a GeoTensor. No I/O has
    # happened yet; the bytes only travel when someone calls `view.load()`.
    view = data_in.read_from_window(window=window, boundless=boundless)

    # The view-vs-GeoTensor dispatch below covers three caller intents:
    #
    #   1. Some hypothetical AsyncGeoData implementation might choose to
    #      return a GeoTensor directly (e.g. an eager in-memory shim). The
    #      isinstance check lets it pass through without a redundant
    #      `await view.load()` — that would try to await on a non-awaitable.
    #   2. Materialise NOW if the caller asked for it via `trigger_load` or
    #      `return_only_data`. This is the single I/O hop in the async path.
    #   3. Otherwise return the lazy view so the caller can fan out / await
    #      at their own cadence (the recommended pattern for
    #      `asyncio.gather`-style concurrency).
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
    """Read a geographic bounding box from an async reader.

    Mirrors :func:`georeader.read.read_from_bounds`. The function converts
    geographic ``bounds`` into a pixel window (in the reader's CRS),
    optionally pads it, rounds it outward, and delegates to
    :func:`read_from_window`. The single ``await`` is the windowed
    materialisation inside ``read_from_window``.

    Args:
        data_in: Async reader.
        bounds: ``(xmin, ymin, xmax, ymax)``. Interpreted in ``crs_bounds``
            (defaults to the reader's CRS).
        crs_bounds: Optional CRS for ``bounds`` (e.g. ``"EPSG:4326"``);
            only the bounds are reprojected — the data stays in the
            reader's CRS.
        pad_add: ``(pad_rows, pad_cols)`` to expand the window before
            reading (useful for interpolation edge context).
        return_only_data: See :func:`read_from_window`.
        trigger_load: See :func:`read_from_window`.
        boundless: See :func:`read_from_window`.

    Returns:
        Same return contract as :func:`read_from_window`.

    Example:
        >>> # Bounds in WGS84 against a UTM raster — only the bounds get
        >>> # reprojected; the returned chip is still in the reader's CRS.
        >>> gt = await asyncread.read_from_bounds(
        ...     reader,
        ...     bounds=(3.00, 41.55, 3.01, 41.56),
        ...     crs_bounds="EPSG:4326",
        ...     trigger_load=True,
        ... )

    See Also:
        :func:`georeader.read.read_from_bounds`: sync counterpart.
        :func:`read_to_crs`: when you also want to warp the *data* (not just
        the bounds) to a different CRS.
    """
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
    """Read the bounding rectangle of a polygon from an async reader.

    Mirrors :func:`georeader.read.read_from_polygon`. Reads the **minimum
    bounding rectangle** containing ``polygon`` — not the polygon shape
    itself. For actual polygon masking, combine the result with
    :func:`rasterio.features.geometry_mask` post-read.

    Useful for irregular AOIs (admin boundaries, watersheds, fields) where
    a rectangular bounds call would over-fetch. ``MultiPolygon`` returns a
    single rectangle containing all parts.

    Args:
        data_in: Async reader.
        polygon: Shapely ``Polygon`` or ``MultiPolygon`` in ``crs_polygon``.
        crs_polygon: Optional CRS for ``polygon`` (e.g. ``"EPSG:4326"``).
        pad_add: ``(pad_rows, pad_cols)`` to expand the window before
            reading.
        return_only_data: See :func:`read_from_window`.
        trigger_load: See :func:`read_from_window`.
        boundless: See :func:`read_from_window`.
        window_surrounding: If True, adds a 1-pixel buffer around the
            polygon's bbox to guarantee complete coverage when polygon
            vertices align exactly with pixel boundaries.

    Returns:
        Same return contract as :func:`read_from_window`.

    Example:
        >>> from shapely.geometry import box
        >>> aoi = box(500200, 4599400, 500800, 4600000)  # UTM 31N
        >>> gt = await asyncread.read_from_polygon(reader, aoi, trigger_load=True)

    See Also:
        :func:`georeader.read.read_from_polygon`: sync counterpart.
    """
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
    """Read a fixed-shape chip centred on a geographic coordinate.

    Mirrors :func:`georeader.read.read_from_center_coords`. Useful for ML
    training-chip extraction around points of interest — the
    ``center_coords`` map to pixel ``(height/2, width/2)`` in the output.

    Args:
        data_in: Async reader.
        center_coords: ``(x, y)`` in ``crs_center_coords`` (defaults to the
            reader's CRS). For WGS84, this is ``(lon, lat)``.
        shape: Output chip size as ``(height, width)`` in pixels.
        crs_center_coords: Optional CRS for ``center_coords``.
        return_only_data: See :func:`read_from_window`.
        trigger_load: See :func:`read_from_window`.
        boundless: See :func:`read_from_window`. When ``True``, chips that
            partially fall outside the raster are padded to maintain
            ``shape`` exactly — handy for fixed-size ML inputs.

    Returns:
        Same return contract as :func:`read_from_window`.

    Example:
        >>> # 256x256 chip centred on Madrid, sampled in WGS84
        >>> gt = await asyncread.read_from_center_coords(
        ...     reader,
        ...     center_coords=(-3.7038, 40.4168),
        ...     shape=(256, 256),
        ...     crs_center_coords="EPSG:4326",
        ...     trigger_load=True,
        ... )

    See Also:
        :func:`georeader.read.read_from_center_coords`: sync counterpart.
    """
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
    """Reproject from an async reader to a target CRS / resolution / extent.

    Mirrors :func:`georeader.read.read_reproject`. Streams **only the input
    window required for the destination grid** — does not pre-load the
    entire raster. The non-I/O code paths (destination grid setup,
    intersection check, dtype handling, per-band warp loop) are shared with
    the sync path via :func:`georeader.read._reproject_setup` and
    :func:`georeader.read._reproject_finalize`, so the warp result is
    bit-identical to ``read.read_reproject`` on the same input.

    The orchestration has three branches:

    1. **No-op fast path** — when source and destination CRS match and the
       grids align, the function skips the warp and does a plain
       ``await read_from_window(...)`` on the source-aligned window. This
       is ~10–100× faster for aligned data.
    2. **Non-intersecting** — when the destination extent does not overlap
       the source, returns the pre-allocated nodata-filled
       :class:`GeoTensor` directly. No I/O.
    3. **Normal path** — windowed read of the input region
       (``await read_from_polygon(..., trigger_load=True)`` with a 3-pixel
       buffer for interpolation edges) followed by the in-process warp.

    Args:
        data_in: Async reader (or pre-loaded :class:`GeoTensor`).
        dst_crs: Destination CRS. ``None`` reuses ``data_in.crs`` (useful
            for resolution-only changes).
        bounds: ``(xmin, ymin, xmax, ymax)`` in ``dst_crs``. Mutually
            exclusive with ``window_out`` — provide one.
        resolution_dst_crs: Target resolution in ``dst_crs`` units. ``float``
            applies to both axes; tuple is ``(res_x, res_y)``.
        dst_transform: Pre-computed output transform (useful for aligning
            to an existing grid). When set, ``bounds`` and ``window_out``
            describe the same grid the transform encodes.
        window_out: Pre-computed output window. Defines output dimensions.
        resampling: Resampling algorithm from :class:`rasterio.warp.Resampling`.
            Default is ``cubic_spline`` (smooth, good for continuous data);
            use ``nearest`` for categorical data.
        dtype_dst: Output dtype. ``None`` matches ``data_in.dtype``.
        return_only_data: If True, return only the numpy array.
        dst_nodata: Fill value for out-of-bounds destination pixels.
            ``None`` uses ``data_in.fill_value_default``.

    Returns:
        :class:`GeoTensor` (or ``np.ndarray`` when ``return_only_data=True``)
        on the destination grid.

    Example:
        >>> # Reproject onto an EPSG:4326 grid at 0.0001° resolution
        >>> gt = await asyncread.read_reproject(
        ...     reader,
        ...     dst_crs="EPSG:4326",
        ...     bounds=(3.00, 41.55, 3.01, 41.56),
        ...     resolution_dst_crs=0.0001,
        ... )

    See Also:
        :func:`georeader.read.read_reproject`: sync counterpart with the
        full parameter and example treatment.
        :func:`read_to_crs`, :func:`read_reproject_like`: thin wrappers
        for the common "just change CRS" and "match a template grid" cases.
    """
    # The three-branch structure mirrors `read.read_reproject` exactly —
    # only the I/O step (Branch 3) uses `await` here. See `_ReprojectPlan`
    # for the branch semantics.
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

    # Branch 1 — Fast path. Aligned grids: warp-free windowed read.
    if plan.fast_path_window is not None:
        gt = await read_from_window(
            data_in,
            plan.fast_path_window,
            return_only_data=return_only_data,
            trigger_load=True,
        )
        return gt  # type: ignore[return-value]

    # Branch 2 — Non-intersecting. Return the nodata-filled destination
    # without any I/O.
    if plan.nonintersecting:
        return GeoTensor(
            plan.destination,
            transform=plan.dst_transform,
            crs=plan.dst_crs,
            fill_value_default=plan.dst_nodata,
        )

    # Branch 3 — Normal path. One await against the reader for ONLY the
    # input region required by the destination grid (3-pixel pad for edge
    # context). When `data_in` is already a GeoTensor the data is in memory
    # so no fetch happens — `read_reproject` accepting a GeoTensor is the
    # "warp this array" entry point and we honour it here too.
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

    # Warp is in-process CPU — no `await`, by design. See module docstring.
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
    """Reproject from an async reader onto another grid's CRS + transform + shape.

    Mirrors :func:`georeader.read.read_reproject_like`. The canonical use
    case is **stack alignment**: read several rasters and force them onto
    the exact pixel grid of a reference :class:`GeoTensor` so they can be
    concatenated, differenced, or fed to a CNN as channels.

    Args:
        data_in: Async reader to reproject.
        data_like: Reference object with ``crs``, ``transform``, ``shape``,
            and ``res`` — typically another :class:`GeoTensor`.
        resolution_dst: Override ``data_like``'s resolution while still
            using its CRS and transform (rare; useful for pyramid building).
        resampling: See :func:`read_reproject`.
        dtype_dst: See :func:`read_reproject`.
        return_only_data: See :func:`read_reproject`.
        dst_nodata: See :func:`read_reproject`.

    Returns:
        :class:`GeoTensor` matching ``data_like``'s grid exactly
        (``crs``, ``transform``, spatial shape).

    Example:
        >>> # Align an async-fetched chip to the same grid as an existing chip
        >>> aligned = await asyncread.read_reproject_like(reader, reference_geotensor)
        >>> assert aligned.crs == reference_geotensor.crs
        >>> assert aligned.shape[-2:] == reference_geotensor.shape[-2:]

    See Also:
        :func:`georeader.read.read_reproject_like`: sync counterpart.
    """
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
    """Reproject an async reader to a different CRS.

    Mirrors :func:`georeader.read.read_to_crs`. Convenience wrapper around
    :func:`read_reproject` for the common case of "just change the CRS" —
    the destination transform and window are computed via
    :func:`~georeader.read.calculate_transform_window` from the source's
    bounds at the requested ``resolution_dst_crs``.

    Args:
        data_in: Async reader.
        dst_crs: Destination CRS (e.g. ``"EPSG:4326"``, ``"EPSG:3857"``).
        resampling: See :func:`read_reproject`.
        resolution_dst_crs: Target resolution in ``dst_crs`` units. ``None``
            picks a resolution that matches the source's pixel size.
        return_only_data: See :func:`read_reproject`.

    Returns:
        :class:`GeoTensor` in ``dst_crs``. When ``data_in.crs == dst_crs``
        the function short-circuits and returns ``data_in`` unchanged (no
        I/O, no warp).

    Example:
        >>> # UTM 31N → Web Mercator
        >>> gt_3857 = await asyncread.read_to_crs(reader, dst_crs="EPSG:3857")
        >>> print(gt_3857.crs)  # 'EPSG:3857'

    See Also:
        :func:`georeader.read.read_to_crs`: sync counterpart.
        :func:`read_reproject_like`: when you have a template grid to match.
    """
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
    """Resample an async reader to a different resolution (same CRS).

    Mirrors :func:`georeader.read.resize`. Useful for downsampling
    high-res imagery, upsampling coarse data, matching resolution across
    multi-source datasets, or building image pyramids.

    For downsampling (``resolution_dst > resolution_src``), the function
    applies a Gaussian anti-aliasing filter by default to prevent
    aliasing artifacts (moiré patterns, jagged edges). For upsampling,
    anti-aliasing is a no-op.

    Args:
        data_in: Async reader (or pre-loaded :class:`GeoTensor`).
        resolution_dst: Target resolution in the source CRS units. ``float``
            applies to both axes; tuple is ``(res_y, res_x)``.
        window_out: Pre-computed output window. ``None`` computes from the
            scale factor (rounded up to ensure complete coverage).
        anti_aliasing: Apply a Gaussian filter before downsampling.
            Strongly recommended for downsampling continuous data.
        anti_aliasing_sigma: Custom Gaussian σ. ``None`` uses ``(scale-1)/2``.
        resampling: Resampling algorithm.
        return_only_data: See :func:`read_reproject`.

    Returns:
        :class:`GeoTensor` at the requested resolution, same CRS as input.

    .. warning::

       When ``anti_aliasing=True`` and the input is a lazy async reader
       being *downsampled*, the reader's **full extent is loaded eagerly**
       (awaited) before the Gaussian filter runs — the filter needs the
       pixels in memory. To stream only a windowed region, pre-load that
       region with :func:`read_from_bounds` (or similar) and pass the
       resulting :class:`GeoTensor` here. Or pass ``anti_aliasing=False``
       to skip the filter entirely.

    Example:
        >>> # Sentinel-2 native 10m → 30m (Landsat-like resolution)
        >>> gt_30m = await asyncread.resize(reader, resolution_dst=30.0)
        >>> print(gt_30m.res)  # (30.0, 30.0)

    See Also:
        :func:`georeader.read.resize`: sync counterpart with the full
        parameter and example treatment.
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
        # `apply_anti_aliasing` is sync: for lazy inputs that need the
        # Gaussian filter (i.e. an actual downsample) it materialises them
        # via `data_in.load()`, which on an AsyncGeoData returns an
        # un-awaited coroutine. Await the full-extent load here first so
        # the shared helper only ever sees in-memory data. Upsampling is
        # a no-op for the filter, so the reader stays lazy in that case.
        downsampling = any(r_or < r_dst for r_or, r_dst in zip(resolution_or, resolution_dst))
        if downsampling and not isinstance(data_in, GeoTensor):
            src = await data_in.load()
        src = apply_anti_aliasing(
            src, anti_aliasing_sigma=anti_aliasing_sigma, resolution_dst=resolution_dst
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
    """Read an XYZ web-map tile from an async reader.

    Mirrors :func:`georeader.read.read_from_tile`. The primary use case is
    serving tiles for Leaflet / OpenLayers / Mapbox / etc. from arbitrary
    COG data: the function fetches **only the bytes for the requested
    tile region** from the source raster, then reprojects that small chip
    into Web Mercator (the standard web-map CRS).

    Tile coordinates follow the OSM Slippy Map / TMS convention:

    - The world is split into ``2**z`` × ``2**z`` tiles at zoom ``z``.
    - ``(x, y) = (0, 0)`` is the top-left (north-west).
    - ``x`` increases eastward; ``y`` increases southward.

    Algorithm:

    1. Convert ``(x, y, z)`` to geographic bounds in Web Mercator via
       :func:`mercantile.xy_bounds`.
    2. Early-return ``None`` if the tile does not intersect the data
       footprint (matches typical tile-server "empty tile" behaviour).
    3. If the reader exposes its own ``read_from_tile`` (sync or async),
       delegate to it for the optimised native-format path.
    4. Otherwise, fall through to :func:`read_from_polygon` (when the tile
       happens to be in the data's CRS and no resize is requested) or
       :func:`read_reproject` (the general case — windowed read of the
       tile region, then warp into the destination CRS at the requested
       output shape).

    Args:
        data: Async reader.
        x: Tile column index (``0`` to ``2**z - 1``).
        y: Tile row index (``0`` to ``2**z - 1``).
        z: Zoom level (typically 0–22).
        dst_crs: Output CRS. Defaults to Web Mercator (``EPSG:3857``), the
            standard for web maps. Set to ``None`` to use the data's
            native CRS.
        out_shape: Output tile dimensions as ``(height, width)``. Defaults
            to ``(256, 256)`` — the de-facto web-tile standard.
        resolution_dst_crs: Override the output resolution explicitly.
            Ignored when ``out_shape`` is provided.
        assert_if_not_intersects: Raise ``AssertionError`` instead of
            returning ``None`` for non-intersecting tiles.

    Returns:
        :class:`GeoTensor` of shape ``(bands, out_shape[0], out_shape[1])``,
        georeferenced to ``dst_crs`` at the tile's geographic location, or
        ``None`` when the tile misses the data and ``assert_if_not_intersects``
        is ``False``.

    Example:
        >>> import mercantile
        >>>
        >>> # Pick a tile at zoom 14 covering a lon/lat point
        >>> t = mercantile.tile(3.0029, 41.5502, 14)
        >>> tile_gt = await asyncread.read_from_tile(reader, x=t.x, y=t.y, z=t.z)
        >>> if tile_gt is not None:
        ...     print(tile_gt.shape)  # (bands, 256, 256)

    See Also:
        :func:`georeader.read.read_from_tile`: sync counterpart.
    """
    bounds_wgs = mercantile.xy_bounds(int(x), int(y), int(z))
    polygon_crs_webmercator = box(
        bounds_wgs.left, bounds_wgs.bottom, bounds_wgs.right, bounds_wgs.top
    )

    intersects = polygon_crs_webmercator.intersects(data.footprint(crs=WEB_MERCATOR_CRS))

    if not intersects:
        assert not assert_if_not_intersects, "Tile does not intersect data"
        return None

    # Optional fast-path delegation.
    #
    # `AsyncGeoData` does NOT require `read_from_tile` on the protocol —
    # it's an opt-in optimisation. A reader can expose one if the
    # underlying format (or a server-side tiling API) makes per-tile
    # reads cheaper than going through the generic windowed-read +
    # warp path below.
    #
    # We dispatch dynamically because:
    #   - sync overrides exist (e.g. RasterioReader.read_from_tile uses
    #     a WarpedVRT — sync, but still cheap). `inspect.iscoroutinefunction`
    #     keeps us from `await`-ing a non-coroutine and crashing.
    #   - async overrides will be the common case for future readers that
    #     wrap a server-side tile endpoint (XYZ source, COG-tile API).
    #
    # `out_shape is None` means "don't resize", which the generic path
    # below handles specifically — skip delegation in that case so we
    # don't have to plumb the no-resize semantics through every
    # custom override.
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

        # Destination grid over the TILE's extent (mirrors the sync
        # `read.read_from_tile`) — not the whole raster's, which is what
        # `calculate_transform_window` would compute.
        polygon_crs_data = window_utils.polygon_to_crs(
            polygon_crs_webmercator, WEB_MERCATOR_CRS, data.crs
        )
        bounds_crs_data = polygon_crs_data.bounds

        in_height, in_width = data.shape[-2:]
        dst_transform, width, height = rasterio.warp.calculate_default_transform(
            data.crs, dst_crs, in_width, in_height, *bounds_crs_data, resolution=resolution_dst_crs
        )
        window_data = rasterio.windows.Window(0, 0, width=width, height=height)

    gt = await read_reproject(
        data, dst_crs=dst_crs, dst_transform=dst_transform, window_out=window_data
    )
    return gt  # type: ignore[return-value]
