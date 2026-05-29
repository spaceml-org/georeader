"""
Async COG reader: thin adapter over ``developmentseed/async-geotiff``.

This module provides :class:`AsyncGeoTIFFReader`, an ``async``-native reader
for Cloud-Optimized GeoTIFFs (COGs). It is a thin adapter on top of
`async-geotiff <https://github.com/developmentseed/async-geotiff>`_ that
provides the :class:`~georeader.abstract_reader.AsyncGeoData` protocol,
the lazy windowed-view pattern that mirrors
:class:`~georeader.rasterio_reader.RasterioReader`, and translation
between georeader's ``GeoTensor`` / ``rasterio.windows.Window`` carriers
and async-geotiff's ``RasterArray`` / ``Window`` types. The actual IFD
walk, tile-fetch math, decompression, and request coalescing all live
upstream. Use it for high-concurrency fan-out workloads (tile servers,
async ML inference) where many reads happen concurrently from one
process.

Sync vs Async
-------------

::

    ┌──────────────────────────────────────────────────────────────────────┐
    │                  RASTERIOREADER vs ASYNCGEOTIFFREADER                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │                                                                       │
    │  RasterioReader (sync, GDAL)        AsyncGeoTIFFReader (async, COG)  │
    │  ──────────────────────────         ────────────────────────────     │
    │                                                                       │
    │  • Sync reads via rasterio          • Async reads via async-geotiff  │
    │  • GDAL VSI / fsspec / opener=      • obspec.AsyncStore transport    │
    │  • Every GDAL driver                • TIFF / COG only                │
    │  • WarpedVRT reprojection           • No warp / no reproject         │
    │  • Fresh open() per read            • Persistent GeoTIFF handle      │
    │                                                                       │
    │  Use for:                           Use for:                          │
    │  • Notebooks, batch scripts         • Tile servers fanning out 100s  │
    │  • Single scenes                    • Async ML inference services    │
    │  • JP2/NetCDF/HDF5/GRIB            • COG-heavy cloud workflows       │
    └──────────────────────────────────────────────────────────────────────┘

Construction
------------

``__init__`` is intentionally cheap — it does not fetch the COG header. Most
users call the async ``open()`` classmethod which performs the IFD fetch::

    from obstore.store import S3Store
    from georeader.async_geotiff_reader import AsyncGeoTIFFReader

    store = S3Store(bucket="my-bucket", region="us-east-1")
    reader = await AsyncGeoTIFFReader.open("scene.tif", store=store)

    # Metadata properties are now sync and instant.
    print(reader.crs, reader.shape, reader.dtype)

    # Reads are async coroutines.
    gt = await reader.load()

Why this reader does not warp itself
------------------------------------

`async-geotiff explicitly disclaims <https://github.com/developmentseed/async-geotiff#anti-features>`_
warping, resampling, and automatic overview selection. The reader keeps that
boundary — it only implements the primitive windowed read. The
reproject/resize/tile family is provided by :mod:`georeader.asyncread`, which
streams **only the input window required for the destination grid** (it does
**not** load the entire raster first) and shares the warp loop with the sync
:mod:`georeader.read` path.

For cross-CRS reads, use :func:`georeader.asyncread.read_to_crs` /
:func:`~georeader.asyncread.read_reproject_like`, or switch to
:class:`~georeader.rasterio_reader.RasterioReader` (which has WarpedVRT
integration on the sync path).

Read by bounds / polygon / center / tile is provided by the
:mod:`georeader.asyncread` module functions, which work with any
:class:`~georeader.abstract_reader.AsyncGeoData` input (sync equivalents in
:mod:`georeader.read` for :class:`~georeader.abstract_reader.GeoData`).

See Also
--------
georeader.abstract_reader.AsyncGeoData : Protocol satisfied by this reader.
georeader.rasterio_reader.RasterioReader : Sync alternative with full GDAL.
georeader.geotensor.GeoTensor : Carrier type returned by every read.

References
----------
- async-geotiff: https://github.com/developmentseed/async-geotiff
- obstore: https://github.com/developmentseed/obstore
- obspec: https://github.com/developmentseed/obspec
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import rasterio.windows
from rasterio import Affine

from georeader import window_utils
from georeader.abstract_reader import AsyncGeoData
from georeader.geotensor import GeoTensor

try:
    from async_geotiff import GeoTIFF, RasterArray
    from async_geotiff import Window as _AGTWindow

    _ASYNC_GEOTIFF_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by environments without the extra
    _ASYNC_GEOTIFF_AVAILABLE = False
    GeoTIFF = None  # type: ignore[assignment,misc]
    RasterArray = None  # type: ignore[assignment,misc]
    _AGTWindow = None  # type: ignore[assignment,misc]


def _require_async_geotiff() -> None:
    """Raise a clear error if the optional ``async-geotiff`` extra is missing."""
    if not _ASYNC_GEOTIFF_AVAILABLE:
        raise ImportError(
            "AsyncGeoTIFFReader requires the optional 'async' extra. "
            "Install with: pip install 'georeader-spaceml[async]'"
        )


class AsyncGeoTIFFReader(AsyncGeoData):
    """Async COG reader. Thin adapter over :class:`async_geotiff.GeoTIFF`.

    Use for high-concurrency fan-out (tile servers, async ML inference
    services). For one-off sync reads, use
    :class:`~georeader.rasterio_reader.RasterioReader` instead.

    The constructor is cheap — it does not fetch the COG header. Call the
    async ``open()`` classmethod (or use the ``async with`` context manager)
    to perform the IFD fetch before reading metadata properties.

    Mirrors :class:`~georeader.rasterio_reader.RasterioReader`'s
    laziness pattern: :meth:`read_from_window` is **sync** and only
    constructs a windowed view (no I/O); :meth:`load` is async and
    performs the actual fetch. The :attr:`window_focus` attribute holds
    the current view's window in absolute pixel coordinates relative to
    the chosen ``overview_level``.

    Args:
        path_or_url: Path or URL relative to the ``store``. For local stores,
            this is the filename inside the store's ``prefix``; for cloud
            stores, the path inside the bucket.
        store: An ``obspec``-compatible async store. ``obstore.store``
            provides ``S3Store`` / ``GCSStore`` / ``AzureStore`` /
            ``LocalStore`` etc. Required — there is no default.
        overview_level: Which overview to read from. ``None`` (default)
            reads at full resolution from the primary IFD. An integer
            ``i`` reads from ``geotiff.overviews[i]`` (0-based).
        window_focus: Initial window focus in absolute pixel coordinates
            against the chosen overview level. ``None`` (default) means
            the full extent. Mostly internal — produced by
            :meth:`read_from_window`.
    """

    def __init__(
        self,
        path_or_url: str,
        *,
        store: Any,
        overview_level: Optional[int] = None,
        window_focus: Optional[rasterio.windows.Window] = None,
    ) -> None:
        _require_async_geotiff()
        self.path_or_url = path_or_url
        self._store = store
        self._overview_level = overview_level
        self._geotiff: Optional[Any] = None
        self.window_focus: Optional[rasterio.windows.Window] = window_focus

    @classmethod
    async def open(
        cls,
        path_or_url: str,
        *,
        store: Any,
        overview_level: Optional[int] = None,
    ) -> "AsyncGeoTIFFReader":
        """Async constructor — fetches and parses the COG header.

        Most users call this rather than ``__init__`` directly. Equivalent to
        ``__init__`` followed by ``await ...._open_geotiff()``.
        """
        self = cls(path_or_url, store=store, overview_level=overview_level)
        await self._open_geotiff()
        return self

    async def _open_geotiff(self) -> None:
        """Fetch the COG header (IFD chain) and cache the GeoTIFF handle."""
        _require_async_geotiff()
        self._geotiff = await GeoTIFF.open(self.path_or_url, store=self._store)

    # ----------------------------------------------------------------- internals
    def _require_open(self) -> Any:
        if self._geotiff is None:
            raise RuntimeError(
                "AsyncGeoTIFFReader not opened — "
                "call `await AsyncGeoTIFFReader.open(...)` or use `async with`."
            )
        return self._geotiff

    @property
    def _level(self) -> Any:
        """The async-geotiff object to read from.

        When ``overview_level is None`` this is the full-resolution
        :class:`async_geotiff.GeoTIFF`; otherwise it's
        ``geotiff.overviews[overview_level]``. Both expose ``transform``,
        ``shape``, ``read(window=...)``, etc.
        """
        gt = self._require_open()
        if self._overview_level is None:
            return gt
        return gt.overviews[self._overview_level]

    @property
    def _raster_window(self) -> rasterio.windows.Window:
        """Full-extent window at the current overview level.

        Counterpart to :attr:`RasterioReader.real_window`. Computed
        per-access rather than cached because the level resolves
        lazily (the IFD handle is shared across overview-pinned views).
        """
        level = self._level
        return rasterio.windows.Window(
            col_off=0, row_off=0, width=level.width, height=level.height,
        )

    # ------------------------------------------------------------------ metadata
    @property
    def crs(self) -> Any:
        return self._require_open().crs

    @property
    def transform(self) -> Affine:
        """Affine transform of the current view.

        When :attr:`window_focus` is set, returns the transform shifted
        to the focus window's origin (matches
        :class:`~georeader.rasterio_reader.RasterioReader`).
        """
        level_transform = self._level.transform
        if self.window_focus is None:
            return level_transform
        return rasterio.windows.transform(self.window_focus, level_transform)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Returns ``(count, height, width)``.

        Reflects the focused window size when :attr:`window_focus` is set.
        Note ``async_geotiff.GeoTIFF.shape`` is just ``(height, width)``;
        the band count lives on ``.count``.
        """
        bands = self._require_open().count
        if self.window_focus is None:
            level = self._level
            return (bands, level.height, level.width)
        return (bands, int(self.window_focus.height), int(self.window_focus.width))

    @property
    def dtype(self) -> Any:
        # async-geotiff returns np.dtype | None directly — no need to re-wrap.
        return self._require_open().dtype

    @property
    def fill_value_default(self) -> Any:
        """Default fill value for out-of-bounds reads.

        Returns the COG's nodata value when set, else ``0`` — matches
        :class:`~georeader.rasterio_reader.RasterioReader`'s default
        (``nodata if not None else 0``) so the read module's
        no-intersection / boundless padding branches behave
        consistently across sync and async readers.
        """
        nodata = self._require_open().nodata
        return nodata if nodata is not None else 0

    @property
    def dims(self) -> list[str]:
        return ["band", "y", "x"]

    # --------------------------------------------------------------------- reads
    def read_from_window(
        self,
        window: rasterio.windows.Window,
        boundless: bool = True,
    ) -> "AsyncGeoTIFFReader":
        """Return a new windowed view of this reader. **Sync, no I/O.**

        Mirrors :meth:`RasterioReader.read_from_window`: the returned
        reader shares the underlying ``async-geotiff`` handle and only
        carries a different :attr:`window_focus`. Call ``await
        view.load()`` to materialise.

        ``window`` is interpreted relative to the current
        :attr:`window_focus` (matches
        :meth:`RasterioReader.set_window` with ``relative=True``).
        ``boundless=False`` intersects the resolved window with the
        underlying raster extent and raises :class:`WindowError` if the
        intersection is empty; ``boundless=True`` permits the focus
        window to extend past the raster (padding is applied later in
        :meth:`load`).
        """
        self._require_open()
        # Translate the requested window from view-local to absolute coords.
        if self.window_focus is None:
            abs_window = window
        else:
            abs_window = rasterio.windows.Window(
                col_off=window.col_off + self.window_focus.col_off,
                row_off=window.row_off + self.window_focus.row_off,
                width=window.width,
                height=window.height,
            )

        if not boundless:
            # Raises WindowError if the windows are disjoint — same
            # contract as RasterioReader.set_window(..., boundless=False).
            abs_window = rasterio.windows.intersection(abs_window, self._raster_window)

        view = AsyncGeoTIFFReader(
            self.path_or_url,
            store=self._store,
            overview_level=self._overview_level,
            window_focus=abs_window,
        )
        view._geotiff = self._geotiff  # share the cached IFD handle
        return view

    async def load(self, boundless: bool = True) -> GeoTensor:
        """Materialise the current view as a :class:`GeoTensor`.

        Reads from :attr:`window_focus` (full extent when ``None``).
        Follows the library padding pattern from
        :meth:`GeoTensor.read_from_window`:
        :func:`window_utils.get_slice_pad` -> async fetch -> ``.pad()``.

        ``boundless=True`` pads up to the focused window's shape with
        :attr:`fill_value_default` (or ``0`` when the COG has no
        nodata); ``boundless=False`` returns the clipped intersection.

        Raises:
            rasterio.windows.WindowError: If the focused window does not
                intersect the raster at all (regardless of ``boundless``).
        """
        raster_window = self._raster_window
        target_window = self.window_focus if self.window_focus is not None else raster_window

        if boundless:
            slice_dict, pad_width = window_utils.get_slice_pad(raster_window, target_window)
            inner_window = rasterio.windows.Window.from_slices(
                slice_dict["y"], slice_dict["x"],
            )
            inner_gt = await self._fetch_window(inner_window)
            if any(p != 0 for p in pad_width["x"] + pad_width["y"]):
                # Fall back to 0 when the COG has no explicit nodata —
                # matches rasterio's C-level boundless padding default.
                fill = self.fill_value_default if self.fill_value_default is not None else 0
                inner_gt = inner_gt.pad(
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=fill,
                )
            return inner_gt

        # boundless=False: clip to the intersection. WindowError on disjoint.
        inner_window = rasterio.windows.intersection(target_window, raster_window)
        return await self._fetch_window(inner_window)

    async def _fetch_window(self, window: rasterio.windows.Window) -> GeoTensor:
        """Read a fully-in-bounds window directly from async-geotiff."""
        agt_window = _AGTWindow(
            col_off=int(window.col_off),
            row_off=int(window.row_off),
            width=int(window.width),
            height=int(window.height),
        )
        arr: Any = await self._level.read(window=agt_window)
        return _rasterarray_to_geotensor(
            arr, fill_value=self.fill_value_default, crs=self.crs,
        )

    # ------------------------------------------------------ overviews & blocks
    def overviews(self) -> list[int]:
        """Return decimation factors for available overviews.

        Mirrors :meth:`RasterioReader.overviews` — e.g. ``[2, 4, 8]``
        means overviews exist at 1/2, 1/4, 1/8 of the primary IFD's
        resolution. Empty list if the COG has no overviews.

        Factors are computed from the full-resolution width divided by
        each overview's width (rounded), since ``async-geotiff`` exposes
        the overview IFDs directly rather than a factor list.
        """
        gt = self._require_open()
        full_width = gt.width
        return [int(round(full_width / ov.width)) for ov in gt.overviews]

    def reader_overview(self, overview_level: int) -> "AsyncGeoTIFFReader":
        """Return a new reader pinned to a specific overview level.

        Mirrors :meth:`RasterioReader.reader_overview`. The new reader
        shares the underlying ``async-geotiff`` handle (no re-open) and
        resets :attr:`window_focus` — converting a focus across
        resolutions is non-trivial and matches the parent's TODO.

        Args:
            overview_level: Overview index. ``0`` is the first overview
                (finest after full res); negative indexes count from the
                end (``-1`` is the coarsest overview, not full res — same
                convention as :meth:`RasterioReader.reader_overview`).
        """
        gt = self._require_open()
        if overview_level < 0:
            overview_level = len(gt.overviews) + overview_level

        view = AsyncGeoTIFFReader(
            self.path_or_url,
            store=self._store,
            overview_level=overview_level,
        )
        view._geotiff = self._geotiff

        if self.window_focus is not None:
            import warnings
            warnings.warn(
                "window_focus is not preserved across overview levels — "
                "returning the overview at full extent.",
                stacklevel=2,
            )
        return view

    def block_windows(
        self, bidx: int = 1,
    ) -> list[tuple[tuple[int, int], rasterio.windows.Window]]:
        """Return the internal COG tile windows at the current overview level.

        Mirrors :meth:`RasterioReader.block_windows`. The returned
        windows are tile-aligned in pixel coordinates relative to the
        current overview level; the last row/column may be smaller than
        ``(tile_height, tile_width)`` if the raster size isn't an exact
        multiple. When :attr:`window_focus` is set, only blocks that
        intersect the focus are returned, and each is clipped to the
        focus extent (matches the parent's behavior).

        Critical for tile-aligned async fan-out — reading non-aligned
        windows triggers partial-tile fetches inside ``async-geotiff``
        and wastes bytes over the wire.

        Args:
            bidx: Band index. Accepted for API parity with
                :meth:`RasterioReader.block_windows`; ignored because
                COG tiling is uniform across bands.
        """
        del bidx  # COG tile grid is uniform across bands
        level = self._level
        tile_w = level.tile_width
        tile_h = level.tile_height
        out: list[tuple[tuple[int, int], rasterio.windows.Window]] = []

        for row_idx, row_off in enumerate(range(0, level.height, tile_h)):
            for col_idx, col_off in enumerate(range(0, level.width, tile_w)):
                width = min(tile_w, level.width - col_off)
                height = min(tile_h, level.height - row_off)
                block = rasterio.windows.Window(
                    col_off=col_off, row_off=row_off, width=width, height=height,
                )
                if self.window_focus is not None:
                    if not rasterio.windows.intersect([self.window_focus, block]):
                        continue
                    block = rasterio.windows.intersection(self.window_focus, block)
                out.append(((row_idx, col_idx), block))

        return out

    # -------------------------------------------------------------- lifecycle
    async def aclose(self) -> None:
        """No-op — obstore pools its own connections; async-geotiff has no resource to release."""
        return None

    async def __aenter__(self) -> "AsyncGeoTIFFReader":
        if self._geotiff is None:
            await self._open_geotiff()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        if self._geotiff is None:
            return (
                f"AsyncGeoTIFFReader(path_or_url={self.path_or_url!r}, "
                f"overview_level={self._overview_level!r}, unopened)"
            )
        # Layout mirrors RasterioReader.__repr__ for consistency.
        transform_indent = "\n" + " " * 29
        transform_str = transform_indent.join(str(self.transform).splitlines())
        return (
            "\n"
            f"         path_or_url:        {self.path_or_url}\n"
            f"         overview_level:     {self._overview_level}\n"
            f"         Shape:              {self.shape}\n"
            f"         Resolution:         {self.res}\n"
            f"         Bounds:             {self.bounds}\n"
            f"         CRS:                {self.crs}\n"
            f"         fill_value_default: {self.fill_value_default}\n"
            f"         Transform:          {transform_str}\n"
        )


def _rasterarray_to_geotensor(
    arr: Any,
    *,
    fill_value: Any,
    crs: Any,
) -> GeoTensor:
    """Translate :class:`async_geotiff.RasterArray` → :class:`GeoTensor`.

    ``RasterArray.data`` is ``(bands, height, width)``. When ``.mask`` is
    present (``True`` means valid in async-geotiff's convention) and a
    ``fill_value`` is available, we substitute the fill where the mask is
    ``False``. The result carries the same ``transform`` as the source plus
    the reader's CRS (``RasterArray.crs`` reaches through to the parent
    GeoTIFF which we already have).
    """
    data: np.ndarray = arr.data
    if arr.mask is not None and fill_value is not None:
        invalid = np.broadcast_to(~arr.mask, data.shape)
        data = np.where(invalid, fill_value, data)
    return GeoTensor(
        values=data,
        transform=arr.transform,
        crs=crs,
        fill_value_default=fill_value,
    )
