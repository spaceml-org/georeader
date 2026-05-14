"""
Async COG reader: thin adapter over ``developmentseed/async-geotiff``.

This module provides :class:`AsyncGeoTIFFReader`, an ``async``-native reader
for Cloud-Optimized GeoTIFFs (COGs). It is a thin (~80-LOC) adapter on top of
`async-geotiff <https://github.com/developmentseed/async-geotiff>`_ that
exposes the same metadata surface as :class:`~georeader.rasterio_reader.RasterioReader`
and conforms to :class:`~georeader.abstract_reader.AsyncGeoData`. Use it for
high-concurrency fan-out workloads (tile servers, async ML inference) where
many reads happen concurrently from one process.

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

Why this reader does not warp
-----------------------------

`async-geotiff explicitly disclaims <https://github.com/developmentseed/async-geotiff#anti-features>`_
warping, resampling, and automatic overview selection. This reader follows
suit: ``read_bounds(target_crs=...)`` raises :class:`NotImplementedError`. For
cross-CRS reads, either fetch in the native CRS and post-warp via
:func:`georeader.read.read_reproject_like`, or use
:class:`~georeader.rasterio_reader.RasterioReader` (which has WarpedVRT
integration on the sync path).

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

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np
import rasterio.windows

from georeader import window_utils
from georeader.abstract_reader import AsyncGeoData
from georeader.geotensor import GeoTensor

if TYPE_CHECKING:
    from rasterio import Affine

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
    """

    def __init__(
        self,
        path_or_url: str,
        *,
        store: Any,
        overview_level: Optional[int] = None,
    ) -> None:
        _require_async_geotiff()
        self.path_or_url = path_or_url
        self._store = store
        self._overview_level = overview_level
        self._geotiff: Optional[Any] = None

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

    # ------------------------------------------------------------------ metadata
    @property
    def crs(self) -> Any:
        return self._require_open().crs

    @property
    def transform(self) -> "Affine":
        return self._level.transform

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Returns ``(count, height, width)``.

        Note ``async_geotiff.GeoTIFF.shape`` is just ``(height, width)``;
        the band count lives on ``.count``.
        """
        level = self._level
        return (self._require_open().count, level.height, level.width)

    @property
    def dtype(self) -> Any:
        # async-geotiff returns np.dtype | None directly — no need to re-wrap.
        return self._require_open().dtype

    @property
    def fill_value_default(self) -> Any:
        return self._require_open().nodata

    @property
    def dims(self) -> list[str]:
        return ["band", "y", "x"]

    # --------------------------------------------------------------------- reads
    async def read_from_window(
        self,
        window: rasterio.windows.Window,
        boundless: bool = True,
    ) -> GeoTensor:
        """Read a window. Returns a fresh :class:`GeoTensor`.

        The ``boundless`` argument is accepted for protocol parity with
        :meth:`georeader.abstract_reader.GeoData.read_from_window` but
        ignored: async-geotiff's :meth:`read` already returns the requested
        window region with the appropriate fill where data is missing.
        """
        del boundless  # accepted for protocol parity, see docstring
        agt_window = _AGTWindow(
            col_off=int(window.col_off),
            row_off=int(window.row_off),
            width=int(window.width),
            height=int(window.height),
        )
        arr: Any = await self._level.read(window=agt_window)
        return _rasterarray_to_geotensor(arr, fill_value=self.fill_value_default, crs=self.crs)

    async def read_from_bounds(
        self,
        bounds: Tuple[float, float, float, float],
        *,
        target_resolution: Optional[Tuple[float, float]] = None,
        target_crs: Any = None,
    ) -> GeoTensor:
        """Read by geographic bounds in the reader's native CRS.

        Raises :class:`NotImplementedError` if ``target_resolution`` or
        ``target_crs`` are set — this reader does not warp or resample (the
        underlying ``async-geotiff`` explicitly disclaims warp). For
        cross-CRS reads, either fetch in the native CRS and post-warp via
        :func:`georeader.read.read_reproject_like`, or use
        :class:`~georeader.rasterio_reader.RasterioReader`.
        """
        if target_crs is not None or target_resolution is not None:
            raise NotImplementedError(
                "AsyncGeoTIFFReader does not warp or resample. "
                "Read in the native CRS, then call georeader.read.read_reproject_like, "
                "or use RasterioReader for WarpedVRT-based on-the-fly warping."
            )
        win = window_utils.window_from_bounds(self, bounds)
        return await self.read_from_window(win)

    async def load(self, boundless: bool = True) -> GeoTensor:
        """Read the whole raster (at the current ``overview_level``).

        ``boundless`` is accepted for protocol parity but ignored — the full
        extent is always inside the raster's bounds.
        """
        del boundless
        level = self._level
        full_window = rasterio.windows.Window(
            col_off=0, row_off=0, width=level.width, height=level.height,
        )
        return await self.read_from_window(full_window)

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
        status = "opened" if self._geotiff is not None else "unopened"
        return (
            f"AsyncGeoTIFFReader(path_or_url={self.path_or_url!r}, "
            f"overview_level={self._overview_level!r}, {status})"
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
