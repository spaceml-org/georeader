"""
Rasterio Reader: Lazy file-backed raster reading with geospatial awareness.

This module provides the RasterioReader class, a lazy wrapper around rasterio
that enables efficient reading of raster data from disk or cloud storage.
Unlike GeoTensor which holds data in memory, RasterioReader only reads data
when explicitly requested, making it ideal for large files and parallel processing.

Key Features
------------

- **Lazy Loading**: Data is read only when `load()` or `read()` is called
- **Multi-file Support**: Read multiple rasters as a time series stack
- **Windowed Reading**: Efficiently read subsets without loading full file
- **Overview Support**: Read from pyramids for quick previews
- **Cloud-native**: Works with COGs on S3, GCS, Azure via GDAL VSI
- **Process-safe**: Opens files fresh each read for parallel processing

Reader vs GeoTensor
-------------------

Choosing between RasterioReader and GeoTensor::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                 RASTERIOREADER vs GEOTENSOR                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  RasterioReader (Lazy)              GeoTensor (In-Memory)               │
    │  ─────────────────────              ────────────────────                │
    │                                                                         │
    │  • Data on disk/cloud               • Data in RAM                       │
    │  • Read on demand                   • Instant access                    │
    │  • Memory efficient                 • Full numpy API                    │
    │  • Parallel-safe                    • Arithmetic operations             │
    │  • Overview/pyramid support         • Broadcasting                      │
    │                                                                         │
    │  Use for:                           Use for:                            │
    │  • Large files                      • Processing pipelines              │
    │  • Cloud data                       • CNN inference                     │
    │  • Tiled processing                 • Index calculations                │
    │  • Quick previews                   • Visualizations                    │
    │                                                                         │
    │  Convert: reader.load() ────────────────────────────────► GeoTensor     │
    └─────────────────────────────────────────────────────────────────────────┘

Time Series / Multi-file Reading
--------------------------------

RasterioReader can stack multiple files as a time dimension::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MULTI-FILE READING                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Input: List of paths                Output array shape                 │
    │  ────────────────────                ──────────────────                 │
    │                                                                         │
    │  paths = [                                                              │
    │    "2023-01.tif",   ─────┐                                              │
    │    "2023-02.tif",   ─────┼──────► stack=True:  (T, C, H, W)             │
    │    "2023-03.tif"    ─────┘                      (3, 4, 1000, 1000)      │
    │  ]                                                                      │
    │                                                                         │
    │  Each file: (4, 1000, 1000)        stack=False: (T×C, H, W)             │
    │  4 bands, 1000×1000 pixels                       (12, 1000, 1000)       │
    │                                                                         │
    │  Requirements for multi-file:                                           │
    │  • Same CRS                                                             │
    │  • Same transform (resolution, origin)                                  │
    │  • Same shape (unless allow_different_shape=True)                       │
    └─────────────────────────────────────────────────────────────────────────┘

Window Focus
------------

set_window() creates a "view" into the raster for efficient subsetting::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WINDOW FOCUS CONCEPT                                 │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Full raster (10000 × 10000)                                            │
    │  ┌────────────────────────────────────────────────────────────────┐     │
    │  │                                                                │     │
    │  │                                                                │     │
    │  │        ┌─────────────────────┐                                 │     │
    │  │        │    window_focus     │  ← reader.set_window(...)       │     │
    │  │        │    (2000 × 2000)    │                                 │     │
    │  │        │                     │  After set_window:              │     │
    │  │        │  ┌───────────┐      │  • reader.shape → (C, 2000, 2000)│    │
    │  │        │  │ read()    │      │  • reader.bounds → window bounds│     │
    │  │        │  │ window    │      │  • read(window=...) is relative │     │
    │  │        │  └───────────┘      │    to window_focus              │     │
    │  │        └─────────────────────┘                                 │     │
    │  │                                                                │     │
    │  └────────────────────────────────────────────────────────────────┘     │
    │                                                                         │
    │  Benefits: • Work with large files efficiently                          │
    │            • Coordinates/bounds reflect the focused region              │
    │            • Tiled processing with consistent interface                 │
    └─────────────────────────────────────────────────────────────────────────┘

Module Contents
---------------

Classes:
    - :class:`RasterioReader`: Main lazy reader class

Quick Start
-----------

Read a local GeoTIFF::

    from georeader.rasterio_reader import RasterioReader

    # Open reader (lazy - no data loaded yet)
    reader = RasterioReader("image.tif")
    print(f"Shape: {reader.shape}, CRS: {reader.crs}")

    # Load into memory as GeoTensor
    gt = reader.load()

Read from cloud storage (COG on S3)::

    reader = RasterioReader("s3://bucket/image.tif")

    # Read only a small window
    window = rasterio.windows.Window(1000, 2000, 512, 512)
    subset = reader.read_from_window(window).load()

Read time series::

    paths = ["2023-01.tif", "2023-02.tif", "2023-03.tif"]
    reader = RasterioReader(paths, stack=True)
    print(f"Time series shape: {reader.shape}")  # (3, C, H, W)

    # Read specific bands and time steps
    subset = reader.isel({"time": [0, 2], "band": [0, 1, 2]})

See Also
--------
georeader.geotensor : In-memory georeferenced arrays
georeader.read : High-level read and reprojection functions
rasterio : Underlying library documentation

References
----------
- Rasterio: https://rasterio.readthedocs.io/
- Cloud Optimized GeoTIFF: https://cogeo.org/
- GDAL VSI: https://gdal.org/user/virtual_file_systems.html
"""
import re
import rasterio
import rasterio.windows
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any, Callable
import warnings
import numbers
from georeader import geotensor

from collections.abc import Iterable
from georeader import window_utils
from georeader.window_utils import window_bounds, get_slice_pad
from shapely.geometry import Polygon
from georeader.abstract_reader import same_extent, GeoData
from georeader.read import WEB_MERCATOR_CRS, SIZE_DEFAULT, window_from_tile, read_from_tile
from numpy.typing import NDArray


# CPL_VSIL_CURL_NON_CACHED configuration option can be set to values like 
# /vsicurl/http://example.com/foo.tif:/vsicurl/http://example.com/some_directory, so that at file handle closing, 
# all cached content related to the mentioned file(s) is no longer cached.
# https://github.com/rasterio/rasterio/issues/1877
# VSICurlClearCache()
# https://github.com/rasterio/rasterio/blob/main/rasterio/_path.py


RIO_ENV_OPTIONS_DEFAULT = geotensor.RIO_ENV_OPTIONS_DEFAULT


# Azure Blob Storage URLs carry a Shared Access Signature (SAS) as a query
# string, e.g.
#   https://<account>.blob.core.windows.net/<container>/<path>.tif?sv=...&se=...&sig=<secret>
# The `sig` parameter is the actual cryptographic signature of the token: anyone
# holding it can access the resource for the lifetime of the SAS. The other
# parameters (signed version `sv`, expiry `se`, permissions `sp`, ...) are not
# secret, so we keep them to preserve useful context (e.g. the expiry time) and
# only mask `sig`. The lookbehind keeps the match surgical so nothing else in the
# URL is altered.
_SAS_SIGNATURE_RE = re.compile(r"(?i)(?<=[?&]sig=)[^&]+")


def mask_sas_token(path: Any) -> Any:
    """Mask the signature of an Azure SAS token embedded in a file path.

    Replaces the value of the ``sig`` query parameter with ``****`` so that the
    secret signature of an Azure Shared Access Signature does not leak into
    ``repr``/log output. Non-string inputs and paths without a SAS signature are
    returned unchanged. The expiry (``se``) and other non-secret parameters are
    preserved.
    """
    if not isinstance(path, str):
        return path
    return _SAS_SIGNATURE_RE.sub("****", path)


# Keys the reader computes itself when calling rasterio.open(). Rejected in
# ``rio_open_kwargs`` at construction: "opener" would bypass the opener/fs
# mutual-exclusion validation, and "mode"/"overview_level" are passed
# positionally/explicitly by some (not all) internal call sites — a user value
# would raise "got multiple values" on some code paths and be silently ignored
# on others.
_FORBIDDEN_RIO_OPEN_KEYS = frozenset({"opener", "mode", "overview_level"})


def _validate_bytes_path_knobs(
    opener: Optional[Any], fs: Optional[Any], rio_open_kwargs: Optional[Dict[str, Any]]
) -> None:
    """Construction-time validation of the opener/fs/rio_open_kwargs knobs.

    Fails fast with a targeted message instead of surfacing as a confusing
    error deep inside ``rasterio.open`` at first read.
    """
    if opener is not None and fs is not None:
        raise ValueError(
            "RasterioReader: pass either `opener=` or `fs=`, not both. "
            "`fs=` is a shortcut for `opener=fs.open`."
        )
    if fs is not None and not callable(getattr(fs, "open", None)):
        # A string ("s3") or any object without .open would otherwise surface
        # as an AttributeError at first read.
        raise TypeError(
            "RasterioReader: `fs=` expects an fsspec-like filesystem object "
            f"with a callable .open method, got {type(fs).__name__!r}. "
            "For a protocol string use fsspec.filesystem(...) first."
        )
    if rio_open_kwargs is not None:
        forbidden = _FORBIDDEN_RIO_OPEN_KEYS.intersection(rio_open_kwargs)
        if forbidden:
            raise ValueError(
                f"RasterioReader: rio_open_kwargs must not contain {sorted(forbidden)}. "
                "Use the dedicated `opener=`/`fs=` arguments and the "
                "`overview_level=` constructor argument instead."
            )


class RasterioReader:
    """
    Lazy file-backed raster reader with geospatial metadata.

    RasterioReader wraps rasterio to provide lazy, memory-efficient access to
    raster files on disk or cloud storage. Data is only read when explicitly
    requested via `load()` or `read()`, making it ideal for large files and
    parallel processing scenarios.

    The class supports reading single files or multiple files as a stacked
    time series. All files must share the same CRS, transform, and shape
    (unless `allow_different_shape=True`).

    Args:
        paths (Union[List[str], str]): Single path or list of paths to raster files.
            Supports local paths, S3 URIs (s3://), GCS URIs (gs://), Azure paths,
            and HTTP URLs for COGs.
        allow_different_shape (bool, optional): If True, allows reading files with
            different shapes (still requires same CRS, transform, band count).
            Defaults to False.
        window_focus (Optional[rasterio.windows.Window], optional): Initial window
            to focus on. All subsequent operations will be relative to this window.
            Defaults to None (full raster).
        fill_value_default (Optional[Union[int, float]], optional): Value for
            out-of-bounds pixels in boundless reads. Defaults to nodata value
            if available, otherwise 0.
        stack (bool, optional): If True and paths is a list, returns 4D arrays
            (T, C, H, W). If False, concatenates along band dimension (T*C, H, W).
            Ignored for single file. Defaults to True.
        indexes (Optional[List[int]], optional): Band indices to read (1-based,
            following rasterio convention). None reads all bands. Defaults to None.
        overview_level (Optional[int], optional): Pyramid level to read from
            (0-based, 0 = first overview, None = full resolution). Useful for
            quick previews. Defaults to None.
        check (bool, optional): Validate that all paths have matching CRS,
            transform, and shape. Defaults to True.
        rio_env_options (Optional[Dict[str, str]], optional): GDAL environment
            options for reading. Defaults to RIO_ENV_OPTIONS_DEFAULT.
        opener (Optional[Callable], optional): Keyword-only. A callable passed
            straight to ``rasterio.open(opener=...)`` for custom byte-range
            transport. Mutually exclusive with ``fs``. When neither ``opener``
            nor ``fs`` is set, rasterio routes bytes through GDAL VSI (the
            default, fastest cloud path).

            .. warning::
                The reader's pickle contract (multiprocessing / joblib /
                Dask) narrows to whatever the callback can do: a lambda,
                a closure over a session, or anything holding locks will
                fail to pickle at ``pool.map``/``submit`` time. Use a
                module-level function (or a picklable fsspec filesystem
                via ``fs=``) when the reader must cross process
                boundaries. Python openers also come with rasterio-side
                constraints: no GDAL VSI chaining (e.g. ``/vsizip/``),
                and the file object is shared with GDAL's threads — avoid
                combining ``opener=`` with multi-threaded GDAL options in
                ``rio_env_options`` (e.g. ``GDAL_NUM_THREADS``).
        fs (Optional[fsspec.AbstractFileSystem], optional): Keyword-only.
            Shortcut equivalent to ``opener=fs.open``. Useful for niche
            backends (FTP, SFTP, GitHub) or custom auth that fsspec speaks
            but GDAL VSI does not. Mutually exclusive with ``opener``. Must
            expose a callable ``.open`` (validated at construction). Most
            fsspec filesystems pickle cleanly, so this is the knob to
            prefer for multiprocessing workloads.
        rio_open_kwargs (Optional[Dict[str, Any]], optional): Keyword-only.
            Arbitrary additional keyword arguments forwarded to every
            ``rasterio.open(...)`` call (e.g. ``{"sharing": False}``). Escape
            hatch for rasterio options not surfaced as first-class kwargs.
            The dict is copied at construction, and the reader-computed keys
            (``opener``, ``mode``, ``overview_level``) are rejected — use
            the dedicated arguments for those.

    Attributes:
        crs (rasterio.crs.CRS): Coordinate reference system.
        transform (rasterio.Affine): Affine transform (reflects window_focus if set).
        shape (Tuple[int, ...]): Array shape as (T, C, H, W) or (C, H, W).
        dtype (str): Data type of the raster.
        count (int): Number of bands being read.
        width (int): Width in pixels (of window_focus if set).
        height (int): Height in pixels (of window_focus if set).
        bounds (Tuple[float, float, float, float]): Geographic bounds (minx, miny, maxx, maxy).
        res (Tuple[float, float]): Pixel resolution (x_res, y_res).
        nodata (Optional[Union[int, float]]): Nodata value from file metadata.
        fill_value_default (Union[int, float]): Fill value for boundless reads.
        dims (List[str]): Dimension names for xarray compatibility.
        attrs (Dict[str, Any]): Extra attributes dictionary.

    Examples:
        Read a single GeoTIFF::

            >>> from georeader.rasterio_reader import RasterioReader
            >>>
            >>> reader = RasterioReader("image.tif")
            >>> print(f"Shape: {reader.shape}")  # (C, H, W)
            Shape: (4, 1000, 1000)
            >>> print(f"CRS: {reader.crs}")
            CRS: EPSG:32630
            >>>
            >>> # Load into memory as GeoTensor
            >>> gt = reader.load()
            >>> print(type(gt))
            <class 'georeader.geotensor.GeoTensor'>

        Read specific bands::

            >>> # Read only RGB bands (1-based indexing)
            >>> reader = RasterioReader("image.tif", indexes=[1, 2, 3])
            >>> print(f"Shape: {reader.shape}")
            Shape: (3, 1000, 1000)

        Read time series from multiple files::

            >>> paths = ["2023-01.tif", "2023-02.tif", "2023-03.tif"]
            >>> reader = RasterioReader(paths, stack=True)
            >>> print(f"Shape: {reader.shape}")  # (T, C, H, W)
            Shape: (3, 4, 1000, 1000)
            >>>
            >>> # Access by dimension names
            >>> january = reader.isel({"time": 0})
            >>> print(f"January shape: {january.shape}")
            January shape: (4, 1000, 1000)

        Read from cloud storage::

            >>> reader = RasterioReader("s3://bucket/cog.tif")
            >>> # Read small window without loading entire file
            >>> window = rasterio.windows.Window(0, 0, 512, 512)
            >>> subset = reader.read_from_window(window).load()

        Use overview for quick preview::

            >>> reader = RasterioReader("large_image.tif", overview_level=2)
            >>> # Much faster read at reduced resolution
            >>> preview = reader.load()

        Set window focus for tiled processing::

            >>> reader = RasterioReader("large_image.tif")
            >>> # Focus on region of interest
            >>> reader.set_window(rasterio.windows.Window(5000, 5000, 2000, 2000))
            >>> print(f"Focused shape: {reader.shape}")
            Focused shape: (4, 2000, 2000)
            >>> # All reads now relative to this window

    Note:
        - Files are opened fresh for each read() call (process-safe)
        - Use `load()` to get a GeoTensor for in-memory operations
        - Band indexing is 1-based (rasterio convention)
        - Overview levels are 0-based (0 = first overview, not full res)

    See Also:
        georeader.geotensor.GeoTensor: In-memory array with geo metadata.
        georeader.read.read: High-level read with reprojection.
        read_out_shape: Read with resampling to target shape.
    """
    def __init__(self, paths:Union[List[str], str], allow_different_shape:bool=False,
                 window_focus:Optional[rasterio.windows.Window]=None,
                 fill_value_default:Optional[Union[int, float]]=None,
                 stack:bool=True, indexes:Optional[List[int]]=None,
                 overview_level:Optional[int]=None, check:bool=True,
                 rio_env_options:Optional[Dict[str, str]]=None,
                 *,
                 opener:Optional[Callable]=None,
                 fs:Optional[Any]=None,
                 rio_open_kwargs:Optional[Dict[str, Any]]=None):

        # Syntactic sugar
        if isinstance(paths, str):
            paths = [paths]
            stack = False

        if rio_env_options is None:
            self.rio_env_options = RIO_ENV_OPTIONS_DEFAULT
        else:
            self.rio_env_options = rio_env_options

        # Bytes-path knobs — at most one of opener / fs may be set. ``opener``
        # is the canonical rasterio.open(opener=...) callback; ``fs`` is an
        # fsspec shortcut equivalent to ``opener=fs.open``. Both default to
        # None, in which case rasterio routes bytes through GDAL VSI.
        # ``rio_open_kwargs`` is an escape hatch for arbitrary additional
        # keyword arguments passed straight to rasterio.open() (e.g.
        # ``{"sharing": False}``).
        _validate_bytes_path_knobs(opener, fs, rio_open_kwargs)
        self._opener = opener
        self._fs = fs
        # Copy: child readers (read_from_window / isel / copy / reader_overview)
        # receive this dict — sharing by reference would let post-construction
        # mutation retroactively change every derived reader.
        self._rio_open_kwargs = dict(rio_open_kwargs) if rio_open_kwargs is not None else None

        self.paths = paths

        self.stack = stack

        # TODO keep just a global nodata of size (T,C,) and fill with these values?
        self.fill_value_default = fill_value_default
        self.overview_level = overview_level
        with rasterio.Env(**self._get_rio_options_path(paths[0])):
            with rasterio.open(paths[0], "r", overview_level=overview_level,
                               **self._resolve_open_kwargs()) as src:
                self.real_transform = src.transform
                self.crs = src.crs
                self.dtype = src.profile["dtype"]
                self.real_count = src.count
                self.real_indexes = list(range(1, self.real_count + 1))
                if self.stack:
                    self.real_shape = (len(self.paths), src.count,) + src.shape
                else:
                    self.real_shape = (len(self.paths) * self.real_count, ) + src.shape

                self.real_width = src.width
                self.real_height = src.height

                self.nodata = src.nodata
                if self.fill_value_default is None:
                    self.fill_value_default = self.nodata if (self.nodata is not None) else 0

                self.res = src.res

        # if (abs(self.real_transform.b) > 1e-6) or (abs(self.real_transform.d) > 1e-6):
        #     warnings.warn(f"transform of {self.paths[0]} is not rectilinear {self.real_transform}. "
        #                   f"The vast majority of the code expect rectilinear transforms. This transform "
        #                   f"could cause unexpected behaviours")

        self.attrs = {}
        self.window_focus = rasterio.windows.Window(row_off=0, col_off=0,
                                                    width=self.real_width, height=self.real_height)
        self.real_window = rasterio.windows.Window(row_off=0, col_off=0,
                                                   width=self.real_width, height=self.real_height)
        self.set_indexes(self.real_indexes, relative=False)
        self.set_window(window_focus, relative=False)

        self.allow_different_shape = allow_different_shape

        if self.stack:
            self.dims = ["time", "band", "y", "x"]
        else:
            self.dims = ["band", "y", "x"]

        self._coords = None

        # Assert all paths have same tranform and crs
        #  (checking width and height will not be needed since we're reading with boundless option but I don't see the point to ignore it)
        if check and len(self.paths) > 1:
            # Mask any Azure SAS token signature before interpolating paths
            # into error/warning text — these messages end up in logs and
            # tracebacks (the masking in __repr__ alone doesn't cover them).
            p0_masked = mask_sas_token(self.paths[0])
            for p in self.paths:
                p_masked = mask_sas_token(p)
                with rasterio.Env(**self._get_rio_options_path(p)):
                    with rasterio.open(p, "r", overview_level=overview_level,
                                       **self._resolve_open_kwargs()) as src:
                        if not src.transform.almost_equals(self.real_transform, 1e-6):
                            raise ValueError(f"Different transform in {p0_masked} and {p_masked}: {self.real_transform} {src.transform}")
                        if not str(src.crs).lower() == str(self.crs).lower():
                            raise ValueError(f"Different CRS in {p0_masked} and {p_masked}: {self.crs} {src.crs}")
                        if self.real_count != src.count:
                            raise ValueError(f"Different number of bands in {p0_masked} and {p_masked} {self.real_count} {src.count}")
                        if src.nodata != self.nodata:
                            warnings.warn(
                                f"Different nodata in {p0_masked} and {p_masked}: {self.nodata} {src.nodata}. This might lead to unexpected behaviour")

                        if (self.real_width != src.width) or (self.real_height != src.height):
                            if allow_different_shape:
                                warnings.warn(f"Different shape in {p0_masked} and {p_masked}: ({self.real_height}, {self.real_width}) ({src.height}, {src.width}) Might lead to unexpected behaviour")
                            else:
                                raise ValueError(f"Different shape in {p0_masked} and {p_masked}: ({self.real_height}, {self.real_width}) ({src.height}, {src.width})")

        self.check = check
        if indexes is not None:
            self.set_indexes(indexes)

    def set_indexes(self, indexes:List[int], relative:bool=True)-> None:
        """
        Set the band indices to read.

        Modifies the reader in-place to read only the specified bands. Band
        indices follow rasterio's 1-based convention. Useful for working with
        subsets of multi-band imagery (e.g., RGB from RGBN).

        Args:
            indexes (List[int]): Band indices to read (1-based, per rasterio
                convention). Must be within valid range for the raster.
            relative (bool, optional): If True, indices are relative to current
                `self.indexes`. If False, indices are absolute (relative to
                full raster). Defaults to True.

        Raises:
            AssertionError: If any index is out of bounds (< 1 or > band count).

        Examples:
            Select specific bands from multi-band raster::

                >>> reader = RasterioReader("image.tif")  # 6-band raster
                >>> print(reader.count)
                6
                >>>
                >>> # Read only RGB (bands 1, 2, 3)
                >>> reader.set_indexes([1, 2, 3], relative=False)
                >>> print(reader.count)
                3
                >>> print(reader.shape)
                (3, 1000, 1000)

            Relative indexing::

                >>> reader = RasterioReader("image.tif", indexes=[2, 3, 4, 5])
                >>> print(reader.indexes)  # Currently reading bands 2-5
                [2, 3, 4, 5]
                >>>
                >>> # Select first two of current selection (bands 2, 3)
                >>> reader.set_indexes([1, 2], relative=True)
                >>> print(reader.indexes)
                [2, 3]

            Combined with constructor::

                >>> # Directly specify bands at creation
                >>> reader = RasterioReader("image.tif", indexes=[4, 3, 2])  # NIR, R, G
                >>> nir_r_g = reader.load()

        Note:
            - Modifies reader in-place
            - Use 1-based indexing (band 1 is the first band)
            - For 0-based indexing, use `isel({"band": [...]})` instead

        See Also:
            set_indexes_by_name: Select bands by description/name.
            isel: Dimension-based selection with 0-based indexing.
        """
        if relative:
            new_indexes = [self.indexes[idx - 1] for idx in indexes]
        else:
            new_indexes = indexes

        # Check if indexes are valid
        assert all((s >= 1) and (s <= self.real_count) for s in new_indexes), \
               f"Indexes (1-based) out of real bounds current: {self.indexes} asked: {new_indexes} number of bands:{self.real_count}"
        
        self.indexes = new_indexes

        assert all((s >= 1) and (s <= self.real_count) for s in
                   self.indexes), f"Indexes out of real bounds current: {self.indexes} asked: {indexes} number of bands:{self.real_count}"

        self.count = len(self.indexes)

    def set_indexes_by_name(self, names:List[str]) -> None:
        """
        Function to set the indexes by the name of the band which is stored in the descriptions attribute

        Args:
            names: List of band names to read
        
        Examples:
            >>> r = RasterioReader("path/to/raster.tif") # Read all bands except the first one.
            >>> # Assume r.descriptions = ["B1", "B2", "B3"]
            >>> r.set_indexes_by_name(["B2", "B3"])

        """
        descriptions = self.descriptions
        if len(self.paths) == 1:
            if self.stack:
                descriptions = descriptions[0]
        else:
            assert all(d == descriptions[0] for d in descriptions), "There are tiffs with different names"
            descriptions = descriptions[0]

        bands = [descriptions.index(b) + 1 for b in names]
        self.set_indexes(bands, relative=False)

    @property
    def shape(self):
        if self.stack:
            return len(self.paths), self.count, self.height, self.width
        return len(self.paths) * self.count, self.height, self.width
    
    def same_extent(self, other:Union[GeoData,'RasterioReader'], precision:float=1e-3) -> bool:
        """
        Check if two GeoData objects have the same extent

        Args:
            other: GeoData object to compare
            precision: precision to compare the bounds

        Returns:
            True if both objects have the same extent

        """
        return same_extent(self, other, precision=precision)

    def set_window(self, window_focus:Optional[rasterio.windows.Window] = None,
                   relative:bool = True, boundless:bool=True)->None:
        """
        Set the window focus for subsequent read operations.

        Modifies the reader in-place to focus on a specific window. All subsequent
        `read()`, `load()`, and `read_from_window()` calls will be relative to this
        window. This enables efficient tiled processing of large rasters.

        Args:
            window_focus (Optional[rasterio.windows.Window], optional): Window to
                focus on. If None, resets to full raster extent. Defaults to None.
            relative (bool, optional): If True, window is relative to current
                `window_focus`. If False, window is absolute (relative to full
                raster). Defaults to True.
            boundless (bool, optional): If True, allows window to extend beyond
                raster bounds. If False, intersects with valid extent. Defaults
                to True.

        Examples:
            Focus on a 1000x1000 region::

                >>> reader = RasterioReader("image.tif")
                >>> print(f"Original: {reader.shape}")
                Original: (4, 5000, 5000)
                >>>
                >>> reader.set_window(rasterio.windows.Window(0, 0, 1000, 1000))
                >>> print(f"Focused: {reader.shape}")
                Focused: (4, 1000, 1000)
                >>>
                >>> gt = reader.load()  # Only reads 1000x1000

            Nested windowing (relative=True)::

                >>> reader = RasterioReader("image.tif")
                >>> # First focus: pixels 1000-2000 in both dims
                >>> reader.set_window(rasterio.windows.Window(1000, 1000, 1000, 1000))
                >>>
                >>> # Second focus: relative to first, so actual 1100-1600
                >>> reader.set_window(rasterio.windows.Window(100, 100, 500, 500))
                >>> print(reader.bounds)  # Shows geographic bounds of 1100-1600 region

            Absolute windowing (relative=False)::

                >>> reader = RasterioReader("image.tif")
                >>> reader.set_window(rasterio.windows.Window(0, 0, 500, 500))
                >>>
                >>> # Override with absolute position
                >>> reader.set_window(
                ...     rasterio.windows.Window(2000, 2000, 500, 500),
                ...     relative=False
                ... )
                >>> # Now focused on pixels 2000-2500, not 500-1000

            Reset to full extent::

                >>> reader.set_window(None)
                >>> print(reader.shape)  # Back to full raster
                (4, 5000, 5000)

        Note:
            - Modifies reader in-place
            - Updates `transform`, `bounds`, `width`, `height` attributes
            - Use `read_from_window()` for a non-mutating alternative

        See Also:
            read_from_window: Create new reader for window (non-mutating).
            isel: Dimension-based selection.
        """
        if window_focus is None:
            self.window_focus = rasterio.windows.Window(row_off=0, col_off=0,
                                                        width=self.real_width, height=self.real_height)
        elif relative:
            self.window_focus = rasterio.windows.Window(col_off=window_focus.col_off + self.window_focus.col_off,
                                                        row_off=window_focus.row_off + self.window_focus.row_off,
                                                        height=window_focus.height, width=window_focus.width)
        else:
            self.window_focus = window_focus

        if not boundless:
            self.window_focus = rasterio.windows.intersection(self.real_window, self.window_focus)

        self.height = self.window_focus.height
        self.width = self.window_focus.width

        self.bounds = window_bounds(self.window_focus, self.real_transform)
        self.transform = rasterio.windows.transform(self.window_focus, self.real_transform)

    def tags(self) -> Union[List[Dict[str, str]], Dict[str, str]]:
        """
        Returns a list with the tags for each tiff file.
        If stack and len(self.paths) == 1 it returns just the dictionary of the tags

        """
        tags = []
        for i, p in enumerate(self.paths):
            with rasterio.Env(**self._get_rio_options_path(p)):
                with rasterio.open(p, mode="r", **self._resolve_open_kwargs()) as src:
                    tags.append(src.tags())

        if (not self.stack) and (len(tags) == 1):
            return tags[0]

        return tags

    def _get_rio_options_path(self, path:str) -> Dict[str, str]:
        options = self.rio_env_options
        return geotensor.get_rio_options_path(options=options, path=path)

    def _resolve_open_kwargs(self) -> Dict[str, Any]:
        """Translate the constructor's opener/fs knobs into rasterio.open kwargs.

        Returns the kwargs dict to splat at every ``rasterio.open(path, ...)``
        call site. When neither ``opener`` nor ``fs`` was set on the
        constructor, returns just ``rio_open_kwargs`` (or empty) — rasterio
        then routes bytes through GDAL VSI as usual. When ``opener`` is set,
        forwards it as ``opener=...``; when ``fs`` is set, equivalent to
        ``opener=self._fs.open``.

        Returns:
            Dict[str, Any]: kwargs suitable for ``**`` splat into
            ``rasterio.open(...)``.
        """
        kwargs: Dict[str, Any] = dict(self._rio_open_kwargs or {})
        if self._opener is not None:
            kwargs["opener"] = self._opener
        elif self._fs is not None:
            kwargs["opener"] = self._fs.open
        return kwargs
    
    # This function does not work for e.g. returning the descriptions of the bands
    # @contextmanager
    # def _rio_open(self, path:str, mode:str="r", overview_level:Optional[int]=None) -> rasterio.DatasetReader:
    #     with rasterio.Env(**self._get_rio_options_path(path)):
    #         with rasterio.open(path, mode=mode, overview_level=overview_level) as src:
    #             yield src

    @property
    def descriptions(self) -> Union[List[List[str]], List[str]]:
        """
        Returns a list with the descriptions for each tiff file. (This is usually the name of the bands of the raster)


        Returns:
            If `stack` it returns the flattened list of descriptions for each tiff file. If not `stack` it returns a list of lists.
        
        Examples:
            >>> r = RasterioReader("path/to/raster.tif") # Raster with band names B1, B2, B3
            >>> r.descriptions # returns ["B1", "B2", "B3"]
        """
        descriptions_all = []
        for i, p in enumerate(self.paths):
            with rasterio.Env(**self._get_rio_options_path(p)):
                with rasterio.open(p, **self._resolve_open_kwargs()) as src:
                    desc = src.descriptions

            if self.stack:
                descriptions_all.append([desc[i-1] for i in self.indexes])
            else:
                descriptions_all.extend([desc[i-1] for i in self.indexes])

        return descriptions_all

    def read_from_window(self, window:rasterio.windows.Window, boundless:bool=True) -> '__class__':
        """
        Create a new reader focused on a sub-window.

        Returns a lazy RasterioReader with its `window_focus` set to the
        specified window. The window is interpreted relative to the current
        `window_focus`. No data is read until `load()` or `read()` is called.

        This is efficient for tiled processing where you want to iterate over
        windows without loading everything into memory.

        Args:
            window (rasterio.windows.Window): Target window, relative to current
                `window_focus`.
            boundless (bool, optional): If True, allows windows extending beyond
                raster bounds (filled with `fill_value_default`). If False,
                window is intersected with valid extent. Defaults to True.

        Returns:
            RasterioReader: New reader focused on the specified window.

        Raises:
            rasterio.windows.WindowError: If `boundless=False` and window doesn't
                intersect the valid raster extent.

        Examples:
            Read a 512x512 tile::

                >>> reader = RasterioReader("large_image.tif")
                >>> window = rasterio.windows.Window(1000, 1000, 512, 512)
                >>> tile_reader = reader.read_from_window(window)
                >>> tile = tile_reader.load()
                >>> print(tile.shape)
                (4, 512, 512)

            Tiled processing pattern::

                >>> reader = RasterioReader("large_image.tif")
                >>> tile_size = 512
                >>> results = []
                >>> for row in range(0, reader.height, tile_size):
                ...     for col in range(0, reader.width, tile_size):
                ...         window = rasterio.windows.Window(col, row, tile_size, tile_size)
                ...         tile = reader.read_from_window(window).load()
                ...         # Process tile...
                ...         results.append(process(tile))

            Bounded vs boundless::

                >>> # Window at edge of 1000x1000 raster
                >>> window = rasterio.windows.Window(900, 900, 200, 200)
                >>>
                >>> # Boundless: full 200x200 with fill values
                >>> sub = reader.read_from_window(window, boundless=True)
                >>> print(sub.shape)
                (4, 200, 200)
                >>>
                >>> # Bounded: clipped to 100x100 valid region
                >>> sub = reader.read_from_window(window, boundless=False)
                >>> print(sub.shape)
                (4, 100, 100)

        Note:
            - Returns a lazy reader, not data
            - Chain multiple operations before final `load()`
            - Preserves band selection from parent reader

        See Also:
            set_window: Modify window focus in-place.
            isel: Slice using dimension names.
            load: Materialize reader to GeoTensor.
        """
        rst_reader = RasterioReader(list(self.paths),
                                    allow_different_shape=self.allow_different_shape,
                                    window_focus=self.window_focus,
                                    fill_value_default=self.fill_value_default,
                                    stack=self.stack,
                                    overview_level=self.overview_level,
                                    check=False,
                                    rio_env_options=self.rio_env_options,
                                    opener=self._opener,
                                    fs=self._fs,
                                    rio_open_kwargs=self._rio_open_kwargs)

        rst_reader.set_window(window, relative=True, boundless=boundless)
        rst_reader.set_indexes(self.indexes, relative=False)
        return rst_reader

    def isel(self, sel: Dict[str, Union[slice, List[int], int]], boundless:bool=True) -> '__class__':
        """
        Create a new reader by selecting along named dimensions.

        Mimics xarray's `DataArray.isel()` for intuitive dimension-based slicing.
        Supports selection on "time", "band", "y", and "x" dimensions. Returns
        a lazy reader; data is not loaded until `load()` is called.

        Args:
            sel (Dict[str, Union[slice, List[int], int]]): Selection dictionary
                mapping dimension names to index selections:
                - "time": int, list of ints, or slice (for multi-file readers)
                - "band": list of ints or slice (NOT single int)
                - "x": slice only
                - "y": slice only
            boundless (bool, optional): If True, spatial slices can extend beyond
                bounds. Defaults to True.

        Returns:
            RasterioReader: New reader with the selection applied.

        Raises:
            NotImplementedError: If dimension not in reader's dims, or if
                unsupported selection type is used.

        Examples:
            Select time steps from multi-file reader::

                >>> paths = ["2023-01.tif", "2023-02.tif", "2023-03.tif"]
                >>> reader = RasterioReader(paths, stack=True)
                >>> print(reader.shape)  # (3, 4, 1000, 1000)
                (3, 4, 1000, 1000)
                >>>
                >>> # Single time step (reduces dimension)
                >>> jan = reader.isel({"time": 0})
                >>> print(jan.shape)
                (4, 1000, 1000)
                >>>
                >>> # Range of time steps
                >>> first_two = reader.isel({"time": slice(0, 2)})
                >>> print(first_two.shape)
                (2, 4, 1000, 1000)

            Select bands::

                >>> reader = RasterioReader("image.tif")  # 4 bands
                >>> rgb = reader.isel({"band": [0, 1, 2]})  # 0-based indexing
                >>> print(rgb.shape)
                (3, 1000, 1000)

            Spatial slicing::

                >>> # Crop to region
                >>> subset = reader.isel({"y": slice(100, 500), "x": slice(200, 600)})
                >>> print(subset.shape)
                (4, 400, 400)

            Combined selection::

                >>> # First time step, RGB bands, spatial subset
                >>> sel = {
                ...     "time": 0,
                ...     "band": [0, 1, 2],
                ...     "y": slice(0, 512),
                ...     "x": slice(0, 512)
                ... }
                >>> subset = reader.isel(sel)
                >>> gt = subset.load()

        Note:
            - "time" dimension only available when `stack=True`
            - Band indexing is 0-based in isel (unlike rasterio's 1-based)
            - Single band selection with int not supported (use list)
            - Spatial dimensions only accept slices, not lists/ints

        See Also:
            read_from_window: Window-based spatial selection.
            set_indexes: Set bands to read (1-based).
            set_window: Set spatial window focus.
        """
        for k in sel:
            if k not in self.dims:
                raise NotImplementedError(f"Axis {k} not in dims: {self.dims}")

        stack = self.stack
        if "time" in sel: # time allowed only if self.stack (would have raised error above)
            if isinstance(sel["time"], Iterable):
                paths = [self.paths[i] for i in sel["time"]]
            elif isinstance(sel["time"], slice):
                paths = self.paths[sel["time"]]
            elif isinstance(sel["time"], numbers.Number):
                paths = [self.paths[sel["time"]]]
                stack = False
            else:
                raise NotImplementedError(f"Don't know how to slice {sel['time']} in dim time")
        else:
            paths = self.paths

        # Band slicing
        if "band" in sel:
            if not self.stack:
                # if `True` returns 4D tensors otherwise it returns 3D tensors concatenated over the first dim
                assert (len(self.paths) == 1) or (len(self.indexes) == 1), f"Dont know how to slice {self.paths} and {self.indexes}"

            if self.stack or (len(self.paths) == 1):
                if isinstance(sel["band"], Iterable):
                    indexes = [self.indexes[i] for i in sel["band"]] # indexes relative to current indexes
                elif isinstance(sel["band"], slice):
                    indexes = self.indexes[sel["band"]]
                elif isinstance(sel["band"], numbers.Number):
                    raise NotImplementedError(f"Slicing band with a single number is not supported (use a list)")
                else:
                    raise NotImplementedError(f"Don't know how to slice {sel['band']} in dim band")
            else:
                indexes = self.indexes
                # len(indexes) == 1 and not self.stack in this case band slicing correspond to paths
                if isinstance(sel["band"], Iterable):
                    paths = [self.paths[i] for i in sel["band"]]
                elif isinstance(sel["band"], slice):
                    paths = self.paths[sel["band"]]
                elif isinstance(sel["band"], numbers.Number):
                    paths = [self.paths[sel["band"]]]
                else:
                    raise NotImplementedError(f"Don't know how to slice {sel['time']} in dim time")
        else:
            indexes = self.indexes

        # Spatial slicing
        slice_ = []
        spatial_shape = (self.height, self.width)
        for _i, spatial_name in enumerate(["y", "x"]):
            if spatial_name in sel:
                if not isinstance(sel[spatial_name], slice):
                    raise NotImplementedError(f"spatial dimension {spatial_name} only accept slice objects")
                slice_.append(sel[spatial_name])
            else:
                slice_.append(slice(0, spatial_shape[_i]))

        rst_reader = RasterioReader(paths, allow_different_shape=self.allow_different_shape,
                                    window_focus=self.window_focus,
                                    fill_value_default=self.fill_value_default,
                                    stack=stack, overview_level=self.overview_level,
                                    check=False,
                                    rio_env_options=self.rio_env_options,
                                    opener=self._opener,
                                    fs=self._fs,
                                    rio_open_kwargs=self._rio_open_kwargs)
        window_current = rasterio.windows.Window.from_slices(*slice_, boundless=boundless,
                                                             width=self.width, height=self.height)

        # Set bands to read
        rst_reader.set_indexes(indexes=indexes, relative=False)

        # set window_current relative to self.window_focus
        rst_reader.set_window(window_current, relative=True)

        return rst_reader

    def __copy__(self) -> '__class__':
        rst = RasterioReader(self.paths, allow_different_shape=self.allow_different_shape,
                              window_focus=self.window_focus,
                              fill_value_default=self.fill_value_default,
                              stack=self.stack, overview_level=self.overview_level,
                              check=False, rio_env_options=self.rio_env_options,
                              opener=self._opener, fs=self._fs,
                              rio_open_kwargs=self._rio_open_kwargs)
        rst.set_indexes(self.indexes, relative=False)
        return rst
    
    def overviews(self, index:int=1, time_index:int=0) -> List[int]:
        """
        Get available overview (pyramid) levels for the raster.

        Queries the raster file for internal overview levels, which enable
        efficient reading at reduced resolutions. This is particularly useful
        for COGs (Cloud Optimized GeoTIFFs).

        Args:
            index (int, optional): Band index to query (1-based). Defaults to 1.
            time_index (int, optional): For multi-file readers, which file to
                query (0-based). Defaults to 0.

        Returns:
            List[int]: Available overview factors (e.g., [2, 4, 8, 16] means
                overviews at 1/2, 1/4, 1/8, 1/16 resolution).

        Examples:
            Check available overviews::

                >>> reader = RasterioReader("cog.tif")
                >>> print(reader.overviews())
                [2, 4, 8, 16]
                >>>
                >>> # Read at 1/4 resolution using overview_level=1
                >>> fast_reader = reader.reader_overview(overview_level=1)

        Note:
            - Overview levels are 0-based for `reader_overview()`:
              level 0 = first overview (factor 2), not full resolution
            - Empty list means no internal overviews (read full res only)

        See Also:
            reader_overview: Create reader at specific overview level.
            RasterioReader: Constructor accepts `overview_level` parameter.
        """
        with rasterio.Env(**self._get_rio_options_path(self.paths[time_index])):
            with rasterio.open(self.paths[time_index], **self._resolve_open_kwargs()) as src:
                return src.overviews(index)
    
    def reader_overview(self, overview_level:int) -> '__class__':
        """
        Create a new reader at a specific overview level.

        Returns a lazy reader configured to read from the specified overview
        (pyramid) level. Useful for fast previews or memory-efficient processing
        of large rasters.

        Args:
            overview_level (int): Overview level to read from.
                - Non-negative: Direct overview index (0 = first overview)
                - Negative: Relative from end (-1 = full resolution, -2 = finest
                  overview, etc.)

        Returns:
            RasterioReader: New reader configured for the specified overview level.

        Examples:
            Read at first overview level::

                >>> reader = RasterioReader("large_cog.tif")
                >>> print(reader.overviews())
                [2, 4, 8]
                >>>
                >>> # Read at 1/2 resolution (first overview)
                >>> preview = reader.reader_overview(0)
                >>> gt = preview.load()

            Use negative indexing::

                >>> # -1 = full resolution (no overview)
                >>> full_res = reader.reader_overview(-1)
                >>>
                >>> # -2 = finest available overview
                >>> finest_overview = reader.reader_overview(-2)

            Fast thumbnail generation::

                >>> # Use coarsest overview for fastest load
                >>> num_overviews = len(reader.overviews())
                >>> thumb_reader = reader.reader_overview(num_overviews - 1)
                >>> thumbnail = thumb_reader.load()

        Note:
            - Overview level 0 is the first overview, not full resolution
            - Use `overview_level=None` in constructor for full resolution
            - Window focus is reset when creating overview reader

        See Also:
            overviews: List available overview levels.
            RasterioReader: Constructor accepts `overview_level` parameter.
        """
        if overview_level < 0:
            overview_level = len(self.overviews()) + overview_level
        
        rst = RasterioReader(self.paths, allow_different_shape=self.allow_different_shape,
                             window_focus=None,
                             fill_value_default=self.fill_value_default,
                             stack=self.stack,
                             indexes=self.indexes,
                             overview_level=overview_level,
                             check=False,
                             rio_env_options=self.rio_env_options,
                             opener=self._opener,
                             fs=self._fs,
                             rio_open_kwargs=self._rio_open_kwargs)

        # if self.window_focus hasn't been changed we're good
        if self.window_focus.width == self.real_width and\
            self.window_focus.height == self.real_height and\
            self.window_focus.col_off == 0 and\
            self.window_focus.row_off == 0:
            return rst
        
        # TODO we need to convert the self.window_focus to the dst crs
        # window_utils.
        warnings.warn("Window focus is not supported in overview level. Returning the overview level with the full raster")
        return rst
    
    def block_windows(self, bidx:int=1, time_idx:int=0) -> List[Tuple[int, rasterio.windows.Window]]:
        """
        return the block windows within the object
        (see https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader.block_windows)

        Args:
            bidx: band index to read (1-based)
            time_idx: time index to read (0-based)

        Returns:
            list of (block_idx, window)

        """
        with rasterio.Env(**self._get_rio_options_path(self.paths[time_idx])):
            with rasterio.open(self.paths[time_idx], **self._resolve_open_kwargs()) as src:
                windows_return = [(block_idx, rasterio.windows.intersection(window, self.window_focus)) for block_idx, window in src.block_windows(bidx) if rasterio.windows.intersect(self.window_focus, window)]

        return windows_return

    def copy(self) -> '__class__':
        return self.__copy__()

    def load(self, boundless:bool=True) -> geotensor.GeoTensor:
        """
        Load all raster data into memory as a GeoTensor.

        Reads the data from disk/cloud and returns an in-memory GeoTensor with
        full geospatial metadata. This is the main method for converting a lazy
        RasterioReader into a materialized array for computation.

        Args:
            boundless (bool, optional): If True, out-of-bounds regions are filled
                with `fill_value_default`. If False, the output is clipped to the
                valid data extent. Defaults to True.

        Returns:
            GeoTensor: In-memory array with transform, CRS, and fill value.

        Examples:
            Load a full raster::

                >>> reader = RasterioReader("image.tif")
                >>> gt = reader.load()
                >>> print(type(gt))
                <class 'georeader.geotensor.GeoTensor'>
                >>> print(gt.shape)
                (4, 1000, 1000)

            Load with window focus::

                >>> reader = RasterioReader("image.tif")
                >>> reader.set_window(rasterio.windows.Window(0, 0, 512, 512))
                >>> gt = reader.load()
                >>> print(gt.shape)  # Only reads the focused region
                (4, 512, 512)

            Load time series::

                >>> reader = RasterioReader(["jan.tif", "feb.tif"], stack=True)
                >>> gt = reader.load()
                >>> print(gt.shape)  # (T, C, H, W)
                (2, 4, 1000, 1000)

            Bounded vs boundless reads::

                >>> # Window extends beyond raster edge
                >>> reader = RasterioReader("image.tif")  # 1000x1000
                >>> reader.set_window(rasterio.windows.Window(-100, -100, 500, 500))
                >>>
                >>> gt_boundless = reader.load(boundless=True)
                >>> print(gt_boundless.shape)  # Full requested size, padded
                (4, 500, 500)
                >>>
                >>> gt_bounded = reader.load(boundless=False)
                >>> print(gt_bounded.shape)  # Clipped to valid extent
                (4, 400, 400)

        Note:
            - For large files, consider using `read_from_window()` for partial reads
            - Memory usage equals array size in dtype
            - The returned GeoTensor can be used for in-memory operations

        See Also:
            read: Lower-level read returning raw numpy array.
            read_from_window: Create reader for subset without loading.
            georeader.geotensor.GeoTensor: The in-memory array class.
        """
        np_data = self.read(boundless=boundless)
        if boundless:
            transform = self.transform
        else:
            # update transform, shape and coords
            window = self.window_focus
            start_col = max(window.col_off, 0)
            end_col = min(window.col_off + window.width, self.real_width)
            start_row = max(window.row_off, 0)
            end_row = min(window.row_off + window.height, self.real_height)
            spatial_shape = (end_row - start_row, end_col - start_col)
            assert np_data.shape[-2:] == spatial_shape, f"Different shapes {np_data.shape[-2:]} {spatial_shape}"

            window_real = rasterio.windows.Window(row_off=start_row, col_off=start_col,
                                                  width=spatial_shape[1], height=spatial_shape[0])
            transform = rasterio.windows.transform(window_real, self.real_transform)

        return geotensor.GeoTensor(np_data, transform=transform, crs=self.crs, fill_value_default=self.fill_value_default)

    @property
    def values(self) -> np.ndarray:
        """
        Load raster data as numpy array (xarray compatibility).

        Property for xarray DataArray compatibility. Equivalent to `read()`.
        Useful when treating RasterioReader like an xarray object.

        Returns:
            np.ndarray: Full raster loaded in memory with shape depending on
                `stack` setting: (T, C, H, W) if stacked, (T*C, H, W) otherwise.

        Examples:
            Access like xarray::

                >>> reader = RasterioReader("image.tif")
                >>> data = reader.values
                >>> print(data.shape)
                (4, 1000, 1000)

        Note:
            Loads entire raster into memory. For large files, consider
            using `read_from_window()` or `isel()` for partial reads.

        See Also:
            read: Equivalent method with optional parameters.
            load: Returns GeoTensor with geospatial metadata.
        """
        return self.read()

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        """
        Get raster footprint as a Shapely polygon.

        Returns the geographic extent of the current window focus as a
        polygon. Can optionally reproject to a different CRS.

        Args:
            crs (Optional[str], optional): Target CRS for the footprint. If None,
                returns polygon in raster's native CRS. Defaults to None.

        Returns:
            Polygon: Shapely polygon representing the raster's spatial extent.

        Examples:
            Get footprint in native CRS::

                >>> reader = RasterioReader("image.tif")
                >>> fp = reader.footprint()
                >>> print(fp.bounds)  # (minx, miny, maxx, maxy)
                (600000.0, 4000000.0, 610000.0, 4010000.0)

            Get footprint in WGS84::

                >>> fp_wgs84 = reader.footprint(crs="EPSG:4326")
                >>> print(fp_wgs84.bounds)
                (-3.5, 36.1, -3.4, 36.2)

            Use with geopandas::

                >>> import geopandas as gpd
                >>> fp = reader.footprint()
                >>> gdf = gpd.GeoDataFrame(geometry=[fp], crs=reader.crs)

        See Also:
            bounds: Get bounds as tuple instead of polygon.
            georeader.window_utils.window_polygon: Underlying function.
        """
        pol = window_utils.window_polygon(rasterio.windows.Window(row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]),
                                          self.transform)
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)
    
    def meshgrid(self, dst_crs:Optional[Any]=None) -> Tuple[NDArray, NDArray]:
        from georeader import griddata
        return griddata.meshgrid(self.transform, self.width, self.height, source_crs=self.crs, dst_crs=dst_crs)
    
    def __repr__(self) -> str:
        # Continuation indent aligns multi-line Affine repr under the
        # value column (9-space indent + 18-char label + ": ").
        transform_indent = "\n" + " " * 29
        transform_str = transform_indent.join(str(self.transform).splitlines())
        paths = [mask_sas_token(p) for p in self.paths]
        return (
            "\n"
            f"         Paths:              {paths}\n"
            f"         Shape:              {self.shape}\n"
            f"         Resolution:         {self.res}\n"
            f"         Bounds:             {self.bounds}\n"
            f"         CRS:                {self.crs}\n"
            f"         nodata:             {self.nodata}\n"
            f"         fill_value_default: {self.fill_value_default}\n"
            f"         Transform:          {transform_str}\n"
        )

    def read(self, **kwargs) -> np.ndarray:
        """
        Read raw pixel data from the raster files.

        Low-level method that reads data as a numpy array without geospatial
        metadata. Opens files fresh each call (process-safe). All windows are
        relative to the current `window_focus`.

        Args:
            **kwargs: Keyword arguments passed to rasterio's read method:
                - window (rasterio.windows.Window): Read window relative to
                    `window_focus`. Defaults to full `window_focus`.
                - boundless (bool): Allow out-of-bounds reads. Defaults to True.
                - fill_value (float): Fill value for boundless. Defaults to
                    `fill_value_default`.
                - indexes (List[int]): Band indices (1-based, relative to current
                    `indexes`). Defaults to all selected bands.
                - out_shape (Tuple[int, int]): Resample to target (H, W). None
                    preserves original resolution.
                - resampling (Resampling): Resampling method for `out_shape`.

        Returns:
            np.ndarray: Array of shape (T, C, H, W) if `stack=True`, else (T*C, H, W).
                Returns None if `boundless=False` and window doesn't intersect raster.

        Examples:
            Read full raster::

                >>> reader = RasterioReader("image.tif")
                >>> data = reader.read()
                >>> print(data.shape)
                (4, 1000, 1000)

            Read specific window::

                >>> window = rasterio.windows.Window(0, 0, 256, 256)
                >>> data = reader.read(window=window)
                >>> print(data.shape)
                (4, 256, 256)

            Read with resampling::

                >>> data = reader.read(out_shape=(512, 512))
                >>> print(data.shape)
                (4, 512, 512)

            Read specific bands::

                >>> # Read only first 2 bands (1-based, relative to selected)
                >>> data = reader.read(indexes=[1, 2])
                >>> print(data.shape)
                (2, 1000, 1000)

        Note:
            - Opens/closes file each call (safe for multiprocessing)
            - Windows are relative to `window_focus`, not full raster
            - For geospatial metadata, use `load()` instead
            - See rasterio API for full kwargs documentation

        See Also:
            load: Returns GeoTensor with geospatial metadata.
            read_from_window: Create new reader focused on window.
        """

        if ("window" in kwargs) and kwargs["window"] is not None:
            window_read = kwargs["window"]
            if isinstance(window_read, tuple):
                window_read = rasterio.windows.Window.from_slices(*window_read,
                                                                  boundless=kwargs.get("boundless", True))

            # Windows are relative to the windows_focus window.
            window = rasterio.windows.Window(col_off=window_read.col_off + self.window_focus.col_off,
                                             row_off=window_read.row_off + self.window_focus.row_off,
                                             height=window_read.height, width=window_read.width)
        else:
            window = self.window_focus

        kwargs["window"] = window

        if "boundless" not in kwargs:
            kwargs["boundless"] = True

        if not rasterio.windows.intersect([self.real_window, window]) and not kwargs["boundless"]:
            return None

        if not kwargs["boundless"]:
            window = window.intersection(self.real_window)

        if "fill_value" not in kwargs:
            kwargs["fill_value"] = self.fill_value_default

        if  kwargs.get("indexes", None) is not None:
            # Indexes are relative to the self.indexes window.
            indexes = kwargs["indexes"]
            if isinstance(indexes, numbers.Number):
                n_bands_read = 1
                kwargs["indexes"] = [self.indexes[kwargs["indexes"] - 1]]
                flat_channels = True
            else:
                n_bands_read = len(indexes)
                kwargs["indexes"] = [self.indexes[i - 1] for i in kwargs["indexes"]]
                flat_channels = False
        else:
            kwargs["indexes"] = self.indexes
            n_bands_read = self.count
            flat_channels = False

        if kwargs.get("out_shape", None) is not None:
            if len(kwargs["out_shape"]) == 2:
                kwargs["out_shape"] = (n_bands_read, ) + kwargs["out_shape"]
            elif len(kwargs["out_shape"]) == 3:
                assert kwargs["out_shape"][0] == n_bands_read, f"Expected to read {n_bands_read} but found out_shape: {kwargs['out_shape']}"
            else:
                raise NotImplementedError(f"Expected out_shape of len 2 or 3 found out_shape: {kwargs['out_shape']}")
            spatial_shape = kwargs["out_shape"][1:]
        else:
            spatial_shape = (window.height, window.width)

        shape = (len(self.paths), n_bands_read) + spatial_shape

        obj_out = np.full(shape, kwargs["fill_value"], dtype=self.dtype)
        if rasterio.windows.intersect([self.real_window, window]):
            pad = None
            if kwargs["boundless"]:
                slice_, pad = get_slice_pad(self.real_window, window)
                need_pad = any(x != 0 for x in pad["x"] + pad["y"])

                #  read and pad instead of using boundless attribute when transform is not rectilinear (otherwise rasterio fails!)
                if (abs(self.real_transform.b) > 1e-6) or (abs(self.real_transform.d) > 1e-6):
                    if need_pad:
                        assert kwargs.get("out_shape", None) is None, "out_shape not compatible with boundless and non rectilinear transform!"
                        kwargs["window"] = rasterio.windows.Window.from_slices(slice_["y"], slice_["x"])
                        kwargs["boundless"] = False
                    else:
                        kwargs["boundless"] = False
                else:
                    #  if transform is rectilinear read boundless if needed
                    kwargs["boundless"] = need_pad
                    pad = None

            for i, p in enumerate(self.paths):
                with rasterio.Env(**self._get_rio_options_path(p)):
                    with rasterio.open(p, "r", overview_level=self.overview_level,
                                       **self._resolve_open_kwargs()) as src:
                    # rasterio.read API: https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader.read
                        read_data = src.read(**kwargs)

                        # Add pad when reading
                        if pad is not None and need_pad:
                            slice_y = slice(pad["y"][0], -pad["y"][1] if pad["y"][1] !=0 else None)
                            slice_x = slice(pad["x"][0], -pad["x"][1] if pad["x"][1] !=0 else None)
                            obj_out[i, :, slice_y, slice_x] = read_data
                        else:
                            obj_out[i] = read_data
                        # pad_list_np = _get_pad_list(pad)
                    #
                    # read_data = np.pad(read_data, tuple(pad_list_np), mode="constant",
                    #                    constant_values=self.fill_value_default)



        if flat_channels:
            obj_out = obj_out[:, 0]

        if not self.stack:
            if obj_out.shape[0] == 1:
                obj_out = obj_out[0]
            else:
                obj_out = np.concatenate([obj_out[i] for i in range(obj_out.shape[0])],
                                         axis=0)

        return obj_out
    
    def read_from_tile(self, x:int, y:int, z:int, 
                       out_shape:Tuple[int,int]=(SIZE_DEFAULT, SIZE_DEFAULT),
                       dst_crs:Optional[Any]=WEB_MERCATOR_CRS) -> geotensor.GeoTensor:
        """
        Read a web mercator tile from a raster.
        
        Tiles are TMS tiles defined as: (https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)

        Args:
            x (int): x coordinate of the tile in the TMS system.
            y (int): y coordinate of the tile in the TMS system.
            z (int): z coordinate of the tile in the TMS system.
            out_shape (Tuple[int, int]): size of the tile to read. Defaults to (read.SIZE_DEFAULT, read.SIZE_DEFAULT).
            dst_crs (Optional[Any], optional): CRS of the output tile. Defaults to read.WEB_MERCATOR_CRS.
            
        Returns:
            geotensor.GeoTensor: geotensor with the tile data.
        """
        window = window_from_tile(self, x, y, z)
        window = window_utils.round_outer_window(window)
        data = read_out_shape(self, out_shape=out_shape, window=window)

        if window_utils.compare_crs(self.crs, dst_crs):
            return data
        
        # window = window_utils.pad_window(window, (1, 1))
        # data = read_out_shape(self, out_shape=size_out, window=window)

        return read_from_tile(data, x, y, z, dst_crs=dst_crs, out_shape=out_shape)
        

def _get_pad_list(pad_width:Dict[str,Tuple[int,int]]):
    pad_list_np = [(0, 0)]
    for k in ["y", "x"]:
        if k in pad_width:
            pad_list_np.append(pad_width[k])
        else:
            pad_list_np.append((0, 0))
    return pad_list_np


def read_out_shape(reader:Union[RasterioReader, rasterio.DatasetReader],
                   size_read:Optional[int]=None,
                   indexes:Optional[Union[List[int], int]]=None,
                   window:Optional[rasterio.windows.Window]=None,
                   out_shape:Optional[Tuple[int, int]]=None,
                   fill_value_default:int=0) -> geotensor.GeoTensor:
    """
    Read raster with resampling to target shape.

    Reads data using rasterio's `out_shape` parameter for efficient resampling.
    When reading COGs, this leverages internal overviews for faster reads at
    reduced resolution. Returns a GeoTensor with correct geospatial metadata
    accounting for the resampling.

    Args:
        reader (Union[RasterioReader, rasterio.DatasetReader]): Source raster
            to read from. Can be a RasterioReader or native rasterio dataset.
        size_read (Optional[int], optional): Target size for the longest dimension.
            Aspect ratio is preserved. Ignored if `out_shape` is provided.
            Defaults to None.
        indexes (Optional[Union[List[int], int]], optional): Band indices to read
            (1-based). None reads all bands. Defaults to None.
        window (Optional[rasterio.windows.Window], optional): Window to read from.
            None reads full extent. Defaults to None.
        out_shape (Optional[Tuple[int, int]], optional): Target (height, width)
            for resampling. Overrides `size_read`. Defaults to None.
        fill_value_default (int, optional): Fill value when nodata is not set
            in the source. Defaults to 0.

    Returns:
        GeoTensor: Resampled data with updated transform reflecting new resolution.

    Raises:
        AssertionError: If both `out_shape` and `size_read` are None.

    Examples:
        Read at reduced resolution for preview::

            >>> reader = RasterioReader("large_image.tif")
            >>> print(reader.shape)  # (4, 10000, 10000)
            (4, 10000, 10000)
            >>>
            >>> # Read with max dimension = 512 pixels
            >>> preview = read_out_shape(reader, size_read=512)
            >>> print(preview.shape)
            (4, 512, 512)

        Read with explicit output shape::

            >>> gt = read_out_shape(reader, out_shape=(256, 256))
            >>> print(gt.shape)
            (4, 256, 256)

        Read specific bands resampled::

            >>> rgb = read_out_shape(reader, size_read=1024, indexes=[1, 2, 3])
            >>> print(rgb.shape)
            (3, 1024, 1024)

        Read window at reduced resolution::

            >>> window = rasterio.windows.Window(0, 0, 5000, 5000)
            >>> subset = read_out_shape(reader, window=window, size_read=256)
            >>> print(subset.shape)
            (4, 256, 256)

    Note:
        - Transform is automatically adjusted for the resampled resolution
        - For COGs, this efficiently uses internal overviews
        - Resolution scaling is uniform (same factor for x and y)

    See Also:
        RasterioReader.read: Low-level read with out_shape parameter.
        get_out_shape: Calculate aspect-preserving output shape.
    """

    if window is None:
        shape = reader.shape[-2:]
    else:
        shape = window.height, window.width

    if out_shape is None:
        assert size_read is not None, f"Both out_shape and size_read are None"
        out_shape = get_out_shape(shape, size_read)
    else:
        assert len(out_shape) == 2, f"Expected 2 dimensions found {out_shape}"

    transform = reader.transform if window is None else rasterio.windows.transform(window, reader.transform)

    if (indexes is not None) and isinstance(indexes, (list, tuple)):
        if len(out_shape) == 2:
            out_shape = (len(indexes),) + out_shape
    
    input_output_factor = (shape[0] / out_shape[-2], shape[1] / out_shape[-1])    
    transform = transform * rasterio.Affine.scale(input_output_factor[1], input_output_factor[0])

    output = reader.read(indexes=indexes, out_shape=out_shape, window=window)

    return geotensor.GeoTensor(output, transform=transform,
                               crs=reader.crs, fill_value_default=getattr(reader, "fill_value_default",
                                                                          reader.nodata if reader.nodata else fill_value_default))




def get_out_shape(shape:Tuple[int, int], size_read:int) -> Tuple[int, int]:
    """
    Calculate aspect-preserving output shape.

    Given an input shape and target size for the longest dimension, computes
    the output shape that preserves the aspect ratio. If the target size is
    larger than both dimensions, returns None (no resampling needed).

    Args:
        shape (Tuple[int, int]): Input (height, width) dimensions.
        size_read (int): Target size for the longest dimension.

    Returns:
        Tuple[int, int]: Output (height, width), or None if no resize needed.

    Examples:
        Square image::

            >>> get_out_shape((1000, 1000), 512)
            (512, 512)

        Landscape image::

            >>> get_out_shape((1000, 2000), 512)
            (256, 512)

        Portrait image::

            >>> get_out_shape((2000, 1000), 512)
            (512, 256)

        Target larger than input::

            >>> get_out_shape((400, 300), 512)
            None  # No resize needed

    See Also:
        read_out_shape: Uses this to compute resampled reads.
    """
    if (size_read >= shape[0]) and (size_read >= shape[1]):
        out_shape = None
    elif shape[0] > shape[1]:
        out_shape = (size_read, int(round(shape[1] / shape[0] * size_read)))
    else:
        out_shape = (int(round(shape[0] / shape[1] * size_read)), size_read)
    return out_shape


def needs_boundless(window_data:rasterio.windows.Window,
                    window_read:rasterio.windows.Window) -> bool:
    """
    Check if a read window requires boundless mode.

    Determines whether reading `window_read` from a raster with extent
    `window_data` requires boundless reading (i.e., the read window extends
    beyond the valid data extent).

    Args:
        window_data (rasterio.windows.Window): Window representing the valid
            data extent of the raster.
        window_read (rasterio.windows.Window): Window to be read.

    Returns:
        bool: True if boundless read is needed (window_read extends beyond
            window_data), False otherwise.

    Examples:
        Window within bounds::

            >>> data_window = rasterio.windows.Window(0, 0, 1000, 1000)
            >>> read_window = rasterio.windows.Window(100, 100, 500, 500)
            >>> needs_boundless(data_window, read_window)
            False

        Window extends beyond bounds::

            >>> data_window = rasterio.windows.Window(0, 0, 1000, 1000)
            >>> read_window = rasterio.windows.Window(900, 900, 200, 200)
            >>> needs_boundless(data_window, read_window)
            True

        Window completely outside::

            >>> data_window = rasterio.windows.Window(0, 0, 1000, 1000)
            >>> read_window = rasterio.windows.Window(2000, 2000, 100, 100)
            >>> needs_boundless(data_window, read_window)
            True

    See Also:
        get_slice_pad: Get slice and padding for boundless reads.
    """
    try:
        slice_, pad = get_slice_pad(window_data, window_read)
        return any(x != 0 for x in pad["x"]+pad["y"])

    except rasterio.windows.WindowError:
        return True
