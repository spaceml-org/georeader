"""Carbon Mapper L2B scene raster wrapper.

:class:`CMImageRaster` exposes every loadable L2B scene asset
(``cmf`` / ``cmf-unortho`` / ``uncertainty`` /
``uncertainty-unortho`` / ``artifact-mask`` / ``rgb`` / ``uas``) as
lazy properties backed by :class:`~georeader.rasterio_reader.RasterioReader`
(or plain text for the ``uas.txt`` sidecar).

Per-plume L3A products (mask, concentrations, IME-clipped
concentrations, RGB, outline) live in
:mod:`~georeader.readers.carbonmapper.image` —
:class:`~georeader.readers.carbonmapper.image.CMPlumeImage` is the
counterpart to this class for plume-level data.

Intentionally NOT wrapped:

- PNG assets (``rgb_png`` etc.) — un-georeferenced, not COGs.
- Per-plume ``con_tif`` from the catalog REST surface — duplicates
  the column-density crop already provided by ``CMPlumeImage``.

Pure raster wrappers — no DB binding, no blob upload. The DB-bound
classes (``CarbonMapperTile``, ``CarbonMapperLocationImage``) and the
analyst notebooks consume them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Optional, cast

import requests
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from georeader import read
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader

# Lazy reads return a windowed `GeoData` (georeader's protocol — covers
# both ``RasterioReader`` and ``GeoTensor``).

from georeader.readers.carbonmapper.api_queries import CMTileItem

BBox = tuple[float, float, float, float]   # (W, S, E, N) in WGS-84
PathLike = str | Path


# ─────────────────────────────────────────────────────────────────────
#  L2B scene raster
# ─────────────────────────────────────────────────────────────────────

#: L2B asset keys exposed as lazy properties on :class:`CMImageRaster`.
#: Includes the orthorectified retrievals (``cmf`` / ``uncertainty``),
#: the un-orthorectified raw-frame variants, the ``artifact-mask``
#: anomaly flag layer, and the ``rgb`` true-colour sibling.
#: ``uas`` (text sidecar) is intentionally not in this tuple — it's
#: surfaced via the :attr:`CMImageRaster.uas` property as a plain
#: string, not as a raster band.
CM_L2B_BANDS: tuple[str, ...] = (
    "cmf", "cmf-unortho",
    "uncertainty", "uncertainty-unortho",
    "artifact-mask", "rgb",
)

#: Asset keys retained on :class:`CMImageRaster.asset_paths` —
#: the raster bands above plus the ``uas`` sidecar.
_CM_L2B_KEYS_ALL: tuple[str, ...] = CM_L2B_BANDS + ("uas",)

#: Default STAC collection for the L2B RGB sibling raster. Carbon Mapper
#: publishes the surface RGB and the CH4 retrieval as **separate STAC
#: collections** with matching ``scene_id`` and matching pixel grids:
#: ``l2b-ch4-mfa-v3a`` carries ``cmf`` / ``uncertainty`` /
#: ``artifact-mask``; ``l2b-rgb-v3a`` carries ``rgb``.
DEFAULT_L2B_RGB_COLLECTION = "l2b-rgb-v3a"


@dataclass(repr=False)
class CMImageRaster:
    """L2B scene exposed as four georeader-backed rasters.

    Lazy: instantiating the dataclass does NOT issue HTTP / blob reads;
    access ``.cmf`` / ``.rgb`` / etc. or call :meth:`read_window` /
    :meth:`read_polygon` to trigger I/O.

    Attributes:
        scene_id: CM L2B item id (e.g. ``"tan20251212t185057c20s4001"``).
        asset_paths: Mapping of band name → URL (``https://``) or
            local / blob path. ``artifact-mask`` may be missing — the
            accessor returns ``None``.
        overview_level: Forwarded to ``RasterioReader``. ``None`` for
            full resolution; integer for COG overviews (faster previews).
    """

    scene_id: str
    asset_paths: Mapping[str, PathLike]
    overview_level: Optional[int] = None

    # ---- Constructors --------------------------------------------------

    @classmethod
    def from_cm_tile_item(cls, item: CMTileItem) -> "CMImageRaster":
        """Build from the lightweight STAC item (Phase 0.2).

        STAC asset keys carry file extensions (``cmf.tif``,
        ``uncertainty.tif``, ``artifact-mask.tif``, ``uas.txt``,
        ``*-unortho.tif`` variants). This method strips the
        appropriate extension and retains every key listed in
        :data:`CM_L2B_BANDS` plus ``uas`` (the text sidecar).
        """
        paths: dict[str, PathLike] = {}
        for key, url in item.asset_urls.items():
            if not url:
                continue
            if key.endswith(".tif"):
                stripped = key[:-4]
            elif key.endswith(".txt"):
                stripped = key[:-4]
            else:
                stripped = key
            if stripped in _CM_L2B_KEYS_ALL:
                paths[stripped] = url
        return cls(scene_id=item.scene_id, asset_paths=paths)

    def with_rgb(self, rgb_item: CMTileItem) -> "CMImageRaster":
        """Return a copy with ``rgb`` merged in from a sibling STAC item.

        The CH4 (``l2b-ch4-mfa-v3a``) and RGB (``l2b-rgb-v3a``) L2B
        collections share ``scene_id`` and pixel grid, but each STAC
        item only exposes its own assets. Fetch both with
        :func:`api_queries.get_tile` (passing ``collection=...``) and
        compose them via this method:

        >>> ir = CMImageRaster.from_cm_tile_item(ch4_item)
        >>> ir = ir.with_rgb(rgb_item)
        >>> ir.rgb is not None
        True

        Raises:
            ValueError: If ``rgb_item.scene_id`` doesn't match
                ``self.scene_id`` (mismatched scenes don't share a grid
                — usually a programming error).
        """
        if rgb_item.scene_id != self.scene_id:
            raise ValueError(
                f"scene_id mismatch: {self.scene_id!r} vs {rgb_item.scene_id!r}"
            )
        # Pick the rgb GeoTIFF (with or without `.tif` extension);
        # ignore everything else on the rgb item.
        new_paths = dict(self.asset_paths)
        for key, url in rgb_item.asset_urls.items():
            if not url:
                continue
            stripped = key[:-4] if key.endswith(".tif") else key
            if stripped == "rgb":
                new_paths["rgb"] = url
                break
        return CMImageRaster(
            scene_id=self.scene_id,
            asset_paths=new_paths,
            overview_level=self.overview_level,
        )

    @classmethod
    def from_local(cls, scene_dir: PathLike) -> "CMImageRaster":
        """Build from a downloaded scene directory.

        Picks up every L2B asset present (``cmf.tif`` / ``rgb.tif`` /
        ``uncertainty.tif`` / ``artifact-mask.tif`` and the
        un-orthorectified variants), plus the ``uas.txt`` sidecar.
        Missing files become absent keys in ``asset_paths``.
        """
        d = Path(scene_dir)
        paths: dict[str, PathLike] = {}
        for band in CM_L2B_BANDS:
            p = d / f"{band}.tif"
            if p.exists():
                paths[band] = str(p)
        uas_path = d / "uas.txt"
        if uas_path.exists():
            paths["uas"] = str(uas_path)
        return cls(scene_id=d.name, asset_paths=paths)

    # ---- Lazy band readers --------------------------------------------

    @cached_property
    def cmf(self) -> RasterioReader:
        """CH4 matched-filter retrieval, orthorectified (ppm·m).
        Always present on L2B-CH4 items."""
        return self._open("cmf")

    @cached_property
    def cmf_unortho(self) -> Optional[RasterioReader]:
        """CH4 retrieval in raw sensor frame (pre-orthorectification).
        ``None`` for older collection variants (e.g. ``mfm-v1``) that
        don't ship the unortho sibling."""
        return self._open_optional("cmf-unortho")

    @cached_property
    def rgb(self) -> Optional[RasterioReader]:
        """3-band uint8 RGB. ``None`` for L2B-CH4 collections (RGB lives
        in a separate STAC collection — fetch and pass via
        ``asset_paths`` or compose via :meth:`with_rgb`)."""
        return self._open_optional("rgb")

    @cached_property
    def uncertainty(self) -> RasterioReader:
        """Companion uncertainty raster aligned with ``cmf``."""
        return self._open("uncertainty")

    @cached_property
    def uncertainty_unortho(self) -> Optional[RasterioReader]:
        """Per-pixel uncertainty in raw sensor frame. ``None`` for
        older collection variants without the unortho sibling."""
        return self._open_optional("uncertainty-unortho")

    @cached_property
    def artifact_mask(self) -> Optional[RasterioReader]:
        """Artefact mask (covers ~25% of scene). Flags un-orthorectified
        strip pixels and geometric anomalies — **not** a cloud mask.
        ``None`` if absent."""
        return self._open_optional("artifact-mask")

    @cached_property
    def uas(self) -> Optional[str]:
        """UAS sensor-metadata sidecar — raw text from ``uas.txt``.

        Lazy-fetched on first access (one HTTP GET if the path is a
        URL, or a file read for local paths) and cached as a string.
        Callers parse the structure as needed; we don't impose a
        schema. Returns ``None`` if no ``uas`` URL/path was supplied.

        Auth: rasterio's curl session is configured via the
        ``GDAL_HTTP_HEADERS`` env var (set by the standard reader
        bootstrap). We re-use that header here so a single
        ``Authorization: Bearer <token>`` setup applies to every
        L2B asset, raster or text alike.
        """
        path = self.asset_paths.get("uas")
        if path is None:
            return None
        sp = str(path)
        if sp.startswith(("http://", "https://")):
            headers: dict[str, str] = {}
            gdal_hdr = os.environ.get("GDAL_HTTP_HEADERS", "")
            if gdal_hdr.lower().startswith("authorization:"):
                headers["Authorization"] = gdal_hdr.split(":", 1)[1].strip()
            r = requests.get(sp, headers=headers, timeout=30)
            r.raise_for_status()
            return r.text
        with open(sp, "r") as fh:
            return fh.read()

    # ---- Geometric metadata (pulled from cmf as the canonical band) ---

    @property
    def crs(self) -> str:
        return str(self.cmf.crs)

    @property
    def transform(self):
        return self.cmf.transform

    @property
    def bounds(self) -> BBox:
        b = self.cmf.bounds
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.cmf.height, self.cmf.width)

    # ---- Read helpers (delegate to georeader.read) --------------------

    def read_polygon(
        self,
        polygon: BaseGeometry,
        *,
        crs_polygon: str = "EPSG:4326",
        bands: Iterable[str] = CM_L2B_BANDS,
    ) -> dict[str, Optional[GeoData]]:
        """Read a polygon clip from the requested bands.

        Args:
            polygon: Clip geometry.
            crs_polygon: CRS of ``polygon``. Defaults to ``"EPSG:4326"``.
            bands: Subset of band names. Bands whose asset is missing
                or whose window has zero overlap return ``None``.

        Returns:
            ``{"cmf": <GeoData>, "rgb": <GeoData>, ...}`` — windowed
            ``RasterioReader`` instances (lazy, satisfying the
            :class:`GeoData` protocol). Call ``.load()`` to materialise
            as :class:`GeoTensor`.
        """
        out: dict[str, Optional[GeoData]] = {}
        for band in bands:
            if self.asset_paths.get(band) is None:
                out[band] = None
                continue
            # `uas` is a text sidecar, not a raster — skip the band
            # reader path. Callers reading text sidecars use the
            # `.uas` property directly.
            if band == "uas":
                continue
            reader = self._open(band)
            # `boundless=False` makes `read_from_polygon` return `None`
            # for windows that don't intersect the raster (e.g. an
            # artifact-mask whose un-orthorectified strip falls outside
            # the requested AOI), instead of allocating a fill-valued
            # tensor the size of the requested window. Real CRS / I/O
            # errors are left to propagate — the prior bare
            # `except Exception` swallowed those silently.
            #
            # `read_from_polygon` returns ``GeoData | NDArray``; with
            # ``return_only_data=False`` the GeoData arm is the one we
            # always hit. ``RasterioReader`` satisfies the ``GeoData``
            # protocol structurally, but ty doesn't currently infer
            # that — cast for clarity.
            result = read.read_from_polygon(
                cast(GeoData, reader),
                polygon=polygon,
                crs_polygon=crs_polygon,
                boundless=False,
            )
            out[band] = cast(GeoData, result) if result is not None else None
        return out

    def read_window(
        self,
        bounds_4326: BBox,
        *,
        bands: Iterable[str] = CM_L2B_BANDS,
    ) -> dict[str, Optional[GeoData]]:
        """Read a WGS-84 bbox window from the requested bands."""
        return self.read_polygon(box(*bounds_4326), bands=bands)

    def read_window_to_crs(
        self,
        bounds_4326: BBox,
        crs_dst: str,
        *,
        bands: Iterable[str] = CM_L2B_BANDS,
    ) -> dict[str, Optional[GeoTensor]]:
        """Read a window then reproject each band to ``crs_dst``.

        Reprojection materialises the data — values are
        :class:`GeoTensor`, not lazy readers.
        """
        crops = self.read_window(bounds_4326, bands=bands)
        # `read_to_crs` returns ``GeoTensor | NDArray``; same narrowing
        # rationale as ``read_from_polygon`` above.
        return {
            band: (
                cast(GeoTensor, read.read_to_crs(geo, crs_dst))
                if geo is not None
                else None
            )
            for band, geo in crops.items()
        }

    # ---- Internals -----------------------------------------------------

    def _open(self, band: str) -> RasterioReader:
        path = self.asset_paths.get(band)
        if path is None:
            raise KeyError(f"Asset {band!r} not present on {self.scene_id}")
        return RasterioReader(str(path), overview_level=self.overview_level)

    def _open_optional(self, band: str) -> Optional[RasterioReader]:
        if self.asset_paths.get(band) is None:
            return None
        return self._open(band)

    # ---- Repr ---------------------------------------------------------

    def __repr__(self) -> str:
        present = [b for b in CM_L2B_BANDS if b in self.asset_paths]
        missing = [b for b in CM_L2B_BANDS if b not in self.asset_paths]
        extra = sorted(set(self.asset_paths) - set(CM_L2B_BANDS))
        ov = self.overview_level if self.overview_level is not None else "full"
        lines = [
            "CMImageRaster",
            f"  scene_id:       {self.scene_id}",
            f"  bands present:  {present or '<none>'}",
        ]
        if missing:
            lines.append(f"  bands missing:  {missing}")
        if extra:
            lines.append(f"  extra keys:     {extra}")
        lines.append(f"  overview_level: {ov}")
        return "\n".join(lines)

    __str__ = __repr__


__all__ = [
    "BBox",
    "CM_L2B_BANDS",
    "CMImageRaster",
    "DEFAULT_L2B_RGB_COLLECTION",
]
