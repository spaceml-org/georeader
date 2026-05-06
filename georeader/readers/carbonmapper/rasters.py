"""Carbon Mapper raster wrappers backed by georeader.

Two product families — **GeoTIFF only**:

- :class:`CMImageRaster` — L2B scene (``cmf`` / ``rgb`` / ``uncertainty`` /
  ``artifact-mask``).
- :class:`CMPlumeRaster` — L3A per-plume mask (``plume_tif`` only).

Both expose lazy :class:`~georeader.rasterio_reader.RasterioReader`
instances per band and helpers that delegate to ``georeader.read``.

Intentionally NOT wrapped:

- PNG assets (``rgb_png`` / ``plume_png`` / ``plume_rgb_png``) —
  un-georeferenced; not COGs.
- Per-plume ``con_tif`` — duplicates a circular crop of the scene's
  ``cmf``. Crop ``CMImageRaster.cmf`` to the plume polygon to get
  the same data without the duplicate asset.

Pure raster wrappers — no DB binding, no blob upload. The DB-bound
classes (``CarbonMapperTile``, ``CarbonMapperLocationImage``) and the
analyst notebooks consume them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Optional, cast

import numpy as np
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from georeader import read, window_utils
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader

# Lazy reads return a windowed `GeoData` (georeader's protocol — covers
# both ``RasterioReader`` and ``GeoTensor``).

from georeader.readers.carbonmapper.api_queries import CMTileItem
from georeader.readers.carbonmapper.plume import CMRawPlume

BBox = tuple[float, float, float, float]   # (W, S, E, N) in WGS-84
PathLike = str | Path


# ─────────────────────────────────────────────────────────────────────
#  L2B scene raster
# ─────────────────────────────────────────────────────────────────────

CM_L2B_BANDS: tuple[str, ...] = ("cmf", "rgb", "uncertainty", "artifact-mask")

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
        ``*-unortho.tif`` variants). This method strips the ``.tif``
        extension on the canonical bands and ignores everything else
        (text sidecars, un-orthorectified variants).
        """
        paths: dict[str, PathLike] = {}
        for key, url in item.asset_urls.items():
            if not url:
                continue
            stripped = key[:-4] if key.endswith(".tif") else key
            if stripped in CM_L2B_BANDS:
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

        Looks for ``cmf.tif`` / ``rgb.tif`` / ``uncertainty.tif`` /
        ``artifact-mask.tif`` under ``scene_dir``; missing files become
        absent keys in ``asset_paths``.
        """
        d = Path(scene_dir)
        paths: dict[str, PathLike] = {}
        for band in CM_L2B_BANDS:
            p = d / f"{band}.tif"
            if p.exists():
                paths[band] = str(p)
        return cls(scene_id=d.name, asset_paths=paths)

    # ---- Lazy band readers --------------------------------------------

    @cached_property
    def cmf(self) -> RasterioReader:
        """CH4 matched-filter retrieval (ppm·m). Always present."""
        return self._open("cmf")

    @cached_property
    def rgb(self) -> Optional[RasterioReader]:
        """3-band uint8 RGB. ``None`` for L2B-CH4 collections (RGB lives
        in a separate STAC collection — fetch and pass via
        ``asset_paths`` if needed)."""
        return self._open_optional("rgb")

    @cached_property
    def uncertainty(self) -> RasterioReader:
        """Companion uncertainty raster aligned with ``cmf``."""
        return self._open("uncertainty")

    @cached_property
    def artifact_mask(self) -> Optional[RasterioReader]:
        """Optional artefact mask (covers ~25% of scene). ``None`` if absent."""
        return self._open_optional("artifact-mask")

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
            reader = self._open(band)
            try:
                # `read_from_polygon` returns ``GeoData | NDArray``;
                # with the default ``return_only_data=False`` the GeoData
                # arm is the one we always hit. ``RasterioReader``
                # satisfies the ``GeoData`` protocol structurally, but
                # ty doesn't currently infer that — cast for clarity.
                out[band] = cast(
                    GeoData,
                    read.read_from_polygon(
                        cast(GeoData, reader),
                        polygon=polygon,
                        crs_polygon=crs_polygon,
                    ),
                )
            except Exception:
                # Zero overlap (e.g., artifact-mask outside its strip) → None.
                out[band] = None
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


# ─────────────────────────────────────────────────────────────────────
#  L3A per-plume raster
# ─────────────────────────────────────────────────────────────────────


_PLUME_RASTER_KEYS: tuple[str, ...] = ("plume_tif", "ime_outline_geojson")


@dataclass(repr=False)
class CMPlumeRaster:
    """L3A per-plume GeoTIFF mask wrapper, backed by georeader.

    Wraps the single per-plume GeoTIFF asset that isn't already a
    redundant crop of the scene:

    - ``plume_tif`` — RGBA GeoTIFF; band 4 is the binary alpha mask.
    - ``ime_outline_geojson`` (optional) — plume polygon as GeoJSON;
      preferred over band-4 extraction when present.

    Out of scope (intentionally not exposed): ``con_tif`` (use
    :meth:`CMImageRaster.read_polygon` on the parent scene's ``cmf``
    instead), PNG siblings, ``rgb_tif`` (always ``None`` in live
    responses today).

    Attributes:
        plume_id: Carbon Mapper plume_id.
        urls: Mapping of asset name → URL or local path. Only
            ``plume_tif`` and (optionally) ``ime_outline_geojson`` are
            read; other keys are ignored.
        overview_level: Forwarded to RasterioReader.
    """

    plume_id: str
    urls: Mapping[str, PathLike]
    overview_level: Optional[int] = None

    # ---- Constructors --------------------------------------------------

    @classmethod
    def from_plume_dict(cls, plume_dict: dict) -> "CMPlumeRaster":
        """Build from a ``/catalog/plume/{id}`` response dict.

        Reads only ``plume_tif`` from the response. PNG and ``con_tif``
        keys present on the response are ignored. Does NOT pull
        ``ime_outline_geojson`` (lives on the L3A STAC item, not the
        catalog object); add it via :meth:`with_outline` if fetched
        separately.
        """
        urls = _filter_raster_keys(plume_dict)
        return cls(plume_id=str(plume_dict.get("plume_id", "")), urls=urls)

    @classmethod
    def from_cmrawplume(cls, raw: CMRawPlume) -> "CMPlumeRaster":
        """Build from the typed :class:`CMRawPlume` model.

        Reads ``raw.plume_tif`` only.
        """
        urls: dict[str, PathLike] = {}
        if raw.plume_tif:
            urls["plume_tif"] = raw.plume_tif
        return cls(plume_id=raw.plume_id, urls=urls)

    def with_outline(self, geojson_url: PathLike) -> "CMPlumeRaster":
        """Return a copy with ``ime_outline_geojson`` populated."""
        new_urls = dict(self.urls)
        new_urls["ime_outline_geojson"] = geojson_url
        return CMPlumeRaster(
            plume_id=self.plume_id,
            urls=new_urls,
            overview_level=self.overview_level,
        )

    # ---- Lazy raster accessor -----------------------------------------

    @cached_property
    def plume_tif(self) -> Optional[RasterioReader]:
        """Band 4 alpha = binary plume mask. ``None`` if URL absent."""
        url = self.urls.get("plume_tif")
        if url is None:
            return None
        return RasterioReader(str(url), overview_level=self.overview_level)

    # ---- Plume polygon — two sources of truth -------------------------

    def polygon_from_alpha(self) -> Optional[BaseGeometry]:
        """Extract the plume polygon from ``plume_tif`` band 4 alpha.

        Loads the alpha band via georeader, runs
        :func:`rasterio.features.shapes` on the boolean mask, dissolves,
        and reprojects to EPSG:4326. Returns ``None`` if ``plume_tif``
        is unavailable or yields no positive pixels.
        """
        reader = self.plume_tif
        if reader is None:
            return None
        geo = reader.load()
        arr = np.asarray(geo.values)
        if arr.ndim != 3 or arr.shape[0] < 4:
            return None
        alpha = arr[3]
        mask = (alpha > 0).astype("uint8")
        if not mask.any():
            return None

        import rasterio.features
        from shapely.geometry import shape
        from shapely.ops import unary_union

        polys = [
            shape(geom)
            for geom, val in rasterio.features.shapes(
                mask, mask=mask.astype(bool), transform=geo.transform,
            )
            if val == 1
        ]
        if not polys:
            return None
        merged = unary_union(polys)
        return window_utils.polygon_to_crs(
            merged, crs_polygon=str(geo.crs), dst_crs="EPSG:4326",
        )

    def polygon_from_geojson(self) -> Optional[BaseGeometry]:
        """Use ``ime_outline_geojson`` directly (preferred — no rasterio).

        Returns ``None`` if the outline URL was not provided.
        """
        url = self.urls.get("ime_outline_geojson")
        if url is None:
            return None

        import json
        from shapely.geometry import shape
        from shapely.ops import unary_union

        # Local file or remote — let rasterio's GDAL-vsicurl-trained env
        # not be involved here; just open as text via fsspec / urllib.
        path = str(url)
        if path.startswith(("http://", "https://")):
            import urllib.request
            with urllib.request.urlopen(path) as resp:
                data = json.load(resp)
        else:
            with open(path, "r") as fh:
                data = json.load(fh)

        if isinstance(data, dict) and data.get("type") == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in data.get("features", [])
                     if f.get("geometry")]
        elif isinstance(data, dict) and data.get("type") == "Feature":
            g = data.get("geometry")
            geoms = [shape(g)] if g else []
        elif isinstance(data, dict) and "type" in data and "coordinates" in data:
            geoms = [shape(data)]
        else:
            return None
        if not geoms:
            return None
        return unary_union(geoms)

    def polygon(self) -> Optional[BaseGeometry]:
        """Best-available polygon: outline GeoJSON if present, else alpha."""
        return self.polygon_from_geojson() or self.polygon_from_alpha()

    # ---- Direct load --------------------------------------------------

    # ---- Repr ---------------------------------------------------------

    def __repr__(self) -> str:
        has_tif = "plume_tif" in self.urls and self.urls["plume_tif"]
        has_outline = (
            "ime_outline_geojson" in self.urls
            and self.urls["ime_outline_geojson"]
        )
        ignored = sorted(set(self.urls) - set(_PLUME_RASTER_KEYS))
        ov = self.overview_level if self.overview_level is not None else "full"
        lines = [
            "CMPlumeRaster",
            f"  plume_id:       {self.plume_id}",
            f"  plume_tif:      {'present' if has_tif else 'absent'}",
            f"  ime_outline:    {'present' if has_outline else 'absent'}",
        ]
        if ignored:
            lines.append(f"  ignored keys:   {ignored}")
        lines.append(f"  overview_level: {ov}")
        return "\n".join(lines)

    __str__ = __repr__

    def load_alpha_mask(self) -> Optional[GeoTensor]:
        """Load just band 4 of ``plume_tif`` as a boolean GeoTensor."""
        reader = self.plume_tif
        if reader is None:
            return None
        geo = reader.load()
        arr = np.asarray(geo.values)
        if arr.ndim != 3 or arr.shape[0] < 4:
            return None
        return GeoTensor(
            values=(arr[3] > 0),
            transform=geo.transform,
            crs=geo.crs,
            fill_value_default=False,
        )


def _filter_raster_keys(d: Mapping) -> dict[str, PathLike]:
    """Pick only the GeoTIFF / GeoJSON keys this module cares about."""
    out: dict[str, PathLike] = {}
    for key in _PLUME_RASTER_KEYS:
        v = d.get(key)
        if v:
            out[key] = v
    return out


__all__ = [
    "BBox",
    "CM_L2B_BANDS",
    "CMImageRaster",
    "CMPlumeRaster",
    "DEFAULT_L2B_RGB_COLLECTION",
]
