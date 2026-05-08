"""Per-plume image bundle wrapper for Carbon Mapper L3A products.

One :class:`CMPlumeImage` is the cropped raster suite for one plume —
binary mask, full column-density crop, IME-clipped retrieval, RGB
context, and the canonical outline polygon. This is the per-plume
counterpart to :class:`~georeader.readers.carbonmapper.rasters.CMImageRaster`
(which wraps L2B scenes).

**CH4-only for this PR.** CO2 lands in a follow-up — the URL pattern
below hardcodes the CH4 collection segment swap.

Why this exists: Carbon Mapper exposes per-plume products under two
parallel collection families:

- **v3a** (``l3a-vis-ch4-mfa-v3a`` / ``l3a-ime-ch4-mfa-v3a``) —
  STAC-registered. Plumes from 2023-10 to 2025-12.
- **v3c** (``l3a-vis-ch4-mfa-v3c`` / ``l3a-ime-ch4-mfa-v3c``) —
  newest data (post 2026-01) but **not** registered in
  ``/stac/collections``. Reachable only via direct asset URLs from
  ``/catalog/plume/{id}``.

A STAC-only wrapper misses the last few months of data. This module
handles both: :meth:`CMPlumeImage.from_plume_id` derives all asset
URLs from the REST catalog response (which has signed CDN URLs for
either version) and rewrites them to the Bearer-aware api gateway
form so the URLs don't expire.

Outline GeoJSON is the canonical source for the plume polygon; if
the fetch fails (network / 404 / malformed body), we fall back to
vectorising the band-4 alpha of ``plume.tif`` and log a warning so
callers notice when they're not on the canonical path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

import numpy as np
import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from georeader import window_utils
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from georeader.readers.carbonmapper.plume import CMRawPlume, Collection

PathLike = str | Path
_log = logging.getLogger(__name__)


# Asset proxy base — Bearer-aware mirror of the signed CDN. Used by
# `_derive_asset_urls` to rewrite the host on URLs returned by
# `/catalog/plume/{id}` so the URL doesn't expire.
_API_ASSET_BASE = "https://api.carbonmapper.org/api/v1/catalog/asset"


# Asset keys this wrapper exposes — one per public property on
# `CMPlumeImage`. Kept as a constant for tests + introspection.
CM_PLUME_IMAGE_ASSETS: tuple[str, ...] = (
    "plume.tif",
    "plume-concentrations.tif",
    "plume-outline.geojson",
    "rgb.tif",
    "ime-cmf-concentrations.tif",
    "ime-cmf-mask.tif",
    "ime-cmf-outline.geojson",
)


# ─────────────────────────────────────────────────────────────────────
#  URL pattern derivation
# ─────────────────────────────────────────────────────────────────────


def _cdn_to_api(cdn_url: str) -> str:
    """Rewrite ``catalog.carbonmapper.org/<path>?<signed-query>`` →
    ``api.carbonmapper.org/api/v1/catalog/asset/<path>``.

    The signed-CDN form embeds a short-lived ``?Expires=...`` in the
    query and shouldn't be persisted; the api gateway form is
    Bearer-aware and stable. Drops the query string in the rewrite.
    """
    parsed = urlparse(cdn_url)
    return f"{_API_ASSET_BASE}{parsed.path}"


def _swap_collection_segment(api_url: str, *, src: str, dst: str) -> str:
    """Replace one collection segment (``l3a-vis-...`` → ``l3a-ime-...``)
    in both the path and the filename tail.

    The asset URL pattern is::

        <base>/<COLL>/<Y>/<M>/<D>/<plume_id>/<plume_id>_<COLL>_<asset>.<ext>

    so the collection ID appears twice — once as a path segment and
    once embedded in the filename. Swap both consistently.
    """
    return api_url.replace(src, dst)


def _derive_asset_urls(catalog_plume: Mapping[str, Any]) -> dict[str, str]:
    """From ``/catalog/plume/{id}`` response, derive ``{asset_name: url}``
    for every per-plume product :class:`CMPlumeImage` exposes.

    Translates signed CDN URLs to the api.carbonmapper.org gateway
    form so Bearer auth applies and the URL doesn't expire.

    Returns 7 keys (one per ``CMPlumeImage`` property):

    - ``plume.tif``                  → :attr:`CMPlumeImage.mask`
    - ``plume-concentrations.tif``   → :attr:`CMPlumeImage.concentrations`
    - ``plume-outline.geojson``      → :attr:`CMPlumeImage.outline` (canonical)
    - ``rgb.tif``                    → :attr:`CMPlumeImage.rgb`

    The remaining three live in the IME sibling collection and are
    derived by swapping ``l3a-vis-ch4-mfa-`` → ``l3a-ime-ch4-mfa-``:

    - ``ime-cmf-concentrations.tif`` → :attr:`CMPlumeImage.ime_concentrations`
    - ``ime-cmf-mask.tif``           → :attr:`CMPlumeImage.ime_mask`
    - ``ime-cmf-outline.geojson``    → :attr:`CMPlumeImage.ime_outline`

    Raises
    ------
    ValueError
        If ``catalog_plume`` is missing the ``plume_tif`` field that
        seeds the URL pattern. Other URL fields (``con_tif``,
        ``rgb_png``, etc.) on the response are ignored — we derive
        every asset from the same prefix to avoid version-mismatch
        bugs (e.g. ``plume_tif`` at v3c but ``con_tif`` at v3b).
    """
    seed = catalog_plume.get("plume_tif")
    if not seed:
        raise ValueError(
            "catalog_plume has no 'plume_tif' URL — can't derive asset "
            f"URLs for plume_id {catalog_plume.get('plume_id')!r}. "
            "Either the plume hasn't been processed yet or the "
            "catalog response is malformed."
        )

    # Rewrite host + strip query, then peel off the `_plume.tif` tail
    # to get the prefix shared by every vis asset.
    api_url = _cdn_to_api(seed)
    m = re.match(r"^(.+)_plume\.tif$", api_url)
    if m is None:
        raise ValueError(
            f"plume_tif URL {api_url!r} doesn't match the expected "
            f"`<prefix>_plume.tif` pattern."
        )
    vis_prefix = m.group(1)

    out: dict[str, str] = {
        "plume.tif":                f"{vis_prefix}_plume.tif",
        "plume-concentrations.tif": f"{vis_prefix}_plume-concentrations.tif",
        "plume-outline.geojson":    f"{vis_prefix}_plume-outline.geojson",
        "rgb.tif":                  f"{vis_prefix}_rgb.tif",
    }

    # IME sibling — same pattern but with the collection segment
    # swapped from `l3a-vis-ch4-mfa-<version>` to
    # `l3a-ime-ch4-mfa-<version>`. The collection ID appears twice
    # in the URL (once as a path segment, once embedded in the
    # filename tail), so swap both. Version-agnostic — works for
    # v3a, v3b, v3c and any future bump.
    if "l3a-vis-ch4-mfa-" in vis_prefix:
        ime_prefix = vis_prefix.replace(
            "l3a-vis-ch4-mfa-", "l3a-ime-ch4-mfa-",
        )
        out["ime-cmf-concentrations.tif"] = (
            f"{ime_prefix}_ime-cmf-concentrations.tif"
        )
        out["ime-cmf-mask.tif"] = f"{ime_prefix}_ime-cmf-mask.tif"
        out["ime-cmf-outline.geojson"] = (
            f"{ime_prefix}_ime-cmf-outline.geojson"
        )
    # If we didn't find the vis collection segment (legacy mf-v1 etc.
    # use a different prefix), skip the IME keys rather than guessing
    # — the IME properties return None in that case.

    return out


# ─────────────────────────────────────────────────────────────────────
#  CMPlumeImage
# ─────────────────────────────────────────────────────────────────────


@dataclass(repr=False)
class CMPlumeImage:
    """Per-plume L3A product bundle for one CH4 plume.

    Lazy: instantiating the dataclass does NOT issue HTTP / blob reads.
    Each property opens its asset on first access and caches the
    result.

    Attributes:
        plume_id: Carbon Mapper plume_id (e.g.
            ``"tan20251212t185057c20s4001-E"``).
        urls: Mapping of asset name → URL (Bearer-aware api gateway
            form preferred; signed CDN URLs work but expire). 5 keys
            expected: ``plume.tif``, ``plume-concentrations.tif``,
            ``plume-outline.geojson``, ``rgb.tif``,
            ``ime-cmf-concentrations.tif``. Missing keys → property
            returns ``None``.
        token: Bearer token required for the api-gateway URLs. May be
            ``None`` if ``urls`` already point at signed CDN URLs or
            local files.
        overview_level: Forwarded to :class:`RasterioReader` —
            ``None`` for full resolution, integer for COG overviews.
        http_timeout: Per-request timeout (seconds) for the outline
            GeoJSON fetch. Defaults to 30s.
    """

    plume_id: str
    urls: Mapping[str, str]
    token: Optional[str] = None
    overview_level: Optional[int] = None
    http_timeout: float = 30.0

    # ---- Constructors --------------------------------------------------

    @classmethod
    def from_plume_id(
        cls,
        plume_id: str,
        *,
        token: str,
        overview_level: Optional[int] = None,
        http_timeout: float = 30.0,
    ) -> CMPlumeImage:
        """Build by fetching ``/catalog/plume/{id}`` then deriving asset URLs.

        One round-trip. Handles v3a (STAC-resident) and v3c (CDN-only)
        plumes uniformly via :func:`_derive_asset_urls`.

        Raises:
            requests.HTTPError: On REST failure (404 etc.).
            ValueError: If the response lacks the ``plume_tif`` URL the
                derivation needs.
        """
        r = requests.get(
            f"https://api.carbonmapper.org/api/v1/catalog/plume/{plume_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=http_timeout,
        )
        r.raise_for_status()
        urls = _derive_asset_urls(r.json())
        return cls(
            plume_id=plume_id, urls=urls, token=token,
            overview_level=overview_level, http_timeout=http_timeout,
        )

    @classmethod
    def from_cmrawplume(
        cls,
        raw: CMRawPlume,
        *,
        token: str,
        overview_level: Optional[int] = None,
        http_timeout: float = 30.0,
    ) -> CMPlumeImage:
        """Build from an already-fetched typed plume.

        Useful when the caller is iterating over a list returned by
        ``api_queries.list_plumes(...)`` and wants the image bundle
        per plume without re-fetching the catalog dict. Derives URLs
        from the asset URLs the typed model already carries.
        """
        if not raw.plume_tif:
            raise ValueError(
                f"CMRawPlume {raw.plume_id!r} has no plume_tif URL — "
                f"can't derive asset URLs."
            )
        seed = {"plume_tif": raw.plume_tif, "plume_id": raw.plume_id}
        urls = _derive_asset_urls(seed)
        return cls(
            plume_id=raw.plume_id, urls=urls, token=token,
            overview_level=overview_level, http_timeout=http_timeout,
        )

    @classmethod
    def from_stac_item(
        cls,
        item: Mapping[str, Any],
        *,
        ime_item: Optional[Mapping[str, Any]] = None,
        token: Optional[str] = None,
        overview_level: Optional[int] = None,
        http_timeout: float = 30.0,
    ) -> CMPlumeImage:
        """Build from STAC items (vis + optional ime sibling).

        Use when the caller is driving STAC search directly. v3a-only
        — v3c plumes have no STAC items, so use :meth:`from_plume_id`
        for those. Pass both the vis and ime items if you have them
        (recommended); the ime sibling is required for the
        :attr:`ime_concentrations` property.
        """
        plume_id = str(item.get("id", ""))
        urls: dict[str, str] = {}
        for asset_key in (
            "plume.tif", "plume-concentrations.tif",
            "plume-outline.geojson", "rgb.tif",
        ):
            href = (item.get("assets") or {}).get(asset_key, {}).get("href")
            if href:
                urls[asset_key] = href
        if ime_item is not None:
            href = (ime_item.get("assets") or {}).get(
                "ime-cmf-concentrations.tif", {},
            ).get("href")
            if href:
                urls["ime-cmf-concentrations.tif"] = href
        return cls(
            plume_id=plume_id, urls=urls, token=token,
            overview_level=overview_level, http_timeout=http_timeout,
        )

    # ---- Lazy raster properties ---------------------------------------

    @cached_property
    def mask(self) -> Optional[RasterioReader]:
        """RGBA GeoTIFF — band 4 alpha is the binary plume mask.

        Backed by :class:`RasterioReader`. Returns ``None`` if
        ``plume.tif`` URL is absent.
        """
        return self._open_optional("plume.tif")

    @cached_property
    def concentrations(self) -> Optional[RasterioReader]:
        """Single-band CH4 column density (ppm·m), full per-plume crop.

        Every pixel in the plume window — including noise floor outside
        the mask. Use :attr:`ime_concentrations` for the IME-clipped
        version.
        """
        return self._open_optional("plume-concentrations.tif")

    @cached_property
    def ime_concentrations(self) -> Optional[RasterioReader]:
        """IME-clipped CH4 column density (ppm·m).

        Only mask-significant pixels — others are NoData. This is the
        raster used to compute the plume's ``emission_auto`` integral.
        Returns ``None`` if the IME sibling URL wasn't provided
        (older constructors / pre-v3 schemas).
        """
        return self._open_optional("ime-cmf-concentrations.tif")

    @cached_property
    def rgb(self) -> Optional[RasterioReader]:
        """3-band uint8 true-colour GeoTIFF (per-plume crop)."""
        return self._open_optional("rgb.tif")

    @cached_property
    def ime_mask(self) -> Optional[RasterioReader]:
        """Binary mask of pixels that contributed to ``emission_auto``.

        From ``ime-cmf-mask.tif`` — a tighter subset of
        :attr:`mask`'s band-4 alpha. Use this when you specifically
        need the IME-significant pixel set (e.g. for re-quantification
        with a different wind product) rather than the broader plume
        polygon.
        """
        return self._open_optional("ime-cmf-mask.tif")

    @cached_property
    def outline(self) -> Optional[BaseGeometry]:
        """Plume polygon in EPSG:4326 (broader — the full plume mask).

        Canonical source: ``plume-outline.geojson`` from the asset
        bundle. If that fetch fails (network / 404 / malformed), falls
        back to vectorising the band-4 alpha of :attr:`mask` and logs
        a warning so callers notice when they're not on the canonical
        path.

        Returns ``None`` if neither source is reachable. For the
        tighter IME-significant polygon used in the
        ``emission_auto`` integral, use :attr:`ime_outline`.
        """
        try:
            geom = self._fetch_outline_geojson(
                "plume-outline.geojson", fallback_to_alpha=True,
            )
            if geom is not None:
                return geom
        except Exception as exc:
            _log.warning(
                "outline GeoJSON fetch failed for plume %s (%s); "
                "falling back to band-4 alpha vectorize",
                self.plume_id, exc,
            )
        return self._polygon_from_alpha()

    @cached_property
    def ime_outline(self) -> Optional[BaseGeometry]:
        """IME-significance polygon in EPSG:4326 (tighter than :attr:`outline`).

        Tracks the ``ime-cmf-outline.geojson`` asset — the polygon
        Carbon Mapper actually integrated over for ``emission_auto``.
        Excludes pixels below the IME detection threshold (which
        :attr:`outline` includes via the broader plume mask).

        Returns ``None`` if the IME outline isn't reachable (older
        plumes, network failure, etc.); does **not** vectorize-fallback
        because :attr:`ime_mask` and :attr:`outline` are both
        readily available substitutes.
        """
        try:
            return self._fetch_outline_geojson(
                "ime-cmf-outline.geojson", fallback_to_alpha=False,
            )
        except Exception as exc:
            _log.warning(
                "ime-cmf-outline.geojson fetch failed for plume %s (%s)",
                self.plume_id, exc,
            )
            return None

    # ---- Auxiliary -----------------------------------------------------

    def load_alpha_mask(self) -> Optional[GeoTensor]:
        """Load just band 4 of :attr:`mask` as a boolean GeoTensor."""
        reader = self.mask
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

    # ---- Internals -----------------------------------------------------

    def _open_optional(self, asset_key: str) -> Optional[RasterioReader]:
        url = self.urls.get(asset_key)
        if url is None:
            return None
        return RasterioReader(str(url), overview_level=self.overview_level)

    def _fetch_outline_geojson(
        self, asset_key: str = "plume-outline.geojson",
        *, fallback_to_alpha: bool = True,
    ) -> Optional[BaseGeometry]:
        """Fetch and parse a GeoJSON outline asset.

        Args:
            asset_key: Which outline to fetch — ``plume-outline.geojson``
                (broader plume polygon) or ``ime-cmf-outline.geojson``
                (tighter IME-significant polygon).
            fallback_to_alpha: Unused at this layer — caller decides
                whether to fall back; documented here so callers can
                see the symmetry with the broader-vs-tight pair.

        Returns ``None`` if the URL isn't on this image; raises on
        fetch / parse failure so the calling property can log + branch.
        """
        del fallback_to_alpha  # caller-side concern
        url = self.urls.get(asset_key)
        if url is None:
            return None

        path = str(url)
        if path.startswith(("http://", "https://")):
            headers = (
                {"Authorization": f"Bearer {self.token}"}
                if self.token else {}
            )
            r = requests.get(path, headers=headers, timeout=self.http_timeout)
            r.raise_for_status()
            data = r.json()
        else:
            with open(path, "r") as fh:
                data = json.load(fh)

        return _parse_geojson_to_geometry(data)

    def _polygon_from_alpha(self) -> Optional[BaseGeometry]:
        """Vectorise band-4 alpha of :attr:`mask` to an EPSG:4326 polygon.

        Fallback path — used only when the canonical
        ``plume-outline.geojson`` is unreachable.
        """
        reader = self.mask
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

        # Lazy import to keep rasterio out of the import path when only
        # the GeoJSON-canonical outline is used.
        import rasterio.features

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

    # ---- Repr ---------------------------------------------------------

    def __repr__(self) -> str:
        present = [k for k in CM_PLUME_IMAGE_ASSETS if k in self.urls]
        missing = [k for k in CM_PLUME_IMAGE_ASSETS if k not in self.urls]
        ov = self.overview_level if self.overview_level is not None else "full"
        lines = [
            "CMPlumeImage",
            f"  plume_id:       {self.plume_id}",
            f"  assets present: {present or '<none>'}",
        ]
        if missing:
            lines.append(f"  assets missing: {missing}")
        lines.append(f"  overview_level: {ov}")
        return "\n".join(lines)

    __str__ = __repr__


# ─────────────────────────────────────────────────────────────────────
#  GeoJSON parsing helper (private)
# ─────────────────────────────────────────────────────────────────────


def _parse_geojson_to_geometry(data: Any) -> Optional[BaseGeometry]:
    """Parse a GeoJSON object (Feature, FeatureCollection, or bare
    Geometry) into a unioned shapely geometry. ``None`` if no geometry
    is recoverable."""
    if not isinstance(data, dict):
        return None
    t = data.get("type")
    geoms: list[BaseGeometry] = []
    if t == "FeatureCollection":
        geoms = [
            shape(f["geometry"]) for f in data.get("features", [])
            if f.get("geometry")
        ]
    elif t == "Feature":
        g = data.get("geometry")
        if g:
            geoms = [shape(g)]
    elif t and "coordinates" in data:
        geoms = [shape(data)]
    if not geoms:
        return None
    return unary_union(geoms)


__all__ = [
    "CM_PLUME_IMAGE_ASSETS",
    "CMPlumeImage",
]
