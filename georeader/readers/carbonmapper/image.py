"""Per-plume image bundle wrapper for Carbon Mapper L3A products.

One :class:`CMPlumeImage` is the cropped raster suite for one plume —
binary mask, full column-density crop, IME-clipped retrieval, RGB
context, and the canonical outline polygon. This is the per-plume
counterpart to :class:`~georeader.readers.carbonmapper.rasters.CMImageRaster`
(which wraps L2B scenes).

Which products the bundle carries is an explicit caller choice — the
``products=`` parameter takes descriptors from
:mod:`~georeader.readers.carbonmapper.products` (defaulting to the 7
GeoTIFF/GeoJSON assets). Collection versioning is resolved per plume
from the record's own ``plume_tif`` URL via
:class:`~georeader.readers.carbonmapper.products.CMCollectionSpec` —
nothing is guessed from hardcoded version lists, so the wrapper works
for any gas / cmf_type / version CM publishes (v3a, v3c, v3d, …).

Why URL derivation instead of STAC: ``/stac/collections`` stops at
``-v3a`` (plumes 2023-10 → 2025-12); every newer collection exists
only in the asset-proxy namespace and is reachable via direct asset
URLs from ``/catalog/plume/{id}``. A STAC-only wrapper would miss all
current data. :meth:`CMPlumeImage.from_plume_id` derives all asset
URLs from the REST catalog response (which has signed CDN URLs for
any version) and rewrites them to the Bearer-aware api gateway form
so the URLs don't expire (verified against the live API, 2026-07).

Outline GeoJSON is the canonical source for the plume polygon; if
the fetch fails (network / 404 / malformed body), we fall back to
vectorising the band-4 alpha of ``plume.tif`` and log a warning so
callers notice when they're not on the canonical path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from georeader import read, window_utils
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from georeader.readers.carbonmapper.plume import CMRawPlume, Collection
from georeader.readers.carbonmapper.products import (
    ALL_PLUME_PRODUCTS,
    DEFAULT_PLUME_PRODUCTS,
    IME_CONCENTRATIONS,
    IME_MASK,
    IME_OUTLINE,
    PLUME_CONCENTRATIONS,
    PLUME_OUTLINE,
    PLUME_TIF,
    RGB_TIF,
    CMCollectionSpec,
    CMProduct,
    CMProductFamily,
    CMProductNotSelected,
    _parse_asset_url,
)

if TYPE_CHECKING:
    # The tile-bridge methods (:meth:`CMPlumeImage.tile` and the
    # ``tile_*`` cropping methods, Phase 2) need :class:`CMImageRaster`
    # for typing. Imported under ``TYPE_CHECKING`` to keep the module
    # importable without dragging in the L2B raster machinery for
    # callers who only use the L3A products.
    from georeader.readers.carbonmapper.rasters import CMImageRaster

_log = logging.getLogger(__name__)


# Asset proxy base — Bearer-aware mirror of the signed CDN. Used by
# `_derive_asset_urls` to rewrite the host on URLs returned by
# `/catalog/plume/{id}` so the URL doesn't expire.
_API_ASSET_BASE = "https://api.carbonmapper.org/api/v1/catalog/asset"


# Default asset keys this wrapper exposes — one per public property on
# `CMPlumeImage`. Kept for backwards compatibility + introspection; the
# authoritative registry lives in
# :mod:`georeader.readers.carbonmapper.products`.
CM_PLUME_IMAGE_ASSETS: tuple[str, ...] = tuple(
    p.key for p in DEFAULT_PLUME_PRODUCTS
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


def _derive_asset_urls(
    catalog_plume: Mapping[str, Any],
    products: Sequence[CMProduct] = DEFAULT_PLUME_PRODUCTS,
) -> dict[str, str]:
    """From ``/catalog/plume/{id}`` response, derive ``{asset_key: url}``
    for the **selected** per-plume products.

    Resolves the :class:`~georeader.readers.carbonmapper.products.CMCollectionSpec`
    from the record's own ``plume_tif`` URL (authoritative — it names
    the run the assets were published under; verified same-version
    across the vis/ime families in the 2026-07 audit), then composes
    one asset-proxy URL per selected product. Translates the signed
    CDN host to the api.carbonmapper.org gateway form so Bearer auth
    applies and the URL doesn't expire.

    Args:
        catalog_plume: The ``/catalog/plume/{id}`` (or annotated-list
            item) mapping. Only ``plume_tif`` and ``plume_id`` are
            consulted; other URL fields (``con_tif``, ``rgb_png``, …)
            are ignored — deriving every asset from one spec avoids
            version-mismatch bugs.
        products: Which products to derive URLs for. Defaults to
            :data:`~georeader.readers.carbonmapper.products.DEFAULT_PLUME_PRODUCTS`
            (the 7 GeoTIFF/GeoJSON assets). Pass
            :data:`~georeader.readers.carbonmapper.products.ALL_PLUME_PRODUCTS`
            to include the PNG quicklooks, or any explicit subset.

    Raises
    ------
    ValueError
        If ``catalog_plume`` is missing the ``plume_tif`` field, its
        URL doesn't match the asset pattern, or a non-L3A product was
        requested.
    """
    seed = catalog_plume.get("plume_tif")
    if not seed:
        raise ValueError(
            "catalog_plume has no 'plume_tif' URL — can't derive asset "
            f"URLs for plume_id {catalog_plume.get('plume_id')!r}. "
            "Either the plume hasn't been processed yet or the "
            "catalog response is malformed."
        )

    parsed = _parse_asset_url(str(seed))
    if parsed is None:
        raise ValueError(
            f"plume_tif URL {seed!r} doesn't match the expected "
            "`.../{collection}/{Y}/{M}/{D}/{plume_id}/"
            "{plume_id}_{collection}_plume.tif` asset pattern."
        )

    # The vis products reuse the record's own collection id verbatim
    # (no recomposition drift). The IME sibling collection is composed
    # via the spec — but only when the record's collection is a real
    # `l3a-vis-*` id. Legacy families (`l3a-ch4-mf-v1`, pre vis/ime
    # split) have no IME sibling: those keys are omitted and the ime_*
    # properties return None.
    ime_collection: Optional[str] = None
    try:
        spec = CMCollectionSpec.from_collection_id(parsed.collection_id)
        if parsed.collection_id == spec.collection_id(CMProductFamily.L3A_VIS):
            ime_collection = spec.collection_id(CMProductFamily.L3A_IME)
    except ValueError:
        pass

    out: dict[str, str] = {}
    for product in products:
        if product.family is CMProductFamily.L3A_VIS:
            collection = parsed.collection_id
        elif product.family is CMProductFamily.L3A_IME:
            if ime_collection is None:
                continue
            collection = ime_collection
        else:
            raise ValueError(
                f"Product {product.key!r} is a per-scene "
                f"({product.family.value}) product — only L3A per-plume "
                "products can be derived from a plume record. Use "
                "CMImageRaster for L2B scene products."
            )
        out[product.key] = product.asset_url(
            None,
            parsed.item_id,
            date=(parsed.yyyy, parsed.mm, parsed.dd),
            collection_id=collection,
        )
    return out


def _spec_or_none(record: Mapping[str, Any]) -> Optional[CMCollectionSpec]:
    """Resolve the record's collection spec; ``None`` when impossible
    (legacy families / sparse records) rather than failing the bundle —
    the spec is an optimisation for :attr:`CMPlumeImage.tile`, not a
    requirement."""
    try:
        return CMCollectionSpec.from_plume_record(record)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────
#  CMPlumeImage
# ─────────────────────────────────────────────────────────────────────


@dataclass(repr=False)
class CMPlumeImage:
    """Per-plume L3A bundle — **outline polygon + L2B tile bridge**.

    The two things consumers actually want from a plume:

    1. **Polygon geometry** for the plume — :attr:`outline` returns a
       shapely ``MultiPolygon`` in EPSG:4326. The plume REST response's
       ``geometry_json`` is only a centroid Point; the canonical
       polygon lives on this L3A asset bundle.
    2. **The L2B parent tile** for full-resolution rasters —
       :attr:`tile` resolves the parent :class:`CMImageRaster`
       transparently for v3a (via STAC) and v3c/v3d (via URL-pattern
       fallback). The :meth:`tile_cmf` / :meth:`tile_rgb` /
       :meth:`tile_uncertainty` methods crop the L2B band by the
       outline polygon at full native resolution — this is the
       analysis-grade workflow.

    The L3A pre-cropped thumbnails (:attr:`mask`,
    :attr:`concentrations`, :attr:`ime_concentrations`,
    :attr:`ime_mask`, :attr:`rgb`) are kept as a convenience for fast
    preview UIs. They are typically **41 × 48 px** for
    ``plume-concentrations.tif`` and as small as **11 × 11 px** for
    ``ime-cmf-concentrations.tif`` — well below analysis quality. For
    any quantitative work, prefer the ``tile_*`` methods.

    Lazy: instantiating the dataclass does NOT issue HTTP / blob reads.
    Each property opens its asset on first access and caches the
    result.

    Recommended workflow (Phase 2 one-liner)::

        img = CMPlumeImage.from_plume_id(plume_id, token=token)
        outline = img.outline                        # MultiPolygon in EPSG:4326
        crop    = img.tile_cmf(pad_px=64)            # GeoTensor, full L2B res
        rgb     = img.tile_rgb(pad_px=64)            # 3-band context image

    Attributes:
        plume_id: Carbon Mapper plume_id (e.g.
            ``"tan20251212t185057c20s4001-E"``).
        urls: Mapping of asset name → URL (Bearer-aware api gateway
            form preferred; signed CDN URLs work but expire). Up to 7
            keys expected: ``plume.tif``, ``plume-concentrations.tif``,
            ``plume-outline.geojson``, ``rgb.tif``,
            ``ime-cmf-concentrations.tif``, ``ime-cmf-mask.tif``,
            ``ime-cmf-outline.geojson``. Missing keys → corresponding
            property returns ``None``.
        token: Bearer token. Required for :attr:`tile` (the L2B asset
            proxy is always Bearer-gated). May be ``None`` for L3A-only
            usage if :attr:`urls` already point at signed CDN URLs or
            local files.
        overview_level: Forwarded to :class:`RasterioReader` —
            ``None`` for full resolution, integer for COG overviews.
        http_timeout: Per-request timeout (seconds) for the outline
            GeoJSON fetch. Defaults to 30s.
        products: The explicit product selection this bundle exposes
            (descriptors from
            :mod:`~georeader.readers.carbonmapper.products`). Accessing
            a product outside the selection raises
            :class:`~georeader.readers.carbonmapper.products.CMProductNotSelected`.
            Defaults to
            :data:`~georeader.readers.carbonmapper.products.DEFAULT_PLUME_PRODUCTS`.
        spec: The plume's resolved
            :class:`~georeader.readers.carbonmapper.products.CMCollectionSpec`.
            Set by the ``from_*`` constructors; enables :attr:`tile` to
            resolve the L2B parent at the **same version** with zero
            probing. ``None`` for hand-built bundles (``tile`` then
            falls back to the probing resolver).
    """

    plume_id: str
    urls: Mapping[str, str]
    token: Optional[str] = None
    overview_level: Optional[int] = None
    http_timeout: float = 30.0
    products: tuple[CMProduct, ...] = DEFAULT_PLUME_PRODUCTS
    spec: Optional[CMCollectionSpec] = None

    def __post_init__(self) -> None:
        self._product_cache: dict[str, Any] = {}

    # ---- Generic product access ----------------------------------------

    def product(self, product: CMProduct) -> Any:
        """Open one selected product (cached per product key).

        Returns the product-kind-specific object — a
        :class:`RasterioReader` for raster products, a shapely geometry
        for vector products, ``bytes`` for PNG quicklooks. Returns
        ``None`` when the product was selected but its URL was not
        available from the source (e.g. :meth:`from_stac_item` with a
        missing sibling item).

        Raises:
            CMProductNotSelected: If ``product`` is not in
                :attr:`products` — select it at construction to use it.
        """
        if product not in self.products:
            raise CMProductNotSelected(product, tuple(self.products))
        if product.key in self._product_cache:
            return self._product_cache[product.key]
        url = self.urls.get(product.key)
        value = None
        if url is not None:
            value = product.open(
                str(url),
                token=self.token,
                overview_level=self.overview_level,
                http_timeout=self.http_timeout,
            )
        self._product_cache[product.key] = value
        return value

    # ---- Constructors --------------------------------------------------

    @classmethod
    def from_plume_id(
        cls,
        plume_id: str,
        *,
        token: str,
        products: Sequence[CMProduct] = DEFAULT_PLUME_PRODUCTS,
        overview_level: Optional[int] = None,
        http_timeout: float = 30.0,
    ) -> CMPlumeImage:
        """Build by fetching ``/catalog/plume/{id}`` then deriving asset URLs.

        One round-trip. Works for any collection version (v3a … v3d and
        future bumps) — the version is resolved from the record itself
        via :class:`CMCollectionSpec`, never guessed.

        Args:
            plume_id: Colloquial plume id.
            token: Bearer token.
            products: Explicit product selection (see
                :mod:`~georeader.readers.carbonmapper.products`).
            overview_level, http_timeout: See class attributes.

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
        record = r.json()
        return cls(
            plume_id=plume_id,
            urls=_derive_asset_urls(record, products),
            token=token,
            overview_level=overview_level, http_timeout=http_timeout,
            products=tuple(products),
            spec=_spec_or_none(record),
        )

    @classmethod
    def from_cmrawplume(
        cls,
        raw: CMRawPlume,
        *,
        token: str,
        products: Sequence[CMProduct] = DEFAULT_PLUME_PRODUCTS,
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
        seed = {
            "plume_tif": raw.plume_tif,
            "plume_id": raw.plume_id,
            "gas": raw.gas,
            "cmf_type": raw.emission_cmf_type,
            "emission_version": raw.emission_version,
        }
        return cls(
            plume_id=raw.plume_id,
            urls=_derive_asset_urls(seed, products),
            token=token,
            overview_level=overview_level, http_timeout=http_timeout,
            products=tuple(products),
            spec=_spec_or_none(seed),
        )

    @classmethod
    def from_stac_item(
        cls,
        item: Mapping[str, Any],
        *,
        ime_item: Optional[Mapping[str, Any]] = None,
        products: Sequence[CMProduct] = DEFAULT_PLUME_PRODUCTS,
        token: Optional[str] = None,
        overview_level: Optional[int] = None,
        http_timeout: float = 30.0,
    ) -> CMPlumeImage:
        """Build from STAC items (vis + optional ime sibling).

        Use when the caller is driving STAC search directly. History
        only — ``/stac/collections`` stops at v3a (plumes ≤ 2025-12);
        use :meth:`from_plume_id` for anything newer. Pass both the vis
        and ime items if you have them (recommended); the ime sibling
        provides the ``ime-*`` product URLs.
        """
        plume_id = str(item.get("id", ""))
        urls: dict[str, str] = {}
        for product in products:
            source = (
                item if product.family == CMProductFamily.L3A_VIS
                else ime_item
            )
            if source is None:
                continue
            href = (source.get("assets") or {}).get(product.key, {}).get("href")
            if href:
                urls[product.key] = href
        spec: Optional[CMCollectionSpec] = None
        collection = item.get("collection")
        if collection:
            try:
                spec = CMCollectionSpec.from_collection_id(str(collection))
            except ValueError:
                spec = None
        return cls(
            plume_id=plume_id, urls=urls, token=token,
            overview_level=overview_level, http_timeout=http_timeout,
            products=tuple(products),
            spec=spec,
        )

    # ---- Lazy raster properties (delegate to the product registry) ----

    @property
    def mask(self) -> Optional[RasterioReader]:
        """L3A pre-cropped RGBA GeoTIFF — band 4 alpha is the binary mask.

        **Thumbnail-grade.** Typically a small per-plume window
        (~30–50 px per side). For analysis-grade rasters at full L2B
        native resolution, prefer :meth:`tile_cmf` cropped by
        :attr:`outline`.

        Backed by :class:`RasterioReader`. Returns ``None`` if
        ``plume.tif`` URL is absent; raises
        :class:`CMProductNotSelected` if ``PLUME_TIF`` wasn't selected.
        """
        return self.product(PLUME_TIF)

    @property
    def concentrations(self) -> Optional[RasterioReader]:
        """L3A pre-cropped CH4 column density thumbnail (ppm·m).

        **Thumbnail-grade — typically ~41 × 48 px.** Covers the full
        per-plume window incl. noise floor outside the mask. Use
        :attr:`ime_concentrations` for the IME-clipped variant.

        For analysis-grade resolution, prefer :meth:`tile_cmf` —
        crops the L2B parent at native pixel grid (~150 × 150 px
        with default padding).
        """
        return self.product(PLUME_CONCENTRATIONS)

    @property
    def ime_concentrations(self) -> Optional[RasterioReader]:
        """L3A IME-clipped CH4 column density (ppm·m).

        **Thumbnail-grade — typically ~11 × 11 px.** Only
        mask-significant pixels (others NoData). This is the raster
        Carbon Mapper uses to compute the plume's ``emission_auto``
        integral — so it's the right thing if you want to reproduce
        the upstream emission calculation, but **the wrong thing if
        you want a recognisable plume image** (pixelates into
        circle-like blobs at this resolution).

        For an analysis-grade plume image at L2B native resolution,
        use :meth:`tile_cmf` cropped by :attr:`outline` instead.

        Returns ``None`` if the IME sibling URL wasn't provided
        (e.g. :meth:`from_stac_item` without ``ime_item``).
        """
        return self.product(IME_CONCENTRATIONS)

    @property
    def rgb(self) -> Optional[RasterioReader]:
        """L3A pre-cropped 3-band uint8 true-colour GeoTIFF.

        **Thumbnail-grade** per-plume crop. For full-resolution true
        colour at L2B native pixel size (analysis-grade context for
        overlays), use :meth:`tile_rgb`.
        """
        return self.product(RGB_TIF)

    @property
    def ime_mask(self) -> Optional[RasterioReader]:
        """L3A binary mask of pixels that contributed to ``emission_auto``.

        From ``ime-cmf-mask.tif`` — a tighter subset of
        :attr:`mask`'s band-4 alpha. **Thumbnail-grade** (same ~11 × 11
        px window as :attr:`ime_concentrations`). Use this when you
        specifically need the IME-significant pixel set (e.g. for
        re-quantification with a different wind product) rather than
        the broader plume polygon.
        """
        return self.product(IME_MASK)

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
            geom = self.product(PLUME_OUTLINE)
            if geom is not None:
                return geom
        except CMProductNotSelected:
            raise
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
            return self.product(IME_OUTLINE)
        except CMProductNotSelected:
            raise
        except Exception as exc:
            _log.warning(
                "ime-cmf-outline.geojson fetch failed for plume %s (%s)",
                self.plume_id, exc,
            )
            return None

    # ---- Tile bridge (Phase 2) ----------------------------------------

    @cached_property
    def scene_id(self) -> str:
        """Parent L2B ``scene_id``, derived from ``plume_id`` (no HTTP).

        Carbon Mapper convention: ``plume_id == "{scene_id}-{part}"``.
        Splitting on the last ``-`` recovers the parent scene id used
        by STAC item lookup and the asset-proxy URL pattern.

        >>> img = CMPlumeImage(plume_id="tan20260331t181625c77s4001-A", urls={})
        >>> img.scene_id
        'tan20260331t181625c77s4001'
        """
        return self.plume_id.rsplit("-", 1)[0]

    @cached_property
    def tile(self) -> CMImageRaster:
        """Parent L2B :class:`CMImageRaster` for this plume (lazy).

        When :attr:`spec` is known (any ``from_*`` constructor), the
        L2B parent is resolved by probing the plume's **own collection
        version first** (usually one probe — same-version pairing),
        with the default candidates as backup for the re-versioned
        case (a v3d L3A plume whose L2B parent still serves at v3c).
        No STAC lookup, no extra catalog calls, and never stale.
        Without a spec (hand-built bundles), falls back to
        :func:`~georeader.readers.carbonmapper.api_queries.get_image_raster_for_plume`
        (STAC first, then candidate probing). Result is cached, so
        subsequent crops reuse the same raster handle.

        Requires :attr:`token` (set at construction). The L3A asset
        URLs on :attr:`urls` are usable without a token only when
        they're signed CDN URLs (which expire); the L2B asset proxy
        always needs Bearer auth.

        Raises:
            ValueError: If :attr:`token` is ``None``.
            RuntimeError: If the spec-less fallback could not resolve
                the L2B parent through either STAC or candidate
                probing.

        >>> img = CMPlumeImage.from_plume_id(  # doctest: +SKIP
        ...     "tan20260331t181625c77s4001-A", token=tok,
        ... )
        >>> img.tile.cmf  # doctest: +SKIP
        <RasterioReader …/l2b-ch4-mfa-v3c/2026/03/31/…/cmf.tif>
        """
        if self.token is None:
            msg = (
                f"CMPlumeImage(plume_id={self.plume_id!r}) has no token — "
                "cannot fetch the L2B tile. Pass token= at construction "
                "or use from_plume_id/from_cmrawplume which take it."
            )
            raise ValueError(msg)

        # Lazy import to avoid pulling the L2B raster machinery (and
        # api_queries) into the L3A-only import path.
        from georeader.readers.carbonmapper.rasters import CMImageRaster

        if self.spec is not None:
            return CMImageRaster.from_scene_id(
                self.scene_id,
                token=self.token,
                spec=self.spec,
                overview_level=self.overview_level,
                http_timeout=self.http_timeout,
            )

        from georeader.readers.carbonmapper.api_queries import (
            get_image_raster_for_plume,
        )

        ir = get_image_raster_for_plume(self.token, self.plume_id)
        if ir is None:
            msg = (
                f"L2B parent tile for plume {self.plume_id!r} is not "
                "reachable. Neither STAC nor the URL-pattern fallback "
                "resolved it. Either the scene hasn't been published or "
                "it uses a collection variant not in the candidate list."
            )
            raise RuntimeError(msg)
        return ir

    def tile_cmf(self, *, pad_px: int = 64) -> GeoTensor:
        """L2B CH4 retrieval cropped to :attr:`outline`, full L2B resolution.

        This is the headline workflow: the analysis-grade
        ``cmf`` raster, tightly cropped to the plume polygon, at the
        L2B's native pixel grid — **not** the pre-cropped L3A
        thumbnail (``plume-concentrations.tif`` is typically
        ~41×48 px, ``ime-cmf-concentrations.tif`` only ~11×11 px).

        Args:
            pad_px: Pixels of context to add on every side around the
                outline bounding rectangle. Default ``64`` shows the
                plume embedded in its surroundings — set to ``0`` for
                a tight crop, larger for more context.

        Returns:
            :class:`GeoTensor` — eagerly loaded, ready to plot.

        Raises:
            ValueError: If :attr:`outline` is ``None`` (canonical fetch
                failed AND band-4 vectorize fallback failed — rare).
            ValueError, RuntimeError: If :attr:`tile` couldn't be
                resolved (see :attr:`tile`).

        >>> crop = img.tile_cmf(pad_px=64)  # doctest: +SKIP
        >>> crop.values.shape  # doctest: +SKIP
        (1, 168, 152)
        """
        return self._crop_tile_band("cmf", pad_px=pad_px)

    def tile_rgb(self, *, pad_px: int = 64) -> GeoTensor:
        """L2B true-colour RGB sibling cropped to :attr:`outline`.

        See :meth:`tile_cmf` for the cropping semantics. Returns the
        3-band RGB raster at L2B native resolution — the "proper image"
        backdrop for a plume polygon overlay.

        Raises:
            KeyError: If the tile has no RGB sibling URL (rare —
                ``with_rgb=True`` is the default on the tile resolver).
        """
        return self._crop_tile_band("rgb", pad_px=pad_px)

    def tile_uncertainty(self, *, pad_px: int = 64) -> GeoTensor:
        """L2B per-pixel uncertainty cropped to :attr:`outline`.

        Companion to :meth:`tile_cmf` — the uncertainty raster shares
        the cmf's pixel grid. See :meth:`tile_cmf` for the cropping
        semantics.
        """
        return self._crop_tile_band("uncertainty", pad_px=pad_px)

    def _crop_tile_band(self, band: str, *, pad_px: int) -> GeoTensor:
        """Shared crop pipeline for the ``tile_*`` methods.

        Cropping rule: take the plume polygon's bounding rectangle in
        the L2B's CRS, add ``pad_px`` pixels on each side, read that
        window. Same algorithm as ``CMImageRaster.read_polygon`` but
        with explicit padding for a single band.
        """
        outline = self.outline
        if outline is None:
            msg = (
                f"CMPlumeImage(plume_id={self.plume_id!r}).outline is None "
                "— canonical plume-outline.geojson fetch failed AND the "
                "band-4 alpha vectorize fallback returned no geometry. "
                "Cannot crop the tile."
            )
            raise ValueError(msg)

        tile = self.tile
        reader = getattr(tile, band.replace("-", "_"), None)
        if reader is None:
            msg = (
                f"L2B tile for plume {self.plume_id!r} has no {band!r} "
                f"asset. Available: {sorted(tile.asset_paths)}."
            )
            raise KeyError(msg)

        # `boundless=False` returns None on zero-overlap windows. The
        # outline is derived from this plume, which came from this
        # tile, so non-overlap implies a data bug — surface it loudly.
        result = read.read_from_polygon(
            reader,
            polygon=outline,
            crs_polygon="EPSG:4326",
            pad_add=(pad_px, pad_px),
            boundless=False,
            trigger_load=True,
        )
        if result is None:
            msg = (
                f"Outline of plume {self.plume_id!r} doesn't overlap its "
                f"parent tile (scene_id={self.scene_id!r}, band={band!r}). "
                "This indicates upstream data inconsistency."
            )
            raise RuntimeError(msg)
        # `trigger_load=True` materialises into a GeoTensor.
        if isinstance(result, GeoTensor):
            return result
        # Fall-back path: result is a GeoData reader, load explicitly.
        return result.load()

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
        selected = [p.key for p in self.products]
        present = [k for k in selected if k in self.urls]
        missing = [k for k in selected if k not in self.urls]
        ov = self.overview_level if self.overview_level is not None else "full"
        lines = [
            "CMPlumeImage",
            f"  plume_id:       {self.plume_id}",
            f"  assets present: {present or '<none>'}",
        ]
        if missing:
            lines.append(f"  assets missing: {missing}")
        if self.spec is not None:
            lines.append(f"  spec:           {self.spec}")
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
