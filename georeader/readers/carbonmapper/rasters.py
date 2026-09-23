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

import logging
import os
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, cast

from rasterio.errors import RasterioIOError
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from georeader import read
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader

# Lazy reads return a windowed `GeoData` (georeader's protocol — covers
# both ``RasterioReader`` and ``GeoTensor``).

from georeader.readers.carbonmapper.api_queries import (
    CMSceneNotPublished,
    CMTileItem,
)
from georeader.readers.carbonmapper.download import _request, _validate_bbox
from georeader.readers.carbonmapper.products import (
    CM_API_ASSET_BASE,
    DEFAULT_SCENE_PRODUCTS,
    CMCollectionSpec,
    CMProduct,
    CMProductFamily,
    rio_env_options_for,
)

BBox = tuple[float, float, float, float]   # (W, S, E, N) in WGS-84
PathLike = str | Path

_log = logging.getLogger(__name__)


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

#: Asset-proxy base (alias of the registry constant).
_API_ASSET_BASE = CM_API_ASSET_BASE

#: L2B collection versions observed on the live API, newest first.
#: Last verified 2026-07-31 — ``v3e`` is the current era; no CH4
#: ``v3b`` scene was ever observed, so it is not probed. Every candidate
#: tuple below is generated from this one sequence, so tracking a new
#: Carbon Mapper version is a single edit that keeps CH4, CO2 and RGB
#: in step.
_L2B_VERSIONS_NEWEST_FIRST: tuple[str, ...] = ("v3e", "v3d", "v3c", "v3a")

#: Matched-filter variants Carbon Mapper has published L2B scenes
#: under, per gas. CO2 moved between the two across versions — the
#: STAC registry (2026-09) lists ``l2b-co2-mfal-v3b..v3e`` and
#: ``l2b-co2-mfa-v3a`` / ``-v3e``, and a v3e CO2 scene serves from
#: ``mfa`` only — so CO2 lookups probe both.
_L2B_CMF_TYPES: dict[str, tuple[str, ...]] = {
    "ch4": ("mfa",),
    "co2": ("mfa", "mfal"),
}

#: Default L2B CH4 collection candidates probed by
#: :meth:`CMImageRaster.from_scene_id` **only when no**
#: :class:`~georeader.readers.carbonmapper.products.CMCollectionSpec`
#: (or explicit ``collection``) is available — i.e. scene-name-only
#: lookups with no plume record to resolve the version from.
#:
#: This list **rots by construction**: it is a hardcoded snapshot of a
#: namespace Carbon Mapper extends without notice (``/stac/collections``
#: lists new versions, but only once they exist — and a scene-name-only
#: lookup would need one STAC call per version to find its own). It is a
#: best-effort last resort, not a source of truth — whenever a plume
#: record is available, prefer the spec path, which reads the version
#: off the record and never goes stale. Older variants (``mfa-v3``,
#: ``mfa-v1``, ``mfm-v1``) can be passed explicitly for historical
#: scenes.
DEFAULT_L2B_CH4_COLLECTION_CANDIDATES: tuple[str, ...] = tuple(
    f"l2b-ch4-mfa-{version}" for version in _L2B_VERSIONS_NEWEST_FIRST
)

#: Default L2B RGB sibling collection candidates probed by
#: :meth:`CMImageRaster.from_scene_id`. Same version-letter ordering
#: as the CH4 candidates, and the same staleness caveat.
DEFAULT_L2B_RGB_COLLECTION_CANDIDATES: tuple[str, ...] = tuple(
    f"l2b-rgb-{version}" for version in _L2B_VERSIONS_NEWEST_FIRST
)


def _version_key(version: str) -> tuple[int, str]:
    """Sort key for ``v<N><letters>`` version strings (``v3e`` > ``v3a``,
    ``v10`` > ``v3e``). Unparseable versions sort first."""
    m = re.fullmatch(r"v(\d+)([a-z]*)", version)
    return (int(m.group(1)), m.group(2)) if m else (-1, version)


def _lag_ordered_versions(version: str) -> tuple[str, ...]:
    """Versions to probe for a scene whose plume record says ``version``.

    1. ``version`` itself — same-version pairing is the common case.
    2. Older known versions, newest first — L2B lags when Carbon Mapper
       re-versions L3A without republishing the scene.
    3. Newer known versions, oldest first — last resort only, so a
       plume never pairs with a later reprocess when its own run exists.
    """
    key = _version_key(version)
    older = [v for v in _L2B_VERSIONS_NEWEST_FIRST if _version_key(v) < key]
    newer = [
        v for v in reversed(_L2B_VERSIONS_NEWEST_FIRST) if _version_key(v) > key
    ]
    return (version, *older, *newer)


def l2b_collection_candidates_for_spec(spec: CMCollectionSpec) -> tuple[str, ...]:
    """L2B collection ids to probe for a plume, in probe order.

    Candidates never leave the spec's gas: a CO2 plume never resolves
    to a CH4 scene raster. Within each version (ordered by
    :func:`_lag_ordered_versions`) the spec's own cmf_type is tried
    first, then the other variants known for that gas.

    >>> l2b_collection_candidates_for_spec(CMCollectionSpec("v3d"))
    ('l2b-ch4-mfa-v3d', 'l2b-ch4-mfa-v3c', 'l2b-ch4-mfa-v3a', 'l2b-ch4-mfa-v3e')
    >>> l2b_collection_candidates_for_spec(
    ...     CMCollectionSpec("v3e", "co2", "mfa", ime_cmf_type="mfal"))[:3]
    ('l2b-co2-mfa-v3e', 'l2b-co2-mfal-v3e', 'l2b-co2-mfa-v3d')
    """
    cmf_types = list(dict.fromkeys(
        [spec.cmf_type]
        + ([spec.ime_cmf_type] if spec.ime_cmf_type else [])
        + list(_L2B_CMF_TYPES.get(spec.gas, ()))
    ))
    return tuple(
        f"l2b-{spec.gas}-{cmf}-{version}"
        for version in _lag_ordered_versions(spec.version)
        for cmf in cmf_types
    )


def rgb_collection_candidates_for_spec(spec: CMCollectionSpec) -> tuple[str, ...]:
    """L2B RGB sibling collection ids to probe, in the same version order
    as :func:`l2b_collection_candidates_for_spec` (RGB is not gas-typed)."""
    return tuple(f"l2b-rgb-{v}" for v in _lag_ordered_versions(spec.version))


def _probe_order(
    spec: CMCollectionSpec | None,
    family: CMProductFamily,
    candidates: Sequence[str] | None,
) -> tuple[str, ...]:
    """Final probe order for one L2B family (``L2B`` or ``L2B_RGB``).

    1. No spec → ``candidates``, or the module defaults when ``None``.
    2. With a spec → the spec's own collection id first, then
       ``candidates`` if given, else the spec-derived same-gas,
       lag-ordered candidates (duplicates dropped).
    """
    rgb = family is CMProductFamily.L2B_RGB
    if spec is None:
        if candidates is not None:
            return tuple(candidates)
        return (
            DEFAULT_L2B_RGB_COLLECTION_CANDIDATES if rgb
            else DEFAULT_L2B_CH4_COLLECTION_CANDIDATES
        )
    first = spec.collection_id(family)
    if candidates is None:
        candidates = (
            rgb_collection_candidates_for_spec(spec) if rgb
            else l2b_collection_candidates_for_spec(spec)
        )
    return (first, *(c for c in candidates if c != first))


def _is_absent_asset_error(exc: Exception) -> bool:
    """``True`` when a raster open failed because the asset doesn't exist
    (HTTP 403/404 from the asset proxy, or a missing local file) rather
    than because of auth, throttling or a server error."""
    msg = str(exc)
    return bool(
        re.search(r"response code:?\s*40[34]\b", msg)
        or "No such file or directory" in msg
    )


def _parse_scene_date(scene_id: str) -> tuple[str, str, str]:
    """Extract ``(YYYY, MM, DD)`` from a Carbon Mapper scene_id.

    Carbon Mapper scene_ids follow the convention
    ``<3-char-instrument><YYYYMMDD>t<HHMMSS>...`` for every instrument
    we've observed (``tan`` / ``emi`` / ``ang`` / ``av3`` / ``GAO``).
    The date sits at positions ``[3:11]``.

    Raises:
        ValueError: If positions ``[3:11]`` aren't an 8-digit date.
    """
    if len(scene_id) < 11:
        raise ValueError(
            f"scene_id {scene_id!r} too short to carry an 8-digit date"
        )
    date_part = scene_id[3:11]
    if not date_part.isdigit():
        raise ValueError(
            f"scene_id {scene_id!r} positions [3:11] = {date_part!r} "
            f"are not an 8-digit date"
        )
    return date_part[:4], date_part[4:6], date_part[6:8]


def _l2b_asset_url(collection: str, scene_id: str, asset: str) -> str:
    """Build the asset-proxy URL for one L2B asset (one scene + one key).

    Pattern::

        {API_ASSET_BASE}/{collection}/{Y}/{M}/{D}/{scene_id}/
            {scene_id}_{collection}_{asset}

    Verified for ``l2b-ch4-mfa-v3a`` / ``v3c`` and ``l2b-rgb-v3a`` /
    ``v3c`` (probe 2026-05-11). The asset suffix carries its extension
    (e.g. ``cmf.tif``, ``uas.txt``).
    """
    yyyy, mm, dd = _parse_scene_date(scene_id)
    return (
        f"{_API_ASSET_BASE}/{collection}/{yyyy}/{mm}/{dd}/"
        f"{scene_id}/{scene_id}_{collection}_{asset}"
    )


def _probe_l2b_collection(
    scene_id: str,
    collection_candidates: Sequence[str],
    *,
    probe_asset: str,
    token: str,
    http_timeout: float,
) -> str | None:
    """Probe candidate L2B collections; return the first that serves ``probe_asset``.

    Issues a range-GET (``Range: bytes=0-1``) per candidate. Behaviour
    per status code:

    - ``200`` / ``206`` — winner, return this collection id.
    - ``404`` — legitimate "this scene isn't in this collection
      variant"; try the next candidate.
    - ``403`` — **also** "this scene isn't in this collection", just
      phrased differently: the asset proxy answers 403 when the
      *collection exists* but does not hold the requested scene, and
      404 when the collection itself is unknown. Verified 2026-07-31
      with one token in one session: ``l2b-ch4-mfa-v3e`` +
      a March-2026 scene → 403, while the same collection + a July-2026
      scene → 206 and a bogus ``l2b-ch4-mfa-v3q`` + the March scene →
      404. So 403 must be skipped like 404, or a candidate list that
      leads with a version newer than the scene aborts the whole chain.
      This is not an auth signal: a bad or absent token returns **401**
      (also verified), which is still surfaced.
    - **Anything else** (401, 429, 5xx) — transient or authentication
      failure that should NOT be silently treated as "not published".
      Surfaced via ``raise_for_status`` so callers see the real error.
    - Transport-level errors (``ConnectionError``, ``Timeout``) — same
      reasoning: surface, don't swallow.

    Returns ``None`` only when every candidate 404'd — i.e. when CM
    has genuinely not published this scene in any of the probed
    collections.

    Used by :meth:`CMImageRaster.from_scene_id` to decide which L2B
    version the parent scene was processed at, without an explicit
    catalog lookup.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Range": "bytes=0-1",
    }
    for coll in collection_candidates:
        url = _l2b_asset_url(coll, scene_id, probe_asset)
        r = _request(
            "GET", url, headers=headers, timeout=http_timeout, stream=True,
        )
        r.close()
        if r.status_code in (200, 206):
            return coll
        if r.status_code in (403, 404):
            # Both mean "not this collection" — see the docstring for the
            # 403-vs-404 split. Auth failures arrive as 401, not 403.
            continue
        # 401 / 429 (after retries) / 5xx — real error, not a data fact.
        r.raise_for_status()
    return None


@dataclass(repr=False)
class CMImageRaster:
    """L2B scene exposed as four georeader-backed rasters.

    Lazy: instantiating the dataclass does NOT issue HTTP / blob reads;
    access ``.cmf`` / ``.rgb`` / etc. or call :meth:`read_window` /
    :meth:`read_polygon` to trigger I/O.

    Attributes:
        scene_id: CM L2B item id (e.g. ``"tan20251212t185057c20s4001"``).
        asset_paths: Mapping of band name → URL (``https://``) or
            local / blob path. Optional bands (``artifact-mask``,
            ``rgb``, the ``*-unortho`` variants) may be missing from the
            mapping **or** from the server — either way the accessor
            returns ``None``.
        overview_level: Forwarded to ``RasterioReader``. ``None`` for
            full resolution; integer for COG overviews (faster previews).
        token: Bearer token for asset-proxy URLs. Applied per read
            (scoped GDAL env), never exported process-wide. ``None``
            falls back to whatever ``GDAL_HTTP_HEADERS`` the caller set.
    """

    scene_id: str
    asset_paths: Mapping[str, PathLike]
    overview_level: Optional[int] = None
    token: Optional[str] = None

    # ---- Constructors --------------------------------------------------

    @classmethod
    def from_cm_tile_item(
        cls, item: CMTileItem, *, token: Optional[str] = None,
    ) -> CMImageRaster:
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
        return cls(scene_id=item.scene_id, asset_paths=paths, token=token)

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
            token=self.token,
        )

    @classmethod
    def from_scene_id(
        cls,
        scene_id: str,
        *,
        token: str,
        spec: CMCollectionSpec | None = None,
        collection: str | None = None,
        rgb_collection: str | None = None,
        products: Sequence[CMProduct] = DEFAULT_SCENE_PRODUCTS,
        l2b_collection_candidates: Sequence[str] | None = None,
        rgb_collection_candidates: Sequence[str] | None = None,
        with_rgb: bool = True,
        overview_level: int | None = None,
        http_timeout: float = 30.0,
    ) -> CMImageRaster:
        """Build by deriving L2B asset URLs from the scene_id (URL-pattern).

        Bypasses STAC entirely — derives every asset URL by templating
        against the verified asset-proxy pattern (see
        :func:`_l2b_asset_url`). Required for 2026 scenes (v3c/v3d)
        which are **not** in ``/stac/collections``.

        Collection resolution, in order of preference:

        1. ``collection`` / ``rgb_collection`` — explicit ids,
           probe-free (you pinned them, they're used verbatim).
        2. ``spec`` — a :class:`CMCollectionSpec` resolved from the
           plume record (``CMCollectionSpec.from_plume_record``). The
           spec's composed collection id is probed **first**. Usually
           one probe suffices (same-version pairing), but the L3A side
           can be re-versioned ahead of the L2B parent (observed: a v3d
           L3A plume whose L2B still serves at v3c — audit §4), so
           :func:`l2b_collection_candidates_for_spec` backs it up with
           older versions (then newer ones) **of the same gas** — a CO2
           plume never resolves to a CH4 scene. Because the spec always
           contributes the record's own version, this path never goes
           stale when Carbon Mapper bumps versions.
        3. Candidate probing (legacy) — for scene-name-only lookups
           with no record available. Probes
           ``l2b_collection_candidates`` (default: the CH4 candidates)
           in order; first 200/206 wins.

        Parameters
        ----------
        scene_id:
            L2B scene name, equal to ``plume_id.rsplit("-", 1)[0]`` for
            any plume that came from this scene. Must follow the
            ``<inst><YYYYMMDD>t<HHMMSS>...`` convention so the date
            can be parsed. (The catalog record's ``scene_id`` field is
            a **UUID** and will not work here.)
        token:
            Bearer token. Required — the asset-proxy URLs return 401
            without it.
        spec:
            Collection spec (gas / cmf_type / version) to compose both
            collection ids from. Its own ids are probed first; explicit
            candidate lists (if given) follow, else the gas-matched
            spec-derived candidates.
        collection:
            Explicit L2B CH4-family collection id (e.g.
            ``"l2b-ch4-mfa-v3d"``). Used verbatim with no probing and
            takes precedence over ``spec`` for the CH4 side (``spec``
            then still orders the RGB candidates if ``rgb_collection``
            is not pinned).
        rgb_collection:
            Explicit RGB sibling collection id (e.g. ``"l2b-rgb-v3d"``).
            Used verbatim with no probing; takes precedence over
            ``spec`` for the RGB side.
        products:
            Which per-scene products to derive URLs for — descriptors
            from :mod:`~georeader.readers.carbonmapper.products`
            (:data:`DEFAULT_SCENE_PRODUCTS` = the 5 rasters + ``uas``).
            The RGB sibling is controlled by ``with_rgb``, not this
            list (it lives in a different collection).
        l2b_collection_candidates:
            L2B collection IDs to probe, in order, when ``collection``
            is not given. ``None`` (default) means the spec-derived
            candidates when ``spec`` is given, else
            :data:`DEFAULT_L2B_CH4_COLLECTION_CANDIDATES`.
        rgb_collection_candidates:
            L2B RGB sibling collection IDs probed identically (on
            ``rgb.tif``). ``None`` (default) means the spec-derived
            candidates when ``spec`` is given, else
            :data:`DEFAULT_L2B_RGB_COLLECTION_CANDIDATES`.
        with_rgb:
            When ``True`` (default), attach the ``rgb`` sibling URL
            (probing the RGB candidates only when its collection isn't
            pinned by ``spec`` / ``rgb_collection``). When ``False``,
            ``self.rgb`` will be ``None``.
        overview_level:
            Forwarded to :class:`RasterioReader`.
        http_timeout:
            Per-probe range-GET timeout (seconds).

        Returns
        -------
        CMImageRaster
            With ``asset_paths`` populated for the selected products
            and, when ``with_rgb=True``, the ``rgb`` sibling URL.

        Raises
        ------
        CMSceneNotPublished
            When probing was needed and every candidate L2B collection
            404'd for ``scene_id`` — the scene either hasn't been
            processed yet or only exists in an unlisted collection
            variant. Catch in ETL paths that want to defer rather
            than error. (Not raised on the explicit ``collection``
            path, which builds URLs without probing — missing scenes
            surface as read errors on first access.)
        ValueError
            When ``scene_id`` doesn't carry an 8-digit date at
            positions ``[3:11]``, or a non-L2B product was requested.

        Examples
        --------
        >>> tile = CMImageRaster.from_scene_id(  # doctest: +SKIP
        ...     "tan20260623t124240c80s4001", token=tok,
        ...     spec=CMCollectionSpec(version="v3d"),
        ... )
        >>> tile.cmf  # doctest: +SKIP
        <RasterioReader …/l2b-ch4-mfa-v3d/2026/06/23/…>
        """
        # The spec's composed ids probe FIRST (the record's own version
        # — usually the right one) with same-gas backups; an explicit
        # `collection` / `rgb_collection` is used verbatim.
        l2b_collection_candidates = _probe_order(
            spec, CMProductFamily.L2B, l2b_collection_candidates,
        )
        rgb_collection_candidates = _probe_order(
            spec, CMProductFamily.L2B_RGB, rgb_collection_candidates,
        )
        l2b_coll = collection
        rgb_coll = rgb_collection

        if l2b_coll is None:
            l2b_coll = _probe_l2b_collection(
                scene_id,
                l2b_collection_candidates,
                probe_asset="cmf.tif",
                token=token,
                http_timeout=http_timeout,
            )
            if l2b_coll is None:
                raise CMSceneNotPublished(scene_id)

        # Build one asset URL per selected product. Keys are the
        # extension-less band names — they must match the lazy-property
        # names exactly (`_open` strips nothing).
        asset_paths: dict[str, PathLike] = {}
        for product in products:
            if product.family is not CMProductFamily.L2B:
                raise ValueError(
                    f"Product {product.key!r} ({product.family.value}) "
                    "is not an L2B per-scene product. Use CMPlumeImage "
                    "for L3A per-plume products. The whole-scene RGB "
                    "sibling lives in a separate collection and is not "
                    "selected via products= — pass with_rgb=True (the "
                    "default) to attach it."
                )
            asset_paths[product.band] = _l2b_asset_url(
                l2b_coll, scene_id, product.key,
            )

        if with_rgb:
            if rgb_coll is None:
                rgb_coll = _probe_l2b_collection(
                    scene_id,
                    rgb_collection_candidates,
                    probe_asset="rgb.tif",
                    token=token,
                    http_timeout=http_timeout,
                )
            if rgb_coll is not None:
                asset_paths["rgb"] = _l2b_asset_url(rgb_coll, scene_id, "rgb.tif")

        return cls(
            scene_id=scene_id,
            asset_paths=asset_paths,
            overview_level=overview_level,
            token=token,
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
        ``None`` if absent — which is common: Carbon Mapper publishes it
        per scene, and no v3e scene checked in 2026-09 carried one."""
        return self._open_optional("artifact-mask")

    @cached_property
    def uas(self) -> Optional[str]:
        """UAS sensor-metadata sidecar — raw text from ``uas.txt``.

        Lazy-fetched on first access (one HTTP GET if the path is a
        URL, or a file read for local paths) and cached as a string.
        Callers parse the structure as needed; we don't impose a
        schema. Returns ``None`` if no ``uas`` URL/path was supplied.

        Auth: :attr:`token` when set; otherwise the ``Authorization``
        line of a process-wide ``GDAL_HTTP_HEADERS`` (which may hold
        several CRLF- or newline-separated headers), so the legacy
        env-var setup keeps working for text and raster assets alike.
        """
        path = self.asset_paths.get("uas")
        if path is None:
            return None
        sp = str(path)
        if sp.startswith(("http://", "https://")):
            headers: dict[str, str] = {}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            else:
                for line in os.environ.get("GDAL_HTTP_HEADERS", "").splitlines():
                    name, _, value = line.partition(":")
                    if name.strip().lower() == "authorization" and value.strip():
                        headers["Authorization"] = value.strip()
                        break
            r = _request("GET", sp, headers=headers, timeout=30)
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
                (absent from :attr:`asset_paths`, or an optional band
                the server doesn't have) or whose window has zero
                overlap return ``None``.

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
            reader = self._reader(band)
            if reader is None:
                out[band] = None
                continue
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
        """Read a WGS-84 bbox window from the requested bands.

        Raises:
            ValueError: If ``bounds_4326`` is malformed or crosses the
                antimeridian (``W > E``) — split such windows in two.
        """
        _validate_bbox(bounds_4326)
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
        return RasterioReader(
            str(path),
            overview_level=self.overview_level,
            rio_env_options=rio_env_options_for(str(path), self.token),
        )

    def _open_optional(self, band: str) -> Optional[RasterioReader]:
        """Open an optional band; ``None`` when it has no path **or** the
        server says it doesn't exist (403/404). Auth, throttling and
        server errors still raise."""
        if self.asset_paths.get(band) is None:
            return None
        try:
            return self._open(band)
        except RasterioIOError as exc:
            if not _is_absent_asset_error(exc):
                raise
            _log.info("Optional L2B asset %r absent for %s", band, self.scene_id)
            return None

    def _reader(self, band: str) -> Optional[RasterioReader]:
        """Reader for ``band`` through its cached accessor (so optional
        bands resolve to ``None`` and each band is opened once); bands
        without an accessor are opened directly."""
        attr = band.replace("-", "_")
        if attr in type(self).__dict__:
            return getattr(self, attr)
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
    "DEFAULT_L2B_CH4_COLLECTION_CANDIDATES",
    "DEFAULT_L2B_RGB_COLLECTION",
    "DEFAULT_L2B_RGB_COLLECTION_CANDIDATES",
    "BBox",
    "CMImageRaster",
    "l2b_collection_candidates_for_spec",
    "rgb_collection_candidates_for_spec",
]
