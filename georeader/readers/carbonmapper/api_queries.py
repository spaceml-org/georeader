"""High-level typed queries over the Carbon Mapper REST + STAC APIs.

This module is the typed, cross-resolution layer that sits between the
raw HTTP wrappers in :mod:`georeader.readers.carbonmapper.download` and consumers
(the Phase 2 ``DailyMonitoringCM`` ETL, analyst notebooks, future
Partner-feed backfills).

Why this exists
---------------
:mod:`download` exposes ~16 low-level endpoint wrappers that return raw
JSON / pandas DataFrames. Every consumer otherwise has to:

1. Pick the right endpoint (``/catalog/plume-csv`` vs
   ``/catalog/plumes/annotated`` vs STAC ``search`` — all three have
   different schemas).
2. Parse the response into something usable.
3. Stitch resources together by hand: plume → ``scene_id`` via
   ``rsplit("-", 1)[0]``, ``scene_id`` → STAC item, plume → source via
   ``/catalog/source/plume/name/{plume_id}``.

This module lifts those patterns into:

- One function per **logical question** (not per HTTP endpoint).
- Typed return values (:class:`CMRawPlume`, :class:`CMTileItem`,
  :class:`CMSource`) — never raw dicts.
- Owned knowledge of the bbox-encoding (``data_model §2.1``) and
  ``source_name`` query-suffix (``data_model §2.2``) quirks.

Failure modes
-------------
The exception hierarchy is part of the contract:

- :class:`CMPlumeNotFound`     — ``get_plume`` 404.
- :class:`CMSourceNotFound`    — ``get_source`` 404.
- :class:`CMSceneNotPublished` — ``get_tile`` / ``get_tile_for_plume``
  404 (CM publishes L2B selectively — ``data_model §5.2``). The
  cross-resolution helper :func:`get_tile_for_plume` *catches* this and
  returns ``None``; the single-resource :func:`get_tile` *re-raises*
  so callers can choose to defer.

Examples
--------
"What does CM know about this plume?":

>>> from georeader.readers.carbonmapper.api_queries import get_plume_context
>>> plume, tile, source = get_plume_context(token, "tan20251212t185057c20s4001-E")
>>> plume.plume_id
'tan20251212t185057c20s4001-E'
>>> tile.scene_id if tile else None
'tan20251212t185057c20s4001'
>>> source.sector if source else None  # may be None if unattributed
'1B2'

"All tiles ever observing this chronic emitter":

>>> from georeader.readers.carbonmapper.api_queries import list_tiles_for_source
>>> tiles = list_tiles_for_source(token, "CH4_1B2_100m_-104.17525_32.49125")
>>> {t.platform for t in tiles}
{'Tanager1', 'EMIT'}

See also
--------
georeader.readers.carbonmapper.download : raw HTTP / JSON wrappers.
georeader.readers.carbonmapper.plume.CMRawPlume : typed plume model.
georeader.readers.carbonmapper.source.CMSource : typed source model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import requests
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from georeader.readers.carbonmapper import download as _dl
from georeader.readers.carbonmapper.plume import CMRawPlume, Gas
from georeader.readers.carbonmapper.products import CMCollectionSpec
from georeader.readers.carbonmapper.source import CMSource, _strip_query_suffix

if TYPE_CHECKING:
    # Lazy import — `rasters.py` imports `CMSceneNotPublished` and
    # `CMTileItem` from this module, so we can't import its symbols at
    # module load without a cycle. `TYPE_CHECKING` keeps the annotation
    # available to checkers without triggering the runtime import.
    from georeader.readers.carbonmapper.rasters import CMImageRaster

BBox = tuple[float, float, float, float]   # (W, S, E, N) WGS-84

_log = logging.getLogger(__name__)

DEFAULT_L2B_COLLECTION = "l2b-ch4-mfa-v3a"


# ─────────────────────────────────────────────────────────────────────
#  Exceptions
# ─────────────────────────────────────────────────────────────────────


class CMAPIError(Exception):
    """Base for everything raised by :mod:`api_queries`.

    Catch this to handle any expected Carbon Mapper API miss in one
    block. ``requests.HTTPError`` for non-404 statuses (e.g. 500, 429)
    propagates unchanged — those are infra issues, not data issues.
    """


class CMPlumeNotFound(CMAPIError):
    """Raised by :func:`get_plume` when the plume is unknown to CM.

    The unmodified ``plume_id`` is preserved on the instance for
    logging.

    Examples
    --------
    >>> try:
    ...     get_plume(token, "tan-does-not-exist")  # doctest: +SKIP
    ... except CMPlumeNotFound as exc:
    ...     log.warning("missing plume", plume_id=exc.plume_id)
    """

    def __init__(self, plume_id: str):
        super().__init__(f"Plume not found: {plume_id}")
        self.plume_id = plume_id


class CMSourceNotFound(CMAPIError):
    """Raised by :func:`get_source` when the source name is unknown.

    The (cleaned, query-suffix-stripped) ``source_name`` is preserved
    on the instance.
    """

    def __init__(self, source_name: str):
        super().__init__(f"Source not found: {source_name}")
        self.source_name = source_name


class CMSceneNotPublished(CMAPIError):
    """Raised when STAC has no L2B item for a given ``scene_id``.

    Carbon Mapper publishes L2B selectively (``data_model.md §5.2``):
    plumes can exist for scenes whose L2B raster has not been (or never
    will be) released. The Phase 2 promotion path defers such plumes
    rather than failing hard.

    The :func:`get_tile` single-resource fetcher *raises* this so
    callers can pick a strategy; the cross-resolution
    :func:`get_tile_for_plume` *catches* it and returns ``None``.
    """

    def __init__(self, scene_id: str):
        super().__init__(f"L2B scene not published: {scene_id}")
        self.scene_id = scene_id


# ─────────────────────────────────────────────────────────────────────
#  Resource models
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CMTileItem:
    """Lightweight Carbon Mapper L2B STAC item — API-only, no DB binding.

    The DB-bound counterpart is ``CarbonMapperTile`` (Phase 1). The
    promotion direction (API → DB) lives on the *DB* side via
    ``CarbonMapperTile.from_cm_tile_item(item, cm_provider=...)``; this
    keeps :mod:`api_queries` free of any database imports.

    Frozen so instances are hashable and safe to use as dict keys when
    deduplicating ``scene_ids`` in cross-resolution queries.

    Attributes
    ----------
    scene_id:
        STAC item id — equivalent to ``plume_id.rsplit("-", 1)[0]`` for
        plumes that came from this scene.
    collection:
        STAC collection id, e.g. ``"l2b-ch4-mfa-v3a"``.
    datetime:
        UTC-aware acquisition time parsed from
        ``properties["datetime"]``.
    platform:
        ``properties["platform"]`` — ``"Tanager1"``, ``"EMIT"``, etc.
    bbox:
        ``(W, S, E, N)`` in WGS-84 decimal degrees.
    geometry:
        Shapely geometry (typically a Polygon) of the scene footprint.
    asset_urls:
        Mapping of asset name → href URL, e.g.
        ``{"cmf": "https://.../cmf.tif", "rgb": ...}``. The L2B CH4
        collection consistently exposes ``cmf``, ``rgb``,
        ``uncertainty``, and ``artifact-mask``.
    properties:
        Full ``properties`` mapping from the STAC item.
    raw:
        Original STAC item dict — useful for fields not yet exposed
        on the dataclass.

    Examples
    --------
    >>> from georeader.readers.carbonmapper.api_queries import CMTileItem
    >>> tile = CMTileItem.from_stac_item({
    ...     "id": "tan20251212t185057c20s4001",
    ...     "collection": "l2b-ch4-mfa-v3a",
    ...     "properties": {"datetime": "2025-12-12T18:50:57Z",
    ...                    "platform": "Tanager1"},
    ...     "bbox": [-103.6, 31.4, -103.4, 31.6],
    ...     "geometry": {"type": "Polygon", "coordinates": [
    ...         [[-103.6, 31.4], [-103.4, 31.4],
    ...          [-103.4, 31.6], [-103.6, 31.6], [-103.6, 31.4]]]},
    ...     "assets": {"cmf": {"href": "https://cm/.../cmf.tif"}},
    ... })
    >>> tile.scene_id, tile.platform
    ('tan20251212t185057c20s4001', 'Tanager1')
    >>> tile.asset_urls["cmf"]
    'https://cm/.../cmf.tif'
    """

    scene_id: str
    collection: str
    datetime: datetime
    platform: str
    bbox: tuple[float, float, float, float]
    geometry: BaseGeometry
    asset_urls: Mapping[str, str]
    properties: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def from_stac_item(cls, item: Mapping[str, Any]) -> "CMTileItem":
        """Build a :class:`CMTileItem` from a raw STAC item dict.

        Tolerates both string and pre-parsed datetime values for
        ``properties["datetime"]`` and falls back to ``utcnow`` if the
        property is missing entirely.

        Parameters
        ----------
        item:
            STAC item dict (Feature shape) as returned by
            :func:`georeader.readers.carbonmapper.download.stac_get_item` or
            :func:`georeader.readers.carbonmapper.download.stac_search`.

        Returns
        -------
        CMTileItem

        Raises
        ------
        ValueError
            If ``item["bbox"]`` is missing or not 4-length.
        """
        props = dict(item.get("properties") or {})
        bbox = tuple(item.get("bbox") or ())
        if len(bbox) != 4:
            raise ValueError(f"STAC item missing 4-tuple bbox: {item.get('id')!r}")

        dt_raw = props.get("datetime")
        if isinstance(dt_raw, datetime):
            dt = dt_raw
        elif isinstance(dt_raw, str):
            dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00"))
        else:
            dt = datetime.now(timezone.utc)

        geom_dict = item.get("geometry") or {}
        if not geom_dict:
            raise ValueError(f"STAC item missing geometry: {item.get('id')!r}")
        geom = shape(geom_dict)  # type: ignore[arg-type]

        assets = item.get("assets") or {}
        asset_urls = {
            name: asset.get("href", "")
            for name, asset in assets.items()
            if isinstance(asset, Mapping)
        }

        return cls(
            scene_id=str(item.get("id", "")),
            collection=str(item.get("collection", "")),
            datetime=dt,
            platform=str(props.get("platform", "")),
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            geometry=geom,
            asset_urls=asset_urls,
            properties=props,
            raw=dict(item),
        )


# ─────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────


def _is_404(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        resp = getattr(exc, "response", None)
        return resp is not None and getattr(resp, "status_code", None) == 404
    return False


def _scene_id_from_plume(plume_id: str) -> str:
    """Derive the parent scene_id from a plume_id.

    ``plume_id = "{scene_id}-{part}"`` per Carbon Mapper convention.
    """
    return plume_id.rsplit("-", 1)[0]


# ─────────────────────────────────────────────────────────────────────
#  Single-resource fetchers
# ─────────────────────────────────────────────────────────────────────


def get_plume(token: str, plume_id: str) -> CMRawPlume:
    """Fetch a single plume by its CM ``plume_id``.

    Wraps ``GET /catalog/plume/{id}`` and parses the result through
    :class:`CMRawPlume`.

    Parameters
    ----------
    token:
        Carbon Mapper Bearer token. Required for non-public fields.
    plume_id:
        Either the colloquial name (e.g.
        ``"tan20251212t185057c20s4001-E"``) or the UUID form.

    Returns
    -------
    CMRawPlume

    Raises
    ------
    CMPlumeNotFound
        When the API returns 404.
    requests.HTTPError
        For non-404 errors (5xx, 429, etc.).

    Examples
    --------
    >>> plume = get_plume(token, "tan20251212t185057c20s4001-E")  # doctest: +SKIP
    >>> plume.plume_id, plume.gas
    ('tan20251212t185057c20s4001-E', 'CH4')
    """
    try:
        raw = _dl.get_plume_by_id(plume_id, token=token)
    except requests.HTTPError as exc:
        if _is_404(exc):
            raise CMPlumeNotFound(plume_id) from exc
        raise
    return CMRawPlume(**raw)


def get_tile(
    token: str,
    scene_id: str,
    *,
    collection: str = DEFAULT_L2B_COLLECTION,
) -> CMTileItem:
    """Fetch a single L2B STAC item by ``scene_id``.

    Wraps ``GET /stac/collections/{collection}/items/{scene_id}``.

    Parameters
    ----------
    token:
        Bearer token (STAC item endpoints accept anonymous reads for
        published items, but auth surfaces additional fields).
    scene_id:
        The L2B scene_id, equal to ``plume_id.rsplit("-", 1)[0]`` for
        any plume that came from this scene.
    collection:
        STAC collection — defaults to :data:`DEFAULT_L2B_COLLECTION`
        (CH4 matched-filter v3a). Override for CO2 or earlier versions.

    Returns
    -------
    CMTileItem

    Raises
    ------
    CMSceneNotPublished
        When the L2B item has not been published yet (HTTP 404).
        Re-raised — not caught — so callers can choose to defer.

    Examples
    --------
    >>> tile = get_tile(token, "tan20251212t185057c20s4001")  # doctest: +SKIP
    >>> tile.platform, list(tile.asset_urls)
    ('Tanager1', ['cmf', 'rgb', 'uncertainty', 'artifact-mask'])
    """
    try:
        raw = _dl.stac_get_item(collection, scene_id, token=token)
    except requests.HTTPError as exc:
        if _is_404(exc):
            raise CMSceneNotPublished(scene_id) from exc
        raise
    return CMTileItem.from_stac_item(raw)


def get_source(token: str, source_name: str) -> CMSource:
    """Fetch a single Carbon Mapper source by its canonical name.

    Strips the source-name query-string suffix (``?plume_gas=...``)
    automatically (``data_model §2.2``) — pass either the dirty or
    clean form.

    Parameters
    ----------
    token:
        Bearer token.
    source_name:
        Canonical or query-suffixed source name, e.g.
        ``"CH4_1B2_100m_-104.17525_32.49125"`` or
        ``"CH4_1B2_100m_-104.17525_32.49125?plume_gas=CH4"``.

    Returns
    -------
    CMSource

    Raises
    ------
    CMSourceNotFound
        When the API returns 404.

    Examples
    --------
    >>> src = get_source(token, "CH4_1B2_100m_-104.17525_32.49125")  # doctest: +SKIP
    >>> src.sector, src.plume_count
    ('1B2', 12)
    """
    cleaned = _strip_query_suffix(source_name)
    try:
        raw = _dl.get_source_by_name(cleaned, token=token)
    except requests.HTTPError as exc:
        if _is_404(exc):
            raise CMSourceNotFound(cleaned) from exc
        raise
    # The single-source endpoint can return either a Feature or properties
    # directly; coerce to a Feature shape so CMSource.from_geojson_feature
    # handles both.
    if "properties" not in raw and "source_name" in raw:
        feature = {"properties": dict(raw),
                   "geometry": {"type": "Point",
                                "coordinates": [raw.get("lon"), raw.get("lat")]}}
    else:
        feature = dict(raw)

    # The /catalog/source/{name} endpoint sometimes returns top-level
    # geometry with null coords and stashes the real centroid under
    # properties.point — fall back to that when the outer geometry is
    # unusable.
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or [None, None]
    if not coords or coords[0] is None or coords[1] is None:
        props = feature.get("properties") or {}
        point = props.get("point") or {}
        pcoords = point.get("coordinates") if isinstance(point, dict) else None
        if pcoords and pcoords[0] is not None and pcoords[1] is not None:
            feature = dict(feature)
            feature["geometry"] = {"type": "Point", "coordinates": list(pcoords)}

    return CMSource.from_geojson_feature(feature)


# ─────────────────────────────────────────────────────────────────────
#  Listing with filters
# ─────────────────────────────────────────────────────────────────────


def list_plumes(
    token: str,
    *,
    bbox: BBox | None = None,
    sectors: list[str] | None = None,
    instruments: list[str] | None = None,
    datetime_min: datetime | None = None,
    datetime_max: datetime | None = None,
    gas: Gas | Literal["CH4"] = Gas.CH4,
    limit: int = 1_000,
) -> list[CMRawPlume]:
    """Materialised list of plumes matching filters.

    Wraps ``/catalog/plumes/annotated`` and converts each row into a
    :class:`CMRawPlume`. The bbox is encoded as repeated keys (REST
    style — see :func:`georeader.readers.carbonmapper.download._rest_bbox_params`).

    Parameters
    ----------
    token:
        Bearer token.
    bbox:
        ``(W, S, E, N)`` WGS-84 spatial filter.
    sectors:
        IPCC sector codes — e.g. ``["1B2", "6A"]``.
    instruments:
        Instrument short codes — e.g. ``["emi", "tan"]`` or
        :class:`Instrument` members like ``[Instrument.EMIT, Instrument.TANAGER]``.
    datetime_min, datetime_max:
        Optional UTC bounds — combined into an RFC 3339 interval.
    gas:
        :data:`Gas.CH4` (default). **CH4-only for this PR**;
        ``Gas.CO2`` lands in a follow-up. Typed as
        ``Gas | Literal["CH4"]`` so plain string call-sites
        (``gas="CH4"``) continue to type-check.
    limit:
        Max rows returned in this call. The API caps at 1 000 per page.

    Returns
    -------
    list[CMRawPlume]

    Examples
    --------
    Permian methane plumes for Q1 2025 from EMIT and Tanager:

    >>> from datetime import datetime, timezone
    >>> plumes = list_plumes(  # doctest: +SKIP
    ...     token,
    ...     bbox=(-104.5, 31.0, -101.5, 33.5),
    ...     instruments=["emi", "tan"],
    ...     datetime_min=datetime(2025, 1, 1, tzinfo=timezone.utc),
    ...     datetime_max=datetime(2025, 4, 1, tzinfo=timezone.utc),
    ...     limit=500,
    ... )
    >>> sum(p.emission_auto or 0 for p in plumes)  # doctest: +SKIP
    412350.0
    """
    dt_range = _build_datetime_range(datetime_min, datetime_max)
    result = _dl.get_plumes_annotated(
        plume_gas=str(gas),
        bbox=bbox,
        datetime_range=dt_range,
        sectors=sectors,
        instruments=instruments,
        limit=limit,
        token=token,
    )
    items = result.get("items", []) if isinstance(result, Mapping) else []
    return [CMRawPlume(**row) for row in items]


def list_tiles(
    token: str,
    *,
    bbox: BBox | None = None,
    datetime_min: datetime | None = None,
    datetime_max: datetime | None = None,
    collection: str = DEFAULT_L2B_COLLECTION,
    limit: int = 1_000,
) -> list[CMTileItem]:
    """Materialised list of L2B STAC items matching filters.

    Wraps ``/stac/search`` (comma-joined STAC bbox encoding).

    Parameters
    ----------
    token:
        Bearer token.
    bbox:
        ``(W, S, E, N)`` WGS-84 spatial filter.
    datetime_min, datetime_max:
        Optional UTC bounds.
    collection:
        STAC collection — defaults to :data:`DEFAULT_L2B_COLLECTION`.
    limit:
        Max items in this call.

    Returns
    -------
    list[CMTileItem]

    Examples
    --------
    >>> tiles = list_tiles(  # doctest: +SKIP
    ...     token, bbox=(-104.5, 31.0, -101.5, 33.5), limit=10,
    ... )
    >>> {t.platform for t in tiles}  # doctest: +SKIP
    {'Tanager1', 'EMIT'}
    """
    dt_range = _build_datetime_range(datetime_min, datetime_max)
    result = _dl.stac_search(
        collections=[collection],
        bbox=bbox,
        datetime_range=dt_range,
        limit=limit,
        token=token,
    )
    features = result.get("features", []) if isinstance(result, Mapping) else []
    return [CMTileItem.from_stac_item(f) for f in features]


def list_sources(
    token: str,
    *,
    bbox: BBox | None = None,
    sectors: list[str] | None = None,
    gas: Gas | Literal["CH4"] = Gas.CH4,
) -> list[CMSource]:
    """List Carbon Mapper sources matching filters.

    Wraps the source listing endpoint (REST Catalog). Each item is
    parsed via :meth:`CMSource.from_geojson_feature`, which strips the
    source-name query-suffix.

    Parameters
    ----------
    token:
        Bearer token.
    bbox:
        ``(W, S, E, N)`` WGS-84 spatial filter (REST repeated-keys
        encoding).
    sectors:
        IPCC sector codes.
    gas:
        :data:`Gas.CH4` (default). **CH4-only for this PR**;
        ``Gas.CO2`` lands in a follow-up.

    Returns
    -------
    list[CMSource]

    Examples
    --------
    Top oil & gas sources in the Permian:

    >>> sources = list_sources(  # doctest: +SKIP
    ...     token,
    ...     bbox=(-104.5, 31.0, -101.5, 33.5),
    ...     sectors=["1B2"],
    ... )
    >>> sorted(sources, key=lambda s: -(s.emission_auto or 0))[:3]  # doctest: +SKIP
    [<CMSource ...>, <CMSource ...>, <CMSource ...>]
    """
    # The `download.get_sources` wrapper actually targets
    # `/plumes/annotated` (see its docstring) — the true source listing
    # lives at `/catalog/sources.geojson` and returns a GeoJSON
    # FeatureCollection. Hit it directly with REST repeated-keys bbox.
    params: list[tuple[str, str]] = [("plume_gas", str(gas))]
    if bbox is not None:
        for v in bbox:
            params.append(("bbox", str(v)))
    if sectors:
        for s in sectors:
            params.append(("sectors", s))
    resp = requests.get(
        f"{_dl.CATALOG_URL}/sources.geojson",
        params=params,
        headers=_dl._headers(token),
        timeout=60,
    )
    resp.raise_for_status()
    fc = resp.json()
    features = fc.get("features", []) if isinstance(fc, Mapping) else []
    return [CMSource.from_geojson_feature(f) for f in features]


# ─────────────────────────────────────────────────────────────────────
#  Cross-resolution
# ─────────────────────────────────────────────────────────────────────


def get_tile_for_plume(
    token: str,
    plume_id: str,
    *,
    collection: str = DEFAULT_L2B_COLLECTION,
) -> CMTileItem | None:
    """Resolve a plume to its parent L2B STAC item.

    Derives the parent ``scene_id`` via
    ``plume_id.rsplit("-", 1)[0]`` and looks up the corresponding
    STAC item.

    Unlike :func:`get_tile`, this helper **catches**
    :class:`CMSceneNotPublished` and returns ``None`` — appropriate for
    consumers (Phase 2 ETL) that want to defer rather than error.

    Parameters
    ----------
    token:
        Bearer token.
    plume_id:
        Colloquial plume id (with the ``-{part}`` suffix).
    collection:
        STAC collection — defaults to :data:`DEFAULT_L2B_COLLECTION`.

    Returns
    -------
    CMTileItem | None
        ``None`` when the L2B scene has not been published yet.

    Examples
    --------
    >>> tile = get_tile_for_plume(token, "tan20251212t185057c20s4001-E")  # doctest: +SKIP
    >>> tile.scene_id if tile else "deferred"  # doctest: +SKIP
    'tan20251212t185057c20s4001'
    """
    scene_id = _scene_id_from_plume(plume_id)
    try:
        return get_tile(token, scene_id, collection=collection)
    except CMSceneNotPublished:
        return None


def get_image_raster_for_scene(
    token: str,
    scene_id: str,
    *,
    collection: str = DEFAULT_L2B_COLLECTION,
    prefer_url_pattern_fallback: bool = True,
    with_rgb: bool = True,
) -> CMImageRaster | None:
    """Resolve a ``scene_id`` to a :class:`CMImageRaster` with rgb sibling.

    Single entry point for "give me the L2B parent tile as a usable
    raster object" — handles both STAC-resident scenes (v3a) and
    URL-pattern-only scenes (v3c, the live 2026 L2B version)
    transparently:

    1. Try STAC item lookup (cheap when it works — one HTTP call for
       the CH4 item, one for the RGB sibling). Builds via
       :meth:`CMImageRaster.from_cm_tile_item` + :meth:`with_rgb`.
    2. If STAC returns 404 AND ``prefer_url_pattern_fallback`` is
       ``True`` (default), fall back to
       :meth:`CMImageRaster.from_scene_id` which probes the verified
       L2B asset-proxy URL pattern (see design doc §4.7).
    3. Return ``None`` only if both paths fail.

    This is the helper :class:`CMPlumeImage` will use (Phase 2) to
    expose ``.tile`` as a lazy property.

    Parameters
    ----------
    token:
        Bearer token.
    scene_id:
        L2B scene id (e.g. ``"tan20260331t181625c77s4001"``).
    collection:
        STAC collection probed first. Defaults to
        :data:`DEFAULT_L2B_COLLECTION` (``l2b-ch4-mfa-v3a``).
    prefer_url_pattern_fallback:
        When ``True`` (default), fall back to URL-pattern derivation
        on STAC 404. Set to ``False`` to keep the v3a-only behaviour
        of :func:`get_tile_for_plume`.
    with_rgb:
        When ``True`` (default), attach the RGB sibling raster.

    Returns
    -------
    CMImageRaster | None
        ``None`` when STAC has no item AND either the fallback is
        disabled or the URL-pattern probes also 404.

    Examples
    --------
    >>> tile = get_image_raster_for_scene(  # doctest: +SKIP
    ...     token, "tan20260331t181625c77s4001",
    ... )
    >>> tile.cmf  # doctest: +SKIP
    <RasterioReader …/l2b-ch4-mfa-v3c/2026/03/31/…/cmf.tif>
    """
    # Lazy import to avoid the rasters → api_queries circular import.
    from georeader.readers.carbonmapper.rasters import CMImageRaster

    # ── 1. STAC path (cheap when it works) ──────────────────────────
    try:
        ch4_item = get_tile(token, scene_id, collection=collection)
        ir = CMImageRaster.from_cm_tile_item(ch4_item)
        if with_rgb:
            try:
                # RGB sibling lives in `l2b-rgb-v3a` (string defined here
                # rather than imported from rasters.py to avoid the
                # rasters→api_queries circular import).
                rgb_item = get_tile(
                    token, scene_id, collection="l2b-rgb-v3a",
                )
                ir = ir.with_rgb(rgb_item)
            except CMSceneNotPublished:
                # CH4 item exists but RGB sibling doesn't — keep the
                # CH4-only raster rather than failing the whole call.
                pass
        return ir
    except CMSceneNotPublished:
        if not prefer_url_pattern_fallback:
            return None
    # Fall through to URL-pattern path.

    # ── 2. URL-pattern fallback (for v3c/v3d scenes not in STAC) ────
    try:
        return CMImageRaster.from_scene_id(
            scene_id, token=token, with_rgb=with_rgb,
        )
    except CMSceneNotPublished:
        return None


def get_image_raster_for_plume(
    token: str,
    plume_id: str,
    *,
    collection: str = DEFAULT_L2B_COLLECTION,
    prefer_url_pattern_fallback: bool = True,
    with_rgb: bool = True,
) -> CMImageRaster | None:
    """Resolve a plume to its parent :class:`CMImageRaster`.

    Sugar over :func:`get_image_raster_for_scene` that derives the
    parent ``scene_id`` from ``plume_id`` first. Same STAC-first /
    URL-pattern-fallback behaviour.

    Parameters
    ----------
    token, collection, prefer_url_pattern_fallback, with_rgb:
        Forwarded to :func:`get_image_raster_for_scene`.
    plume_id:
        Colloquial plume id (with the ``-{part}`` suffix).

    Returns
    -------
    CMImageRaster | None

    Examples
    --------
    >>> tile = get_image_raster_for_plume(  # doctest: +SKIP
    ...     token, "tan20260331t181625c77s4001-A",
    ... )
    >>> tile is not None  # doctest: +SKIP
    True
    """
    scene_id = _scene_id_from_plume(plume_id)

    # Preferred path: one catalog fetch resolves the CMCollectionSpec
    # (gas / cmf_type / version) from the plume's own record, which
    # names the L2B parent collection at the same version — verified
    # pairing, no probing, and it never goes stale when Carbon Mapper
    # bumps versions. Falls back to the STAC-first/probe-second dance
    # only when the record is unavailable or unparseable.
    try:
        record = _dl.get_plume_by_id(plume_id, token=token)
        spec = CMCollectionSpec.from_plume_record(record)
    except (requests.HTTPError, requests.ConnectionError, ValueError) as exc:
        _log.debug(
            "Could not resolve CMCollectionSpec for plume %s (%s); "
            "falling back to STAC + candidate probing", plume_id, exc,
        )
        spec = None

    if spec is not None:
        from georeader.readers.carbonmapper.rasters import CMImageRaster

        return CMImageRaster.from_scene_id(
            scene_id, token=token, spec=spec, with_rgb=with_rgb,
        )

    return get_image_raster_for_scene(
        token, scene_id,
        collection=collection,
        prefer_url_pattern_fallback=prefer_url_pattern_fallback,
        with_rgb=with_rgb,
    )


def get_source_for_plume(
    token: str,
    plume_id: str,
) -> CMSource | None:
    """Resolve a plume to its attributed Carbon Mapper source.

    Wraps ``/catalog/source/plume/name/{plume_id}`` — the *by-name*
    endpoint, which returns the cleaned ``source_name`` (preferred over
    the UUID-keyed sibling for colloquial ``plume_id`` strings).

    Returns ``None`` when CM has not attributed the plume to a source
    (HTTP 404). Other HTTP errors propagate.

    Parameters
    ----------
    token:
        Bearer token.
    plume_id:
        Colloquial plume id.

    Returns
    -------
    CMSource | None

    Examples
    --------
    >>> src = get_source_for_plume(token, "tan20251212t185057c20s4001-E")  # doctest: +SKIP
    >>> src.source_name if src else "unattributed"  # doctest: +SKIP
    'CH4_1B2_100m_-104.0_32.0'
    """
    try:
        raw = _dl.get_source_for_plume_name(plume_id, token=token)
    except requests.HTTPError as exc:
        if _is_404(exc):
            return None
        raise
    if not raw:
        return None
    if "geometry" not in raw and "properties" not in raw:
        feature = {
            "properties": dict(raw),
            "geometry": {"type": "Point",
                         "coordinates": [raw.get("lon"), raw.get("lat")]},
        }
    else:
        feature = dict(raw)

    # The endpoint occasionally returns a plume-shaped payload (no
    # source_name, null top-level geometry) when CM has not yet
    # attributed the plume — treat as unattributed.
    props = feature.get("properties") or {}
    if not props.get("source_name") and not feature.get("source_name"):
        return None

    # Fall back to properties.point when the outer geometry is null
    # (same quirk as get_source).
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or [None, None]
    if not coords or coords[0] is None or coords[1] is None:
        point = props.get("point") or {}
        pcoords = point.get("coordinates") if isinstance(point, dict) else None
        if pcoords and pcoords[0] is not None and pcoords[1] is not None:
            feature["geometry"] = {"type": "Point", "coordinates": list(pcoords)}

    return CMSource.from_geojson_feature(feature)


def get_plume_context(
    token: str,
    plume_id: str,
) -> tuple[CMRawPlume, CMTileItem | None, CMSource | None]:
    """Single-call fetch of a plume plus its parent tile and source.

    The most common notebook / ETL question is *"give me everything CM
    knows about this plume"*. This helper batches the three independent
    REST/STAC calls behind a single name and surfaces the contracts as
    a typed tuple.

    Failure modes are asymmetric:

    - The plume itself **must** exist — ``CMPlumeNotFound`` propagates.
    - Tile resolution returns ``None`` when the scene has not been
      published to L2B (``CMSceneNotPublished`` caught internally).
    - Source resolution returns ``None`` when CM has not attributed
      the plume (404 caught internally).

    Parameters
    ----------
    token:
        Bearer token.
    plume_id:
        Colloquial plume id.

    Returns
    -------
    (CMRawPlume, CMTileItem | None, CMSource | None)

    Raises
    ------
    CMPlumeNotFound
        When the plume itself is unknown.

    Examples
    --------
    Notebook exploration:

    >>> plume, tile, source = get_plume_context(  # doctest: +SKIP
    ...     token, "tan20251212t185057c20s4001-E",
    ... )
    >>> print(f"emission: {plume.emission_auto:.0f} kg/h")  # doctest: +SKIP
    emission: 1240 kg/h
    >>> if source:                                          # doctest: +SKIP
    ...     print(f"source {source.source_name} sector {source.sector}")
    """
    plume = get_plume(token, plume_id)
    tile = get_tile_for_plume(token, plume_id)
    source = get_source_for_plume(token, plume_id)
    return plume, tile, source


def list_plumes_for_tile(
    token: str,
    scene_id: str,
    *,
    gas: Gas | Literal["CH4"] = Gas.CH4,
) -> list[CMRawPlume]:
    """All plumes attributed to a given L2B scene.

    Carbon Mapper plume_ids embed the scene_id —
    ``plume_id = "{scene_id}-{part}"`` — so we filter the annotated
    plumes listing client-side by prefix.

    Parameters
    ----------
    token:
        Bearer token.
    scene_id:
        L2B scene id, e.g. ``"tan20251212t185057c20s4001"``.
    gas:
        :data:`Gas.CH4` (default). **CH4-only for this PR**;
        ``Gas.CO2`` lands in a follow-up.

    Returns
    -------
    list[CMRawPlume]

    Note
    ----
    The current implementation pulls a 1 000-plume page and filters
    in Python. For high-volume scenes that may miss tail rows; pass a
    bbox filter or use :func:`list_plumes` directly when completeness
    matters.

    Examples
    --------
    >>> plumes = list_plumes_for_tile(  # doctest: +SKIP
    ...     token, "tan20251212t185057c20s4001",
    ... )
    >>> [p.plume_id[-1] for p in plumes]  # doctest: +SKIP
    ['A', 'B', 'C', 'E']
    """
    result = _dl.get_plumes_annotated(
        plume_gas=str(gas),
        limit=1_000,
        token=token,
    )
    items = result.get("items", []) if isinstance(result, Mapping) else []
    prefix = f"{scene_id}-"
    return [
        CMRawPlume(**row)
        for row in items
        if str(row.get("plume_id", "")).startswith(prefix)
    ]


def list_plumes_for_source(
    token: str,
    source_name: str,
    *,
    limit: int = 10_000,
) -> list[CMRawPlume]:
    """All plumes attributed to a Carbon Mapper source.

    Wraps ``/catalog/source-plumes-csv/{source_name}``. The CSV
    endpoint is single-shot (no pagination) — the result is fully
    materialised.

    Strips the ``?...`` query suffix from ``source_name`` automatically
    (``data_model §2.2``).

    Parameters
    ----------
    token:
        Bearer token.
    source_name:
        Canonical or query-suffixed source name.
    limit:
        Cap the returned list. Defaults to 10 000 — CM sources rarely
        exceed a few hundred plumes, so this is just a safety cap.

    Returns
    -------
    list[CMRawPlume]

    Examples
    --------
    >>> plumes = list_plumes_for_source(  # doctest: +SKIP
    ...     token, "CH4_1B2_100m_-104.17525_32.49125",
    ... )
    >>> len(plumes), plumes[0].plume_id[:3]  # doctest: +SKIP
    (47, 'tan')
    """
    import io
    import pandas as pd

    cleaned = _strip_query_suffix(source_name)
    csv_text = _dl.get_source_plumes_csv(cleaned, token=token)
    if not csv_text:
        return []
    df = pd.read_csv(io.StringIO(csv_text))
    if limit and len(df) > limit:
        df = df.head(limit)
    # CSV -> dict gives `float('nan')` for empty cells. Pydantic
    # str-typed fields like `sensitivity_mode` reject NaN; coerce
    # NaNs to None so optional fields fall back to their defaults.
    rows = df.to_dict(orient="records")
    cleaned: list[CMRawPlume] = []
    for row in rows:
        sane = {k: (None if isinstance(v, float) and v != v else v)
                for k, v in row.items()}
        cleaned.append(CMRawPlume(**sane))
    return cleaned


def list_tiles_for_source(
    token: str,
    source_name: str,
    *,
    collection: str = DEFAULT_L2B_COLLECTION,
) -> list[CMTileItem]:
    """All distinct parent L2B tiles touched by a source's plumes.

    Implementation:

    1. :func:`list_plumes_for_source` — every plume attributed to the
       source.
    2. ``{plume_id.rsplit("-", 1)[0] for ...}`` — distinct scene_ids.
    3. ``stac_search(ids=[...])`` — resolve to STAC items.

    Useful for tile-level backfill: given a chronic emitter, fetch
    every L2B scene that ever observed it, regardless of whether
    plumes were detected on a given pass.

    Parameters
    ----------
    token:
        Bearer token.
    source_name:
        Canonical or query-suffixed source name.
    collection:
        STAC collection — defaults to :data:`DEFAULT_L2B_COLLECTION`.

    Returns
    -------
    list[CMTileItem]
        Empty list if the source has no plumes.

    Examples
    --------
    >>> tiles = list_tiles_for_source(  # doctest: +SKIP
    ...     token, "CH4_1B2_100m_-104.17525_32.49125",
    ... )
    >>> sorted({t.platform for t in tiles})  # doctest: +SKIP
    ['EMIT', 'Tanager1']
    """
    plumes = list_plumes_for_source(token, source_name)
    scene_ids = sorted({_scene_id_from_plume(p.plume_id) for p in plumes})
    if not scene_ids:
        return []
    result = _dl.stac_search(
        collections=[collection], ids=scene_ids, limit=len(scene_ids), token=token,
    )
    features = result.get("features", []) if isinstance(result, Mapping) else []
    return [CMTileItem.from_stac_item(f) for f in features]


# ─────────────────────────────────────────────────────────────────────
#  Helpers (private)
# ─────────────────────────────────────────────────────────────────────


def _build_datetime_range(
    dt_min: datetime | None, dt_max: datetime | None,
) -> str | None:
    """Build an RFC 3339 datetime-range string from optional bounds."""
    if dt_min is None and dt_max is None:
        return None
    lo = dt_min.isoformat().replace("+00:00", "Z") if dt_min else ".."
    hi = dt_max.isoformat().replace("+00:00", "Z") if dt_max else ".."
    return f"{lo}/{hi}"


__all__ = [
    "BBox",
    "CMAPIError",
    "CMPlumeNotFound",
    "CMSceneNotPublished",
    "CMSourceNotFound",
    "CMTileItem",
    "DEFAULT_L2B_COLLECTION",
    "get_plume",
    "get_plume_context",
    "get_source",
    "get_source_for_plume",
    "get_tile",
    "get_tile_for_plume",
    "list_plumes",
    "list_plumes_for_source",
    "list_plumes_for_tile",
    "list_sources",
    "list_tiles",
    "list_tiles_for_source",
]
