"""
download.py
===========

Carbon Mapper Data Platform API client for the marsml pipeline.

Provides typed wrappers around three Carbon Mapper APIs:

    1. **REST Catalog API**  — plumes, sources, scenes, plume CSV, assets
    2. **STAC API**          — spatiotemporal search across collections
    3. **Asset Download**    — GeoTIFF retrievals, RGB imagery, plume PNGs

Authentication
--------------
Most *read* endpoints work without a token, but some (scenes, related
plumes, STAC tokens) require a Bearer token.  Use
:func:`obtain_token` or
:meth:`~georeader.readers.carbonmapper.config.CarbonMapperConfig.refresh_access_token`
to obtain one from credentials in ``config/carbonmapper_token.json``.

References
----------
- API Docs      : https://api.carbonmapper.org/api/v1/docs
- STAC Root     : https://api.carbonmapper.org/api/v1/stac/
- Registration  : https://data.carbonmapper.org
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, cast
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://api.carbonmapper.org/api/v1"
CATALOG_URL = f"{BASE_URL}/catalog"
STAC_URL = f"{BASE_URL}/stac"

BBox = tuple[float, float, float, float]   # (W, S, E, N) WGS-84


def _rest_bbox_params(bbox: BBox | None) -> dict[str, list[str]]:
    """Encode a bounding box for the **REST Catalog** API.

    The Carbon Mapper REST Catalog requires the bbox as **four repeated
    query keys** (``?bbox=W&bbox=S&bbox=E&bbox=N``); the comma-joined
    STAC encoding is rejected with HTTP 422. ``requests`` serialises a
    list-valued query parameter as repeated keys, so the returned shape
    can be merged directly into a normal ``params`` dict.

    Parameters
    ----------
    bbox:
        ``(west_lon, south_lat, east_lon, north_lat)`` in WGS-84 decimal
        degrees, or ``None`` to omit the bbox.

    Returns
    -------
    dict[str, list[str]]
        ``{}`` if ``bbox`` is ``None``; otherwise
        ``{"bbox": ["W", "S", "E", "N"]}`` ready for ``params.update(...)``.

    Raises
    ------
    ValueError
        If ``bbox`` is not length 4.

    Examples
    --------
    >>> _rest_bbox_params(None)
    {}
    >>> _rest_bbox_params((-104.5, 31.0, -101.5, 33.5))
    {'bbox': ['-104.5', '31.0', '-101.5', '33.5']}

    Typical caller pattern:

    >>> params = {"limit": 25}
    >>> params.update(_rest_bbox_params((-104.5, 31.0, -101.5, 33.5)))
    >>> # ?limit=25&bbox=-104.5&bbox=31.0&bbox=-101.5&bbox=33.5
    """
    if bbox is None:
        return {}
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (W, S, E, N); got {bbox!r}")
    return {"bbox": [str(v) for v in bbox]}


def _stac_bbox_param(bbox: BBox | None) -> dict[str, str]:
    """Encode a bounding box for the **STAC** API.

    STAC's ``/search`` and ``/collections/{id}/items`` endpoints require
    the bbox as a single **comma-joined** key
    (``?bbox=W,S,E,N``) per the OGC API – Features specification. This
    is the opposite encoding from the REST Catalog (see
    :func:`_rest_bbox_params`).

    Parameters
    ----------
    bbox:
        ``(west_lon, south_lat, east_lon, north_lat)`` in WGS-84 decimal
        degrees, or ``None`` to omit the bbox.

    Returns
    -------
    dict[str, str]
        ``{}`` if ``bbox`` is ``None``; otherwise
        ``{"bbox": "W,S,E,N"}`` ready for ``params.update(...)``.

    Raises
    ------
    ValueError
        If ``bbox`` is not length 4.

    Examples
    --------
    >>> _stac_bbox_param(None)
    {}
    >>> _stac_bbox_param((-104.5, 31.0, -101.5, 33.5))
    {'bbox': '-104.5,31.0,-101.5,33.5'}
    """
    if bbox is None:
        return {}
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (W, S, E, N); got {bbox!r}")
    return {"bbox": ",".join(str(v) for v in bbox)}


def _headers(token: str | None = None) -> dict[str, str]:
    """Build request headers, optionally with Bearer auth."""
    h: dict[str, str] = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _get(url: str, params: dict | None = None, token: str | None = None) -> dict | list | str:
    """GET with basic error handling and rate-limit back-off."""
    resp = requests.get(url, params=params, headers=_headers(token), timeout=60)
    if resp.status_code == 429:
        wait = min(int(resp.headers.get("Retry-After", 5)), 300)
        logger.warning("Rate-limited; sleeping %d s", wait)
        time.sleep(wait)
        resp = requests.get(url, params=params, headers=_headers(token), timeout=60)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type:
        return resp.json()
    return resp.text


def _post(url: str, body: dict, token: str | None = None) -> dict:
    """POST JSON with basic error handling."""
    resp = requests.post(url, json=body, headers=_headers(token), timeout=60)
    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════════
# 1.  REST CATALOG API
# ═══════════════════════════════════════════════════════════════════════════


# ---- Authentication ------------------------------------------------------


def obtain_token(email: str, password: str) -> dict:
    """
    Exchange credentials for a JWT access/refresh token pair.

    Parameters
    ----------
    email:
        Registered Carbon Mapper account e-mail address.
    password:
        Account password.

    Returns
    -------
    dict
        A mapping with at least two keys:

        - ``"access"``  — short-lived JWT bearer token (use in API calls).
        - ``"refresh"`` — long-lived refresh token (use with :func:`refresh_token`).

    Examples
    --------
    >>> tokens = obtain_token("user@example.com", "s3cret")
    >>> access_token = tokens["access"]
    >>> data = get_plumes_annotated(plume_gas="CH4", limit=5, token=access_token)
    """
    return _post(f"{BASE_URL}/token/pair", {"email": email, "password": password})


def refresh_token(refresh: str) -> dict:
    """
    Refresh an expired access token using a refresh token.

    Parameters
    ----------
    refresh:
        The ``"refresh"`` value previously returned by :func:`obtain_token`.

    Returns
    -------
    dict
        A mapping with a new ``"access"`` token (and optionally a new
        ``"refresh"`` token if the server rotates them).

    Examples
    --------
    >>> tokens = obtain_token("user@example.com", "s3cret")
    >>> new_tokens = refresh_token(tokens["refresh"])
    >>> access_token = new_tokens["access"]
    """
    return _post(f"{BASE_URL}/token/refresh", {"refresh": refresh})


# ---- Plumes (L4A — the main public endpoint) ----------------------------


def get_plumes_annotated(
    *,
    plume_gas: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    datetime_range: str | None = None,
    sectors: list[str] | None = None,
    instruments: list[str] | None = None,
    emission_min: int | None = None,
    emission_max: int | None = None,
    qualities: list[str] | None = None,
    has_phme: bool | None = None,
    source_name: str | None = None,
    limit: int = 25,
    offset: int = 0,
    sort: str = "desc",
    token: str | None = None,
) -> dict:
    """
    Fetch annotated plumes with emissions, wind, and image URLs.

    This is the primary public endpoint — no auth required for published
    data, though authenticated requests may see additional fields.

    Parameters
    ----------
    plume_gas:
        Gas species to filter by.  ``"CH4"`` (methane) or ``"CO2"``
        (carbon dioxide).  Omit to return both gases.
    bbox:
        Bounding-box spatial filter as
        ``(west_lon, south_lat, east_lon, north_lat)`` in WGS 84 decimal
        degrees.  Example: ``(-104.5, 31.0, -101.5, 33.5)`` for the
        Permian Basin.
    datetime_range:
        RFC 3339 time interval string, e.g.
        ``"2024-01-01T00:00:00Z/2024-06-01T00:00:00Z"``.  Either bound
        may be replaced with ``".."`` to indicate open-ended.
    sectors:
        One or more IPCC sector codes to filter by.  Common values:
        ``"1B2"`` (Oil & Gas), ``"6A"`` (Solid Waste), ``"1B1a"``
        (Coal Mining), ``"4B"`` (Livestock), ``"other"``.
    instruments:
        Instrument identifiers to restrict results to.  Recognised values:
        ``"emi"`` (EMIT on the ISS), ``"tan"`` (Tanager-1 satellite),
        ``"ang"`` (AVIRIS-NG airborne), ``"GAO"`` (Global Airborne
        Observatory), ``"av3"`` (AVIRIS-3 airborne).
    emission_min:
        Lower bound on ``emission_auto`` in **kg/hr** (inclusive).
    emission_max:
        Upper bound on ``emission_auto`` in **kg/hr** (inclusive).
    qualities:
        Plume quality flags to include.  One or more of ``"good"``,
        ``"questionable"``, ``"bad"``.
    has_phme:
        When ``True``, return only plumes flagged as a Potentially Harmful
        Methane Event (PHME).  When ``False``, exclude them.  ``None``
        (default) applies no PHME filter.
    source_name:
        Return only plumes associated with a specific emission source, e.g.
        ``"CH4_6A_100m_-74.00656_40.71283"``.
    limit:
        Maximum number of plumes to return in a single response.  The
        server caps this at **1 000** per request.  Use *offset* or
        :func:`paginate_plumes` for larger result sets.
    offset:
        Zero-based index of the first result to return.  Use with *limit*
        for manual pagination.
    sort:
        Result ordering direction.  ``"desc"`` (default) returns the most
        recently detected plumes first; ``"asc"`` returns oldest first.
    token:
        Bearer token for authenticated requests (see :func:`obtain_token`).
        Not required for publicly available data.

    Returns
    -------
    dict
        A mapping with the following top-level keys:

        - ``"items"`` — list of plume dicts.  Each dict contains:

          - ``"plume_id"``                    — unique identifier string.
          - ``"gas"``                          — ``"CH4"`` or ``"CO2"``.
          - ``"geometry_json"``               — GeoJSON geometry (Point or Polygon).
          - ``"emission_auto"``               — estimated emission rate in **kg/hr**.
          - ``"emission_uncertainty_auto"``   — 1-σ uncertainty in kg/hr.
          - ``"wind_speed_avg_auto"``         — average wind speed in m/s.
          - ``"wind_direction_avg_auto"``     — meteorological wind direction in degrees.
          - ``"plume_png"``, ``"plume_tif"``, ``"con_tif"``, ``"rgb_png"``,
            ``"rgb_tif"``, ``"plume_rgb_png"`` — pre-signed download URLs for
            raster assets (may be ``null`` if unavailable).
          - ``"sector"``                      — IPCC sector code.
          - ``"plume_quality"``               — quality flag.
          - ``"instrument"``                  — sensor identifier.
          - ``"platform"``                    — e.g. ``"ISS"``, ``"satellite"``.
          - ``"scene_timestamp"``             — ISO 8601 acquisition time (UTC).

        - ``"total_count"`` — total number of matching plumes (across all pages).
        - ``"bbox_count"``  — number of matching plumes inside the requested bbox.

    Examples
    --------
    Fetch the 10 most recent good-quality CH4 plumes in the Permian Basin:

    >>> result = get_plumes_annotated(
    ...     plume_gas="CH4",
    ...     bbox=(-104.5, 31.0, -101.5, 33.5),
    ...     qualities=["good"],
    ...     limit=10,
    ... )
    >>> print(result["total_count"])
    >>> for p in result["items"]:
    ...     print(p["plume_id"], p["emission_auto"], "kg/hr")

    Filter by sector and emission threshold (Oil & Gas, ≥ 500 kg/hr):

    >>> result = get_plumes_annotated(
    ...     plume_gas="CH4",
    ...     sectors=["1B2"],
    ...     emission_min=500,
    ...     limit=25,
    ... )

    Authenticated request for additional fields:

    >>> tokens = obtain_token("user@example.com", "s3cret")
    >>> result = get_plumes_annotated(
    ...     plume_gas="CH4",
    ...     limit=5,
    ...     token=tokens["access"],
    ... )
    """
    params: dict[str, Any] = {"limit": limit, "offset": offset, "sort": sort}
    if plume_gas:
        params["plume_gas"] = plume_gas
    params.update(_rest_bbox_params(bbox))
    if datetime_range:
        params["datetime"] = datetime_range
    if sectors:
        params["sectors"] = sectors
    if instruments:
        params["instruments"] = instruments
    if emission_min is not None:
        params["emission_min"] = emission_min
    if emission_max is not None:
        params["emission_max"] = emission_max
    if qualities:
        params["qualities"] = qualities
    if has_phme is not None:
        params["has_phme"] = str(has_phme).lower()
    if source_name:
        params["source_name"] = source_name
    return cast(dict, _get(f"{CATALOG_URL}/plumes/annotated", params=params, token=token))


def get_plume_by_id(plume_id: str, token: str | None = None) -> dict:
    """
    Get a single plume by its UUID or colloquial name.

    Parameters
    ----------
    plume_id:
        The unique plume identifier.  Either the UUID form
        (``"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"``) or the colloquial
        name derived from the scene (e.g. ``"emi20240420t101448p07050-A"``).
    token:
        Optional Bearer token for authenticated requests.

    Returns
    -------
    dict
        A single plume dict with the same schema as each element of
        ``get_plumes_annotated()["items"]``.

    Examples
    --------
    >>> plume = get_plume_by_id("emi20240420t101448p07050-A")
    >>> print(plume["emission_auto"], "kg/hr")
    """
    return cast(dict, _get(f"{CATALOG_URL}/plume/{plume_id}", token=token))


def get_plumes_csv(
    *,
    plume_gas: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    datetime_range: str | None = None,
    sectors: list[str] | None = None,
    instruments: list[str] | None = None,
    limit: int = 50_000,
    offset: int = 0,
    token: str | None = None,
) -> str:
    """
    Download plume data as CSV text.

    Returns the same plumes as :func:`get_plumes_annotated` but serialised
    as a comma-separated text string suitable for writing directly to a
    ``.csv`` file or loading into :func:`pandas.read_csv`.

    Parameters
    ----------
    plume_gas:
        Gas species filter — ``"CH4"`` or ``"CO2"``.
    bbox:
        Bounding-box spatial filter as
        ``(west_lon, south_lat, east_lon, north_lat)`` in WGS 84.
    datetime_range:
        RFC 3339 time interval, e.g.
        ``"2024-01-01T00:00:00Z/2024-06-01T00:00:00Z"``.
    sectors:
        IPCC sector codes to include, e.g. ``["1B2", "6A"]``.
    instruments:
        Instrument identifiers, e.g. ``["emi", "tan"]``.
    limit:
        Maximum rows per request.  The server caps at **10 000** when
        asset-URL columns (``plume_tif``, ``rgb_png``, etc.) are included,
        or **50 000** when they are excluded.
    offset:
        Zero-based row offset for pagination.
    token:
        Optional Bearer token for authenticated requests.

    Returns
    -------
    str
        Raw CSV text with a header row followed by one row per plume.
        Columns include: ``plume_id``, ``lat`` (→ ``plume_latitude``),
        ``lon`` (→ ``plume_longitude``), ``datetime``,
        ``sector``, ``gas``, ``emission_auto``,
        ``emission_uncertainty_auto``, wind fields, and asset URLs.
        Pass raw column names directly to ``CMRawPlume`` — e.g.
        ``CMRawPlume(**row)`` from a ``pandas.read_csv`` iteration.

    Examples
    --------
    Save a bulk CSV for the Permian Basin to disk:

    >>> csv_text = get_plumes_csv(
    ...     plume_gas="CH4",
    ...     bbox=(-104.5, 31.0, -101.5, 33.5),
    ...     limit=10_000,
    ... )
    >>> from pathlib import Path
    >>> Path("permian_ch4.csv").write_text(csv_text)

    Load into pandas:

    >>> import io, pandas as pd
    >>> df = pd.read_csv(io.StringIO(csv_text))

    .. note::
        When file-URL columns are included the server caps *limit* at
        10 000.  For larger exports paginate with *offset* or filter with
        *bbox* / *datetime_range*.
    """
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if plume_gas:
        params["plume_gas"] = plume_gas
    params.update(_rest_bbox_params(bbox))
    if datetime_range:
        params["datetime"] = datetime_range
    if sectors:
        params["sectors"] = sectors
    if instruments:
        params["instruments"] = instruments
    return cast(str, _get(f"{CATALOG_URL}/plume-csv", params=params, token=token))


# ---- Sources (L4B — aggregated emission sources) -------------------------


def get_sources(
    *,
    source_gas: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    sectors: list[str] | None = None,
    plume_count_min: int | None = None,
    limit: int = 25,
    offset: int = 0,
    token: str | None = None,
) -> dict:
    """
    Fetch emission sources (clusters of plumes at a location).

    An *emission source* groups all plumes detected at the same geographic
    location into a persistent point source record.  Each source includes
    aggregated statistics across all associated plumes.

    Parameters
    ----------
    source_gas:
        Gas species to filter by — ``"CH4"`` or ``"CO2"``.
    bbox:
        Bounding-box spatial filter as
        ``(west_lon, south_lat, east_lon, north_lat)`` in WGS 84.
    sectors:
        IPCC sector codes to include, e.g. ``["1B2"]`` for Oil & Gas.
    plume_count_min:
        Return only sources with at least this many associated plumes.
    limit:
        Maximum number of sources to return per page (max 1 000).
    offset:
        Zero-based pagination offset.
    token:
        Optional Bearer token for authenticated requests.

    Returns
    -------
    dict
        Same structure as :func:`get_plumes_annotated`.  Each item
        includes: ``source_name``, lat/lon, ``sector``, ``plume_count``,
        persistence, ``emission_auto`` (persistence-weighted average), and
        first/last detection date ranges.

    Notes
    -----
    The sources endpoint shares ``/catalog/plumes/annotated`` filtered by
    ``source_name``.  For a full source listing use the STAC search or the
    CSV endpoint grouped by source.

    Examples
    --------
    >>> sources = get_sources(source_gas="CH4", sectors=["1B2"], limit=10)
    >>> for s in sources["items"]:
    ...     print(s["source_name"], s.get("plume_count"), "plumes")
    """
    # The catalog API surfaces sources through plume queries filtered by
    # source_name.  A dedicated /sources endpoint may be available with auth.
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if source_gas:
        params["plume_gas"] = source_gas
    params.update(_rest_bbox_params(bbox))
    if sectors:
        params["sectors"] = sectors
    if plume_count_min is not None:
        params["plume_count_min"] = plume_count_min
    return cast(dict, _get(f"{CATALOG_URL}/plumes/annotated", params=params, token=token))


# ---- Scenes (L2 — observation footprints) --------------------------------


def get_scenes(
    *,
    instrument: str | None = None,
    plume_gas: str | None = None,
    has_plume_emissions: bool | None = None,
    cloud_cover_pct_max: float | None = None,
    limit: int = 25,
    offset: int = 0,
    token: str | None = None,
) -> dict:
    """
    List observation scenes (satellite or aircraft flight strips).

    A *scene* is a single sensor overpass or flight strip captured by one
    of Carbon Mapper's instruments.  Scenes describe the spatial footprint
    and metadata for the raw L2 observations from which plumes are derived.

    .. note::
        This endpoint requires authentication (see :func:`obtain_token`).

    Parameters
    ----------
    instrument:
        Sensor identifier to filter by.  One of ``"emi"`` (EMIT/ISS),
        ``"tan"`` (Tanager-1), ``"ang"`` (AVIRIS-NG), ``"GAO"``
        (Global Airborne Observatory), or ``"av3"`` (AVIRIS-3).
    plume_gas:
        Gas species of interest — ``"CH4"`` or ``"CO2"``.
    has_plume_emissions:
        When ``True``, return only scenes that contain at least one
        detected plume with quantified emissions.
    cloud_cover_pct_max:
        Maximum allowable cloud cover percentage (0 – 100).
    limit:
        Maximum number of scenes to return per page.
    offset:
        Zero-based pagination offset.
    token:
        Bearer token (required — see :func:`obtain_token`).

    Returns
    -------
    dict
        A mapping with ``"items"`` (list of scene dicts) and
        ``"total_count"``.  Each scene dict includes scene ID, instrument,
        platform, acquisition datetime, spatial extent (GeoJSON), and
        cloud-cover percentage.

    Examples
    --------
    >>> tokens = obtain_token("user@example.com", "s3cret")
    >>> scenes = get_scenes(
    ...     instrument="emi",
    ...     has_plume_emissions=True,
    ...     limit=5,
    ...     token=tokens["access"],
    ... )
    >>> for s in scenes["items"]:
    ...     print(s.get("scene_id"), s.get("datetime"))
    """
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if instrument:
        params["instrument"] = instrument
    if plume_gas:
        params["plume_gas"] = plume_gas
    if has_plume_emissions is not None:
        params["has_plume_emissions"] = str(has_plume_emissions).lower()
    if cloud_cover_pct_max is not None:
        params["cloud_cover_pct_max"] = cloud_cover_pct_max
    return cast(dict, _get(f"{CATALOG_URL}/scenes", params=params, token=token))


# ---- Asset Download (retrievals, RGB, plume TIF/PNG) ---------------------


def download_asset(asset_key: str, dest: Path | str, token: str | None = None) -> Path:
    """
    Download a raster asset (GeoTIFF or PNG) by its storage key.

    Parameters
    ----------
    asset_key:
        The path portion after ``/catalog/asset/``.  For example::

            l2b-ch4-mf-v1/2016/10/08/ang20161008t211637/ang20161008t211637_l2b-ch4-mf-v1_cmf.tif

        Asset keys are available in STAC item ``assets[name]["href"]``
        entries and can be derived from the plume ``plume_tif`` /
        ``con_tif`` / ``rgb_tif`` URLs.
    dest:
        Local file path where the asset will be written.  Parent
        directories are created automatically.
    token:
        Optional Bearer token for authenticated access.

    Returns
    -------
    Path
        The resolved path of the downloaded file.

    Examples
    --------
    >>> download_asset(
    ...     "l2b-ch4-mf-v1/2016/10/08/ang20161008t211637/ang20161008t211637_l2b-ch4-mf-v1_cmf.tif",
    ...     dest="./retrieval.tif",
    ... )

    .. note::
        For plume dicts returned by :func:`get_plumes_annotated`, prefer
        :func:`download_plume_assets` which handles all assets at once.
    """
    dest = Path(dest)
    url = f"{CATALOG_URL}/asset/{asset_key}"
    resp = requests.get(url, headers=_headers(token), timeout=120, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info("Downloaded %s → %s (%d bytes)", asset_key, dest, dest.stat().st_size)
    return dest


def download_plume_assets(plume: dict, dest_dir: Path | str) -> dict[str, Path]:
    """
    Download all available raster assets for a single plume.

    Given a plume dict returned by :func:`get_plumes_annotated`, download
    every non-null asset URL into *dest_dir* and return a mapping of asset
    type to local file path.

    Parameters
    ----------
    plume:
        A single plume dict as returned in
        ``get_plumes_annotated()["items"]``.  The function inspects the
        ``"plume_png"``, ``"plume_tif"``, ``"con_tif"``, ``"rgb_png"``,
        ``"rgb_tif"``, and ``"plume_rgb_png"`` keys for download URLs.
    dest_dir:
        Directory into which assets are downloaded.  Created automatically
        if it does not already exist.

    Returns
    -------
    dict[str, Path]
        Mapping of asset type → local :class:`~pathlib.Path` for each
        successfully downloaded asset.  Assets that are missing (``null``)
        or whose download fails are omitted.  Example::

            {
                "plume_png": Path("./plumes/emi20240420t101448p07050-A_plume.png"),
                "plume_tif": Path("./plumes/emi20240420t101448p07050-A_plume.tif"),
                "con_tif": Path("./plumes/emi20240420t101448p07050-A_con.tif"),
                "rgb_png": Path("./plumes/emi20240420t101448p07050-A_rgb.png"),
            }

    Examples
    --------
    .. code-block:: python

        result = get_plumes_annotated(plume_gas="CH4", limit=1, qualities=["good"])
        plume = result["items"][0]
        downloaded = download_plume_assets(plume, "./plume_data/")
        for asset_type, path in downloaded.items():
            print(asset_type, "→", path)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    plume_name = plume.get("plume_id", "unknown")
    downloaded: dict[str, Path] = {}

    asset_keys = ["plume_png", "plume_tif", "con_tif", "rgb_png", "rgb_tif", "plume_rgb_png"]
    for key in asset_keys:
        url = plume.get(key)
        if not url:
            continue
        suffix = ".tif" if key.endswith("tif") else ".png"
        short = key.replace("_tif", "").replace("_png", "")
        local = dest_dir / f"{plume_name}_{short}{suffix}"
        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(local, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            downloaded[key] = local
            logger.info("  %s → %s", key, local)
        except requests.RequestException as exc:
            logger.warning("  %s download failed: %s", key, exc)
    return downloaded


# ═══════════════════════════════════════════════════════════════════════════
# 2.  STAC API  (SpatioTemporal Asset Catalog)
# ═══════════════════════════════════════════════════════════════════════════

# The STAC API provides access to the full collection hierarchy:
#
#   L2B  — atmospheric retrievals (CH4/CO2 matched-filter results, RGB)
#   L2C  — plume detections (CNN salience maps)
#   L3A  — preliminary plume images (quick-looks, IME)
#   L3B  — fully processed plumes
#   L4A  — plume emissions (with wind/quantification)
#
# Each collection has versioned sub-collections, e.g.:
#   l2b-ch4-mfa-v3, l2b-rgb-v3a, l4a-combined-ch4-v3a, etc.


def stac_get_catalog() -> dict:
    """Get the STAC root catalog with links to all collections."""
    return cast(dict, _get(f"{STAC_URL}/"))


def stac_list_collections() -> list[dict]:
    """List all STAC collections with their metadata."""
    resp = _get(f"{STAC_URL}/collections")
    if isinstance(resp, dict):
        return cast("list[dict]", resp.get("collections", resp))
    return cast("list[dict]", resp)


def stac_get_collection(collection_id: str) -> dict:
    """
    Get metadata for a single collection.

    Key collections
    ---------------
    - ``l2b-ch4-mfa-v3``       — CH4 matched-filter atmospheric retrievals (GeoTIFF)
    - ``l2b-co2-mfal-v3a``     — CO2 retrievals (log-normal variant)
    - ``l2b-rgb-v3a``          — simultaneous RGB surface imagery (GeoTIFF)
    - ``l2c-ch4-v0``           — CH4 plume detections
    - ``l2c-co2-v0``           — CO2 plume detections
    - ``l3a-ime-ch4-mfa-v3a``  — preliminary plume IME estimates
    - ``l4a-combined-ch4-v3a`` — final combined CH4 plume emissions
    - ``l4a-co2-mfal-v3a``     — final CO2 plume emissions
    """
    return cast(dict, _get(f"{STAC_URL}/collections/{collection_id}"))


def stac_get_items(
    collection_id: str,
    *,
    limit: int = 10,
    bbox: tuple[float, float, float, float] | None = None,
    datetime_range: str | None = None,
    token: str | None = None,
) -> dict:
    """
    Get items from a STAC collection (OGC API Features compliant).

    Parameters
    ----------
    collection_id:
        Identifier of the STAC collection to query, e.g.
        ``"l4a-combined-ch4-v3a"`` or ``"l2b-rgb-v3a"``.
    limit:
        Maximum number of items to return.
    bbox:
        Bounding-box spatial filter as
        ``(west_lon, south_lat, east_lon, north_lat)`` in WGS 84.
    datetime_range:
        RFC 3339 time interval string.
    token:
        Optional Bearer token for authenticated requests.

    Returns
    -------
    dict
        A GeoJSON FeatureCollection.  Each Feature has ``"assets"``
        containing download links for GeoTIFFs, PNGs, and other raster
        products, with ``"href"`` and media-type annotations.

    Examples
    --------
    >>> items = stac_get_items("l4a-combined-ch4-v3a", limit=5)
    >>> for feat in items["features"]:
    ...     print(feat["id"], feat.get("properties", {}).get("datetime"))
    """
    params: dict[str, Any] = {"limit": limit}
    params.update(_stac_bbox_param(bbox))
    if datetime_range:
        params["datetime"] = datetime_range
    return cast(dict, _get(f"{STAC_URL}/collections/{collection_id}/items", params=params, token=token))


def stac_get_item(
    collection_id: str,
    item_id: str,
    *,
    token: str | None = None,
) -> dict:
    """Get a single STAC item by collection ID and item ID.

    Wraps ``GET /stac/collections/{collection_id}/items/{item_id}``. The
    item ID for a Carbon Mapper L2B raster matches the *scene_id* — i.e.
    ``plume_id.rsplit("-", 1)[0]``.

    Parameters
    ----------
    collection_id:
        STAC collection identifier, e.g. ``"l2b-ch4-mfa-v3a"`` for the
        L2B CH4 matched-filter rasters.
    item_id:
        Item identifier (= scene_id for L2B rasters).
    token:
        Optional Bearer token. STAC item endpoints generally accept
        unauthenticated requests for published items.

    Returns
    -------
    dict
        STAC Item GeoJSON (a single Feature with ``"id"``, ``"bbox"``,
        ``"geometry"``, ``"properties"``, ``"assets"``).

    Raises
    ------
    requests.HTTPError
        ``404`` when the scene has not been published to L2B yet — the
        higher-level :func:`georeader.readers.carbonmapper.api_queries.get_tile`
        translates this into ``CMSceneNotPublished``.

    Examples
    --------
    >>> item = stac_get_item(
    ...     "l2b-ch4-mfa-v3a",
    ...     "tan20251212t185057c20s4001",
    ... )
    >>> sorted(item["assets"])
    ['artifact-mask', 'cmf', 'rgb', 'uncertainty']
    """
    return cast(dict, _get(
        f"{STAC_URL}/collections/{collection_id}/items/{item_id}",
        token=token,
    ))


def get_source_by_name(source_name: str, *, token: str | None = None) -> dict:
    """Get a single Carbon Mapper source by its deterministic name.

    A *source* is a DBSCAN cluster of plumes detected at the same
    geographic location. Source names follow the pattern
    ``{gas}_{sector}_{footprint_m}m_{lon}_{lat}``, e.g.
    ``"CH4_1B2_100m_-104.17525_32.49125"``.

    Wraps ``GET /catalog/source/{source_name}``.

    Parameters
    ----------
    source_name:
        Canonical source name. **Strip any ``?...`` query-string
        suffix first** — see :func:`georeader.readers.carbonmapper.source._strip_query_suffix`.
    token:
        Optional Bearer token.

    Returns
    -------
    dict
        Source record as a GeoJSON Feature (or, for some endpoints,
        a flat dict). The high-level
        :func:`georeader.readers.carbonmapper.api_queries.get_source` normalises
        both shapes into a typed :class:`CMSource`.
    """
    return cast(dict, _get(f"{CATALOG_URL}/source/{source_name}", token=token))


def get_source_for_plume_name(
    plume_id: str, *, token: str | None = None,
) -> dict:
    """Look up the Carbon Mapper source attributed to a plume.

    Uses the *by-name* endpoint
    (``GET /catalog/source/plume/name/{plume_id}``), which returns the
    cleaned ``source_name`` — preferred over the UUID-keyed sibling
    ``/catalog/source/plume/{plume_uuid}`` when working with the
    colloquial plume_id strings (e.g. ``"emi20240420t101448p07050-A"``).

    Parameters
    ----------
    plume_id:
        Colloquial plume identifier — ``"{instrument}{datetime}-{part}"``.
    token:
        Optional Bearer token.

    Returns
    -------
    dict
        Source record (GeoJSON Feature or flat dict). Returns ``404``
        when CM has not yet attributed the plume to a source. The
        high-level :func:`georeader.readers.carbonmapper.api_queries.get_source_for_plume`
        translates ``404`` into ``None``.

    Examples
    --------
    >>> raw = get_source_for_plume_name("tan20251212t185057c20s4001-E")
    >>> raw.get("source_name")
    'CH4_1B2_100m_-104.0_32.0'
    """
    return cast(dict, _get(f"{CATALOG_URL}/source/plume/name/{plume_id}", token=token))


def get_source_plumes_csv(
    source_name: str, *, token: str | None = None,
) -> str:
    """List all plumes attributed to a Carbon Mapper source, as CSV text.

    Wraps ``GET /catalog/source-plumes-csv/{source_name}``. The CSV
    schema matches :func:`get_plumes_csv` — each row can be passed
    directly to :class:`georeader.readers.carbonmapper.plume.CMRawPlume`.

    Parameters
    ----------
    source_name:
        Canonical source name. Strip any ``?...`` suffix first.
    token:
        Optional Bearer token.

    Returns
    -------
    str
        Raw CSV text (header + one row per plume). Empty string is
        possible for sources with no published plumes.

    Examples
    --------
    Load straight into pandas:

    >>> import io, pandas as pd
    >>> csv_text = get_source_plumes_csv("CH4_1B2_100m_-104.17525_32.49125")
    >>> df = pd.read_csv(io.StringIO(csv_text))
    """
    return cast(str, _get(f"{CATALOG_URL}/source-plumes-csv/{source_name}", token=token))


def stac_search(
    *,
    collections: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    datetime_range: str | None = None,
    ids: list[str] | None = None,
    limit: int = 10,
    token: str | None = None,
) -> dict:
    """
    Cross-collection STAC item search.

    Searches across one or more STAC collections using spatial and
    temporal filters and returns matching items as a GeoJSON
    FeatureCollection.

    Parameters
    ----------
    collections:
        List of STAC collection IDs to search.  If ``None``, all
        collections are searched.  Example:
        ``["l2b-ch4-mfa-v3", "l4a-combined-ch4-v3a"]``.
    bbox:
        Bounding-box spatial filter as
        ``(west_lon, south_lat, east_lon, north_lat)`` in WGS 84.
    datetime_range:
        RFC 3339 time interval string, e.g.
        ``"2024-01-01T00:00:00Z/2024-06-01T00:00:00Z"``.
    limit:
        Maximum number of items to return.
    token:
        Optional Bearer token for authenticated requests.

    Returns
    -------
    dict
        A GeoJSON FeatureCollection mapping.  Key fields:

        - ``"type"``     — ``"FeatureCollection"``.
        - ``"features"`` — list of STAC item GeoJSON Features.  Each
          Feature has:

          - ``"id"``         — item ID.
          - ``"geometry"``   — GeoJSON geometry of the scene footprint.
          - ``"properties"`` — item metadata (datetime, collection, etc.).
          - ``"assets"``     — dict of named assets, each with an
            ``"href"`` download URL and media type.

        - ``"context"``  — pagination info (``matched``, ``returned``).

    Examples
    --------
    Search CH4 retrievals in the Permian Basin:

    >>> result = stac_search(
    ...     collections=["l2b-ch4-mfa-v3"],
    ...     bbox=(-104.5, 31.0, -101.5, 33.5),
    ...     datetime_range="2024-01-01T00:00:00Z/2024-06-01T00:00:00Z",
    ...     limit=5,
    ... )
    >>> for feat in result["features"]:
    ...     print(feat["id"], list(feat["assets"].keys()))

    Search across multiple collections simultaneously:

    >>> result = stac_search(
    ...     collections=["l4a-combined-ch4-v3a", "l2b-rgb-v3a"],
    ...     bbox=(-104.5, 31.0, -101.5, 33.5),
    ...     limit=3,
    ... )
    """
    params: dict[str, Any] = {"limit": limit}
    if collections:
        params["collections"] = ",".join(collections)
    if ids:
        params["ids"] = ",".join(ids)
    params.update(_stac_bbox_param(bbox))
    if datetime_range:
        params["datetime"] = datetime_range
    return cast(dict, _get(f"{STAC_URL}/search", params=params, token=token))


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Convenience / Pipeline Helpers
# ═══════════════════════════════════════════════════════════════════════════


def paginate_plumes(
    *,
    plume_gas: str = "CH4",
    bbox: tuple[float, float, float, float] | None = None,
    datetime_range: str | None = None,
    max_plumes: int = 100,
    page_size: int = 50,
    token: str | None = None,
) -> list[dict]:
    """
    Auto-paginate through the annotated plumes endpoint.

    Repeatedly calls :func:`get_plumes_annotated` with increasing offsets
    until *max_plumes* items have been collected or the server reports no
    more results.

    Parameters
    ----------
    plume_gas:
        Gas species — ``"CH4"`` (default) or ``"CO2"``.
    bbox:
        Bounding-box spatial filter as
        ``(west_lon, south_lat, east_lon, north_lat)`` in WGS 84.
    datetime_range:
        RFC 3339 time interval, e.g.
        ``"2024-01-01T00:00:00Z/2024-06-01T00:00:00Z"``.
    max_plumes:
        Maximum total number of plume dicts to return (across all pages).
    page_size:
        Number of plumes to request per API call.  Must not exceed 1 000.
    token:
        Optional Bearer token for authenticated requests.

    Returns
    -------
    list[dict]
        A flat list of plume dicts (same schema as
        ``get_plumes_annotated()["items"]``), up to *max_plumes* entries.

    Examples
    --------
    Collect up to 500 recent CH4 plumes in the Permian Basin:

    >>> plumes = paginate_plumes(
    ...     plume_gas="CH4",
    ...     bbox=(-104.5, 31.0, -101.5, 33.5),
    ...     max_plumes=500,
    ...     page_size=100,
    ... )
    >>> print(f"Collected {len(plumes)} plumes")

    Export to GeoJSON:

    >>> from pathlib import Path
    >>> export_plumes_to_geojson(plumes, Path("permian_ch4.geojson"))
    """
    all_items: list[dict] = []
    offset = 0
    while len(all_items) < max_plumes:
        batch_size = min(page_size, max_plumes - len(all_items))
        result = get_plumes_annotated(
            plume_gas=plume_gas,
            bbox=bbox,
            datetime_range=datetime_range,
            limit=batch_size,
            offset=offset,
            token=token,
        )
        items = result.get("items", [])
        if not items:
            break
        all_items.extend(items)
        offset += len(items)
        total = result.get("total_count", 0)
        if offset >= total:
            break
    return all_items[:max_plumes]


def export_plumes_to_geojson(plumes: list[dict], output: Path | str) -> Path:
    """
    Convert a list of plume dicts into a GeoJSON FeatureCollection and
    write to *output*.
    """
    output = Path(output)
    features = []
    for p in plumes:
        geom = p.get("geometry_json")
        props = {k: v for k, v in p.items() if k != "geometry_json"}
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    fc = {"type": "FeatureCollection", "features": features}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fc, indent=2))
    logger.info("Wrote %d features → %s", len(features), output)
    return output

