"""Typed model for a Carbon Mapper *source* (DBSCAN cluster of plumes).

A Carbon Mapper *source* groups all plumes detected at the same
geographic location into a persistent point-source record. Sources are
addressed by a deterministic name of the form
``{gas}_{sector}_{footprint_m}m_{lon}_{lat}`` — e.g.
``"CH4_1B2_100m_-104.17525_32.49125"``.

This module is the **API-side** typed view of a Carbon Mapper source.
Downstream consumers may persist it into their own tables, but this
package deliberately does not assume any particular DB schema.

Notable quirks handled here
---------------------------
- ``/catalog/sources.geojson`` features sometimes return ``source_name``
  with a stray query-string fragment appended
  (``"...?plume_gas=CH4&bbox=..."``). :func:`_strip_query_suffix` removes
  it; :meth:`CMSource.from_geojson_feature` calls it always so callers
  never see the dirty form.
- The endpoints return either a GeoJSON Feature (with ``properties`` /
  ``geometry``) or a flat dict; the higher-level
  :mod:`georeader.readers.carbonmapper.api_queries` normalises these
  before invoking :meth:`from_geojson_feature`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from shapely.geometry import Point

from georeader.readers.carbonmapper.plume import _parse_iso_datetime, _to_float


def _strip_query_suffix(source_name: str) -> str:
    """Strip a stray ``?...`` query-string fragment from a ``source_name``.

    ``/catalog/sources.geojson`` features sometimes return
    ``source_name`` with a query-string fragment appended (e.g.
    ``"CH4_6A_500m_-117.26768_34.59375?plume_gas=CH4&bbox=..."``). The
    fragment is not part of the canonical name and must be removed
    before using the value as a key into other endpoints
    (``/catalog/source/{name}``,
    ``/catalog/source-plumes-csv/{name}``).

    This function is **idempotent** — calling it twice yields the same
    result — and a no-op for empty / suffix-free names.

    Parameters
    ----------
    source_name:
        Source name as returned by the API (may or may not have a
        ``?...`` suffix).

    Returns
    -------
    str
        The portion before the first ``?``, or the input unchanged if
        no ``?`` is present.

    Examples
    --------
    >>> _strip_query_suffix("CH4_1B2_100m_-104_32?plume_gas=CH4&bbox=1")
    'CH4_1B2_100m_-104_32'
    >>> _strip_query_suffix("CH4_1B2_100m_-104_32")  # already clean
    'CH4_1B2_100m_-104_32'
    >>> _strip_query_suffix("")
    ''
    """
    if not source_name:
        return source_name
    return source_name.split("?", 1)[0]


@dataclass(frozen=True)
class CMSource:
    """Typed view of a Carbon Mapper source (cluster of plumes).

    Frozen — instances are immutable and hashable. The ``raw`` dict
    captures the full upstream properties payload so consumers can
    reach for fields not yet exposed on the dataclass without round-
    tripping through the API.

    Attributes
    ----------
    source_name:
        Canonical name (no ``?...`` suffix). Stable across CM API
        revisions for the same physical site.
    gas:
        Gas species — typically ``"CH4"`` or ``"CO2"``.
    sector:
        IPCC sector code, e.g. ``"1B2"`` (Oil & Gas), ``"6A"`` (Solid
        Waste), ``"1B1a"`` (Coal Mining).
    point:
        Centroid as a Shapely :class:`shapely.geometry.Point` in WGS-84.
    plume_count:
        Number of plumes Carbon Mapper has attributed to this source.
    persistence:
        Carbon Mapper's persistence metric (overpasses-with-detection /
        total-overpasses), in ``[0, 1]``.
    emission_auto:
        Persistence-weighted average emission rate in ``kg/h``. ``None``
        when CM has not produced an aggregate estimate.
    emission_uncertainty_auto:
        Companion uncertainty for ``emission_auto``, in ``kg/h``.
    first_observation, last_observation:
        Earliest and latest detection datetimes (UTC-aware).
    raw:
        Original ``properties`` mapping from the API response.

    Examples
    --------
    Parse from a ``/catalog/sources.geojson`` feature:

    >>> feature = {
    ...     "properties": {
    ...         "source_name": "CH4_1B2_100m_-104.17525_32.49125?plume_gas=CH4",
    ...         "sector": "1B2", "gas": "CH4",
    ...         "plume_count": 12, "persistence": 0.42,
    ...         "emission_auto": 250.0,
    ...     },
    ...     "geometry": {"type": "Point",
    ...                  "coordinates": [-104.17525, 32.49125]},
    ... }
    >>> src = CMSource.from_geojson_feature(feature)
    >>> src.source_name              # query suffix stripped
    'CH4_1B2_100m_-104.17525_32.49125'
    >>> src.point.x, src.point.y
    (-104.17525, 32.49125)
    >>> src.plume_count, src.sector
    (12, '1B2')
    """

    source_name: str
    gas: str
    sector: str
    point: Point
    plume_count: int
    persistence: float
    emission_auto: float | None = None
    emission_uncertainty_auto: float | None = None
    first_observation: datetime | None = None
    last_observation: datetime | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_geojson_feature(cls, feature: dict) -> "CMSource":
        """Parse a ``/catalog/sources.geojson`` feature into a CMSource.

        Always strips the ``source_name`` query-string suffix
        (``?plume_gas=...``) — this is the canonical strip site, so
        downstream code can treat ``CMSource.source_name`` as clean.

        Parameters
        ----------
        feature:
            GeoJSON Feature dict with at least ``"geometry"`` (Point)
            and ``"properties"`` (with ``source_name`` and friends).

        Returns
        -------
        CMSource
            Typed source record with the suffix stripped.

        Raises
        ------
        ValueError
            If ``feature["geometry"]`` does not carry a Point coordinate
            pair.

        Examples
        --------
        >>> feature = {
        ...     "properties": {"source_name": "x?bbox=1", "sector": "1B2",
        ...                    "gas": "CH4", "plume_count": 1,
        ...                    "persistence": 0.5},
        ...     "geometry": {"type": "Point", "coordinates": [-100.0, 30.0]},
        ... }
        >>> CMSource.from_geojson_feature(feature).source_name
        'x'
        """
        props = dict(feature.get("properties") or {})
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or (None, None)
        lon, lat = (coords + [None, None])[:2] if isinstance(coords, list) else (None, None)

        if lon is None or lat is None:
            raise ValueError(
                f"feature is missing Point coordinates: {feature!r}"
            )

        return cls(
            source_name=_strip_query_suffix(str(props.get("source_name", ""))),
            gas=str(props.get("gas", "") or ""),
            sector=str(props.get("sector", "") or ""),
            point=Point(float(lon), float(lat)),
            plume_count=int(props.get("plume_count") or 0),
            persistence=float(props.get("persistence") or 0.0),
            emission_auto=_to_float(props.get("emission_auto")),
            emission_uncertainty_auto=_to_float(
                props.get("emission_uncertainty_auto")
            ),
            first_observation=_parse_iso_datetime(props.get("first_observation")),
            last_observation=_parse_iso_datetime(props.get("last_observation")),
            raw=props,
        )
