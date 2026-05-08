"""
plume.py
========

Unified Pydantic model for Carbon Mapper plume records.

Handles payloads from **both** Carbon Mapper API formats:

- **CSV bulk export** (``/api/v1/catalog/plume-csv``) — provides
  ``plume_latitude``, ``plume_longitude``, ``datetime``, ``plume_bounds``.
- **Annotated plume JSON** (``/api/v1/catalog/plumes/annotated``) —
  provides ``geometry_json``, ``scene_timestamp``, ``validated``,
  ``has_phme``.

All fields except ``plume_id`` are optional so that the model can be
constructed from either format without validation errors.

This module is the **API-side** typed view of a Carbon Mapper plume
record. Downstream consumers (e.g. UNEP IMEO MARS) may persist the
record into their own tables / views; field-level docstrings below
mirror the column comments on the
``src_carbon_mapper_plumes`` SQL view in pysat
(`UNEP-IMEO-MARS/pysat <https://github.com/UNEP-IMEO-MARS/pysat>`_),
so the upstream API and one downstream staging view share a single
source of truth.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping from Carbon Mapper instrument short codes (the prefix of every
#: ``plume_id`` and ``scene_id``) to the canonical satellite label used
#: by Carbon Mapper's STAC catalogue.
#:
#: Examples
#: --------
#: >>> sorted(CM_INSTRUMENT_TO_SATELLITE)
#: ['ang', 'av3', 'emi', 'gao', 'tan']
#: >>> CM_INSTRUMENT_TO_SATELLITE["emi"]
#: 'EMIT'
CM_INSTRUMENT_TO_SATELLITE: dict[str, str] = {
    "tan": "Tanager1",
    "ang": "AVIRISNG",
    "av3": "AVIRIS3",
    "emi": "EMIT",
    "gao": "GAO",
}

#: ``mars_plumes`` columns that downstream MARS-style consumers populate
#: from a Carbon Mapper record. Kept here as documentation of the
#: contract; not consumed by anything in this package.
CARBONMAPPER_PLUME_PARAMS = [
    "ch4_fluxrate",
    "ch4_fluxrate_std",
    "geometry",
    "lon",
    "lat",
    "detection_institution",
    "validated",
    "validator_user",
    "wind_u",
    "wind_v",
    "metadata",
]

#: Human-readable instrument names indexed by Carbon Mapper short code.
#: Keys are lowercase to match the prefix convention used by ``plume_id``
#: / ``scene_id`` and :data:`CM_INSTRUMENT_TO_SATELLITE`. Lookups via
#: :attr:`CMRawPlume.instrument_name` lowercase the key defensively, so
#: any upstream variant in case (``"GAO"`` etc.) still resolves.
CARBONMAPPER_INSTRUMENTS: dict[str, str] = {
    "emi": "EMIT",
    "tan": "Tanager-1",
    "ang": "AVIRIS-NG",
    "gao": "Global Airborne Observatory",
    "av3": "AVIRIS-3",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float(v: Any) -> float | None:
    """Coerce *v* to float, returning ``None`` for missing/unconvertible values."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(val: str | None) -> datetime | None:
    """Parse an ISO-8601 datetime string to a UTC-aware :class:`datetime`.

    Handles trailing ``Z``, short timezone offsets (``+00``), and naive
    datetimes (assumed UTC).
    """
    if not val:
        return None
    s = val.strip().replace("Z", "+00:00")
    m = re.search(r"([+-]\d{2})$", s)
    if m:
        s = s + ":00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, AttributeError):
        return None


def _parse_bounds(
    bounds: Union[str, List[float], Tuple[float, float, float, float], None],
) -> Tuple[float, float, float, float] | None:
    """Parse a bounds value (string, list, or tuple) into a 4-float tuple."""
    if bounds is None:
        return None
    if isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        try:
            return float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3])
        except (TypeError, ValueError):
            return None
    if isinstance(bounds, str):
        try:
            arr = json.loads(bounds)
            if isinstance(arr, list) and len(arr) == 4:
                return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        except Exception:
            try:
                parts = [p.strip(" []") for p in bounds.split(",")]
                if len(parts) == 4:
                    return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                return None
    return None


def decompose_wind(
    speed: float | None,
    direction_deg: float | None,
) -> tuple[float | None, float | None]:
    """Convert wind speed + meteorological direction to (u, v) components.

    Meteorological convention: 0° = wind *from* North, 90° = wind *from* East.
    Returns the eastward (u) and northward (v) wind vector components.
    """
    if speed is None or direction_deg is None:
        return None, None
    direction_rad = math.radians(direction_deg)
    wind_u = -speed * math.sin(direction_rad)
    wind_v = -speed * math.cos(direction_rad)
    return wind_u, wind_v


# ---------------------------------------------------------------------------
# Unified model
# ---------------------------------------------------------------------------


class CMRawPlume(BaseModel):
    """Unified Carbon Mapper plume model.

    Accepts payloads from both the CSV bulk-export endpoint and the
    annotated plume JSON endpoint. Only ``plume_id`` is required — all
    other fields default to ``None`` so either format can be parsed
    without errors.

    Geometry is built automatically from whichever source is available:

    1. ``geometry_json`` (GeoJSON dict) — Point geometries are buffered
       by 0.001° to produce a small polygon.
    2. ``plume_bounds`` (bounding box) — converted to a ``shapely.box``.

    Note that ``geometry`` here is **not** the retrieved plume mask
    polygon — it's just the API's reported point/bounds. For the
    authoritative plume polygon, use
    :meth:`~georeader.readers.carbonmapper.rasters.CMPlumeRaster.polygon`,
    which extracts it from the L3A ``plume_tif`` band-4 alpha mask.

    Downstream MARS staging-view counterpart
    ----------------------------------------
    UNEP IMEO MARS persists this record into ``src_plume_staging_hist``
    and exposes it via the **``src_carbon_mapper_plumes`` view** (defined
    in `pysat sql/view01_raw_carbon_mapper_plumes_view.sql
    <https://github.com/UNEP-IMEO-MARS/pysat/blob/main/sql/view01_raw_carbon_mapper_plumes_view.sql>`_).
    Field-level docstrings below mirror that view's
    ``COMMENT ON COLUMN`` statements.

    Mapping reference (CMRawPlume field → SQL view column):

    ============================  =====================================
    ``CMRawPlume`` field          ``src_carbon_mapper_plumes`` column
    ============================  =====================================
    ``plume_id``                  ``source_id``
    ``datetime_str`` /            ``tile_date``
      ``scene_timestamp``
    ``published_at_str``          ``published_at``
    ``modified_str``              ``modified``
    ``plume_latitude``            ``lat``
    ``plume_longitude``           ``lon``
    ``plume_bounds_raw``          ``plume_bounds``
    ``wind_source_auto``          ``wind_source``
    ``wind_speed_avg_auto``       ``wind_speed_m_s``
    ``wind_speed_std_auto``       ``wind_speed_std_m_s``
    ``wind_direction_avg_auto``   ``wind_direction_deg``
    ``wind_direction_std_auto``   ``wind_direction_std_deg``
    ``emission_auto``             ``emission_rate_kg_h``
    ``emission_uncertainty_auto`` ``emission_rate_uncertainty_kg_h``
    ``ipcc_sector``               ``sector``
    ``con_tif``                   ``concentration_tif``
    ``rgb_tif``, ``rgb_png``      same names
    ``plume_tif``, ``plume_png``  same names
    ============================  =====================================
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    # Field descriptions below mirror the column comments on the
    # ``src_carbon_mapper_plumes`` view in pysat
    # (sql/view01_raw_carbon_mapper_plumes_view.sql) so the upstream API
    # docs, the staging-table view, and this in-memory model all share
    # one source of truth. Keep them in sync if the SQL view's COMMENT
    # ON COLUMN statements change.

    # --- Core identifiers ---
    plume_id: str = Field(
        description=(
            "Unique plume identifier in the format "
            "``{platform}{YYYYMMDD}{HHMMSS}-{part}``. The first three "
            "characters represent the platform (e.g. ``gao`` for Global "
            "Airborne Observatory) followed by the acquisition date and "
            "time in ISO 8601 UTC format. The ``-{part}`` suffix (e.g. "
            "``-A``) retains key information from the original radiance "
            "filename and indicates the order of multiple plumes "
            "detected in the same image."
        ),
    )
    gas: str | None = Field(
        default="CH4",
        description="The gas molecule detected during imaging operations.",
    )

    # --- Coordinates (CSV: required; JSON: derived from geometry_json) ---
    plume_latitude: float | None = Field(
        default=None, alias="plume_latitude",
        description="Latitude estimate of plume origin (decimal degrees, EPSG:4326).",
    )
    plume_longitude: float | None = Field(
        default=None, alias="plume_longitude",
        description="Longitude estimate of plume origin (decimal degrees, EPSG:4326).",
    )

    # --- Timestamps ---
    # CSV format uses "datetime"; annotated format uses "scene_timestamp"
    datetime_str: str | None = Field(
        default=None, alias="datetime",
        description=(
            "Acquisition time (UTC ISO 8601). Maps to the SQL view's "
            "``tile_date`` column. Set on CSV-format payloads; the "
            "annotated-JSON endpoint uses ``scene_timestamp`` instead."
        ),
    )
    scene_timestamp: str | None = Field(
        default=None,
        description=(
            "Acquisition time (UTC ISO 8601) — annotated-JSON variant of "
            "``datetime``. Either field may be populated, never both."
        ),
    )
    scene_id: str | None = Field(
        default=None,
        description=(
            "Parent L2B scene id, equivalent to "
            "``plume_id.rsplit('-', 1)[0]``. Same string used as the "
            "STAC item id in the ``l2b-ch4-mfa-v3a`` collection."
        ),
    )
    published_at_str: str | None = Field(
        default=None, alias="published_at",
        description="Date and time the observation was published (UTC).",
    )
    modified_str: str | None = Field(
        default=None, alias="modified",
        description="Date and time the observation was last modified (UTC).",
    )

    # --- Emissions ---
    emission_auto: float | None = Field(
        default=None,
        description=(
            "Quantified emission rate of the plume [kg/hr], estimated "
            "using the Integrated Methane Enhancement (IME) method "
            "(Duren et al. 2019, *California's Methane Super-Emitters*, "
            "Nature)."
        ),
    )
    emission_uncertainty_auto: float | None = Field(
        default=None,
        description=(
            "Uncertainty in the emission rate [± kg/hr range], derived "
            "from uncertainty in IME and wind speed."
        ),
    )

    # --- Wind ---
    wind_speed_avg_auto: float | None = Field(
        default=None,
        description="Mean wind speed at the plume site [m/s].",
    )
    wind_speed_std_auto: float | None = Field(
        default=None,
        description="Standard deviation of wind speed [m/s].",
    )
    wind_direction_avg_auto: float | None = Field(
        default=None,
        description="Wind direction at the plume site [degrees].",
    )
    wind_direction_std_auto: float | None = Field(
        default=None,
        description="Standard deviation of wind direction [degrees].",
    )
    wind_source_auto: str | None = Field(
        default=None,
        description=(
            "Wind reanalysis source (e.g. ``HRRR``, ``ECMWF_IFS``, "
            "``ERA5``). Indicates which forecast/reanalysis product fed "
            "the IME quantification."
        ),
    )

    # --- Instrument / platform ---
    instrument: str | None = Field(
        default=None,
        description=(
            "Three-character sensor abbreviation: ``ang`` (AVIRIS-NG), "
            "``av3`` (AVIRIS-3), ``emi`` (EMIT), ``tan`` (Tanager-1), "
            "``gao`` (GAO)."
        ),
    )
    platform: str | None = Field(
        default=None,
        description="Unique name of the platform the instrument is attached to.",
    )
    provider: str | None = Field(
        default=None,
        description="Short description of the data provider's name.",
    )

    # --- Classification / metadata ---
    ipcc_sector: str | None = Field(
        default=None, alias="ipcc_sector",
        description=(
            "IPCC emissions sector (e.g. ``1B2`` for Oil & Gas) when "
            "Carbon Mapper attributes one. Reference: "
            "https://www.ipcc-nggip.iges.or.jp/public/gl/guidelin/ch1ri.pdf"
        ),
    )
    sector: str | None = Field(
        default=None,
        description=(
            "Carbon Mapper free-text sector category. Often a "
            "human-readable wrapper around ``ipcc_sector`` (e.g. "
            '``"Oil & Gas (1B2)"``).'
        ),
    )
    emission_cmf_type: str | None = Field(
        default=None, alias="emission_cmf_type",
        description=(
            "Statistical column-wise atmospheric retrieval algorithm "
            "used to threshold methane / carbon dioxide plumes from "
            "background concentrations (e.g. ``mfa``)."
        ),
    )
    mission_phase: str | None = Field(
        default=None,
        description=(
            "Operational mission phase, such as ``first_light`` or "
            "``production``."
        ),
    )
    emission_version: str | None = Field(
        default=None,
        description=(
            "Version label for the algorithm + calibration applied to "
            "produce this emission record. Pairs with reprocessing "
            "campaigns."
        ),
    )
    processing_software: str | None = Field(
        default=None,
        description=(
            "Software version used by the provider to process the raw "
            "satellite data (e.g. ``cmpro: 3.41.4``)."
        ),
    )
    gsd: float | None = Field(
        default=None,
        description=(
            "Native ground sample distance — the distance on the ground "
            "represented by the center-to-center spacing of pixels in "
            "the sensor's raw radiance data [meters]."
        ),
    )
    sensitivity_mode: str | None = Field(
        default=None,
        description=(
            "The sensor's configured detection threshold and "
            "radiometric settings, which affect signal-to-noise ratio "
            "(SNR), exposure time, and spectral fidelity."
        ),
    )
    off_nadir: float | None = Field(
        default=None,
        description=(
            "Angle between the satellite's sensor line of sight and the "
            "point directly below the satellite (nadir) [degrees]. "
            "Carbon Mapper publishes this on the plume; the equivalent "
            "STAC property at the L2B scene level is ``view:off_nadir``."
        ),
    )

    # --- Quality & validation (annotated JSON) ---
    plume_quality: str | None = Field(
        default=None,
        description=(
            "CM-side quality flag for the plume retrieval. Presence "
            "implies the record was reviewed by Carbon Mapper's "
            "pipeline."
        ),
    )
    validated: bool | None = Field(
        default=None,
        description="CM-side validation flag (annotated JSON only).",
    )
    validator_user: str | None = Field(
        default=None,
        description="Validator user id from the CM annotated payload.",
    )
    has_phme: bool | None = Field(
        default=None,
        description=(
            "Whether the plume has been Plume Height + Mass Estimated. "
            "Annotated JSON only."
        ),
    )
    detection_institution: str | None = Field(
        default=None,
        description="Detection institution string from the CM annotated payload.",
    )

    # --- Source linkage (annotated JSON) ---
    source_id: str | None = Field(
        default=None,
        description=(
            "Carbon Mapper-assigned emission-source id. Joins to the CM "
            "API's source endpoint."
        ),
    )
    source_name: str | None = Field(
        default=None,
        description=(
            "Carbon Mapper source-name string (e.g. "
            "``CH4_1B2_100m_-104.17525_32.49125``)."
        ),
    )

    # --- Assets ---
    plume_tif: str | None = Field(
        default=None,
        description=(
            "HTTPS link to a GeoTIFF of the delineated plume (L3A "
            "alpha-banded mask). "
            ":meth:`~georeader.readers.carbonmapper.rasters.CMPlumeRaster.polygon`"
            " extracts the polygon from band 4 of this file — the "
            "authoritative source for the retrieved plume shape."
        ),
    )
    plume_png: str | None = Field(
        default=None,
        description="HTTPS link to a PNG visualisation of the delineated plume.",
    )
    con_tif: str | None = Field(
        default=None,
        description=(
            "HTTPS link to a GeoTIFF pixel map of unsmoothed "
            "concentration values [ppm·m]. The L2B-tile-level "
            "equivalent is the ``cmf`` asset on the parent STAC item."
        ),
    )
    rgb_tif: str | None = Field(
        default=None,
        description=(
            "HTTPS link to a 3-band, natural-colour, full-strip "
            "surface-reflectance GeoTIFF. The L2B-tile-level sibling "
            "lives in the ``l2b-rgb-v3a`` STAC collection."
        ),
    )
    rgb_png: str | None = Field(
        default=None,
        description=(
            "HTTPS link to a natural-colour, full-strip "
            "surface-reflectance PNG."
        ),
    )
    plume_rgb_png: str | None = Field(
        default=None,
        description="HTTPS link to a PNG of the plume overlaid on RGB.",
    )

    # --- Geometry sources ---
    geometry_json: dict | None = Field(
        default=None,
        description=(
            "Raw GeoJSON geometry dict from the CM payload — typically "
            "a Point or coarse Polygon. **Not** the retrieved plume "
            "polygon; for that, use ``CMPlumeRaster.polygon()`` against "
            "``plume_tif``."
        ),
    )
    plume_bounds_raw: Optional[Union[str, List[float], Tuple[float, float, float, float]]] = Field(
        default=None, alias="plume_bounds",
        description="Geographic bounds encompassing the plume image (W, S, E, N).",
    )

    # --- Derived ---
    geometry: BaseGeometry | None = Field(
        default=None,
        description=(
            "Shapely geometry built from ``geometry_json`` (preferred) "
            "or ``plume_bounds`` at validation time. **Not** the "
            "retrieved plume mask — same caveat as ``geometry_json``."
        ),
    )

    # ------------------------------------------------------------------ #
    # Field validators                                                     #
    # ------------------------------------------------------------------ #

    @field_validator(
        "plume_latitude",
        "plume_longitude",
        "gsd",
        "off_nadir",
        "emission_auto",
        "emission_uncertainty_auto",
        "wind_speed_avg_auto",
        "wind_speed_std_auto",
        "wind_direction_avg_auto",
        "wind_direction_std_auto",
        mode="before",
    )
    @classmethod
    def _coerce_float(cls, v: Any) -> float | None:
        return _to_float(v)

    @field_validator("validated", "has_phme", mode="before")
    @classmethod
    def _coerce_bool(cls, v: Any) -> bool | None:
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    # ------------------------------------------------------------------ #
    # Model validator                                                      #
    # ------------------------------------------------------------------ #

    @model_validator(mode="after")
    def _build_geometry(self) -> "CMRawPlume":
        """Build shapely geometry from ``geometry_json`` or ``plume_bounds``."""
        geom: BaseGeometry | None = None

        # Priority 1: GeoJSON
        if self.geometry_json:
            try:
                geom = shape(self.geometry_json)
            except Exception:
                geom = None
            geom_type = self.geometry_json.get("type", "")
            if geom_type == "Point" and geom is not None:
                # Buffer by ~111 m to get a small polygon
                geom = geom.buffer(0.001)
            # Fill lat/lon from Point coordinates if not set
            if (self.plume_latitude is None or self.plume_longitude is None) and geom_type == "Point":
                coords = self.geometry_json.get("coordinates")
                if coords and len(coords) >= 2:
                    object.__setattr__(self, "plume_longitude", float(coords[0]))
                    object.__setattr__(self, "plume_latitude", float(coords[1]))

        # Priority 2: Bounding box
        if geom is None:
            b = _parse_bounds(self.plume_bounds_raw)
            if b is not None:
                try:
                    geom = box(*b)
                except Exception:
                    geom = None

        object.__setattr__(self, "geometry", geom)
        return self

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def observation_datetime(self) -> datetime | None:
        """Parse observation time from ``datetime_str`` or ``scene_timestamp``."""
        return _parse_iso_datetime(self.datetime_str) or _parse_iso_datetime(self.scene_timestamp)

    @property
    def published_at(self) -> datetime | None:
        return _parse_iso_datetime(self.published_at_str)

    @property
    def modified_at(self) -> datetime | None:
        return _parse_iso_datetime(self.modified_str)

    @property
    def lat(self) -> float | None:
        return self.plume_latitude

    @property
    def lon(self) -> float | None:
        return self.plume_longitude

    @property
    def geometry_wkt(self) -> str | None:
        return self.geometry.wkt if self.geometry is not None else None

    @property
    def wind_u(self) -> float | None:
        """Eastward wind component (m/s), meteorological convention."""
        u, _ = decompose_wind(self.wind_speed_avg_auto, self.wind_direction_avg_auto)
        return u

    @property
    def wind_v(self) -> float | None:
        """Northward wind component (m/s), meteorological convention."""
        _, v = decompose_wind(self.wind_speed_avg_auto, self.wind_direction_avg_auto)
        return v

    @property
    def instrument_name(self) -> str | None:
        """Human-readable instrument name from :data:`CARBONMAPPER_INSTRUMENTS`.

        The lookup is case-insensitive — upstream payloads occasionally
        report ``"GAO"`` while ``plume_id`` prefixes are lowercase, so
        the table key is normalised at lookup time rather than relying
        on every caller to lowercase first.
        """
        if self.instrument is None:
            return None
        return CARBONMAPPER_INSTRUMENTS.get(
            self.instrument.lower(), self.instrument,
        )

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_source_dict(self) -> Dict[str, Any]:
        """Serialise to a dict suitable for round-tripping through :meth:`from_raw`."""
        d: Dict[str, Any] = {"plume_id": self.plume_id, "gas": self.gas}

        # Coordinates
        if self.plume_latitude is not None:
            d["plume_latitude"] = self.plume_latitude
        if self.plume_longitude is not None:
            d["plume_longitude"] = self.plume_longitude

        # Timestamps
        if self.datetime_str is not None:
            d["datetime"] = self.datetime_str
        if self.scene_timestamp is not None:
            d["scene_timestamp"] = self.scene_timestamp
        if self.scene_id is not None:
            d["scene_id"] = self.scene_id
        if self.published_at_str is not None:
            d["published_at"] = self.published_at_str
        if self.modified_str is not None:
            d["modified"] = self.modified_str

        # Emissions
        d["emission_auto"] = self.emission_auto
        d["emission_uncertainty_auto"] = self.emission_uncertainty_auto

        # Wind
        d["wind_speed_avg_auto"] = self.wind_speed_avg_auto
        d["wind_speed_std_auto"] = self.wind_speed_std_auto
        d["wind_direction_avg_auto"] = self.wind_direction_avg_auto
        d["wind_direction_std_auto"] = self.wind_direction_std_auto
        d["wind_source_auto"] = self.wind_source_auto

        # Instrument / platform
        d["instrument"] = self.instrument
        d["platform"] = self.platform
        d["provider"] = self.provider

        # Classification
        if self.ipcc_sector is not None:
            d["ipcc_sector"] = self.ipcc_sector
        if self.sector is not None:
            d["sector"] = self.sector
        d["emission_cmf_type"] = self.emission_cmf_type
        d["mission_phase"] = self.mission_phase
        d["emission_version"] = self.emission_version
        d["processing_software"] = self.processing_software
        d["gsd"] = self.gsd
        d["sensitivity_mode"] = self.sensitivity_mode
        d["off_nadir"] = self.off_nadir

        # Quality / validation
        if self.plume_quality is not None:
            d["plume_quality"] = self.plume_quality
        if self.validated is not None:
            d["validated"] = self.validated
        if self.validator_user is not None:
            d["validator_user"] = self.validator_user
        if self.has_phme is not None:
            d["has_phme"] = self.has_phme
        if self.detection_institution is not None:
            d["detection_institution"] = self.detection_institution

        # Source linkage
        if self.source_id is not None:
            d["source_id"] = self.source_id
        if self.source_name is not None:
            d["source_name"] = self.source_name

        # Assets
        d["plume_tif"] = self.plume_tif
        d["plume_png"] = self.plume_png
        d["con_tif"] = self.con_tif
        d["rgb_tif"] = self.rgb_tif
        d["rgb_png"] = self.rgb_png
        if self.plume_rgb_png is not None:
            d["plume_rgb_png"] = self.plume_rgb_png

        # Geometry sources
        if self.geometry_json is not None:
            d["geometry_json"] = self.geometry_json
        if self.plume_bounds_raw is not None:
            d["plume_bounds"] = self.plume_bounds_raw

        return d

    # ------------------------------------------------------------------ #
    # Factory classmethods                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_raw(cls, raw: Union[str, Dict[str, Any]]) -> "CMRawPlume":
        """Create from a JSON string or dict (CSV row or annotated-plume payload)."""
        if isinstance(raw, str):
            raw = json.loads(raw)
        return cls(**raw)

    # ------------------------------------------------------------------ #
    # Representation                                                       #
    # ------------------------------------------------------------------ #

    def _short_wkt_preview(self, max_len: int = 160) -> str | None:
        if not self.geometry:
            return None
        txt = self.geometry.wkt.replace("\n", " ").strip()
        return txt if len(txt) <= max_len else txt[: max_len - 3] + "..."

    def __str__(self) -> str:
        geom = self.geometry
        geom_type = getattr(geom, "geom_type", None)
        area = round(geom.area, 6) if geom is not None else None
        dt = self.observation_datetime.isoformat() if self.observation_datetime else None
        return (
            f"{self.__class__.__name__}\n"
            f"  plume_id: {self.plume_id}\n"
            f"  observation_datetime (UTC): {dt}\n"
            f"  lat: {self.lat}\n"
            f"  lon: {self.lon}\n"
            f"  instrument: {self.instrument}\n"
            f"  platform: {self.platform}\n"
            f"  geometry_type: {geom_type}\n"
            f"  geometry_area_deg2: {area}\n"
            f"  emission_auto: {self.emission_auto}\n"
            f"  emission_uncertainty_auto: {self.emission_uncertainty_auto}\n"
            f"  wind_speed_avg_auto: {self.wind_speed_avg_auto}\n"
            f"  wind_direction_avg_auto: {self.wind_direction_avg_auto}\n"
            f"  gas: {self.gas}\n"
            f"  validated: {self.validated}\n"
        )

    def __repr__(self) -> str:
        geom = self.geometry
        geom_type = getattr(geom, "geom_type", None)
        area = geom.area if geom is not None else None
        return (
            f"{self.__class__.__name__}(\n"
            f"  plume_id={self.plume_id!r},\n"
            f"  lat={self.lat},\n"
            f"  lon={self.lon},\n"
            f"  gas={self.gas!r},\n"
            f"  instrument={self.instrument!r},\n"
            f"  platform={self.platform!r},\n"
            f"  emission_auto={self.emission_auto},\n"
            f"  emission_uncertainty_auto={self.emission_uncertainty_auto},\n"
            f"  wind_speed_avg_auto={self.wind_speed_avg_auto},\n"
            f"  wind_direction_avg_auto={self.wind_direction_avg_auto},\n"
            f"  validated={self.validated},\n"
            f"  geometry_type={geom_type},\n"
            f"  geometry_area_deg2={area},\n"
            f"  geometry_wkt_preview={self._short_wkt_preview()!r}\n"
            f")"
        )


# Backward-compat alias — older marsml imports referenced this name.
CarbonMapperPlumeRaw = CMRawPlume
