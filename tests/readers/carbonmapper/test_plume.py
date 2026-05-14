"""Tests for georeader.readers.carbonmapper.plume — unified CarbonMapper plume model."""

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from georeader.readers.carbonmapper.plume import (
    CARBONMAPPER_INSTRUMENTS,
    CMRawPlume,
    CarbonMapperPlumeRaw,
    Collection,
    Gas,
    Instrument,
    _parse_iso_datetime,
    _to_float,
    decompose_wind,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _csv_raw(**overrides) -> dict:
    """Minimal CSV-format plume dict."""
    base = {
        "plume_id": "tan20240101t120000p00001-A",
        "plume_latitude": 31.456,
        "plume_longitude": -102.123,
        "datetime": "2024-01-01T12:00:00Z",
        "gas": "CH4",
        "instrument": "tan",
        "platform": "Tanager-1",
        "emission_auto": 500.0,
        "plume_bounds": [-102.2, 31.4, -102.0, 31.5],
    }
    base.update(overrides)
    return base


def _json_raw(**overrides) -> dict:
    """Minimal annotated-JSON-format plume dict."""
    base = {
        "plume_id": "emi20240420t101448p07050-A",
        "gas": "CH4",
        "geometry_json": {"type": "Point", "coordinates": [-102.123, 31.456]},
        "emission_auto": 1234.5,
        "emission_uncertainty_auto": 200.0,
        "wind_speed_avg_auto": 5.2,
        "wind_direction_avg_auto": 270.0,
        "plume_quality": "good",
        "validated": True,
        "has_phme": False,
        "detection_institution": "Carbon Mapper",
        "instrument": "emi",
        "scene_timestamp": "2024-04-20T10:14:48Z",
        "scene_id": "emi20240420t101448",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestToFloat:
    def test_none(self):
        assert _to_float(None) is None

    def test_empty_string(self):
        assert _to_float("") is None

    def test_int(self):
        assert _to_float(42) == 42.0

    def test_string_number(self):
        assert _to_float("3.14") == pytest.approx(3.14)

    def test_non_numeric(self):
        assert _to_float("abc") is None


class TestParseIsoDatetime:
    def test_none(self):
        assert _parse_iso_datetime(None) is None

    def test_z_suffix(self):
        dt = _parse_iso_datetime("2024-04-20T10:14:48Z")
        assert dt is not None
        assert dt.year == 2024

    def test_short_offset(self):
        dt = _parse_iso_datetime("2024-04-20T10:14:48+02")
        assert dt is not None

    def test_naive_assumed_utc(self):
        dt = _parse_iso_datetime("2024-04-20T10:14:48")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_invalid(self):
        assert _parse_iso_datetime("not-a-date") is None


class TestDecomposeWind:
    def test_none_inputs(self):
        assert decompose_wind(None, 180.0) == (None, None)
        assert decompose_wind(5.0, None) == (None, None)

    def test_from_north(self):
        u, v = decompose_wind(10.0, 0.0)
        assert u == pytest.approx(0.0, abs=1e-10)
        assert v == pytest.approx(-10.0, abs=1e-10)

    def test_from_west(self):
        u, v = decompose_wind(10.0, 270.0)
        assert u == pytest.approx(10.0, abs=1e-10)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_magnitude_preserved(self):
        u, v = decompose_wind(7.5, 135.0)
        assert math.sqrt(u**2 + v**2) == pytest.approx(7.5, abs=1e-10)


# ---------------------------------------------------------------------------
# Alias
# ---------------------------------------------------------------------------


class TestAlias:
    def test_alias_is_same_class(self):
        assert CarbonMapperPlumeRaw is CMRawPlume


# ---------------------------------------------------------------------------
# Construction from CSV format
# ---------------------------------------------------------------------------


class TestCSVFormat:
    def test_basic_csv_plume(self):
        plume = CMRawPlume.from_raw(_csv_raw())
        assert plume.plume_id == "tan20240101t120000p00001-A"
        assert plume.lat == pytest.approx(31.456)
        assert plume.lon == pytest.approx(-102.123)

    def test_csv_geometry_from_bounds(self):
        plume = CMRawPlume.from_raw(_csv_raw())
        assert plume.geometry is not None
        assert plume.geometry.geom_type == "Polygon"

    def test_csv_observation_datetime(self):
        plume = CMRawPlume.from_raw(_csv_raw())
        assert plume.observation_datetime is not None
        assert plume.observation_datetime.year == 2024

    def test_csv_round_trip(self):
        plume = CMRawPlume.from_raw(_csv_raw())
        d = plume.to_source_dict()
        plume2 = CMRawPlume.from_raw(d)
        assert plume2.plume_id == plume.plume_id
        assert plume2.emission_auto == plume.emission_auto

    def test_csv_from_json_string(self):
        raw_str = json.dumps(_csv_raw())
        plume = CMRawPlume.from_raw(raw_str)
        assert plume.plume_id == "tan20240101t120000p00001-A"

    def test_csv_fields_present(self):
        plume = CMRawPlume.from_raw(_csv_raw(
            ipcc_sector="Oil & Gas (1B2)",
            emission_cmf_type="auto",
            mission_phase="phase1",
            gsd=7.5,
        ))
        assert plume.ipcc_sector == "Oil & Gas (1B2)"
        assert plume.emission_cmf_type == "auto"
        assert plume.mission_phase == "phase1"
        assert plume.gsd == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Construction from annotated JSON format
# ---------------------------------------------------------------------------


class TestAnnotatedJSONFormat:
    def test_basic_json_plume(self):
        plume = CMRawPlume.from_raw(_json_raw())
        assert plume.plume_id == "emi20240420t101448p07050-A"
        assert plume.gas == "CH4"

    def test_json_geometry_from_geojson_point(self):
        plume = CMRawPlume.from_raw(_json_raw())
        assert plume.geometry is not None
        # Point gets buffered to Polygon
        assert plume.geometry.geom_type == "Polygon"

    def test_json_lat_lon_derived_from_point(self):
        plume = CMRawPlume.from_raw(_json_raw())
        assert plume.lat == pytest.approx(31.456)
        assert plume.lon == pytest.approx(-102.123)

    def test_json_polygon_geometry(self):
        raw = _json_raw(geometry_json={
            "type": "Polygon",
            "coordinates": [[
                [-102.124, 31.455],
                [-102.122, 31.455],
                [-102.122, 31.457],
                [-102.124, 31.457],
                [-102.124, 31.455],
            ]],
        })
        plume = CMRawPlume.from_raw(raw)
        assert plume.geometry.geom_type == "Polygon"

    def test_json_observation_datetime_from_scene_timestamp(self):
        plume = CMRawPlume.from_raw(_json_raw())
        dt = plume.observation_datetime
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 4

    def test_json_round_trip(self):
        plume = CMRawPlume.from_raw(_json_raw())
        d = plume.to_source_dict()
        plume2 = CMRawPlume.from_raw(d)
        assert plume2.plume_id == plume.plume_id
        assert plume2.validated == plume.validated

    def test_json_quality_fields(self):
        plume = CMRawPlume.from_raw(_json_raw())
        assert plume.plume_quality == "good"
        assert plume.validated is True
        assert plume.has_phme is False
        assert plume.detection_institution == "Carbon Mapper"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


class TestValidators:
    def test_string_emission_coerced(self):
        plume = CMRawPlume.from_raw(_json_raw(emission_auto="999.9"))
        assert plume.emission_auto == pytest.approx(999.9)

    def test_non_numeric_emission_becomes_none(self):
        plume = CMRawPlume.from_raw(_json_raw(emission_auto="N/A"))
        assert plume.emission_auto is None

    def test_validated_string_true(self):
        plume = CMRawPlume.from_raw(_json_raw(validated="true"))
        assert plume.validated is True

    def test_validated_string_false(self):
        plume = CMRawPlume.from_raw(_json_raw(validated="no"))
        assert plume.validated is False

    def test_validated_none(self):
        plume = CMRawPlume.from_raw(_json_raw(validated=None))
        assert plume.validated is None

    def test_has_phme_string_yes(self):
        plume = CMRawPlume.from_raw(_json_raw(has_phme="yes"))
        assert plume.has_phme is True


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_wind_u_v(self):
        plume = CMRawPlume.from_raw(_json_raw(
            wind_speed_avg_auto=10.0,
            wind_direction_avg_auto=270.0,
        ))
        assert plume.wind_u == pytest.approx(10.0, abs=1e-10)
        assert plume.wind_v == pytest.approx(0.0, abs=1e-10)

    def test_wind_none_when_missing(self):
        plume = CMRawPlume.from_raw(_json_raw(wind_speed_avg_auto=None))
        assert plume.wind_u is None
        assert plume.wind_v is None

    def test_instrument_name_known(self):
        plume = CMRawPlume.from_raw(_json_raw(instrument="emi"))
        assert plume.instrument_name == "EMIT"

    def test_instrument_name_unknown(self):
        plume = CMRawPlume.from_raw(_json_raw(instrument="xyz"))
        assert plume.instrument_name == "xyz"

    def test_instrument_name_none(self):
        plume = CMRawPlume.from_raw(_json_raw(instrument=None))
        assert plume.instrument_name is None

    def test_geometry_wkt(self):
        plume = CMRawPlume.from_raw(_csv_raw())
        assert plume.geometry_wkt is not None
        assert "POLYGON" in plume.geometry_wkt

    def test_observation_datetime_prefers_datetime_str(self):
        """When both datetime_str and scene_timestamp exist, datetime_str wins."""
        plume = CMRawPlume.from_raw({
            "plume_id": "t",
            "datetime": "2024-01-01T00:00:00Z",
            "scene_timestamp": "2025-06-01T00:00:00Z",
        })
        assert plume.observation_datetime.year == 2024


# ---------------------------------------------------------------------------
# Note: ``CMRawPlume.from_source_id`` and ``from_plume_staging_record``
# (the marsml-coupled DB classmethods) live downstream in marsml — they
# are intentionally not exposed by the georeader port. Tests for those
# stay in marsml.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# String representations
# ---------------------------------------------------------------------------


class TestRepr:
    def test_str_contains_plume_id(self):
        plume = CMRawPlume.from_raw(_json_raw())
        assert "emi20240420t101448p07050-A" in str(plume)

    def test_repr_contains_plume_id(self):
        plume = CMRawPlume.from_raw(_json_raw())
        assert "emi20240420t101448p07050-A" in repr(plume)


# ---------------------------------------------------------------------------
# scene_id / scene_uuid / version derived properties
# ---------------------------------------------------------------------------


class TestSceneIdProperty:
    """`scene_id` is derived from `plume_id` (not the API's UUID field)."""

    def test_scene_id_strips_part_suffix(self):
        plume = CMRawPlume.from_raw(_json_raw())
        # "emi20240420t101448p07050-A" → "emi20240420t101448p07050"
        assert plume.scene_id == "emi20240420t101448p07050"

    def test_scene_id_for_tanager(self):
        plume = CMRawPlume.from_raw(_csv_raw(
            plume_id="tan20251212t185057c20s4001-E",
        ))
        assert plume.scene_id == "tan20251212t185057c20s4001"

    def test_scene_id_handles_two_hyphens(self):
        """If the plume_id ever gains an extra hyphen segment, rsplit
        only strips the last one."""
        plume = CMRawPlume.from_raw(_csv_raw(
            plume_id="tan20251212t185057c20s4001-foo-A",
        ))
        assert plume.scene_id == "tan20251212t185057c20s4001-foo"

    def test_scene_uuid_holds_api_uuid(self):
        """The API's `scene_id` (UUID) lands on the renamed `scene_uuid`
        field via the `alias`."""
        plume = CMRawPlume.from_raw(_json_raw(
            scene_id="64a51834-5fe5-40e0-aadd-e0c5944850c3",
        ))
        assert plume.scene_uuid == "64a51834-5fe5-40e0-aadd-e0c5944850c3"
        # And the property returns the parseable form, not the UUID
        assert plume.scene_id == "emi20240420t101448p07050"


class TestVersionProperty:
    def test_version_re_exposes_emission_version(self):
        plume = CMRawPlume.from_raw(_json_raw(emission_version="v3a"))
        assert plume.version == "v3a"

    def test_version_v3c(self):
        plume = CMRawPlume.from_raw(_json_raw(emission_version="v3c"))
        assert plume.version == "v3c"

    def test_version_none_when_absent(self):
        plume = CMRawPlume.from_raw(_csv_raw())   # no emission_version
        assert plume.version is None


# ---------------------------------------------------------------------------
# Enums (Gas / Instrument / Collection)
# ---------------------------------------------------------------------------


class TestGasEnum:
    def test_members(self):
        assert Gas.CH4.value == "CH4"
        assert Gas.CO2.value == "CO2"

    def test_str_serialises_to_value(self):
        assert str(Gas.CH4) == "CH4"

    def test_construct_from_string(self):
        assert Gas("CH4") is Gas.CH4

    def test_satisfies_str_protocol(self):
        # StrEnum inherits from str — usable as a string anywhere
        assert isinstance(Gas.CH4, str)
        assert "CH4".startswith(Gas.CH4)


class TestInstrumentEnum:
    def test_members(self):
        assert Instrument.TANAGER.value == "tan"
        assert Instrument.EMIT.value == "emi"
        assert Instrument.AVIRIS_NG.value == "ang"
        assert Instrument.AVIRIS_3.value == "av3"
        assert Instrument.GAO.value == "GAO"   # upstream uppercase

    def test_case_insensitive_lookup(self):
        # Both lowercase and uppercase free-form strings resolve to the
        # canonical member via the `_missing_` hook
        assert Instrument("tan") is Instrument.TANAGER
        assert Instrument("TAN") is Instrument.TANAGER
        assert Instrument("Tan") is Instrument.TANAGER

    def test_gao_normalises_both_cases(self):
        assert Instrument("GAO") is Instrument.GAO
        assert Instrument("gao") is Instrument.GAO

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            Instrument("not-an-instrument")


class TestCollectionEnum:
    def test_v3a_is_stac_resident(self):
        assert Collection.L3A_VIS_V3A.is_stac_resident
        assert Collection.L3A_IME_V3A.is_stac_resident
        assert Collection.L2B_V3A.is_stac_resident
        assert Collection.L2B_RGB_V3A.is_stac_resident

    def test_v3c_is_not_stac_resident(self):
        assert not Collection.L3A_VIS_V3C.is_stac_resident
        assert not Collection.L3A_IME_V3C.is_stac_resident

    def test_version_property(self):
        assert Collection.L3A_VIS_V3A.version == "v3a"
        assert Collection.L3A_VIS_V3C.version == "v3c"
        assert Collection.L2B_V3A.version == "v3a"

    def test_string_value(self):
        assert Collection.L3A_VIS_V3A.value == "l3a-vis-ch4-mfa-v3a"
        assert Collection.L3A_IME_V3C.value == "l3a-ime-ch4-mfa-v3c"
        assert Collection.L2B_RGB_V3A.value == "l2b-rgb-v3a"
