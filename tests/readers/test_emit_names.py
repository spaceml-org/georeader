"""Tests for EMIT product-name parsing and DAAC link building in georeader.readers.emit.

Pure string tests, no network and no fixture files. NASA LP DAAC closed the v001
collections on 2026-08-31 and publishes v002 (L2A mask: v003) with names that drop
the ``_<orbit>_<scene>`` suffix. v001 links must stay byte-identical to georeader 2.3.5.
"""

from __future__ import annotations
from datetime import datetime, timezone

import pytest

from georeader.readers import emit


DAAC = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected"

# Real v001 L1B RAD ids (2022 to the last v001 acquisition on 2026-08-31).
V001_IDS = [
    "EMIT_L1B_RAD_001_20220827T060753_2223904_013",
    "EMIT_L1B_RAD_001_20230824T175349_2323612_024",
    "EMIT_L1B_RAD_001_20231024T070157_2329705_004",
    "EMIT_L1B_RAD_001_20231102T172835_2330611_005",
    "EMIT_L1B_RAD_001_20231201T183549_2333512_016",
    "EMIT_L1B_RAD_001_20240625T065045_2417705_028",
    "EMIT_L1B_RAD_001_20241005T065118_2427905_013",
    "EMIT_L1B_RAD_001_20241005T082651_2427906_021",
    "EMIT_L1B_RAD_001_20241202T071056_2433705_010",
    "EMIT_L1B_RAD_001_20260831T012520_2624301_059",
]

V002_ID = "EMIT_L1B_RAD_002_20260831T012520"


# ── parse_product_name ────────────────────────────────────────────────────


class TestParseProductName:
    def test_v001_id(self):
        pid = emit.parse_product_name("EMIT_L1B_RAD_001_20220827T060753_2223904_013")
        assert pid == emit.EMITProductID("L1B", "RAD", "001", "20220827T060753", "2223904", "013")

    def test_v002_id_has_no_orbit_scene(self):
        pid = emit.parse_product_name(V002_ID)
        assert (pid.level, pid.product, pid.version, pid.dt) == ("L1B", "RAD", "002", "20260831T012520")
        assert pid.orbit is None and pid.scene is None

    @pytest.mark.parametrize("name", [
        f"{V002_ID}.nc",
        f"/data/emit/{V002_ID}.nc",
        f"az://container/emit/{V002_ID}.nc",
    ])
    def test_filenames_and_paths(self, name):
        assert emit.parse_product_name(name).name == V002_ID

    def test_acquisition_is_aware_utc(self):
        pid = emit.parse_product_name(V002_ID)
        assert pid.acquisition == datetime(2026, 8, 31, 1, 25, 20, tzinfo=timezone.utc)

    @pytest.mark.parametrize("name, product", [
        ("EMIT_L1B_OBS_002_20260831T012520.nc", ("L1B", "OBS", "002")),
        ("EMIT_L2A_MASK_003_20260831T012520.nc", ("L2A", "MASK", "003")),
        ("EMIT_L2A_RFL_001_20220827T060753_2223904_013", ("L2A", "RFL", "001")),
        ("EMIT_L2B_CH4ENH_002_20220827T060753_2223904_013.tif", ("L2B", "CH4ENH", "002")),
    ])
    def test_other_products(self, name, product):
        pid = emit.parse_product_name(name)
        assert (pid.level, pid.product, pid.version) == product

    def test_v_prefixed_version(self):
        pid = emit.parse_product_name("EMIT_L2B_CH4ENH_V001_20220827T060753_2223904_013")
        assert pid.version == "001"
        assert pid.name == "EMIT_L2B_CH4ENH_001_20220827T060753_2223904_013"

    @pytest.mark.parametrize("name", [
        "EMIT_L1B_RAD_001_20220827T060753_2223904",  # truncated suffix
        "EMIT_L1B_RAD_20220827T060753",              # no version
        "PRS_L1_STD_OFFL_20220827060753",            # foreign prefix
        "EMIT_L1B_RAD_001_20220827",                 # truncated datetime
    ])
    def test_rejects_non_emit_names(self, name):
        with pytest.raises(ValueError, match="Not an EMIT product name"):
            emit.parse_product_name(name)

    def test_with_product_keeps_acquisition(self):
        pid = emit.parse_product_name(V002_ID).with_product("L2A", "MASK", "003")
        assert pid.name == "EMIT_L2A_MASK_003_20260831T012520"

    @pytest.mark.parametrize("name", V001_IDS + [V002_ID])
    def test_name_round_trip(self, name):
        assert emit.parse_product_name(name).name == name


# ── Link builders ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("tile", V001_IDS)
def test_links_v001_unchanged(tile):
    """Expected strings are the georeader 2.3.5 outputs, spelled out as templates."""
    suffix = tile[len("EMIT_L1B_RAD_001_"):]  # <dt>_<orbit>_<scene>
    assert emit.get_radiance_link(tile) == f"{DAAC}/EMITL1BRAD.001/{tile}/{tile}.nc"
    assert emit.get_radiance_link(f"/tmp/{tile}.nc") == f"{DAAC}/EMITL1BRAD.001/{tile}/{tile}.nc"
    assert emit.get_obs_link(tile) == f"{DAAC}/EMITL1BRAD.001/{tile}/EMIT_L1B_OBS_001_{suffix}.nc"
    assert emit.get_l2amask_link(tile) == (
        f"{DAAC}/EMITL2ARFL.001/EMIT_L2A_RFL_001_{suffix}/EMIT_L2A_MASK_001_{suffix}.nc"
    )


def test_links_v001_from_companion_product():
    """Any product of the acquisition resolves to the L1B links, as in 2.3.5."""
    rfl = "EMIT_L2A_RFL_001_20220827T060753_2223904_013"
    rad = "EMIT_L1B_RAD_001_20220827T060753_2223904_013"
    assert emit.get_radiance_link(rfl) == f"{DAAC}/EMITL1BRAD.001/{rad}/{rad}.nc"


def test_links_v002():
    assert emit.get_radiance_link(f"{V002_ID}.nc") == f"{DAAC}/EMITL1BRAD.002/{V002_ID}/{V002_ID}.nc"
    assert emit.get_obs_link(f"{V002_ID}.nc") == (
        f"{DAAC}/EMITL1BRAD.002/{V002_ID}/EMIT_L1B_OBS_002_20260831T012520.nc"
    )
    assert emit.get_l2amask_link(f"{V002_ID}.nc") == (
        f"{DAAC}/EMITL2AMASK.003/EMIT_L2A_MASK_003_20260831T012520/EMIT_L2A_MASK_003_20260831T012520.nc"
    )


def test_ch4enh_v001_points_at_v002():
    """CH4ENH v001 was emptied in 2024-11; v002 covers every v001 L1B scene."""
    tile = "EMIT_L1B_RAD_001_20260831T012520_2624301_059"
    ch4 = "EMIT_L2B_CH4ENH_002_20260831T012520_2624301_059"
    assert emit.get_ch4enhancement_link(tile) == f"{DAAC}/EMITL2BCH4ENH.002/{ch4}/{ch4}.tif"


def test_ch4enh_v002_is_none():
    assert emit.get_ch4enhancement_link(V002_ID) is None


@pytest.mark.parametrize("builder", [
    emit.get_radiance_link, emit.get_obs_link, emit.get_l2amask_link, emit.get_ch4enhancement_link,
])
def test_non_emit_name_raises(builder):
    with pytest.raises(ValueError, match="Not an EMIT product name"):
        builder("S2A_MSIL1C_20220827T060753.SAFE")


@pytest.mark.parametrize("builder", [emit.get_l2amask_link, emit.get_ch4enhancement_link])
def test_unknown_version_raises(builder):
    with pytest.raises(ValueError, match="Unknown EMIT L1B version '003'"):
        builder("EMIT_L1B_RAD_003_20270101T000000")


# ── product_name_from_params / split_product_name ────────────────────────


def test_product_name_from_params_v001_default():
    assert emit.product_name_from_params("emit20220810t064957", "2222205", "033") == (
        "EMIT_L1B_RAD_001_20220810T064957_2222205_033"
    )


def test_product_name_from_params_v002():
    assert emit.product_name_from_params("emit20220810t064957", version="002") == (
        "EMIT_L1B_RAD_002_20220810T064957"
    )


def test_product_name_from_params_v001_needs_orbit_scene():
    with pytest.raises(ValueError, match="orbit and daac_scene_number"):
        emit.product_name_from_params("emit20220810t064957")


def test_split_product_name_v001():
    assert emit.split_product_name("EMIT_L1B_RAD_001_20220810T064957_2222205_033") == (
        "emit20220810t064957", "2222205", "033",
        datetime(2022, 8, 10, 6, 49, 57, tzinfo=timezone.utc),
    )


def test_split_product_name_v002_has_no_orbit_scene():
    assert emit.split_product_name(f"{V002_ID}.nc") == (
        "emit20260831t012520", None, None,
        datetime(2026, 8, 31, 1, 25, 20, tzinfo=timezone.utc),
    )
