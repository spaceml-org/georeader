"""Tests for the explicit Carbon Mapper product registry.

URL expectations are copied verbatim from live-API probes in the
2026-07 audit (``docs/carbonmapper/api_audit_2026-07.md``) — every
"expected" URL below returned 200/206 with a Bearer token on
2026-07-06.
"""

import json

import pytest

from georeader.readers.carbonmapper import products as P
from georeader.readers.carbonmapper.image import CMPlumeImage, _derive_asset_urls
from georeader.readers.carbonmapper.products import (
    CM_API_ASSET_BASE,
    CMCollectionSpec,
    CMProductFamily,
    CMProductNotSelected,
    _parse_asset_url,
    parse_item_date,
    product_for_key,
)

# The June-2026 Tanager plume used as the audit's seed. Its plume_tif,
# and the IME URL its record's own con_tif field pointed at.
PID_V3D = "tan20260623t124240c80s4001-A"
SEED_PLUME_TIF = (
    "https://catalog.carbonmapper.org/l3a-vis-ch4-mfa-v3d/2026/06/23/"
    f"{PID_V3D}/{PID_V3D}_l3a-vis-ch4-mfa-v3d_plume.tif"
    "?Expires=9999&Signature=abc"
)
AUDIT_CON_TIF = (
    f"{CM_API_ASSET_BASE}/l3a-ime-ch4-mfa-v3d/2026/06/23/"
    f"{PID_V3D}/{PID_V3D}_l3a-ime-ch4-mfa-v3d_ime-cmf-concentrations.tif"
)


# ─── CMCollectionSpec ────────────────────────────────────────────────


class TestCollectionSpec:
    def test_composes_all_families_same_version(self):
        spec = CMCollectionSpec(version="v3d")
        assert spec.collection_id(CMProductFamily.L3A_VIS) == "l3a-vis-ch4-mfa-v3d"
        assert spec.collection_id(CMProductFamily.L3A_IME) == "l3a-ime-ch4-mfa-v3d"
        assert spec.collection_id(CMProductFamily.L2B) == "l2b-ch4-mfa-v3d"
        assert spec.collection_id(CMProductFamily.L2B_RGB) == "l2b-rgb-v3d"

    def test_co2_and_cmf_type_variants(self):
        spec = CMCollectionSpec(version="v3a", gas="co2", cmf_type="mfal")
        assert spec.collection_id(CMProductFamily.L3A_IME) == "l3a-ime-co2-mfal-v3a"
        # RGB is not gas-typed.
        assert spec.collection_id(CMProductFamily.L2B_RGB) == "l2b-rgb-v3a"

    @pytest.mark.parametrize(
        "cid, expected",
        [
            ("l3a-vis-ch4-mfa-v3d", CMCollectionSpec("v3d", "ch4", "mfa")),
            ("l3a-ime-co2-mfal-v3a", CMCollectionSpec("v3a", "co2", "mfal")),
            ("l2b-ch4-mfm-v1", CMCollectionSpec("v1", "ch4", "mfm")),
            ("l2b-rgb-v3a", CMCollectionSpec("v3a")),
            ("l3a-vis-ch4-mf-v002", CMCollectionSpec("v002", "ch4", "mf")),
        ],
    )
    def test_from_collection_id(self, cid, expected):
        assert CMCollectionSpec.from_collection_id(cid) == expected

    def test_from_collection_id_rejects_legacy_family(self):
        # `l3a-ch4-mf-v1` predates the vis/ime split — no family match.
        with pytest.raises(ValueError, match="Unrecognised"):
            CMCollectionSpec.from_collection_id("l3a-ch4-mf-v1")

    def test_from_plume_record_prefers_url(self):
        record = {
            "plume_id": PID_V3D,
            "plume_tif": SEED_PLUME_TIF,
            # Deliberately contradictory fields — the URL must win.
            "gas": "CH4", "cmf_type": "mfa", "emission_version": "v1",
        }
        assert CMCollectionSpec.from_plume_record(record).version == "v3d"

    def test_from_plume_record_falls_back_to_fields(self):
        record = {
            "plume_id": PID_V3D,
            "gas": "CH4", "cmf_type": "mfa", "emission_version": "v3d",
        }
        spec = CMCollectionSpec.from_plume_record(record)
        assert spec == CMCollectionSpec(version="v3d", gas="ch4", cmf_type="mfa")

    def test_from_plume_record_legacy_url_falls_back_to_fields(self):
        legacy = (
            "https://catalog.carbonmapper.org/l3a-ch4-mf-v1/2019/06/15/"
            "ang20190615t184217-A/ang20190615t184217-A_l3a-ch4-mf-v1_plume.tif"
        )
        record = {
            "plume_id": "ang20190615t184217-A",
            "plume_tif": legacy,
            "gas": "CH4", "cmf_type": "mf", "emission_version": "v1",
        }
        spec = CMCollectionSpec.from_plume_record(record)
        assert spec.cmf_type == "mf"
        assert spec.version == "v1"

    def test_from_plume_record_unresolvable_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            CMCollectionSpec.from_plume_record({"plume_id": "x"})


# ─── URL parsing helpers ─────────────────────────────────────────────


class TestParseAssetUrl:
    def test_parses_signed_cdn_url(self):
        parsed = _parse_asset_url(SEED_PLUME_TIF)
        assert parsed is not None
        assert parsed.collection_id == "l3a-vis-ch4-mfa-v3d"
        assert (parsed.yyyy, parsed.mm, parsed.dd) == ("2026", "06", "23")
        assert parsed.item_id == PID_V3D
        assert parsed.key == "plume.tif"

    def test_rejects_non_asset_url(self):
        assert _parse_asset_url("https://catalog.carbonmapper.org/foo/bar.tif") is None


class TestParseItemDate:
    def test_scene_name(self):
        assert parse_item_date("tan20260623t124240c80s4001") == ("2026", "06", "23")

    def test_uuid_scene_id_raises(self):
        # The catalog record's `scene_id` field is a UUID — it must be
        # rejected loudly, not silently mis-parsed.
        with pytest.raises(ValueError, match="UUID"):
            parse_item_date("3faede0d-45dd-43e4-a1b4-ef4c0d69acf2")


# ─── Registry integrity ──────────────────────────────────────────────


class TestRegistry:
    def test_default_plume_products_match_legacy_bundle(self):
        assert tuple(p.key for p in P.DEFAULT_PLUME_PRODUCTS) == (
            "plume.tif", "plume-concentrations.tif", "plume-outline.geojson",
            "rgb.tif", "ime-cmf-concentrations.tif", "ime-cmf-mask.tif",
            "ime-cmf-outline.geojson",
        )

    def test_scene_products_match_l2b_bands(self):
        assert tuple(p.band for p in P.DEFAULT_SCENE_PRODUCTS) == (
            "cmf", "cmf-unortho", "uncertainty", "uncertainty-unortho",
            "artifact-mask", "uas",
        )

    def test_product_identity_is_family_scoped(self):
        # rgb.tif exists in two families — two distinct products.
        assert P.RGB_TIF != P.SCENE_RGB
        assert product_for_key("rgb.tif", family=CMProductFamily.L3A_VIS) is P.RGB_TIF
        assert product_for_key("rgb.tif", family=CMProductFamily.L2B_RGB) is P.SCENE_RGB
        with pytest.raises(KeyError, match="Ambiguous"):
            product_for_key("rgb.tif")
        with pytest.raises(KeyError, match="No Carbon Mapper product"):
            product_for_key("nope.tif")

    def test_names_are_python_identifiers(self):
        for prod in P.ALL_PRODUCTS:
            assert prod.name.isidentifier(), prod.key


# ─── asset_url against audit-verified ground truth ───────────────────


class TestAssetUrl:
    def test_ime_concentrations_matches_record_con_tif(self):
        """The composed IME URL must be byte-identical to what the
        record's own con_tif field pointed at (audit §1)."""
        spec = CMCollectionSpec(version="v3d")
        url = P.IME_CONCENTRATIONS.asset_url(spec, PID_V3D)
        assert url == AUDIT_CON_TIF

    def test_l2b_cmf_same_version_pairing(self):
        """Audit §4: the v3d plume's L2B parent serves at v3d."""
        spec = CMCollectionSpec(version="v3d")
        scene = PID_V3D.rsplit("-", 1)[0]
        url = P.CMF.asset_url(spec, scene)
        assert url == (
            f"{CM_API_ASSET_BASE}/l2b-ch4-mfa-v3d/2026/06/23/"
            f"{scene}/{scene}_l2b-ch4-mfa-v3d_cmf.tif"
        )

    def test_explicit_collection_id_override(self):
        url = P.PLUME_TIF.asset_url(
            None, PID_V3D, collection_id="l3a-ch4-mf-v1",
        )
        assert "/l3a-ch4-mf-v1/" in url
        assert url.endswith(f"{PID_V3D}_l3a-ch4-mf-v1_plume.tif")

    def test_requires_spec_or_collection(self):
        with pytest.raises(ValueError, match="spec or an"):
            P.PLUME_TIF.asset_url(None, PID_V3D)


# ─── Kind-specific open() ────────────────────────────────────────────


class TestOpenDispatch:
    def test_text_product_local(self, tmp_path):
        f = tmp_path / "uas.txt"
        f.write_text("sensor: tanager")
        assert P.UAS.open(str(f)) == "sensor: tanager"

    def test_quicklook_product_local(self, tmp_path):
        f = tmp_path / "plume.png"
        f.write_bytes(b"\x89PNG...")
        assert P.PLUME_PNG.open(str(f)) == b"\x89PNG..."

    def test_vector_product_local(self, tmp_path):
        geojson = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
            },
        }
        f = tmp_path / "plume-outline.geojson"
        f.write_text(json.dumps(geojson))
        geom = P.PLUME_OUTLINE.open(str(f))
        assert geom is not None and geom.is_valid


# ─── Explicit selection on CMPlumeImage ──────────────────────────────


class TestProductSelection:
    def _record(self):
        return {"plume_id": PID_V3D, "plume_tif": SEED_PLUME_TIF}

    def test_derive_respects_selection(self):
        urls = _derive_asset_urls(
            self._record(), products=(P.PLUME_TIF, P.IME_MASK),
        )
        assert set(urls) == {"plume.tif", "ime-cmf-mask.tif"}
        assert urls["ime-cmf-mask.tif"].startswith(CM_API_ASSET_BASE)

    def test_derive_all_products_includes_quicklooks(self):
        urls = _derive_asset_urls(self._record(), products=P.ALL_PLUME_PRODUCTS)
        assert "plume-rgb.png" in urls
        assert urls["plume-rgb.png"].endswith("_l3a-vis-ch4-mfa-v3d_plume-rgb.png")

    def test_derive_rejects_scene_products(self):
        with pytest.raises(ValueError, match="per-scene"):
            _derive_asset_urls(self._record(), products=(P.CMF,))

    def test_unselected_product_raises(self):
        img = CMPlumeImage(
            plume_id=PID_V3D,
            urls={"plume.tif": "/tmp/nonexistent.tif"},
            products=(P.PLUME_TIF,),
        )
        with pytest.raises(CMProductNotSelected, match="rgb.tif"):
            _ = img.rgb
        with pytest.raises(CMProductNotSelected):
            img.product(P.IME_MASK)

    def test_selected_but_absent_returns_none(self):
        img = CMPlumeImage(plume_id=PID_V3D, urls={})
        assert img.ime_concentrations is None
