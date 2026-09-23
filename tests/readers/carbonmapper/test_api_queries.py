"""Tests for ``georeader.readers.carbonmapper.api_queries``."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests

from georeader.readers.carbonmapper import api_queries as aq
from georeader.readers.carbonmapper.plume import CMRawPlume
from georeader.readers.carbonmapper.source import CMSource


# ─────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────


def _stac_item(scene_id: str = "tan20251212t185057c20s4001") -> dict:
    return {
        "id": scene_id,
        "collection": "l2b-ch4-mfa-v3a",
        "properties": {
            "datetime": "2025-12-12T18:50:57Z",
            "platform": "Tanager1",
        },
        "bbox": [-103.6, 31.4, -103.4, 31.6],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-103.6, 31.4],
                [-103.4, 31.4],
                [-103.4, 31.6],
                [-103.6, 31.6],
                [-103.6, 31.4],
            ]],
        },
        "assets": {
            "cmf":           {"href": "https://cm/.../cmf.tif"},
            "rgb":           {"href": "https://cm/.../rgb.tif"},
            "uncertainty":   {"href": "https://cm/.../uncertainty.tif"},
            "artifact-mask": {"href": "https://cm/.../artifact-mask.tif"},
        },
    }


def _http_error(status: int) -> requests.HTTPError:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    err = requests.HTTPError(f"{status}", response=resp)
    return err


# ─────────────────────────────────────────────────────────────────────
#  CMTileItem.from_stac_item
# ─────────────────────────────────────────────────────────────────────


def test_cmtileitem_from_stac_parses_basics():
    item = aq.CMTileItem.from_stac_item(_stac_item())
    assert item.scene_id == "tan20251212t185057c20s4001"
    assert item.collection == "l2b-ch4-mfa-v3a"
    assert item.platform == "Tanager1"
    assert item.bbox == (-103.6, 31.4, -103.4, 31.6)
    assert item.datetime == datetime(2025, 12, 12, 18, 50, 57, tzinfo=timezone.utc)
    assert set(item.asset_urls) == {"cmf", "rgb", "uncertainty", "artifact-mask"}
    assert item.geometry.geom_type == "Polygon"


def test_cmtileitem_datetime_falls_back_to_start_datetime():
    item = _stac_item()
    item["properties"] = {
        "datetime": None, "start_datetime": "2025-12-12T18:50:57Z",
    }
    tile = aq.CMTileItem.from_stac_item(item)
    assert tile.datetime == datetime(2025, 12, 12, 18, 50, 57, tzinfo=timezone.utc)


def test_cmtileitem_without_any_datetime_raises():
    """No fabricated `now()` timestamps — an item without a time is bad data."""
    item = _stac_item()
    item["properties"] = {"datetime": None}
    with pytest.raises(ValueError, match="no datetime"):
        aq.CMTileItem.from_stac_item(item)


def test_cmtileitem_rejects_missing_bbox():
    bad = _stac_item()
    bad["bbox"] = [1, 2]
    with pytest.raises(ValueError):
        aq.CMTileItem.from_stac_item(bad)


# ─────────────────────────────────────────────────────────────────────
#  Single-resource fetchers
# ─────────────────────────────────────────────────────────────────────


def test_get_plume_returns_typed_model(monkeypatch):
    monkeypatch.setattr(
        aq._dl,
        "get_plume_by_id",
        lambda plume_id, token=None: {
            "plume_id": plume_id,
            "plume_latitude": 31.5,
            "plume_longitude": -103.5,
            "gas": "CH4",
        },
    )
    plume = aq.get_plume("tok", "tan-foo-A")
    assert isinstance(plume, CMRawPlume)
    assert plume.plume_id == "tan-foo-A"


def test_get_plume_404_raises_plume_not_found(monkeypatch):
    def boom(plume_id, token=None):
        raise _http_error(404)
    monkeypatch.setattr(aq._dl, "get_plume_by_id", boom)
    with pytest.raises(aq.CMPlumeNotFound):
        aq.get_plume("tok", "tan-missing-A")


def test_get_tile_returns_cmtileitem(monkeypatch):
    monkeypatch.setattr(
        aq._dl, "stac_get_item",
        lambda coll, item_id, token=None: _stac_item(item_id),
    )
    tile = aq.get_tile("tok", "tan-foo")
    assert isinstance(tile, aq.CMTileItem)
    assert tile.scene_id == "tan-foo"


def test_get_tile_404_raises_scene_not_published(monkeypatch):
    def boom(coll, item_id, token=None):
        raise _http_error(404)
    monkeypatch.setattr(aq._dl, "stac_get_item", boom)
    with pytest.raises(aq.CMSceneNotPublished) as exc_info:
        aq.get_tile("tok", "tan-missing")
    assert exc_info.value.scene_id == "tan-missing"


def test_get_source_strips_query_string_suffix(monkeypatch):
    raw_feature = {
        "properties": {
            "source_name": "CH4_1B2_100m_-104.17525_32.49125?plume_gas=CH4",
            "sector": "1B2",
            "gas": "CH4",
            "plume_count": 12,
            "persistence": 0.42,
            "emission_auto": 250.0,
        },
        "geometry": {
            "type": "Point",
            "coordinates": [-104.17525, 32.49125],
        },
    }
    captured = {}

    def fake(name, token=None):
        captured["name"] = name
        return raw_feature

    monkeypatch.setattr(aq._dl, "get_source_by_name", fake)
    src = aq.get_source(
        "tok", "CH4_1B2_100m_-104.17525_32.49125?plume_gas=CH4"
    )
    assert captured["name"] == "CH4_1B2_100m_-104.17525_32.49125"
    assert src.source_name == "CH4_1B2_100m_-104.17525_32.49125"


def test_get_source_404_raises_source_not_found(monkeypatch):
    def boom(name, token=None):
        raise _http_error(404)
    monkeypatch.setattr(aq._dl, "get_source_by_name", boom)
    with pytest.raises(aq.CMSourceNotFound):
        aq.get_source("tok", "missing")


# ─────────────────────────────────────────────────────────────────────
#  Cross-resolution
# ─────────────────────────────────────────────────────────────────────


def _vis_record(plume_id: str, coll: str, **fields) -> dict:
    """Minimal plume record whose `plume_tif` names ``coll``."""
    y, m, d = plume_id[3:7], plume_id[7:9], plume_id[9:11]
    return {
        "plume_id": plume_id,
        "plume_tif": (
            f"https://catalog.carbonmapper.org/{coll}/{y}/{m}/{d}/"
            f"{plume_id}/{plume_id}_{coll}_plume.tif"
        ),
        **fields,
    }


def test_get_tile_for_plume_uses_record_version(monkeypatch):
    """A v3e plume resolves to the v3e STAC item — the old fixed v3a
    default returned None for every plume after 2025-12."""
    pid = "tan20260824t065735c39s4001-B"
    monkeypatch.setattr(
        aq._dl, "get_plume_by_id",
        lambda plume_id, token=None: _vis_record(pid, "l3a-vis-ch4-mfa-v3e"),
    )
    tried: list[str] = []

    def fake(token, scene_id, *, collection):
        tried.append(collection)
        assert scene_id == "tan20260824t065735c39s4001"
        return aq.CMTileItem.from_stac_item(_stac_item(scene_id))

    monkeypatch.setattr(aq, "get_tile", fake)
    assert aq.get_tile_for_plume("tok", pid) is not None
    assert tried == ["l2b-ch4-mfa-v3e"]


def test_get_tile_for_plume_co2_never_tries_ch4(monkeypatch):
    pid = "tan20260823t091609c53s4001-A"
    monkeypatch.setattr(
        aq._dl, "get_plume_by_id",
        lambda plume_id, token=None: _vis_record(
            pid, "l3a-vis-co2-mfa-v3e", gas="CO2", cmf_type="mfal",
        ),
    )
    tried: list[str] = []

    def unpublished(token, scene_id, *, collection):
        tried.append(collection)
        raise aq.CMSceneNotPublished(scene_id)

    monkeypatch.setattr(aq, "get_tile", unpublished)
    assert aq.get_tile_for_plume("tok", pid) is None
    assert tried[:2] == ["l2b-co2-mfa-v3e", "l2b-co2-mfal-v3e"]
    assert all("-co2-" in c for c in tried)


def test_get_tile_for_plume_unknown_plume_uses_defaults(monkeypatch):
    def not_found(plume_id, token=None):
        raise _http_error(404)

    monkeypatch.setattr(aq._dl, "get_plume_by_id", not_found)

    def boom(token, scene_id, *, collection):
        raise aq.CMSceneNotPublished(scene_id)

    monkeypatch.setattr(aq, "get_tile", boom)
    assert aq.get_tile_for_plume("tok", "tan-foo-A") is None


def test_get_tile_for_plume_explicit_collection_skips_record(monkeypatch):
    monkeypatch.setattr(
        aq._dl, "get_plume_by_id",
        lambda *a, **kw: pytest.fail("record must not be fetched"),
    )
    captured = {}

    def fake(token, scene_id, *, collection):
        captured.update(scene_id=scene_id, collection=collection)
        return aq.CMTileItem.from_stac_item(_stac_item(scene_id))

    monkeypatch.setattr(aq, "get_tile", fake)
    aq.get_tile_for_plume(
        "tok", "tan20251212t185057c20s4001-E", collection="l2b-ch4-mfa-v3a",
    )
    assert captured == {
        "scene_id": "tan20251212t185057c20s4001",
        "collection": "l2b-ch4-mfa-v3a",
    }


@pytest.mark.parametrize("status", [401, 429, 503])
def test_get_tile_for_plume_record_errors_propagate(monkeypatch, status):
    def fail(plume_id, token=None):
        raise _http_error(status)

    monkeypatch.setattr(aq._dl, "get_plume_by_id", fail)
    monkeypatch.setattr(
        aq, "get_tile", lambda *a, **kw: pytest.fail("no STAC after a record error"),
    )
    with pytest.raises(requests.HTTPError):
        aq.get_tile_for_plume("tok", "tan20251212t185057c20s4001-E")


def test_get_source_for_plume_returns_none_on_404(monkeypatch):
    def boom(plume_id, token=None):
        raise _http_error(404)
    monkeypatch.setattr(aq._dl, "get_source_for_plume_name", boom)
    assert aq.get_source_for_plume("tok", "tan-foo-A") is None


# ─────────────────────────────────────────────────────────────────────
#  get_image_raster_for_scene / get_image_raster_for_plume
# ─────────────────────────────────────────────────────────────────────


class TestGetImageRasterForScene:
    """STAC-first, URL-pattern fallback for 2026 plumes (Phase 1)."""

    def test_stac_path_wins_when_available(self, monkeypatch):
        """v3a STAC items resolve cleanly — no URL-pattern probe."""
        from georeader.readers.carbonmapper.rasters import CMImageRaster

        # Both CH4 and RGB STAC lookups succeed.
        def fake_get_tile(token, scene_id, *, collection):
            return aq.CMTileItem.from_stac_item(_stac_item(scene_id))

        # Sentinel: from_scene_id should NOT be called when STAC wins.
        def boom(*args, **kwargs):
            raise AssertionError("URL-pattern fallback called on STAC hit")

        monkeypatch.setattr(aq, "get_tile", fake_get_tile)
        monkeypatch.setattr(CMImageRaster, "from_scene_id", boom)

        ir = aq.get_image_raster_for_scene("tok", "tan20251212t185057c20s4001")
        assert isinstance(ir, CMImageRaster)
        assert ir.scene_id == "tan20251212t185057c20s4001"
        assert "cmf" in ir.asset_paths
        assert "rgb" in ir.asset_paths  # RGB sibling attached

    def test_url_pattern_fallback_on_stac_miss(self, monkeypatch):
        """v3c/v3d scenes — STAC 404s, URL-pattern probe succeeds."""
        from georeader.readers.carbonmapper.rasters import CMImageRaster

        def stac_404(token, scene_id, *, collection):
            raise aq.CMSceneNotPublished(scene_id)

        sentinel_ir = CMImageRaster(
            scene_id="tan20260331t181625c77s4001",
            asset_paths={"cmf": "https://x/cmf.tif", "rgb": "https://x/rgb.tif"},
        )

        captured = {}

        def fake_from_scene_id(scene_id, *, token, with_rgb=True, **kw):
            captured["scene_id"] = scene_id
            captured["with_rgb"] = with_rgb
            return sentinel_ir

        monkeypatch.setattr(aq, "get_tile", stac_404)
        monkeypatch.setattr(CMImageRaster, "from_scene_id", fake_from_scene_id)

        ir = aq.get_image_raster_for_scene(
            "tok", "tan20260331t181625c77s4001",
        )
        assert ir is sentinel_ir
        assert captured["scene_id"] == "tan20260331t181625c77s4001"
        assert captured["with_rgb"] is True

    def test_fallback_disabled_returns_none(self, monkeypatch):
        """Set ``prefer_url_pattern_fallback=False`` for the old STAC-only
        behaviour."""
        from georeader.readers.carbonmapper.rasters import CMImageRaster

        def stac_404(token, scene_id, *, collection):
            raise aq.CMSceneNotPublished(scene_id)

        def boom(*args, **kwargs):
            raise AssertionError("fallback called despite disable")

        monkeypatch.setattr(aq, "get_tile", stac_404)
        monkeypatch.setattr(CMImageRaster, "from_scene_id", boom)

        ir = aq.get_image_raster_for_scene(
            "tok", "tan20260331t181625c77s4001",
            prefer_url_pattern_fallback=False,
        )
        assert ir is None

    def test_both_paths_404_returns_none(self, monkeypatch):
        """STAC 404 AND URL-pattern 404 — no exception, just ``None``."""
        from georeader.readers.carbonmapper.rasters import CMImageRaster

        def stac_404(token, scene_id, *, collection):
            raise aq.CMSceneNotPublished(scene_id)

        def fallback_404(scene_id, *, token, **kw):
            raise aq.CMSceneNotPublished(scene_id)

        monkeypatch.setattr(aq, "get_tile", stac_404)
        monkeypatch.setattr(CMImageRaster, "from_scene_id", fallback_404)

        assert aq.get_image_raster_for_scene(
            "tok", "tan20260331t181625c77s4001",
        ) is None

    def test_stac_ch4_hits_rgb_misses_returns_ch4_only(self, monkeypatch):
        """STAC has the CH4 item but not the RGB sibling — keep CH4."""
        from georeader.readers.carbonmapper.rasters import CMImageRaster

        def fake_get_tile(token, scene_id, *, collection):
            if collection == "l2b-rgb-v3a":
                raise aq.CMSceneNotPublished(scene_id)
            return aq.CMTileItem.from_stac_item(_stac_item(scene_id))

        monkeypatch.setattr(aq, "get_tile", fake_get_tile)

        ir = aq.get_image_raster_for_scene(
            "tok", "tan20251212t185057c20s4001",
        )
        assert isinstance(ir, CMImageRaster)
        # cmf retained from the CH4 STAC item; rgb is from the CH4 item's
        # own asset_urls (since this fixture includes it) — the sibling
        # merge didn't happen, but the CH4 item's bundled rgb stays.
        assert "cmf" in ir.asset_paths

    def test_with_rgb_false_skips_sibling_lookup(self, monkeypatch):
        """``with_rgb=False`` → only one STAC call, RGB not requested."""
        calls = []

        def fake_get_tile(token, scene_id, *, collection):
            calls.append(collection)
            return aq.CMTileItem.from_stac_item(_stac_item(scene_id))

        monkeypatch.setattr(aq, "get_tile", fake_get_tile)
        aq.get_image_raster_for_scene(
            "tok", "tan20251212t185057c20s4001", with_rgb=False,
        )
        assert calls == ["l2b-ch4-mfa-v3a"]


class TestGetImageRasterForPlume:
    @staticmethod
    def _record_unavailable(monkeypatch):
        """Force the record-driven spec path to miss (404) so the
        resolver falls back to the STAC/probe path."""
        def boom(plume_id, token=None):
            raise _http_error(404)
        monkeypatch.setattr(aq._dl, "get_plume_by_id", boom)

    def test_record_spec_path_skips_stac(self, monkeypatch):
        """Preferred path: one catalog fetch resolves the spec; the
        L2B parent is probed at the plume's own version first (no
        STAC lookup, defaults as backup — 2026-07 audit)."""
        plume_id = "tan20260623t124240c80s4001-A"
        record = {
            "plume_id": plume_id,
            "plume_tif": (
                "https://catalog.carbonmapper.org/l3a-vis-ch4-mfa-v3d/"
                f"2026/06/23/{plume_id}/{plume_id}_l3a-vis-ch4-mfa-v3d_plume.tif"
            ),
        }
        monkeypatch.setattr(
            aq._dl, "get_plume_by_id", lambda pid, token=None: record,
        )

        def no_stac(*a, **kw):
            raise AssertionError("STAC path must not be used")
        monkeypatch.setattr(aq, "get_image_raster_for_scene", no_stac)

        probes: list[str] = []

        def fake_probe(url, **kw):
            probes.append(url)
            resp = MagicMock()
            resp.status_code = 206
            return resp
        monkeypatch.setattr(requests, "get", fake_probe)

        ir = aq.get_image_raster_for_plume("tok", plume_id)
        # Spec version probed first for CH4 + RGB (one probe each).
        assert len(probes) == 2
        assert "l2b-ch4-mfa-v3d" in str(ir.asset_paths["cmf"])
        assert "l2b-rgb-v3d" in str(ir.asset_paths["rgb"])

    def test_derives_scene_id_from_plume_id(self, monkeypatch):
        self._record_unavailable(monkeypatch)
        captured = {}

        def fake_for_scene(token, scene_id, **kw):
            captured["scene_id"] = scene_id
            return None

        monkeypatch.setattr(aq, "get_image_raster_for_scene", fake_for_scene)
        aq.get_image_raster_for_plume("tok", "tan20260331t181625c77s4001-A")
        assert captured["scene_id"] == "tan20260331t181625c77s4001"

    @pytest.mark.parametrize("status", [401, 429, 500])
    def test_record_errors_propagate_without_fallback(self, monkeypatch, status):
        """A throttled / unauthorised record fetch used to be swallowed
        and fall through to a STAC call plus up to 8 probes, ending in a
        false "not published". It must surface instead."""
        def fail(plume_id, token=None):
            raise _http_error(status)

        monkeypatch.setattr(aq._dl, "get_plume_by_id", fail)
        monkeypatch.setattr(
            aq, "get_image_raster_for_scene",
            lambda *a, **kw: pytest.fail("fallback must not run"),
        )
        with pytest.raises(requests.HTTPError):
            aq.get_image_raster_for_plume("tok", "tan20260331t181625c77s4001-A")

    def test_spec_path_all_probes_miss_returns_none(self, monkeypatch):
        """The contract is `CMImageRaster | None` — the spec path must
        not leak CMSceneNotPublished."""
        pid = "tan20260623t124240c80s4001-A"
        monkeypatch.setattr(
            aq._dl, "get_plume_by_id",
            lambda plume_id, token=None: _vis_record(pid, "l3a-vis-ch4-mfa-v3d"),
        )
        resp = MagicMock()
        resp.status_code = 404
        monkeypatch.setattr(requests, "get", lambda url, **kw: resp)
        assert aq.get_image_raster_for_plume("tok", pid) is None

    def test_explicit_collection_skips_record_and_is_honoured(self, monkeypatch):
        monkeypatch.setattr(
            aq._dl, "get_plume_by_id",
            lambda *a, **kw: pytest.fail("record must not be fetched"),
        )
        captured = {}

        def fake_for_scene(token, scene_id, **kw):
            captured.update(kw)
            return None

        monkeypatch.setattr(aq, "get_image_raster_for_scene", fake_for_scene)
        aq.get_image_raster_for_plume(
            "tok", "tan20260331t181625c77s4001-A", collection="l2b-ch4-mfa-v3c",
        )
        assert captured["collection"] == "l2b-ch4-mfa-v3c"

    def test_forwards_kwargs(self, monkeypatch):
        self._record_unavailable(monkeypatch)
        captured = {}

        def fake_for_scene(token, scene_id, **kw):
            captured.update(kw)
            return None

        monkeypatch.setattr(aq, "get_image_raster_for_scene", fake_for_scene)
        aq.get_image_raster_for_plume(
            "tok", "tan20260331t181625c77s4001-A",
            collection="l2b-ch4-mfa-v3",
            prefer_url_pattern_fallback=False,
            with_rgb=False,
        )
        assert captured == {
            "collection": "l2b-ch4-mfa-v3",
            "prefer_url_pattern_fallback": False,
            "with_rgb": False,
        }


def test_get_plume_context_combines_three_calls(monkeypatch):
    fake_plume = CMRawPlume(
        plume_id="tan-foo-A", plume_latitude=31.5, plume_longitude=-103.5,
    )
    fake_tile = aq.CMTileItem.from_stac_item(_stac_item("tan-foo"))
    fake_source = CMSource.from_geojson_feature({
        "properties": {
            "source_name": "CH4_1B2_100m_-104_32",
            "sector": "1B2", "gas": "CH4",
            "plume_count": 1, "persistence": 0.1,
        },
        "geometry": {"type": "Point", "coordinates": [-104.0, 32.0]},
    })

    monkeypatch.setattr(aq, "get_plume", lambda *a, **kw: fake_plume)
    monkeypatch.setattr(aq, "get_tile_for_plume", lambda *a, **kw: fake_tile)
    monkeypatch.setattr(aq, "get_source_for_plume", lambda *a, **kw: fake_source)

    p, t, s = aq.get_plume_context("tok", "tan-foo-A")
    assert p is fake_plume
    assert t is fake_tile
    assert s is fake_source


def test_get_plume_context_tolerates_missing_optionals(monkeypatch):
    fake_plume = CMRawPlume(
        plume_id="tan-foo-A", plume_latitude=31.5, plume_longitude=-103.5,
    )
    monkeypatch.setattr(aq, "get_plume", lambda *a, **kw: fake_plume)
    monkeypatch.setattr(aq, "get_tile_for_plume", lambda *a, **kw: None)
    monkeypatch.setattr(aq, "get_source_for_plume", lambda *a, **kw: None)

    p, t, s = aq.get_plume_context("tok", "tan-foo-A")
    assert p is fake_plume
    assert t is None
    assert s is None


def test_list_tiles_for_source_dedups_scenes(monkeypatch):
    plumes = [
        CMRawPlume(plume_id="tan-A-1"),
        CMRawPlume(plume_id="tan-A-2"),
        CMRawPlume(plume_id="tan-B-1"),
        CMRawPlume(plume_id="tan-B-2"),
        CMRawPlume(plume_id="tan-A-3"),
    ]
    monkeypatch.setattr(
        aq, "list_plumes_for_source", lambda *a, **kw: plumes,
    )
    seen_ids: list[list[str]] = []

    def fake_search(*, collections, ids, limit, token):
        seen_ids.append(list(ids))
        return {"features": [_stac_item(s) for s in ids]}

    monkeypatch.setattr(aq._dl, "stac_search", fake_search)
    tiles = aq.list_tiles_for_source("tok", "CH4_1B2_...")
    assert len(tiles) == 2
    assert sorted(seen_ids[0]) == ["tan-A", "tan-B"]


def test_list_tiles_for_source_uses_plume_version_and_chunks(monkeypatch):
    """Scenes are looked up in their plume's own-version collection (the
    old fixed v3a default found nothing for 2026 plumes), and ids are
    sent in bounded chunks."""
    plumes = [
        CMRawPlume(**_vis_record(
            f"tan20260824t0657{i:02d}c39s4001-A", "l3a-vis-ch4-mfa-v3e",
        ))
        for i in range(aq._STAC_IDS_PER_REQUEST + 5)
    ]
    monkeypatch.setattr(aq, "list_plumes_for_source", lambda *a, **kw: plumes)
    calls: list[tuple[list[str], list[str]]] = []

    def fake_search(*, collections, ids, limit, token):
        calls.append((list(collections), list(ids)))
        return {"features": [_stac_item(s) for s in ids]}

    monkeypatch.setattr(aq._dl, "stac_search", fake_search)
    tiles = aq.list_tiles_for_source("tok", "CH4_1B2_...")
    assert len(tiles) == len(plumes)
    assert len(calls) == 2
    assert all(colls == ["l2b-ch4-mfa-v3e"] for colls, _ in calls)
    assert max(len(ids) for _, ids in calls) <= aq._STAC_IDS_PER_REQUEST


def test_list_tiles_defaults_to_all_known_versions_and_dedupes(monkeypatch):
    captured = {}

    def fake_search(*, collections, bbox, datetime_range, limit, token):
        captured["collections"] = collections
        old = _stac_item("tan-A")
        new = _stac_item("tan-A")
        new["collection"] = "l2b-ch4-mfa-v3e"
        return {"features": [old, new, _stac_item("tan-B")]}

    monkeypatch.setattr(aq._dl, "stac_search", fake_search)
    tiles = aq.list_tiles("tok")
    assert "l2b-ch4-mfa-v3e" in captured["collections"]
    assert "l2b-ch4-mfa-v3a" in captured["collections"]
    assert [(t.scene_id, t.collection) for t in tiles] == [
        ("tan-A", "l2b-ch4-mfa-v3e"), ("tan-B", "l2b-ch4-mfa-v3a"),
    ]


def test_get_image_raster_for_scene_rgb_follows_collection_version(monkeypatch):
    seen: list[str] = []

    def fake_get_tile(token, scene_id, *, collection):
        seen.append(collection)
        item = _stac_item(scene_id)
        item["collection"] = collection
        return aq.CMTileItem.from_stac_item(item)

    monkeypatch.setattr(aq, "get_tile", fake_get_tile)
    aq.get_image_raster_for_scene(
        "tok", "tan20260824t065735c39s4001", collection="l2b-ch4-mfa-v3e",
    )
    assert seen == ["l2b-ch4-mfa-v3e", "l2b-rgb-v3e"]


def test_list_tiles_for_source_empty_when_no_plumes(monkeypatch):
    monkeypatch.setattr(aq, "list_plumes_for_source", lambda *a, **kw: [])
    monkeypatch.setattr(
        aq._dl, "stac_search",
        lambda **kw: pytest.fail("stac_search should not be called"),
    )
    assert aq.list_tiles_for_source("tok", "any") == []


# ─── 2026-07 source-detail API drift ─────────────────────────────────


class TestSourceDetailDrift:
    """`/catalog/source/{name}` now returns a nested detail shape:
    stats under `source`, full annotated plume records under `plumes`,
    centroid under `point`. The old flat shape (and the
    source-plumes-csv route, which now 400s upstream) must keep
    working as fallbacks."""

    NESTED = {
        "source_name": "CH4_1B2_250m_-104.11776_32.02621",
        "point": {"type": "Point", "coordinates": [-104.11776, 32.02621]},
        "source": {
            "gas": "CH4", "sector": "1B2", "persistence": 0.75,
            "emission_auto": 244.56, "emission_uncertainty_auto": 96.55,
        },
        "plumes": [
            {"plume_id": "ang20191006t180341-1", "gas": "CH4",
             "emission_auto": 338.9},
            {"plume_id": "tan20260311t190317c10s4001-D", "gas": "CH4",
             "emission_auto": 970.0},
        ],
        "scenes": [{"scene_name": "ang20191006t180341"}],
        "detection_dates": ["2019-10-06", "2026-03-11"],
        "observation_dates": ["2019-10-06", "2020-01-01", "2026-03-11"],
    }

    def test_get_source_flattens_nested_shape(self, monkeypatch):
        monkeypatch.setattr(
            aq._dl, "get_source_by_name", lambda name, token=None: dict(self.NESTED),
        )
        src = aq.get_source("tok", self.NESTED["source_name"])
        assert src.source_name == "CH4_1B2_250m_-104.11776_32.02621"
        assert src.sector == "1B2"
        assert src.plume_count == 2
        assert src.persistence == 0.75
        assert src.emission_auto == 244.56
        assert (round(src.point.x, 5), round(src.point.y, 5)) == (-104.11776, 32.02621)

    def test_list_plumes_for_source_uses_embedded_records(self, monkeypatch):
        monkeypatch.setattr(
            aq._dl, "get_source_by_name", lambda name, token=None: dict(self.NESTED),
        )

        def no_csv(*a, **kw):
            raise AssertionError("CSV route must not be used when plumes are embedded")
        monkeypatch.setattr(aq._dl, "get_source_plumes_csv", no_csv)

        plumes = aq.list_plumes_for_source("tok", self.NESTED["source_name"])
        assert [p.plume_id for p in plumes] == [
            "ang20191006t180341-1", "tan20260311t190317c10s4001-D",
        ]

    def test_list_plumes_for_source_csv_fallback(self, monkeypatch):
        """Pre-drift shape (no embedded `plumes`) falls back to CSV."""
        monkeypatch.setattr(
            aq._dl, "get_source_by_name",
            lambda name, token=None: {"source_name": name, "plume_count": 1},
        )
        monkeypatch.setattr(
            aq._dl, "get_source_plumes_csv",
            lambda name, token=None: "plume_id,gas\ntan20260101t000000c00s4001-A,CH4\n",
        )
        plumes = aq.list_plumes_for_source("tok", "CH4_1B2_100m_0_0")
        assert len(plumes) == 1
        assert plumes[0].plume_id == "tan20260101t000000c00s4001-A"

    def test_list_plumes_for_source_404(self, monkeypatch):
        def boom(name, token=None):
            resp = MagicMock()
            resp.status_code = 404
            raise requests.HTTPError("404", response=resp)
        monkeypatch.setattr(aq._dl, "get_source_by_name", boom)
        with pytest.raises(aq.CMSourceNotFound):
            aq.list_plumes_for_source("tok", "CH4_1B2_500m_-103.93835_32.06406")

class TestListPlumesDateAxes:
    """spaceml-org/georeader#64 — publication-date filtering."""

    def _capture(self, monkeypatch):
        captured: dict = {}

        def fake(**kwargs):
            captured.update(kwargs)
            return {"items": []}

        monkeypatch.setattr(aq._dl, "get_plumes_annotated", fake)
        return captured

    def test_published_at_bounds_become_interval(self, monkeypatch):
        cap = self._capture(monkeypatch)
        aq.list_plumes(
            "tok",
            published_at_min=datetime(2026, 3, 1, tzinfo=timezone.utc),
            published_at_max=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        assert cap["published_at_range"] == (
            "2026-03-01T00:00:00Z/2026-04-01T00:00:00Z"
        )
        assert cap["datetime_range"] is None

    def test_published_at_min_only_is_open_ended(self, monkeypatch):
        cap = self._capture(monkeypatch)
        aq.list_plumes(
            "tok", published_at_min=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
        assert cap["published_at_range"] == "2026-03-01T00:00:00Z/.."

    def test_axes_are_independent(self, monkeypatch):
        cap = self._capture(monkeypatch)
        aq.list_plumes(
            "tok",
            datetime_min=datetime(2025, 11, 1, tzinfo=timezone.utc),
            published_at_max=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
        assert cap["datetime_range"] == "2025-11-01T00:00:00Z/.."
        assert cap["published_at_range"] == "../2026-04-01T00:00:00Z"

    def test_unset_by_default(self, monkeypatch):
        cap = self._capture(monkeypatch)
        aq.list_plumes("tok")
        assert cap["published_at_range"] is None

    def test_naive_datetimes_are_utc_with_z(self, monkeypatch):
        cap = self._capture(monkeypatch)
        aq.list_plumes("tok", datetime_min=datetime(2026, 3, 1))
        assert cap["datetime_range"] == "2026-03-01T00:00:00Z/.."


class TestListPlumesPagination:
    """`list_plumes` used to make one request and silently truncate."""

    @staticmethod
    def _rows(start: int, n: int) -> list[dict]:
        return [{"plume_id": f"tan-p{i}-A"} for i in range(start, start + n)]

    def test_pages_until_limit(self, monkeypatch):
        calls: list[tuple[int, int]] = []

        def fake(**kw):
            calls.append((kw["offset"], kw["limit"]))
            return {"items": self._rows(kw["offset"], kw["limit"]),
                    "total_count": 5_000}

        monkeypatch.setattr(aq._dl, "get_plumes_annotated", fake)
        plumes = aq.list_plumes("tok", limit=2_500)
        assert len(plumes) == 2_500
        assert calls == [(0, 1_000), (1_000, 1_000), (2_000, 500)]

    def test_stops_on_short_page_without_total_count(self, monkeypatch):
        calls: list[int] = []

        def fake(**kw):
            calls.append(kw["offset"])
            n = kw["limit"] if kw["offset"] == 0 else 7
            return {"items": self._rows(kw["offset"], n)}

        monkeypatch.setattr(aq._dl, "get_plumes_annotated", fake)
        plumes = aq.list_plumes("tok", limit=5_000)
        assert len(plumes) == 1_007
        assert calls == [0, 1_000]

    def test_stops_at_total_count(self, monkeypatch):
        calls: list[int] = []

        def fake(**kw):
            calls.append(kw["offset"])
            return {"items": self._rows(kw["offset"], kw["limit"]),
                    "total_count": 1_000}

        monkeypatch.setattr(aq._dl, "get_plumes_annotated", fake)
        assert len(aq.list_plumes("tok", limit=5_000)) == 1_000
        assert calls == [0]

    def test_gas_none_queries_both_gases(self, monkeypatch):
        cap = {}

        def fake(**kw):
            cap.update(kw)
            return {"items": []}

        monkeypatch.setattr(aq._dl, "get_plumes_annotated", fake)
        aq.list_plumes("tok", gas=None)
        assert cap["plume_gas"] is None
        aq.list_plumes("tok", gas="CO2")
        assert cap["plume_gas"] == "CO2"


def test_list_plumes_for_tile_filters_by_scene_day_and_prefix(monkeypatch):
    """Used to pull the newest 1 000 CH4 plumes catalog-wide and filter
    by prefix — `[]` for any older scene, and never CO2."""
    scene = "tan20250601t101010c01s4001"
    captured = {}

    def fake(**kw):
        captured.update(kw)
        return {"items": [
            {"plume_id": f"{scene}-A", "gas": "CH4"},
            {"plume_id": f"{scene}-B", "gas": "CO2"},
            {"plume_id": "tan20250601t999999c01s4001-A", "gas": "CH4"},
        ]}

    monkeypatch.setattr(aq._dl, "get_plumes_annotated", fake)
    plumes = aq.list_plumes_for_tile("tok", scene)
    assert [p.plume_id for p in plumes] == [f"{scene}-A", f"{scene}-B"]
    assert captured["plume_gas"] is None
    assert captured["datetime_range"] == (
        "2025-05-31T00:00:00Z/2025-06-03T00:00:00Z"
    )


def test_list_sources_retries_429(monkeypatch):
    from georeader.readers.carbonmapper import download as _dl

    monkeypatch.setattr(_dl, "_sleep", lambda s: None)
    throttled = MagicMock(status_code=429, headers={"Retry-After": "1"})
    ok = MagicMock(status_code=200, headers={})
    ok.json.return_value = {"type": "FeatureCollection", "features": []}
    seq = iter([throttled, ok])
    monkeypatch.setattr(requests, "get", lambda *a, **kw: next(seq))
    assert aq.list_sources("tok", gas=None) == []
