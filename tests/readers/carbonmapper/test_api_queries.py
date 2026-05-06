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


def test_get_tile_for_plume_derives_scene_id(monkeypatch):
    captured = {}

    def fake(token, scene_id, *, collection):
        captured["scene_id"] = scene_id
        return aq.CMTileItem.from_stac_item(_stac_item(scene_id))

    monkeypatch.setattr(aq, "get_tile", fake)
    aq.get_tile_for_plume("tok", "tan20251212t185057c20s4001-E")
    assert captured["scene_id"] == "tan20251212t185057c20s4001"


def test_get_tile_for_plume_returns_none_when_unpublished(monkeypatch):
    def boom(token, scene_id, *, collection):
        raise aq.CMSceneNotPublished(scene_id)

    monkeypatch.setattr(aq, "get_tile", boom)
    assert aq.get_tile_for_plume("tok", "tan-foo-A") is None


def test_get_source_for_plume_returns_none_on_404(monkeypatch):
    def boom(plume_id, token=None):
        raise _http_error(404)
    monkeypatch.setattr(aq._dl, "get_source_for_plume_name", boom)
    assert aq.get_source_for_plume("tok", "tan-foo-A") is None


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


def test_list_tiles_for_source_empty_when_no_plumes(monkeypatch):
    monkeypatch.setattr(aq, "list_plumes_for_source", lambda *a, **kw: [])
    monkeypatch.setattr(
        aq._dl, "stac_search",
        lambda **kw: pytest.fail("stac_search should not be called"),
    )
    assert aq.list_tiles_for_source("tok", "any") == []
