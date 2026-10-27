"""Tests for STAC item paging in download.py."""

from __future__ import annotations

from georeader.readers.carbonmapper import download as dl

NEXT = [{"rel": "next", "href": "https://example.invalid/next"}]
FAKE_BEARER = "not-a-real-token"


def _fake_collection(monkeypatch, n_items: int, *, next_on_last: bool = False):
    """Patch dl._get with a collection of ``n_items`` that honours limit/offset."""
    calls: list[dict] = []

    def fake_get(url, params=None, token=None):
        calls.append({"url": url, "params": dict(params or {}), "token": token})
        offset = (params or {}).get("offset", 0)
        limit = (params or {}).get("limit", 10)
        ids = list(range(offset, min(offset + limit, n_items)))
        has_more = offset + limit < n_items or (next_on_last and ids)
        return {
            "type": "FeatureCollection",
            "features": [{"id": f"item-{i}"} for i in ids],
            "links": NEXT if has_more else [],
        }

    monkeypatch.setattr(dl, "_get", fake_get)
    return calls


def test_stac_get_items_omits_offset_on_the_first_page(monkeypatch):
    calls = _fake_collection(monkeypatch, 3)
    dl.stac_get_items("coll", limit=5)
    assert "offset" not in calls[0]["params"], f"Expected no offset found {calls[0]['params']}"


def test_stac_get_items_sends_offset(monkeypatch):
    calls = _fake_collection(monkeypatch, 30)
    dl.stac_get_items("coll", limit=10, offset=20)
    assert calls[0]["params"]["offset"] == 20, f"Expected offset 20 found {calls[0]['params']}"


def test_paginate_stac_items_follows_next_links(monkeypatch):
    calls = _fake_collection(monkeypatch, 250)
    features = dl.paginate_stac_items("coll", page_size=100, token=FAKE_BEARER)

    assert [f["id"] for f in features] == [f"item-{i}" for i in range(250)], "Expected every item once"
    assert [c["params"].get("offset", 0) for c in calls] == [0, 100, 200]
    assert all(c["params"]["limit"] == 100 for c in calls)
    assert all(c["token"] == FAKE_BEARER for c in calls), "Expected the token on every page"
    assert calls[0]["url"].endswith("/collections/coll/items")


def test_paginate_stac_items_stops_on_an_empty_page(monkeypatch):
    # A server that keeps a next link on its last page must not loop forever.
    calls = _fake_collection(monkeypatch, 100, next_on_last=True)
    features = dl.paginate_stac_items("coll", page_size=100)
    assert len(features) == 100, f"Expected 100 found {len(features)}"
    assert len(calls) == 2, f"Expected 2 calls (last one empty) found {len(calls)}"


def test_paginate_stac_items_caps_at_max_items(monkeypatch):
    calls = _fake_collection(monkeypatch, 250)
    features = dl.paginate_stac_items("coll", page_size=100, max_items=150)
    assert len(features) == 150, f"Expected 150 found {len(features)}"
    assert [c["params"]["limit"] for c in calls] == [100, 50]


def test_paginate_stac_items_pauses_before_each_call(monkeypatch):
    _fake_collection(monkeypatch, 150)
    sleeps: list[float] = []
    monkeypatch.setattr(dl.time, "sleep", sleeps.append)
    dl.paginate_stac_items("coll", page_size=100, pause_s=2.0)
    assert sleeps == [2.0, 2.0], f"Expected one pause per call found {sleeps}"
