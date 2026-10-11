"""Tests for bbox encoding helpers and call sites in download.py."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from georeader.readers.carbonmapper import download as dl


def test_rest_bbox_params_returns_repeated_keys():
    params = dl._rest_bbox_params((-104.5, 31.0, -101.5, 33.5))
    # requests serialises a list value as repeated keys.
    assert params == {"bbox": ["-104.5", "31.0", "-101.5", "33.5"]}


def test_rest_bbox_params_none():
    assert dl._rest_bbox_params(None) == {}


def test_rest_bbox_params_wrong_length_raises():
    with pytest.raises(ValueError):
        dl._rest_bbox_params((1, 2, 3))


def test_stac_bbox_param_comma_joined():
    params = dl._stac_bbox_param((-104.5, 31.0, -101.5, 33.5))
    assert params == {"bbox": "-104.5,31.0,-101.5,33.5"}


def test_stac_bbox_param_none():
    assert dl._stac_bbox_param(None) == {}


def test_stac_bbox_param_wrong_length_raises():
    with pytest.raises(ValueError):
        dl._stac_bbox_param((1, 2, 3, 4, 5))


def _capture_get_url(monkeypatch):
    """Patch dl._get to capture the prepared URL and return an empty payload."""
    captured: dict = {}

    def fake_get(url, params=None, token=None):
        import requests

        req = requests.Request("GET", url, params=params).prepare()
        captured["url"] = req.url
        captured["params"] = params
        return {"items": [], "features": []}

    monkeypatch.setattr(dl, "_get", fake_get)
    return captured


def test_get_plumes_annotated_uses_repeated_bbox_keys(monkeypatch):
    cap = _capture_get_url(monkeypatch)
    dl.get_plumes_annotated(bbox=(-104.5, 31.0, -101.5, 33.5))
    qs = parse_qs(urlsplit(cap["url"]).query)
    assert qs["bbox"] == ["-104.5", "31.0", "-101.5", "33.5"]


def test_get_plumes_csv_uses_repeated_bbox_keys(monkeypatch):
    cap = _capture_get_url(monkeypatch)
    dl.get_plumes_csv(bbox=(-104.5, 31.0, -101.5, 33.5))
    qs = parse_qs(urlsplit(cap["url"]).query)
    assert qs["bbox"] == ["-104.5", "31.0", "-101.5", "33.5"]


def test_get_sources_uses_repeated_bbox_keys(monkeypatch):
    cap = _capture_get_url(monkeypatch)
    dl.get_sources(bbox=(-104.5, 31.0, -101.5, 33.5))
    qs = parse_qs(urlsplit(cap["url"]).query)
    assert qs["bbox"] == ["-104.5", "31.0", "-101.5", "33.5"]


def test_stac_search_uses_comma_joined_bbox(monkeypatch):
    cap = _capture_get_url(monkeypatch)
    dl.stac_search(bbox=(-104.5, 31.0, -101.5, 33.5))
    qs = parse_qs(urlsplit(cap["url"]).query)
    assert qs["bbox"] == ["-104.5,31.0,-101.5,33.5"]


def test_stac_get_items_uses_comma_joined_bbox(monkeypatch):
    cap = _capture_get_url(monkeypatch)
    dl.stac_get_items("l2b-ch4-mfa-v3a", bbox=(-104.5, 31.0, -101.5, 33.5))
    qs = parse_qs(urlsplit(cap["url"]).query)
    assert qs["bbox"] == ["-104.5,31.0,-101.5,33.5"]


# ─── Date-axis params (spaceml-org/georeader#64) ────────────────────
# `datetime` filters observation time (scene_timestamp); publication /
# ingest polling needs the separate documented params.


def test_get_plumes_annotated_date_axis_params(monkeypatch):
    cap = _capture_get_url(monkeypatch)
    dl.get_plumes_annotated(
        datetime_range="2026-03-01T00:00:00Z/2026-03-31T23:59:59Z",
        published_at_range="2026-04-01T00:00:00Z/..",
        created_at_range="../2026-05-01T00:00:00Z",
        modified_at_range="2026-05-01T00:00:00Z/2026-06-01T00:00:00Z",
    )
    qs = parse_qs(urlsplit(cap["url"]).query)
    assert qs["datetime"] == ["2026-03-01T00:00:00Z/2026-03-31T23:59:59Z"]
    assert qs["published_at_datetime"] == ["2026-04-01T00:00:00Z/.."]
    assert qs["created_at"] == ["../2026-05-01T00:00:00Z"]
    assert qs["modified_at"] == ["2026-05-01T00:00:00Z/2026-06-01T00:00:00Z"]


def test_get_plumes_annotated_date_axes_omitted_when_unset(monkeypatch):
    cap = _capture_get_url(monkeypatch)
    dl.get_plumes_annotated(plume_gas="CH4")
    qs = parse_qs(urlsplit(cap["url"]).query)
    for key in ("datetime", "published_at_datetime", "created_at", "modified_at"):
        assert key not in qs


def test_get_plumes_csv_date_axis_params(monkeypatch):
    cap = _capture_get_url(monkeypatch)
    dl.get_plumes_csv(
        published_at_range="2026-04-01T00:00:00Z/..",
        created_at_range="../2026-05-01T00:00:00Z",
        modified_at_range="2026-05-01T00:00:00Z/2026-06-01T00:00:00Z",
    )
    qs = parse_qs(urlsplit(cap["url"]).query)
    assert qs["published_at_datetime"] == ["2026-04-01T00:00:00Z/.."]
    assert qs["created_at"] == ["../2026-05-01T00:00:00Z"]
    assert qs["modified_at"] == ["2026-05-01T00:00:00Z/2026-06-01T00:00:00Z"]


# ─── bbox validation ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bbox, match",
    [
        ((179.5, 10.0, -179.5, 11.0), "antimeridian"),
        ((-104.0, 33.0, -103.0, 32.0), "south > north"),
        ((-104.0, -95.0, -103.0, 32.0), r"\[-90, 90\]"),
    ],
)
def test_bbox_validation_rejects_malformed_boxes(bbox, match):
    for encode in (dl._rest_bbox_params, dl._stac_bbox_param):
        with pytest.raises(ValueError, match=match):
            encode(bbox)


# ─── Rate-limit handling ──────────────────────────────────────────────


class _Resp:
    def __init__(self, status: int, headers: dict | None = None, payload=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload
        self.closed = False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(str(self.status_code), response=self)

    def json(self):
        return self._payload


@pytest.mark.parametrize(
    "header, expected",
    [
        ("7", 7.0),
        ("2.5", 2.5),               # float — `int()` used to crash on this
        ("999999", dl.MAX_RATE_LIMIT_WAIT_S),
        ("garbage", 5.0),           # unparseable → exponential default
        (None, 5.0),
    ],
)
def test_retry_after_seconds(header, expected):
    resp = _Resp(429, {"Retry-After": header} if header else {})
    assert dl._retry_after_seconds(resp, attempt=0) == expected


def test_retry_after_http_date():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    when = datetime.now(timezone.utc) + timedelta(seconds=30)
    resp = _Resp(429, {"Retry-After": format_datetime(when, usegmt=True)})
    assert 25 <= dl._retry_after_seconds(resp, attempt=0) <= 30


def test_request_retries_429_then_succeeds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(dl, "_sleep", sleeps.append)
    seq = iter([_Resp(429, {"Retry-After": "1"}), _Resp(429), _Resp(200)])
    monkeypatch.setattr(dl.requests, "get", lambda *a, **kw: next(seq))
    assert dl._request("GET", "https://x").status_code == 200
    assert sleeps == [1.0, 10.0]


def test_request_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(dl, "_sleep", lambda s: None)
    calls: list[int] = []

    def always_429(*a, **kw):
        calls.append(1)
        return _Resp(429)

    monkeypatch.setattr(dl.requests, "get", always_429)
    assert dl._request("GET", "https://x").status_code == 429
    assert len(calls) == dl.MAX_RATE_LIMIT_RETRIES + 1


def test_post_retries_429(monkeypatch):
    """Token calls had no rate-limit handling at all."""
    monkeypatch.setattr(dl, "_sleep", lambda s: None)
    seq = iter([_Resp(429), _Resp(200, payload={"access": "a"})])
    monkeypatch.setattr(dl.requests, "post", lambda *a, **kw: next(seq))
    assert dl.obtain_token("e", "p") == {"access": "a"}


# ─── paginate_plumes ──────────────────────────────────────────────────


def _page_stub(monkeypatch, pages: list[dict]):
    calls: list[dict] = []
    it = iter(pages)

    def fake(**kw):
        calls.append(kw)
        return next(it)

    monkeypatch.setattr(dl, "get_plumes_annotated", fake)
    return calls


def test_paginate_continues_without_total_count(monkeypatch):
    """Used to stop after page 1 whenever `total_count` was absent."""
    rows = [{"plume_id": str(i)} for i in range(120)]
    calls = _page_stub(monkeypatch, [
        {"items": rows[:50]}, {"items": rows[50:100]}, {"items": rows[100:]},
    ])
    out = dl.paginate_plumes(max_plumes=500, page_size=50)
    assert len(out) == 120
    assert [c["offset"] for c in calls] == [0, 50, 100]


def test_paginate_forwards_all_filters(monkeypatch):
    calls = _page_stub(monkeypatch, [{"items": [], "total_count": 0}])
    dl.paginate_plumes(
        published_at_range="2026-03-01T00:00:00Z/..",
        sectors=["1B2"], instruments=["tan"],
    )
    assert calls[0]["published_at_range"] == "2026-03-01T00:00:00Z/.."
    assert calls[0]["sectors"] == ["1B2"]
    assert calls[0]["instruments"] == ["tan"]
