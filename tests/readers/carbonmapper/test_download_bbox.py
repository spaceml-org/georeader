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
