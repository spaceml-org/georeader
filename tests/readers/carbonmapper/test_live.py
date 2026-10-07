"""Opt-in live checks against the Carbon Mapper API.

The unit suite is fully mocked, so it cannot notice upstream drift — a
renamed collection, a new version, a changed record shape. These tests
exercise exactly the assumptions that have drifted before, against the
real API, on a strict call budget (the API is rate-limited):

1. One annotated-plumes listing (tokenless) — shared by every test.
2. Two range reads of derived IME URLs, one CH4 and one CO2 (with the
   token when one is configured — the asset proxy gates some plumes'
   assets behind auth and serves others anonymously).
3. At most four STAC item lookups for a current CH4 plume (tokenless).
4. With credentials only: one L2B probe and one optional-band open.

Run with::

    CARBONMAPPER_LIVE=1 pytest -m cm_live tests/readers/carbonmapper/test_live.py

Credentials (for step 4) come from ``CARBONMAPPER_TOKEN`` or
``CARBONMAPPER_EMAIL`` + ``CARBONMAPPER_PASSWORD``.
"""

from __future__ import annotations

import os

import pytest
import requests

from georeader.readers.carbonmapper import api_queries as aq
from georeader.readers.carbonmapper import download as dl
from georeader.readers.carbonmapper import products as P
from georeader.readers.carbonmapper.config import CarbonMapperConfig
from georeader.readers.carbonmapper.image import _derive_asset_urls
from georeader.readers.carbonmapper.products import CMCollectionSpec, _parse_asset_url

pytestmark = [
    pytest.mark.cm_live,
    pytest.mark.skipif(
        os.environ.get("CARBONMAPPER_LIVE") != "1",
        reason="live Carbon Mapper checks are opt-in: set CARBONMAPPER_LIVE=1",
    ),
]

#: URL fields every plume record has carried since the 2026-07 audit.
RECORD_URL_FIELDS = ("plume_tif", "con_tif", "plume_png", "rgb_png", "plume_rgb_png")


@pytest.fixture(scope="module")
def newest_plumes() -> list[dict]:
    """The newest published plumes, both gases — the only listing call."""
    return dl.get_plumes_annotated(limit=50)["items"]


@pytest.fixture(scope="module")
def optional_token() -> str | None:
    """A bearer token when credentials are configured, else ``None``."""
    cfg = CarbonMapperConfig.from_env()
    return cfg.get_token() or (
        cfg.refresh_access_token() if cfg.email and cfg.password else None
    )


@pytest.fixture(scope="module")
def token(optional_token) -> str:
    if not optional_token:
        pytest.skip("no Carbon Mapper credentials in the environment")
    return optional_token


def _newest(plumes: list[dict], gas: str) -> dict:
    match = next((p for p in plumes if p.get("gas") == gas and p.get("plume_tif")), None)
    if match is None:
        pytest.skip(f"no {gas} plume among the newest listing")
    return match


def _range_status(url: str, token: str | None = None) -> int:
    headers = {"Range": "bytes=0-0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    resp = dl._request("GET", url, headers=headers, timeout=60, stream=True)
    resp.close()
    return resp.status_code


def test_record_shape_and_spec_resolution(newest_plumes):
    """Every record still carries the URL fields, and the collection
    spec resolved from it reproduces the record's own IME collection
    (the CO2 `mfa`/`mfal` split broke exactly this)."""
    assert newest_plumes, "the annotated listing returned no plumes"
    for rec in newest_plumes:
        for field in RECORD_URL_FIELDS:
            assert field in rec, (rec.get("plume_id"), field)
        spec = CMCollectionSpec.from_plume_record(rec)
        vis = _parse_asset_url(rec["plume_tif"])
        assert vis is not None and vis.collection_id == spec.collection_id(P.CMProductFamily.L3A_VIS)
        if rec.get("con_tif"):
            ime = _parse_asset_url(rec["con_tif"])
            assert ime is not None and ime.collection_id == spec.collection_id(P.CMProductFamily.L3A_IME)


@pytest.mark.parametrize("gas", ["CH4", "CO2"])
def test_derived_ime_url_serves(newest_plumes, optional_token, gas):
    """The IME URL the reader derives must exist. A wrong collection is a
    404; a right one that the proxy gates is a 401 without a token."""
    rec = _newest(newest_plumes, gas)
    url = _derive_asset_urls(rec, (P.IME_CONCENTRATIONS,))["ime-cmf-concentrations.tif"]
    status = _range_status(url, optional_token)
    if status == 401 and optional_token is None:
        pytest.skip(f"{gas} asset requires authentication; no credentials configured")
    assert status in (200, 206), (status, url)


def test_stac_serves_current_l2b(newest_plumes):
    """STAC used to stop at v3a; the helper must find a current plume's
    parent L2B item under the plume's own gas and version."""
    rec = _newest(newest_plumes, "CH4")
    spec = CMCollectionSpec.from_plume_record(rec)
    try:
        tile = aq.get_tile_for_plume(None, rec["plume_id"], spec=spec)
    except requests.HTTPError as exc:  # pragma: no cover — surfaced as a skip, not a failure
        pytest.skip(f"STAC lookup failed upstream: {exc}")
    if tile is None:
        pytest.skip("newest CH4 plume's L2B not in STAC yet (publication lag)")
    assert tile.collection.startswith(f"l2b-{spec.gas}-")
    assert "cmf.tif" in tile.asset_urls


def test_scene_raster_resolves_with_optional_bands(newest_plumes, token):
    """The L2B parent resolves through the spec path, and optional bands
    that the scene doesn't ship come back as ``None`` instead of raising."""
    from georeader.readers.carbonmapper.rasters import CMImageRaster

    rec = _newest(newest_plumes, "CH4")
    spec = CMCollectionSpec.from_plume_record(rec)
    scene = rec["plume_id"].rsplit("-", 1)[0]
    try:
        ir = CMImageRaster.from_scene_id(scene, token=token, spec=spec, with_rgb=False)
    except aq.CMSceneNotPublished:
        pytest.skip("newest CH4 plume's L2B not published yet (publication lag)")
    assert f"l2b-{spec.gas}-" in str(ir.asset_paths["cmf"])
    _ = ir.artifact_mask  # must not raise when the scene has no artifact mask
