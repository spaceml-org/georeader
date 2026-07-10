"""Tests for ``georeader.readers.carbonmapper.image``."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds as t_from_bounds

from georeader.rasterio_reader import RasterioReader

from georeader.readers.carbonmapper.image import (
    CM_PLUME_IMAGE_ASSETS,
    CMPlumeImage,
    _cdn_to_api,
    _derive_asset_urls,
    _parse_geojson_to_geometry,
)
from georeader.readers.carbonmapper.plume import (
    CMRawPlume,
    Collection,
)


# ─── Fixtures ───────────────────────────────────────────────────────


PID_V3A = "tan20251212t185057c20s4001-E"
PID_V3C = "tan20260417t192203c08s4001-B"


def _signed_cdn_url(coll: str, plume_id: str) -> str:
    """Build a realistic signed-CDN URL for a plume_tif asset."""
    y = plume_id[3:7]; m = plume_id[7:9]; d = plume_id[9:11]
    return (
        f"https://catalog.carbonmapper.org/{coll}/"
        f"{y}/{m}/{d}/{plume_id}/{plume_id}_{coll}_plume.tif"
        f"?Expires=1778232132&Signature=foo&Key-Pair-Id=bar"
    )


def _api_url(coll: str, plume_id: str, asset: str) -> str:
    """Build the api-gateway form (what `_derive_asset_urls` produces)."""
    y = plume_id[3:7]; m = plume_id[7:9]; d = plume_id[9:11]
    return (
        f"https://api.carbonmapper.org/api/v1/catalog/asset/{coll}/"
        f"{y}/{m}/{d}/{plume_id}/{plume_id}_{coll}_{asset}"
    )


def _catalog_response(plume_id: str, vis_coll: str) -> dict:
    """Minimal `/catalog/plume/{id}` response — only the fields the
    derivation cares about."""
    return {
        "plume_id": plume_id,
        "plume_tif": _signed_cdn_url(vis_coll, plume_id),
    }


# ─── _cdn_to_api ────────────────────────────────────────────────────


class TestCdnToApi:
    def test_strips_query_string(self):
        signed = (
            "https://catalog.carbonmapper.org/l3a-vis-ch4-mfa-v3c/"
            "2026/04/17/foo/foo_l3a-vis-ch4-mfa-v3c_plume.tif?Expires=123"
        )
        assert _cdn_to_api(signed) == (
            "https://api.carbonmapper.org/api/v1/catalog/asset/"
            "l3a-vis-ch4-mfa-v3c/2026/04/17/foo/"
            "foo_l3a-vis-ch4-mfa-v3c_plume.tif"
        )

    def test_rewrites_host(self):
        url = _cdn_to_api(
            "https://catalog.carbonmapper.org/coll/a/b/c.tif"
        )
        assert url.startswith("https://api.carbonmapper.org/api/v1/catalog/asset/")
        assert "?" not in url


# ─── _derive_asset_urls — core URL pattern ─────────────────────────


class TestDeriveAssetUrlsV3A:
    """v3a is STAC-resident — URLs derived from catalog dict."""

    def test_returns_7_keys(self):
        urls = _derive_asset_urls(
            _catalog_response(PID_V3A, "l3a-vis-ch4-mfa-v3a")
        )
        assert set(urls) == {
            "plume.tif", "plume-concentrations.tif",
            "plume-outline.geojson", "rgb.tif",
            "ime-cmf-concentrations.tif",
            "ime-cmf-mask.tif", "ime-cmf-outline.geojson",
        }

    def test_plume_tif_round_trip(self):
        urls = _derive_asset_urls(
            _catalog_response(PID_V3A, "l3a-vis-ch4-mfa-v3a")
        )
        assert urls["plume.tif"] == _api_url(
            "l3a-vis-ch4-mfa-v3a", PID_V3A, "plume.tif",
        )

    def test_concentrations_url(self):
        urls = _derive_asset_urls(
            _catalog_response(PID_V3A, "l3a-vis-ch4-mfa-v3a")
        )
        assert urls["plume-concentrations.tif"] == _api_url(
            "l3a-vis-ch4-mfa-v3a", PID_V3A, "plume-concentrations.tif",
        )

    def test_outline_url(self):
        urls = _derive_asset_urls(
            _catalog_response(PID_V3A, "l3a-vis-ch4-mfa-v3a")
        )
        assert urls["plume-outline.geojson"] == _api_url(
            "l3a-vis-ch4-mfa-v3a", PID_V3A, "plume-outline.geojson",
        )

    def test_ime_concentrations_url_swaps_collection(self):
        """IME sibling URL swaps `l3a-vis-` → `l3a-ime-` in both
        the path segment and the filename tail."""
        urls = _derive_asset_urls(
            _catalog_response(PID_V3A, "l3a-vis-ch4-mfa-v3a")
        )
        ime = urls["ime-cmf-concentrations.tif"]
        assert "l3a-ime-ch4-mfa-v3a" in ime
        assert "l3a-vis-ch4-mfa-v3a" not in ime
        assert ime.endswith(
            f"{PID_V3A}_l3a-ime-ch4-mfa-v3a_ime-cmf-concentrations.tif",
        )


class TestDeriveAssetUrlsV3C:
    """v3c is NOT in STAC — same URL pattern, different collection segment."""

    def test_handles_v3c_collection(self):
        urls = _derive_asset_urls(
            _catalog_response(PID_V3C, "l3a-vis-ch4-mfa-v3c")
        )
        assert urls["plume.tif"].endswith(
            f"{PID_V3C}_l3a-vis-ch4-mfa-v3c_plume.tif",
        )
        assert urls["ime-cmf-concentrations.tif"].endswith(
            f"{PID_V3C}_l3a-ime-ch4-mfa-v3c_ime-cmf-concentrations.tif",
        )

    def test_returns_all_keys_for_v3c(self):
        urls = _derive_asset_urls(
            _catalog_response(PID_V3C, "l3a-vis-ch4-mfa-v3c")
        )
        assert set(urls) == set(CM_PLUME_IMAGE_ASSETS)
        # Verify the new ime keys carry the v3c collection segment too
        assert "v3c" in urls["ime-cmf-mask.tif"]
        assert "v3c" in urls["ime-cmf-outline.geojson"]


class TestDeriveAssetUrlsErrors:
    def test_missing_plume_tif_raises(self):
        with pytest.raises(ValueError, match="no 'plume_tif'"):
            _derive_asset_urls({"plume_id": PID_V3A})

    def test_unrecognised_collection_skips_ime_only(self):
        """A legacy mf-v1 plume has no `l3a-vis-ch4-mfa-` prefix —
        URL pattern can't construct the IME sibling."""
        seed = {
            "plume_id": "tan20180101t000000p00001-A",
            "plume_tif": (
                "https://catalog.carbonmapper.org/l3a-ch4-mf-v1/"
                "2018/01/01/foo/foo_l3a-ch4-mf-v1_plume.tif?x=1"
            ),
        }
        urls = _derive_asset_urls(seed)
        assert "plume.tif" in urls
        assert "rgb.tif" in urls
        assert "ime-cmf-concentrations.tif" not in urls

    def test_v3b_intermediate_works(self):
        """v3b is the intermediate version (Dec 2025-era plumes); not
        in our enum but the version-agnostic regex swap handles it."""
        seed = {
            "plume_id": "tan20251212t185057c20s4001-E",
            "plume_tif": (
                "https://catalog.carbonmapper.org/l3a-vis-ch4-mfa-v3b/"
                "2025/12/12/tan20251212t185057c20s4001-E/"
                "tan20251212t185057c20s4001-E_l3a-vis-ch4-mfa-v3b"
                "_plume.tif?x=1"
            ),
        }
        urls = _derive_asset_urls(seed)
        assert "v3b" in urls["plume.tif"]
        assert "v3b" in urls["ime-cmf-concentrations.tif"]
        assert "l3a-ime-ch4-mfa-v3b" in urls["ime-cmf-concentrations.tif"]

    def test_malformed_plume_tif_raises(self):
        seed = {
            "plume_id": PID_V3A,
            "plume_tif": "https://catalog.carbonmapper.org/foo/bar.tif",
        }
        with pytest.raises(ValueError, match="asset pattern"):
            _derive_asset_urls(seed)


# ─── _parse_geojson_to_geometry ────────────────────────────────────


class TestParseGeoJsonToGeometry:
    def test_polygon_feature(self):
        feat = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }
        geom = _parse_geojson_to_geometry(feat)
        assert geom is not None
        assert geom.geom_type == "Polygon"
        assert geom.area == pytest.approx(1.0)

    def test_feature_collection(self):
        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
                        ],
                    },
                },
            ],
        }
        geom = _parse_geojson_to_geometry(fc)
        assert geom is not None
        assert geom.area == pytest.approx(1.0)

    def test_bare_geometry(self):
        bare = {
            "type": "Polygon",
            "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
        }
        geom = _parse_geojson_to_geometry(bare)
        assert geom is not None
        assert geom.area == pytest.approx(4.0)

    def test_returns_none_on_garbage(self):
        assert _parse_geojson_to_geometry({}) is None
        assert _parse_geojson_to_geometry("not a dict") is None
        assert _parse_geojson_to_geometry(
            {"type": "FeatureCollection", "features": []},
        ) is None


# ─── CMPlumeImage construction ─────────────────────────────────────


class TestFromCmRawPlume:
    def test_builds_from_cmrawplume(self):
        raw = CMRawPlume(
            plume_id=PID_V3A,
            plume_tif=_signed_cdn_url("l3a-vis-ch4-mfa-v3a", PID_V3A),
        )
        img = CMPlumeImage.from_cmrawplume(raw, token="tok")
        assert img.plume_id == PID_V3A
        assert img.token == "tok"
        assert "plume.tif" in img.urls
        assert "ime-cmf-concentrations.tif" in img.urls

    def test_raises_when_plume_tif_absent(self):
        raw = CMRawPlume(plume_id=PID_V3A)
        with pytest.raises(ValueError, match="no plume_tif URL"):
            CMPlumeImage.from_cmrawplume(raw, token="tok")

    @pytest.mark.parametrize(
        "seed",
        [
            # gateway form — the trigger for spaceml-org/georeader#65:
            # the pre-registry code re-applied the gateway prefix to a
            # plume_tif that was already gateway-form, producing
            # `/catalog/asset/api/v1/catalog/asset/...` 404s.
            _api_url("l3a-vis-ch4-mfa-v3b", PID_V3A, "plume.tif"),
            # already-doubled input parses (tail-anchored) and is repaired
            _api_url("l3a-vis-ch4-mfa-v3b", PID_V3A, "plume.tif").replace(
                "/api/v1/catalog/asset/",
                "/api/v1/catalog/asset/api/v1/catalog/asset/",
                1,
            ),
        ],
        ids=["gateway-seed", "doubled-seed"],
    )
    def test_no_double_gateway_prefix(self, seed):
        """Regression: spaceml-org/georeader#65."""
        raw = CMRawPlume(plume_id=PID_V3A, plume_tif=seed)
        img = CMPlumeImage.from_cmrawplume(raw, token="tok")
        assert all(
            u.count("/catalog/asset/") == 1 for u in img.urls.values()
        )
        assert img.urls["plume-outline.geojson"] == _api_url(
            "l3a-vis-ch4-mfa-v3b", PID_V3A, "plume-outline.geojson",
        )


class TestFromPlumeId:
    def test_one_round_trip(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = _catalog_response(
                PID_V3A, "l3a-vis-ch4-mfa-v3a",
            )
            mock_get.return_value.raise_for_status = MagicMock()
            img = CMPlumeImage.from_plume_id(PID_V3A, token="tok")

        # One HTTP call to /catalog/plume/{id}
        assert mock_get.call_count == 1
        assert "/catalog/plume/" + PID_V3A in mock_get.call_args[0][0]
        assert mock_get.call_args[1]["headers"] == {
            "Authorization": "Bearer tok",
        }
        assert img.plume_id == PID_V3A
        assert set(img.urls) == set(CM_PLUME_IMAGE_ASSETS)

    def test_v3c_handled_transparently(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = _catalog_response(
                PID_V3C, "l3a-vis-ch4-mfa-v3c",
            )
            mock_get.return_value.raise_for_status = MagicMock()
            img = CMPlumeImage.from_plume_id(PID_V3C, token="tok")
        assert "v3c" in img.urls["plume.tif"]


class TestFromStacItem:
    def _stac_item(self, plume_id: str, *, with_outline: bool = True) -> dict:
        assets = {
            "plume.tif":               {"href": "https://x/plume.tif"},
            "plume-concentrations.tif":{"href": "https://x/plume-concentrations.tif"},
            "plume-rgb.png":           {"href": "https://x/plume-rgb.png"},
            "rgb.tif":                 {"href": "https://x/rgb.tif"},
        }
        if with_outline:
            assets["plume-outline.geojson"] = {
                "href": "https://x/plume-outline.geojson",
            }
        return {"id": plume_id, "assets": assets}

    def test_builds_from_vis_item(self):
        item = self._stac_item(PID_V3A)
        img = CMPlumeImage.from_stac_item(item)
        assert img.plume_id == PID_V3A
        assert "plume.tif" in img.urls
        assert "plume-concentrations.tif" in img.urls
        assert "plume-outline.geojson" in img.urls
        assert "ime-cmf-concentrations.tif" not in img.urls

    def test_with_ime_sibling(self):
        vis = self._stac_item(PID_V3A)
        ime = {
            "id": PID_V3A,
            "assets": {
                "ime-cmf-concentrations.tif": {
                    "href": "https://x/ime-cmf-concentrations.tif",
                },
            },
        }
        img = CMPlumeImage.from_stac_item(vis, ime_item=ime)
        assert "ime-cmf-concentrations.tif" in img.urls


# ─── Lazy property opens (with on-disk fixture rasters) ────────────


def _write_rgba_geotiff(path, *, w=8, h=8):
    """4-band uint8 GeoTIFF — sim plume.tif (band 4 = mask)."""
    arr = np.zeros((4, h, w), dtype="uint8")
    arr[3, 2:6, 2:6] = 255   # binary mask in central 4x4 block
    with rasterio.open(
        str(path), "w", driver="GTiff",
        count=4, dtype="uint8",
        width=w, height=h,
        transform=t_from_bounds(-104, 31, -103, 32, w, h),
        crs="EPSG:4326",
    ) as dst:
        dst.write(arr)


def _write_singleband_geotiff(path, *, fill=42.0, w=8, h=8):
    arr = np.full((1, h, w), fill, dtype="float32")
    with rasterio.open(
        str(path), "w", driver="GTiff",
        count=1, dtype="float32",
        width=w, height=h,
        transform=t_from_bounds(-104, 31, -103, 32, w, h),
        crs="EPSG:4326",
    ) as dst:
        dst.write(arr)


class TestLazyProperties:
    def test_mask_returns_rasterio_reader(self, tmp_path):
        p = tmp_path / "plume.tif"
        _write_rgba_geotiff(p)
        img = CMPlumeImage(plume_id=PID_V3A, urls={"plume.tif": str(p)})
        assert isinstance(img.mask, RasterioReader)
        # Repeat access is cached (same instance)
        assert img.mask is img.mask

    def test_concentrations_returns_rasterio_reader(self, tmp_path):
        p = tmp_path / "plume-concentrations.tif"
        _write_singleband_geotiff(p)
        img = CMPlumeImage(
            plume_id=PID_V3A,
            urls={"plume-concentrations.tif": str(p)},
        )
        reader = img.concentrations
        assert isinstance(reader, RasterioReader)
        # Verify we can load it
        gt = reader.load()
        assert gt.values.shape == (1, 8, 8)

    def test_ime_concentrations_returns_none_if_absent(self):
        img = CMPlumeImage(plume_id=PID_V3A, urls={})
        assert img.ime_concentrations is None

    def test_rgb_returns_none_if_absent(self):
        img = CMPlumeImage(plume_id=PID_V3A, urls={})
        assert img.rgb is None

    def test_ime_mask_returns_rasterio_reader(self, tmp_path):
        p = tmp_path / "ime-cmf-mask.tif"
        _write_singleband_geotiff(p, fill=1.0)
        img = CMPlumeImage(
            plume_id=PID_V3A, urls={"ime-cmf-mask.tif": str(p)},
        )
        assert isinstance(img.ime_mask, RasterioReader)

    def test_ime_mask_returns_none_if_absent(self):
        img = CMPlumeImage(plume_id=PID_V3A, urls={})
        assert img.ime_mask is None


class TestImeOutline:
    """`ime_outline` is the IME-significant polygon — tighter than
    `outline` (which uses the broader plume mask)."""

    def test_ime_outline_uses_geojson_when_present(self, tmp_path):
        outline_path = tmp_path / "ime-outline.geojson"
        feat = {
            "type": "Polygon",
            "coordinates": [
                [[-103.6, 31.4], [-103.4, 31.4],
                 [-103.4, 31.6], [-103.6, 31.6], [-103.6, 31.4]],
            ],
        }
        outline_path.write_text(json.dumps(feat))
        img = CMPlumeImage(
            plume_id=PID_V3A,
            urls={"ime-cmf-outline.geojson": str(outline_path)},
        )
        geom = img.ime_outline
        assert geom is not None
        assert geom.area == pytest.approx(0.04)

    def test_ime_outline_returns_none_when_absent(self):
        img = CMPlumeImage(plume_id=PID_V3A, urls={})
        assert img.ime_outline is None

    def test_ime_outline_does_not_fall_back_to_alpha(self, tmp_path):
        """Unlike `outline`, `ime_outline` returns `None` on fetch
        failure rather than vectorising — `outline` is the canonical
        broader fallback."""
        plume_path = tmp_path / "plume.tif"
        _write_rgba_geotiff(plume_path)
        img = CMPlumeImage(
            plume_id=PID_V3A,
            urls={
                "plume.tif": str(plume_path),
                "ime-cmf-outline.geojson": "https://bogus.invalid/x.geojson",
            },
            http_timeout=0.001,
        )
        assert img.ime_outline is None


# ─── Outline canonical / vectorize fallback ────────────────────────


class TestOutlineCanonical:
    def test_outline_uses_geojson_when_present(self, tmp_path):
        # Local geojson file — bypass the HTTP path
        outline_path = tmp_path / "outline.geojson"
        feat = {
            "type": "Polygon",
            "coordinates": [
                [[-103.6, 31.4], [-103.4, 31.4],
                 [-103.4, 31.6], [-103.6, 31.6], [-103.6, 31.4]],
            ],
        }
        outline_path.write_text(json.dumps(feat))
        img = CMPlumeImage(
            plume_id=PID_V3A,
            urls={"plume-outline.geojson": str(outline_path)},
        )
        geom = img.outline
        assert geom is not None
        # Box-shaped polygon → area is 0.04 deg² (0.2 × 0.2)
        assert geom.area == pytest.approx(0.04)

    def test_outline_falls_back_to_alpha_on_geojson_failure(
        self, tmp_path, caplog,
    ):
        """If outline URL is missing, fall back to band-4 vectorize.
        (Empty `urls` is the simplest "fetch failure" case — no URL at
        all, so `_fetch_outline_geojson` returns None silently.)"""
        plume_path = tmp_path / "plume.tif"
        _write_rgba_geotiff(plume_path)
        img = CMPlumeImage(
            plume_id=PID_V3A,
            urls={"plume.tif": str(plume_path)},
        )
        geom = img.outline
        assert geom is not None
        # Vectorized 4×4 alpha block → non-empty geometry
        assert not geom.is_empty

    def test_outline_returns_none_when_neither_source_available(self):
        img = CMPlumeImage(plume_id=PID_V3A, urls={})
        assert img.outline is None

    def test_outline_logs_warning_on_geojson_fetch_error(
        self, tmp_path, caplog,
    ):
        """Bad URL → warning logged + falls back to alpha."""
        import logging
        plume_path = tmp_path / "plume.tif"
        _write_rgba_geotiff(plume_path)
        img = CMPlumeImage(
            plume_id=PID_V3A,
            urls={
                "plume.tif": str(plume_path),
                "plume-outline.geojson": "https://bogus.invalid/x.geojson",
            },
            http_timeout=0.001,   # force quick failure
        )
        with caplog.at_level(logging.WARNING):
            geom = img.outline
        assert geom is not None      # fallback succeeded
        assert any(
            "outline GeoJSON fetch failed" in r.message
            for r in caplog.records
        )


# ─── Repr ──────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_lists_present_and_missing(self):
        img = CMPlumeImage(
            plume_id=PID_V3A,
            urls={"plume.tif": "x", "rgb.tif": "y"},
        )
        text = repr(img)
        assert "CMPlumeImage" in text
        assert PID_V3A in text
        assert "assets present" in text
        assert "assets missing" in text


# ─── Sanity: Collection enum ─────────────────────────────────────


class TestCollectionUsage:
    """Spot-check that the URL-derivation handles every Collection
    member correctly."""

    @pytest.mark.parametrize("vis_coll,ime_coll", [
        (Collection.L3A_VIS_V3A, Collection.L3A_IME_V3A),
        (Collection.L3A_VIS_V3C, Collection.L3A_IME_V3C),
    ])
    def test_collection_pair_round_trip(self, vis_coll, ime_coll):
        urls = _derive_asset_urls(_catalog_response(PID_V3A, vis_coll.value))
        assert vis_coll.value in urls["plume.tif"]
        assert ime_coll.value in urls["ime-cmf-concentrations.tif"]


# ─── Tile bridge (Phase 2) ────────────────────────────────────────


class TestSceneIdDerivation:
    """`CMPlumeImage.scene_id` is the parent L2B scene_id derived
    from the plume_id by trimming the ``-{part}`` suffix."""

    @pytest.mark.parametrize("plume_id, expected_scene", [
        ("tan20260331t181625c77s4001-A", "tan20260331t181625c77s4001"),
        ("tan20251212t185057c20s4001-E", "tan20251212t185057c20s4001"),
        # Multi-hyphen plume id (rsplit keeps everything before the last)
        ("emi20250515t190623-ext-B", "emi20250515t190623-ext"),
    ])
    def test_rsplit_recovers_parent(self, plume_id, expected_scene):
        img = CMPlumeImage(plume_id=plume_id, urls={})
        assert img.scene_id == expected_scene


class TestTileBridge:
    """`CMPlumeImage.tile` is the lazy L2B parent raster.

    Wraps ``api_queries.get_image_raster_for_plume`` (Phase 1) so
    the v3a STAC path and the v3c URL-pattern fallback are both
    transparent here.
    """

    def test_tile_requires_token(self):
        img = CMPlumeImage(plume_id="tan-foo-A", urls={"plume.tif": "x"})
        with pytest.raises(ValueError, match="no token"):
            _ = img.tile

    def test_tile_calls_get_image_raster_for_plume(self):
        from georeader.readers.carbonmapper import api_queries

        sentinel = MagicMock(name="CMImageRaster_sentinel")
        with patch.object(
            api_queries, "get_image_raster_for_plume",
            return_value=sentinel,
        ) as mock_call:
            img = CMPlumeImage(
                plume_id="tan-foo-A", urls={}, token="tok",
            )
            result = img.tile
        assert result is sentinel
        mock_call.assert_called_once_with("tok", "tan-foo-A")

    def test_tile_is_cached(self):
        from georeader.readers.carbonmapper import api_queries

        sentinel = MagicMock(name="CMImageRaster_sentinel")
        with patch.object(
            api_queries, "get_image_raster_for_plume",
            return_value=sentinel,
        ) as mock_call:
            img = CMPlumeImage(
                plume_id="tan-foo-A", urls={}, token="tok",
            )
            _ = img.tile
            _ = img.tile
            _ = img.tile
        # Cached: only one underlying API call regardless of repeat access.
        assert mock_call.call_count == 1

    def test_tile_raises_when_unresolved(self):
        from georeader.readers.carbonmapper import api_queries

        with patch.object(
            api_queries, "get_image_raster_for_plume", return_value=None,
        ):
            img = CMPlumeImage(
                plume_id="tan-foo-A", urls={}, token="tok",
            )
            with pytest.raises(RuntimeError, match="not reachable"):
                _ = img.tile


class TestTileCrop:
    """`tile_cmf` / `tile_rgb` / `tile_uncertainty` crop the L2B
    band by the plume outline polygon at full L2B native resolution.
    """

    @staticmethod
    def _make_image_with_tile_and_outline(monkeypatch, *, with_rgb=True):
        """Build a CMPlumeImage with mocked outline + mocked tile + mocked
        ``read.read_from_polygon``, returning the image and the recorded
        call-args dict."""
        from georeader.geotensor import GeoTensor
        from georeader.readers.carbonmapper import image as _image

        # Fake outline polygon — `tile_*` only checks it's not None.
        outline_geom = MagicMock(name="outline_polygon")

        # Fake tile with the three bands we want to crop. RasterioReader
        # objects are duck-typed by `read.read_from_polygon`.
        fake_tile = MagicMock(name="CMImageRaster")
        fake_tile.cmf = MagicMock(name="cmf_reader")
        fake_tile.rgb = MagicMock(name="rgb_reader") if with_rgb else None
        fake_tile.uncertainty = MagicMock(name="uncertainty_reader")
        fake_tile.asset_paths = {
            "cmf": "https://x/cmf.tif",
            "uncertainty": "https://x/unc.tif",
            **({"rgb": "https://x/rgb.tif"} if with_rgb else {}),
        }
        fake_tile.scene_id = "tan-foo"

        # Recorded crop. `trigger_load=True` so the wrapper sees a
        # ready-to-return GeoTensor and skips the `.load()` fallback.
        sentinel_geo = GeoTensor(
            values=np.zeros((1, 16, 16), dtype="float32"),
            transform=t_from_bounds(0, 0, 100, 100, 16, 16),
            crs="EPSG:32613",
        )
        captured = {}

        def fake_read_from_polygon(data_in, **kw):
            captured["data_in"] = data_in
            captured.update(kw)
            return sentinel_geo

        monkeypatch.setattr(_image.read, "read_from_polygon", fake_read_from_polygon)

        img = CMPlumeImage(
            plume_id="tan-foo-A", urls={}, token="tok",
        )
        # Inject the mocked tile + outline by populating the
        # cached_property slots directly.
        img.__dict__["tile"] = fake_tile
        img.__dict__["outline"] = outline_geom

        return img, captured, sentinel_geo, fake_tile

    def test_tile_cmf_crops_with_default_pad(self, monkeypatch):
        img, captured, sentinel_geo, fake_tile = (
            self._make_image_with_tile_and_outline(monkeypatch)
        )
        result = img.tile_cmf()
        assert result is sentinel_geo
        # Sourced from the tile's cmf reader.
        assert captured["data_in"] is fake_tile.cmf
        # Default pad = 64 on both axes.
        assert captured["pad_add"] == (64, 64)
        assert captured["crs_polygon"] == "EPSG:4326"
        assert captured["boundless"] is False

    def test_tile_rgb_crops_rgb_band(self, monkeypatch):
        img, captured, _, fake_tile = (
            self._make_image_with_tile_and_outline(monkeypatch)
        )
        img.tile_rgb(pad_px=8)
        assert captured["data_in"] is fake_tile.rgb
        assert captured["pad_add"] == (8, 8)

    def test_tile_uncertainty_crops_uncertainty_band(self, monkeypatch):
        img, captured, _, fake_tile = (
            self._make_image_with_tile_and_outline(monkeypatch)
        )
        img.tile_uncertainty(pad_px=0)
        assert captured["data_in"] is fake_tile.uncertainty
        assert captured["pad_add"] == (0, 0)

    def test_tile_rgb_raises_when_sibling_missing(self, monkeypatch):
        img, _, _, _ = self._make_image_with_tile_and_outline(
            monkeypatch, with_rgb=False,
        )
        with pytest.raises(KeyError, match="no 'rgb' asset"):
            img.tile_rgb()

    def test_raises_when_outline_none(self, monkeypatch):
        img, _, _, _ = self._make_image_with_tile_and_outline(monkeypatch)
        # Override outline back to None.
        img.__dict__["outline"] = None
        with pytest.raises(ValueError, match="outline is None"):
            img.tile_cmf()

    def test_raises_on_zero_overlap(self, monkeypatch):
        from georeader.readers.carbonmapper import image as _image

        img, _, _, _ = self._make_image_with_tile_and_outline(monkeypatch)
        # Make read_from_polygon return None (zero-overlap path).
        monkeypatch.setattr(
            _image.read, "read_from_polygon", lambda *a, **kw: None,
        )
        with pytest.raises(RuntimeError, match="doesn't overlap"):
            img.tile_cmf()
