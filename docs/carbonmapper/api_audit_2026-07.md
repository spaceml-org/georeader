# Carbon Mapper API audit — 2026-07-06

A point-in-time verification of every product/URL assumption the
`georeader.readers.carbonmapper` subpackage makes, against the live API
(`https://api.carbonmapper.org/api/v1`). Run as a single rate-limited pass
(~25 spaced requests, every response cached); the raw responses are not
committed but the probe method is reproducible from this document.

Legend: ✅ assumption verified · ❌ assumption wrong / stale · ➕ exists
upstream but not modelled by the reader.

## 1. Plume record (`/catalog/plumes/annotated`, `/catalog/plume/{id}`)

Both endpoints return the **same 35 fields** (verified on a June-2026
Tanager plume; list-vs-detail divergence previously suspected is gone).

- ✅ Asset-URL fields are exactly `plume_tif`, `con_tif`, `plume_png`,
  `rgb_png`, `plume_rgb_png` — **no `rgb_tif` field** (the plain geo RGB
  exists only as a *derived asset*, see §3).
- ✅ `geometry_json` is a Point; the only areal field is `plume_bounds`.
  No outline geometry in the record itself.
- ✅ `scene_id` **is a UUID** (e.g. `3faede0d-45dd-…`), not the
  `tan20260623t…` scene name. The reader already handles this correctly:
  `CMRawPlume.scene_uuid` carries the wire value and the `scene_id`
  property derives the parseable scene name from `plume_id`.
  `plume_id.rsplit("-", 1)[0]` remains the only public source of the
  scene name.
- ✅ `con_tif` points at the **IME** product
  (`…/l3a-ime-ch4-mfa-v3d/…_ime-cmf-concentrations.tif`) — byte-identical
  URL to what `_derive_asset_urls` synthesizes via the vis→ime segment
  swap. The swap trick is validated end-to-end.
- ➕ Fields present upstream and useful for version resolution:
  `cmf_type` (`"mfa"`), `emission_version` (`"v3d"`), `gas` — together
  they compose the L3A collection id
  `l3a-vis-{gas}-{cmf_type}-{emission_version}` with **no probing
  needed**. Also unmodelled: `gsd`, `off_nadir`, `is_offshore`,
  `mission_phase`, `sensitivity_mode`, `status`, `published_at`,
  `publication_sources`, `hide_emission`, `plume_quality`,
  `processing_software`, `emission_cmf_type`, `collection`
  (confusingly = the *L2C detection* collection, `l2c-ch4-v0`, not the
  L3A one).
- ❗ `wind_source_auto` is **not** in the public record (relevant to
  marsml #344 — that field must come from the CSV export / partner feed,
  not this endpoint).

## 2. Collections (STAC registry vs asset proxy)

`GET /stac/collections` returns 86 collections — but the registry stops
at `…-v3a`. The current-era collections (`…-v3c`, `…-v3d`) exist **only
in the asset-proxy namespace** (`/catalog/asset/{collection}/…`), not in
STAC.

- ❌ This resolves a long-standing contradiction: georeader's probe
  comments ("v3c verified 2026-05-11") and marsml `simulate.py`'s
  "non-existent v3c" were both right *in different namespaces*. STAC
  search cannot see any 2026-era L2B scene; `get_tile` /
  `get_image_raster_for_scene`'s STAC-first path always falls through to
  URL probing for current data.
- ➕ Families the reader does not expose at all: `l4a-*` (source-level,
  e.g. `l4a-combined-ch4-v3a`), `l2c-*` (detections), `l3c-attribution`,
  `l2-cloud-mask-planet`.

## 3. L3A per-plume assets

STAC item assets for `l3a-vis-ch4-mfa-v3a` and live 206-probes of the
derived URLs for a v3d plume agree:

- ✅ vis: `plume.tif`, `plume-concentrations.tif`,
  `plume-outline.geojson`, `rgb.tif` — all four of the reader's derived
  names serve (206) at the api-gateway host. **`plume-outline.geojson`
  and `rgb.tif` are real** (they exist as assets even though the record
  has no URL field for them).
- ✅ ime: `ime-cmf-concentrations.tif`, `ime-cmf-mask.tif`,
  `ime-cmf-outline.geojson` all serve.
- ✅ `_cdn_to_api` host rewrite (signed CDN → Bearer-aware gateway) works.
- ➕ PNG quicklooks exist and are unmodelled: `plume.png`, `rgb.png`,
  `plume-rgb.png` (vis) and `ime-cmf-concentrations.png`,
  `ime-cmf-mask.png` (ime).
- A garbage asset name 404s (control), so the 206 results are meaningful.

## 4. L2B scene assets

- ✅ Asset keys in `l2b-ch4-mfa-v3a` STAC items match `CM_L2B_BANDS`
  exactly: `cmf`, `cmf-unortho`, `uncertainty`, `uncertainty-unortho`,
  `artifact-mask` (+ `uas.txt` sidecar). `rgb` lives in the separate
  `l2b-rgb-*` sibling collection, as the reader assumes.
- ✅ URL pattern
  `{base}/{coll}/{Y}/{M}/{D}/{scene}/{scene}_{coll}_{asset}` verified for
  v3c and v3d.
- ❌ **Version pairing is same-version, not cross-version**: a v3d L3A
  plume's L2B parent serves at `l2b-ch4-mfa-v3d` (v3c 404s), a v3c-era
  plume's at `v3c`. The `rasters.py` comment claiming "v3d L3A plumes
  whose L2B parent is still v3c" is wrong.
- ❌ `DEFAULT_L2B_CH4_COLLECTION_CANDIDATES = ("l2b-ch4-mfa-v3c",
  "l2b-ch4-mfa-v3a")` is **stale**: June-2026 scenes are v3d, so the
  default probe misses L2B data that exists. This is the structural flaw
  of hardcoded candidate lists — they silently rot every time CM bumps a
  version. The record itself (`emission_version` / the collection segment
  in `plume_tif`) names the right version with no probing.
- ✅ `l2b-rgb-v3d` also serves (same-version RGB sibling).

## 5. Endpoints

- ✅ `/catalog/plumes/annotated`, `/catalog/plume/{id}`, STAC
  `/collections`, `/collections/{id}/items`, and the
  `/catalog/asset/…` proxy all behave as the reader assumes (Bearer via
  `POST /token/pair`).
- ❌ `/catalog/scenes` (`download.get_scenes`) returns **401 with a
  standard authenticated account** — not usable as documented.

## Consequences for the refactor

1. Product identity should be **explicit per-product classes** (this
   audit's asset tables are the ground truth), selected by the caller —
   not hardcoded tuples with silent `None` fallbacks.
2. Collection/version must be **resolved from the plume record**
   (`plume_tif` URL segment, or `gas`+`cmf_type`+`emission_version`),
   same-version for the L2B parent; keep candidate probing only as an
   explicit opt-in for recordless (scene-name-only) lookups.
3. STAC can only serve ≤ v3a history; anything current must go through
   the asset proxy. The STAC-first/probe-second dance should be explicit
   in the API, not a silent fallback.
