# `examples/` — data for the notebook integration tests

The notebooks under [`docs/`](../docs) double as **integration tests**: they are
executed with [`nbmake`](https://github.com/treebeardtech/nbmake) via

```bash
make test-notebooks
```

Most notebooks need a raster, a vector file, a cloud credential or network
access that is not always available. To make the suite portable, every notebook
reads its local inputs from **this `examples/` folder** (resolved at run time,
so it works no matter which directory the notebook is launched from), and the
test harness ([`docs/conftest.py`](../docs/conftest.py)) **skips** any notebook
whose inputs are missing. Drop the files below into `examples/` (or provide the
relevant credentials) and the corresponding notebooks start running.

> Data files placed here are **git-ignored** on purpose (large rasters /
> non-redistributable licences). Only this README and `.gitignore` are tracked.

## How a notebook is gated

A notebook runs only if **all** of its requirement groups are satisfied; a group
is satisfied if *any* of: a listed file exists under `examples/`, a listed
environment variable is set (the notebook then downloads / authenticates by
itself), or a listed credential file exists. Notebooks marked
**"runs as-is (network)"** below are never skipped — they only need outbound
network access.

## File ↔ notebook map

| File in `examples/` | Used by notebook(s) | Source / how to obtain | Status |
|---|---|---|---|
| `PRISMA/PRS_L1_STD_OFFL_20241109073054_20241109073059_0001.he5` | `docs/prisma_with_cloudsen12.ipynb` | IMEO Azure container (auto-download, see *Credentials*) | ❌ missing |
| `PRISMA/PRS_L1_STD_OFFL_20230929102749_20230929102753_0001.he5` | `docs/simultaneous_prisma_emit.ipynb` | IMEO Azure container (auto-download) | ❌ missing |
| `EnMAP/ENMAP01-____L1B-DT0000074101_20240511T080843Z_001_V010402_20240514T093550Z/` | `docs/enmap_with_cloudsen12.ipynb` | IMEO Azure container (auto-download) | ❌ missing |
| `EMIT/EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc` | `docs/emit_explore.ipynb` | NASA Earthdata (auto-download) | ❌ missing |
| `EMIT/EMIT_L1B_RAD_001_20230929T122534_2327208_039.nc` | `docs/simultaneous_prisma_emit.ipynb` | NASA Earthdata (auto-download) | ❌ missing |
| `S2L1C.tif` | `docs/read_overlapping_probav_and_sentinel2.ipynb` | A Sentinel-2 L1C RGB crop (export your own) | ❌ missing |
| `ang20190928t185111-4_r6871_c424_rgb.tif` | `docs/reading_overlapping_sentinel2_aviris.ipynb` | AVIRIS-NG Permian 2019 RGB crop | ❌ missing |
| `liria.geojson` | `docs/Sentinel-2/query_mosaic_s2_images.ipynb` | Any small AOI polygon — draw one at [geojson.io](https://geojson.io) | ❌ missing |

## Notebooks that need no local file

| Notebook | Requirement | How it runs |
|---|---|---|
| `docs/read_S2_SAFE_from_bucket.ipynb` | network | Streams a Sentinel-2 SAFE from the **public** Google bucket (`GS_NO_SIGN_REQUEST=YES`). Runs as-is (network). |
| `docs/Sentinel-2/explore_metadata_s2.ipynb` | network | Same public bucket. Runs as-is (network). |
| `docs/geotensor_numpy_api.ipynb` | network | Reads a public GeoTIFF from Hugging Face. Runs as-is (network). |
| `docs/advanced/tiling_and_stitching.ipynb` | network + `cloudsen12_models` | Public bucket SAFE + cloud-detection model (installed in-notebook). Runs as-is (network). |
| `docs/Sentinel-2/run_in_gee_image.ipynb` | Earth Engine + `cloudsen12_models` | Needs an Earth Engine account — *skipped* unless EE credentials are present. |
| `docs/Sentinel-2/s2_mosaic_from_gee.ipynb` | Earth Engine | *Skipped* unless EE credentials are present. |
| `docs/Sentinel-2/convert_to_radiance.ipynb` | Earth Engine + public bucket | *Skipped* unless EE credentials are present. |
| `docs/carbonmapper/api_explore.ipynb` | Carbon Mapper API token | *Skipped* unless a Carbon Mapper token is present. |
| `docs/carbonmapper/products_explore.ipynb` | Carbon Mapper API token | *Skipped* unless a Carbon Mapper token is present. |
| `docs/read_overlapping_probav_and_sentinel2.ipynb` | network + `S2L1C.tif` | Proba-V product is downloaded at run time from the public VITO data pool; the Sentinel-2 crop must be in `examples/`. |

## Credentials

Set these as environment variables locally (e.g. `export SAS_TOKEN=...`) or as
**GitHub Actions repository secrets** exported before `make test-notebooks`.
When a credential is present, the notebook downloads its own data and the test
runs; otherwise the test is skipped.

| Provider | Notebooks | Variables / files |
|---|---|---|
| **IMEO Azure** (PRISMA / EnMAP) | `prisma_with_cloudsen12`, `enmap_with_cloudsen12`, `simultaneous_prisma_emit` | `SAS_TOKEN`, `AZURE_STORAGE_ACCOUNT`, `CONTAINER_NAME` (same variables as the [`marshsi`](https://github.com/UNEP-IMEO-MARS/marshsi) package) |
| **NASA Earthdata** (EMIT) | `emit_explore`, `simultaneous_prisma_emit` | `EARTHDATA_TOKEN` (bearer token from <https://urs.earthdata.nasa.gov/profile>) **or** `~/.georeader/auth_emit.json` `{"user": "...", "password": "..."}` |
| **Carbon Mapper** | `carbonmapper/api_explore`, `carbonmapper/products_explore` | `CARBONMAPPER_TOKEN` (or `CARBONMAPPER_EMAIL` + `CARBONMAPPER_PASSWORD`) **or** `~/.georeader/auth_carbonmapper.json` |
| **Google Earth Engine** | `run_in_gee_image`, `s2_mosaic_from_gee`, `convert_to_radiance` | `EARTHENGINE_TOKEN` **or** a pre-authorized `~/.config/earthengine/credentials` |

## `cloudsen12_models`

Four notebooks (`tiling_and_stitching`, `run_in_gee_image`,
`prisma_with_cloudsen12`, `enmap_with_cloudsen12`) run CloudSEN12 cloud-detection
models via the [`cloudsen12_models`](https://github.com/IPL-UV/cloudsen12_models)
package. It is **not** a `georeader` dependency (it depends on `georeader`, which
would be circular), so each of those notebooks installs it in-kernel with a
`%pip install cloudsen12_models` cell. CI/offline runs therefore need network
access for that install.

## Next steps (not yet done)

- Host the `examples/` rasters on a shared store (Hugging Face dataset or a
  private Azure container) so the data files can be fetched automatically.
- Run `make test-notebooks` in CI (GitHub Actions / Jenkins / Azure ML) with the
  credentials above wired as secrets.
