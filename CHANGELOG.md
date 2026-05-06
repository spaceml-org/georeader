# Changelog

## Unreleased — 2.0 line

### Features

* **Carbon Mapper reader** (new): typed access to the Carbon Mapper
  STAC catalogue and plume API at `georeader.readers.carbonmapper`.
  Provides `CMRawPlume` (Pydantic model accepting both CSV bulk-export
  and annotated-JSON payloads), `CMSource` (cluster-of-plumes
  dataclass), `CMImageRaster` / `CMPlumeRaster` (lazy
  `RasterioReader`-backed band accessors with polygon extraction from
  the L3A alpha mask), `CarbonMapperConfig` (file-based token
  persistence), and the typed query layer (`get_tile`, `get_plume`,
  `get_source`, `list_*`, exception hierarchy). Gated behind the
  `[carbonmapper]` install extra (`pip install
  'georeader-spaceml[carbonmapper]'`) which adds `pydantic` and
  `requests`. No Azure SDK dependency — token loading via Azure Key
  Vault is left to downstream consumers (e.g. UNEP IMEO MARS).

## [1.5.12](https://github.com/spaceml-org/georeader/compare/v1.5.11...v1.5.12) (2026-01-28)


### Bug Fixes

* handle NaN values in GLT coordinates by replacing with zeros. Failing for product EMIT_L1B_RAD_001_20260125T232556_2602515_010 ([#33](https://github.com/spaceml-org/georeader/issues/33)) ([ebcb30c](https://github.com/spaceml-org/georeader/commit/ebcb30cfe149eb905c90f2c2ce60170b82fc02df))

## [1.5.11](https://github.com/spaceml-org/georeader/compare/v1.5.10...v1.5.11) (2026-01-27)


### Bug Fixes

* added missing dependency and fixed broken test. ([570926b](https://github.com/spaceml-org/georeader/commit/570926b4797688dea3474614edd682f00060c702))
* added more test coverage and add missing dependency ([4dbd142](https://github.com/spaceml-org/georeader/commit/4dbd1420f001742c74466473c8ea1ed21dc16bd6))
* regenerated poetry lock file for scikit-image inclusion ([7b21e28](https://github.com/spaceml-org/georeader/commit/7b21e28afd6f2fad0053ed8dd403cf743c2dfb04))

## [1.5.10](https://github.com/spaceml-org/georeader/compare/v1.5.9...v1.5.10) (2026-01-26)


### Bug Fixes

* force release for the georeader package. ([821f8fe](https://github.com/spaceml-org/georeader/commit/821f8fe429131a791fc68ae25751dd379c38b30a))

## Refactor

* Updated the EMIT to use the xarray package instead of the native netcdf
