# Changelog

## [2.0.0](https://github.com/spaceml-org/georeader/compare/v1.5.12...v1.6.0) (2026-05-14)


### Features

* version 2.0 geotensor to implement numpy API ([#21](https://github.com/spaceml-org/georeader/issues/21)) ([00e46f8](https://github.com/spaceml-org/georeader/commit/00e46f80bd812e81e2f6c6b5db86d234ec81bcfd))

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
