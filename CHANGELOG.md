# Changelog

## [2.3.5](https://github.com/spaceml-org/georeader/compare/v2.3.4...v2.3.5) (2026-07-06)


### Bug Fixes

* **readers:** bundle S2 spectral response function (SRF) document as package data so `read_srf` works offline ([6067514](https://github.com/spaceml-org/georeader/commit/60675148c62f677fad0f3123c1a9b22eef89f339)), closes [#72](https://github.com/spaceml-org/georeader/issues/72)

## [2.3.4](https://github.com/spaceml-org/georeader/compare/v2.3.3...v2.3.4) (2026-07-01)


### Bug Fixes

* **ee_query:** drop degenerate whole-globe footprints ([#69](https://github.com/spaceml-org/georeader/issues/69)) ([e57e41f](https://github.com/spaceml-org/georeader/commit/e57e41ff58fb5e9354c9826417416ca95ae20845))


### Documentation

* clean up CHANGELOG ([#70](https://github.com/spaceml-org/georeader/issues/70)) ([be7e2ce](https://github.com/spaceml-org/georeader/commit/be7e2ce52fc810af637659acdf020eebc17e0780))

## [2.3.3](https://github.com/spaceml-org/georeader/compare/v2.3.2...v2.3.3) (2026-06-24)


### Features

* **rasterio_reader:** mask Azure SAS token signature in __repr__ ([#66](https://github.com/spaceml-org/georeader/issues/66)) ([4f32d81](https://github.com/spaceml-org/georeader/commit/4f32d81ee51b986891354e2183fe168f778015a2))

## [2.3.2](https://github.com/spaceml-org/georeader/compare/v2.3.1...v2.3.2) (2026-06-12)


### Features

* **enmap:** add opt-in radiance cache to EnMAP reader ([#62](https://github.com/spaceml-org/georeader/issues/62)) ([84e847a](https://github.com/spaceml-org/georeader/commit/84e847a91c20822487792d5e2c970eeef13ef6b8))

## [2.3.1](https://github.com/spaceml-org/georeader/compare/v2.3.0...v2.3.1) (2026-06-05)


### Bug Fixes

* turn notebooks into skippable integration tests and fix bugs of GeoTensor 2.0 version on setters ([#59](https://github.com/spaceml-org/georeader/issues/59)) ([122705c](https://github.com/spaceml-org/georeader/commit/122705c0a43599f97fba909cfaa2fcd53d9acb8e))

## [2.3.0](https://github.com/spaceml-org/georeader/compare/v2.2.0...v2.3.0) (2026-05-29)


### Features

* **emit:** fix EMITImage clone propagation; add slice push-down + opt-in radiance cache ([#56](https://github.com/spaceml-org/georeader/issues/56)) ([4fd1018](https://github.com/spaceml-org/georeader/commit/4fd101811dc921dacd348597409183e23d0e4038))

## [2.2.0](https://github.com/spaceml-org/georeader/compare/v2.1.0...v2.2.0) (2026-05-14)


### Features

* **readers:** add Carbon Mapper reader subpackage ([158e57f](https://github.com/spaceml-org/georeader/commit/158e57f03b0cde11d93478611a9ce73e367b3dd6))

## [2.1.0](https://github.com/spaceml-org/georeader/compare/v1.6.0...v2.1.0) (2026-05-14)


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
