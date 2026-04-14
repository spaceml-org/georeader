# RasterioReader

Lazy-loading implementation of the [GeoData protocol](../modules/read_module.md) backed by `rasterio`. Supports reading from local files and cloud storage (GCS, S3, Azure).

**Tutorials:**

- [Reading Sentinel-2 from the public bucket](../read_S2_SAFE_from_bucket.ipynb)
- [Reading overlapping rasters](../read_overlapping_probav_and_sentinel2.ipynb)
- [VSIL cache problem](../advanced/error_read_write_in_remote_path.md)

::: georeader.rasterio_reader
