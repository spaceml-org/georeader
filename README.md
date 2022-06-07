# georeader

Package to read data from rasters with very few dependencies, compatible with cloud platforms and with lazy loading.

## Install

```bash
git clone 
cd georeader
pip install -e .
```

## Getting started

```python
from georeader.rasterio_reader import RasterioReader



```
This package is work in progress. The API might change without notice. Use it with care.

## TODOs
 * Fix and run tests.
 * Finish `xarray` wrapper (cast function) (finish `GeoDataArray` class)
 * `GeoTensor.resize` with `kornia` if inner tensor is a `torch.Tensor`.

## Potential features

* `read_tile` function + example of serving images.
* Add `matplotlib` plotting functions.
* Readers of standard format for Landsat-8? Read from s3 Landsat bucket?

## Examples

* Tutorial basic usage
* Show MISR dataset
* Example of serving images
* Example of building a torch Dataset


