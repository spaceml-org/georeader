# georeader

Package to read data from rasters that is thread and process save, lightweight and with lazy loading.

## Install

```bash
git clone 
cd georeader
pip install -e .
```
This package is work in progress. The API might change without notice. Use it with care.

## TODOs
 * Fix and run tests.
 * ml  `xarray` wrapper (cast function) (finish `GeoDataArray` class)
 * `GeoTensor.resize` with `kornia` if inner tensor is a `torch.Tensor`.

## Potential features

* `read_tile` function + example of serving images.
* Add extent polygon (out of that area values are invalids)?
* Add `matplotlib` plotting functions.
* Read boundless for non-rectilinear transforms. [Required to read AVIRIS data!]
* Readers of standard format for Landsat-8?

## Examples

* Tutorial basic usage
* Example of serving images
* Example of building a torch Dataset
* Example of reading from Google Bucket collections


