# georeader

Package to read data from rasters that is thread and process save, lightweight and with lazy loading.

WIP

## TODOs
 * Fix and run tests.

## Potential features

* Readers of standard format for Proba-V, Sentinel-2, Landsat-8.
* `read_tile` function + example of serving images.
* `xarray` wrapper (cast function) (`GeoDataArray`)
* Add extent polygon (out of that area values are invalids)?
* Add `matplotlib` plotting functions.
* Read boundless for non-rectilinear transforms.

Examples folder:
* Tutorial basic usage
* Example of serving images
* Example of building a torch Dataset
* Example of reading from Google Bucket collections


