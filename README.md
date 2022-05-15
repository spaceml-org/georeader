# georeader

Package to read data from rasters that is thread and process save, lightweight and with lazy loading.

WIP

## TODOs
 * Fix and run tests.
 * `xarray` wrapper (cast function) (`GeoDataArray`)

## Potential features

* Readers of standard format for Sentinel-2 and Landsat-8.
* `read_tile` function + example of serving images.
* Add extent polygon (out of that area values are invalids)?
* Add `matplotlib` plotting functions.
* Read boundless for non-rectilinear transforms. [Required to read AVIRIS data!]

## Examples

* Tutorial basic usage
* Example of serving images
* Example of building a torch Dataset
* Example of reading from Google Bucket collections


