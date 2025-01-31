## Google Earth Engine functions

We have implemented functions to query and export arbitrarly large images from the Google Earth Engine. 
Functions to export images or cubes are in module `georeader.readers.ee_image` and functions to query Sentinel-1, Sentinel-2 and Landsat are
available in `georeader.readers.ee_query`.

::: georeader.readers.ee_image
    options:
      members:
        - export_image
        - export_cube

::: georeader.readers.ee_query
    options:
      members:
        - query
        - query_s1
        - query_landsat_457
