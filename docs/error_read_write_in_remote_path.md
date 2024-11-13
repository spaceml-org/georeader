# Reading from remote location: GDAL VSIL caching issue

Reading a raster from a remote location (e.g. Google, Amazon or Azure bucket) fails if the file is read and modified by other (non-GDAL) process. This happens even if the file is closed after reading and then modified. This is because [GDAL has a global cache when reading data from remote locations](https://gdal.org/en/latest/user/configoptions.html#networking-options). This notebook shows this problem and a fix implemented in `RasterioReader` to skip the caching (called `read_with_CPL_VSIL_CURL_NON_CACHED`). A direct fix in GDAL would be to call [`VSICurlClearCache`](https://gdal.org/en/latest/doxygen/cpl__vsi_8h.html#a6b22260317edc475793c4165957742b6) or [`VSICurlPartialClearCache`](https://gdal.org/en/latest/doxygen/cpl__vsi_8h.html#a6bc83d16f0f279f601059a218ad2c55c) C functions; however these are not mapped in `rasterio`.


```python
import os
import fsspec

os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "????"
fs = fsspec.filesystem("az")
fs
```




    <adlfs.spec.AzureBlobFileSystem at 0x7fdf4429a020>



This example will read a 3-band rgb COG and modify it by a one band COG with a different shape. We will see that when the COG is modified we can't read the file again if we don't set the `read_with_CPL_VSIL_CURL_NON_CACHED` option. We will see either a read error (if we haven't loaded the data) or we will see the old cached data (if we have load the data).


```python
# rasterio_reader.RIO_ENV_OPTIONS_DEFAULT["CPL_CURL_VERBOSE"] = "YES"
rgb_file = "az://mycontainer/rgb.tif"
one_band_file = "az://mycontainer/one_band.tif"
assert fs.exists(rgb_file)
assert fs.exists(one_band_file)

filepath = "az://mycontainer/removeme.tif"
if fs.exists(filepath):
    fs.delete(filepath)
```


```python
from georeader.rasterio_reader import RasterioReader

fs.copy(rgb_file, filepath)
rst = RasterioReader(filepath)
rst
```




     
             Paths: ['az://mycontainer/removeme.tif']
             Transform: | 30.00, 0.00, 740744.72|
    | 0.00,-30.00, 4287081.96|
    | 0.00, 0.00, 1.00|
             Shape: (3, 1176, 1168)
             Resolution: (30.0, 30.0)
             Bounds: (740744.717767204, 4251801.958449183, 775784.717767204, 4287081.958449183)
             CRS: EPSG:32641
             nodata: 0.0
             fill_value_default: 0.0
            



Reading now fails (we haven't loaded the data).


```python
fs.copy(one_band_file, filepath)
rst = RasterioReader(filepath)
data_rst = rst.load()
data_rst
```


    ---------------------------------------------------------------------------

    CPLE_AppDefinedError                      Traceback (most recent call last)

    CPLE_AppDefinedError: LZWDecode:Corrupted LZW table at scanline 0

    
    The above exception was the direct cause of the following exception:


    CPLE_AppDefinedError                      Traceback (most recent call last)

    CPLE_AppDefinedError: TIFFReadEncodedTile() failed.

    
    The above exception was the direct cause of the following exception:


    CPLE_AppDefinedError                      Traceback (most recent call last)

    File rasterio/_io.pyx:969, in rasterio._io.DatasetReaderBase._read()


    File rasterio/_io.pyx:199, in rasterio._io.io_multi_band()


    File rasterio/_io.pyx:205, in rasterio._io.io_multi_band()


    File rasterio/_err.pyx:325, in rasterio._err.StackChecker.exc_wrap_int()


    CPLE_AppDefinedError: removeme.tif, band 1: IReadBlock failed at X offset 0, Y offset 0: TIFFReadEncodedTile() failed.

    
    The above exception was the direct cause of the following exception:


    RasterioIOError                           Traceback (most recent call last)

    Cell In[4], line 3
          1 fs.copy(one_band_file, filepath)
          2 rst = RasterioReader(filepath)
    ----> 3 data_rst = rst.load()
          4 data_rst


    File ~/git/georeader/georeader/rasterio_reader.py:554, in RasterioReader.load(self, boundless)
        546 def load(self, boundless:bool=True) -> geotensor.GeoTensor:
        547     """
        548     Load all raster in memory in an GeoTensor object
        549 
       (...)
        552 
        553     """
    --> 554     np_data = self.read(boundless=boundless)
        555     if boundless:
        556         transform = self.transform


    File ~/git/georeader/georeader/rasterio_reader.py:704, in RasterioReader.read(self, **kwargs)
        699 for i, p in enumerate(self.paths):
        700     # with rasterio.Env(**options):
        701     #     with rasterio.open(p, "r", overview_level=self.overview_level) as src:
        702     with self._rio_open(p, overview_level=self.overview_level) as src:
        703         # rasterio.read API: https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader.read
    --> 704         read_data = src.read(**kwargs)
        706         # Add pad when reading
        707         if pad is not None and need_pad:


    File rasterio/_io.pyx:644, in rasterio._io.DatasetReaderBase.read()


    File rasterio/_io.pyx:972, in rasterio._io.DatasetReaderBase._read()


    RasterioIOError: Read failed. See previous exception for details.



```python
rst.rio_env_options
```




    {'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
     'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES',
     'GDAL_CACHEMAX': 2000000000,
     'GDAL_HTTP_MULTIPLEX': 'YES'}




```python
fs.delete(filepath)
```

Reading and loading the data will return the old data instead of the new one. (The rgb raster had 3 bands and the other one only one).


```python
filepath = "az://mycontainer/removeme2.tif"
if fs.exists(filepath):
    fs.delete(filepath)

fs.copy(rgb_file, filepath)
rst_mem = RasterioReader(filepath).load()
rst_mem
```




     
             Transform: | 30.00, 0.00, 740744.72|
    | 0.00,-30.00, 4287081.96|
    | 0.00, 0.00, 1.00|
             Shape: (3, 1176, 1168)
             Resolution: (30.0, 30.0)
             Bounds: (740744.717767204, 4251801.958449183, 775784.717767204, 4287081.958449183)
             CRS: EPSG:32641
             fill_value_default: 0.0
            




```python
fs.copy(one_band_file, filepath)
rst = RasterioReader(filepath)
data_rst = rst.load()
data_rst
```




     
             Transform: | 30.00, 0.00, 740744.72|
    | 0.00,-30.00, 4287081.96|
    | 0.00, 0.00, 1.00|
             Shape: (3, 1176, 1168)
             Resolution: (30.0, 30.0)
             Bounds: (740744.717767204, 4251801.958449183, 775784.717767204, 4287081.958449183)
             CRS: EPSG:32641
             fill_value_default: 0.0
            




```python
fs.delete(filepath)
```

Adding option  `read_with_CPL_VSIL_CURL_NON_CACHED` fixes the problem, we see that after the second copy the raster has 1 channel instead of 3.


```python
from georeader import rasterio_reader
rasterio_reader.RIO_ENV_OPTIONS_DEFAULT["read_with_CPL_VSIL_CURL_NON_CACHED"] = True
```


```python
filepath = "az://mycontainer/removeme3.tif"
if fs.exists(filepath):
    fs.delete(filepath)

fs.copy(rgb_file, filepath)
rst_mem = RasterioReader(filepath).load()
rst_mem
```




     
             Transform: | 30.00, 0.00, 740744.72|
    | 0.00,-30.00, 4287081.96|
    | 0.00, 0.00, 1.00|
             Shape: (3, 1176, 1168)
             Resolution: (30.0, 30.0)
             Bounds: (740744.717767204, 4251801.958449183, 775784.717767204, 4287081.958449183)
             CRS: EPSG:32641
             fill_value_default: 0.0
            




```python
fs.copy(one_band_file, filepath)
rst = RasterioReader(filepath)
data_rst = rst.load()
data_rst
```




     
             Transform: | 30.00, 0.00, 255639.31|
    | 0.00,-30.00, 3165851.00|
    | 0.00, 0.00, 1.00|
             Shape: (1, 1262, 1369)
             Resolution: (30.0, 30.0)
             Bounds: (255639.3130302397, 3127990.9952690094, 296709.31303023966, 3165850.9952690094)
             CRS: EPSG:32640
             fill_value_default: -1.0
            




```python
fs.delete(filepath)
```

If we don't load the data we don't have the error and the file is correctly readed:


```python
filepath = "az://mycontainer/removeme3.tif"
if fs.exists(filepath):
    fs.delete(filepath)

fs.copy(rgb_file, filepath)
rst = RasterioReader(filepath)
rst
```




     
             Paths: ['az://mycontainer/removeme3.tif']
             Transform: | 30.00, 0.00, 740744.72|
    | 0.00,-30.00, 4287081.96|
    | 0.00, 0.00, 1.00|
             Shape: (3, 1176, 1168)
             Resolution: (30.0, 30.0)
             Bounds: (740744.717767204, 4251801.958449183, 775784.717767204, 4287081.958449183)
             CRS: EPSG:32641
             nodata: 0.0
             fill_value_default: 0.0
            




```python
fs.copy(one_band_file, filepath)
rst = RasterioReader(filepath)
data_rst = rst.load()
data_rst
```




     
             Transform: | 30.00, 0.00, 255639.31|
    | 0.00,-30.00, 3165851.00|
    | 0.00, 0.00, 1.00|
             Shape: (1, 1262, 1369)
             Resolution: (30.0, 30.0)
             Bounds: (255639.3130302397, 3127990.9952690094, 296709.31303023966, 3165850.9952690094)
             CRS: EPSG:32640
             fill_value_default: -1.0
            
```python
rst.rio_env_options
```




    {'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
     'GDAL_HTTP_MERGE_CONSECUTIVE_RANGES': 'YES',
     'GDAL_CACHEMAX': 2000000000,
     'GDAL_HTTP_MULTIPLEX': 'YES',
     'read_with_CPL_VSIL_CURL_NON_CACHED': True}




