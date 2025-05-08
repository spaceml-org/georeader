[![Article DOI:10.1038/s41598-023-47595-7](https://img.shields.io/badge/Article%20DOI-10.1038%2Fs41598.023.47595.7-blue)](https://doi.org/10.1038/s41598-023-47595-7)  [![GitHub release (latest SemVer including pre-releases)](https://img.shields.io/github/v/release/spaceml-org/georeader?sort=semver)](https://github.com/spaceml-org/georeader/releases) [![PyPI](https://img.shields.io/pypi/v/georeader-spaceml)](https://pypi.org/project/georeader-spaceml/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/georeader-spaceml)](https://pypi.org/project/georeader-spaceml/) [![PyPI - License](https://img.shields.io/pypi/l/georeader-spaceml)](https://github.com/spaceml-org/georeader/blob/main/LICENSE) [![docs](https://badgen.net/badge/docs/spaceml-org.github.io%2Fgeoreader/blue)](https://spaceml-org.github.io/georeader/)

# <img src="https://raw.githubusercontent.com/spaceml-org/georeader/main/docs/images/logo_georeader.png" alt="Logo" width='7%'>  georeader

**georeader** is a package to process raster data from different satellite missions. **georeader** makes easy to [read specific areas of your image](https://github.com/spaceml-org/georeader/blob/main/docs/read_S2_SAFE_from_bucket.ipynb), to [reproject images from different satellites to a common grid](https://github.com/spaceml-org/georeader/blob/main/docs/reading_overlapping_sentinel2_aviris.ipynb)  ([`georeader.read`](https://spaceml-org.github.io/georeader/modules/read_module/)), to go from vector to raster formats ([`georeader.vectorize`](https://spaceml-org.github.io/georeader/modules/vectorize_module/) and [`georeader.rasterize`](https://spaceml-org.github.io/georeader/modules/rasterize_module/)) or to do [radiance to reflectance conversions](https://spaceml-org.github.io/georeader/enmap_with_cloudsen12/) ([`georeader.reflectance`](https://spaceml-org.github.io/georeader/modules/reflectance_module/)). 

**georeader** is mainly used to process satellite data for scientific usage, to create ML-ready datasets and to implement *end-to-end* operational inference pipelines ([e.g. the Kherson Dam Break floodmap](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_postprocess_inference.html)). See [**georeader** concepts and protocols for basic concepts and API](https://spaceml-org.github.io/georeader/modules/read_module/).

## Install

The core package dependencies are `numpy`, `rasterio`, `shapely` and `geopandas`.

```bash
pip install georeader-spaceml
```

## Getting started

> Read from a Sentinel-2 image a fixed size subimage on an specific `lon,lat` location:
 
```python
from georeader.rasterio_reader import RasterioReader
from georeader import read

# S2 image from WorldFloodsv2 dataset
s2url = "https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2/resolve/main/test/S2/EMSR264_18MIANDRIVAZODETAIL_DEL_v2.tif"
rst = RasterioReader(s2url)

# lazy loading bands
rst_rgb = rst.isel({"band": [3, 2, 1]}) # 1-based list as in rasterio

cords_read = (45.43, -19.53) # long, lat
crs_cords = "EPSG:4326"

# See also read.read_from_bounds, read.read_from_polygon for different ways of croping an image
data = read.read_from_center_coords(rst_rgb,
                                    cords_read, shape=(504, 1040),
                                    crs_center_coords=crs_cords)

data_memory = data.load() # this loads the data to memory

data_memory # GeoTensor object
```
```
>>  Transform: | 10.00, 0.00, 539910.00|
| 0.00,-10.00, 7842990.00|
| 0.00, 0.00, 1.00|
         Shape: (3, 504, 1040)
         Resolution: (10.0, 10.0)
         Bounds: (539910.0, 7837950.0, 550310.0, 7842990.0)
         CRS: EPSG:32738
         fill_value_default: 0
```

```python
from georeader import plot
plot.show((data_memory / 3_500).clip(0, 1))

```
<img src="https://raw.githubusercontent.com/spaceml-org/georeader/main/docs/modules/pngs/sample_read.png" alt="awesome georeader" width="50%">


Saving the `GeoTensor` as a COG GeoTIFF: 

```python
from georeader.save import save_cog

# Supports writing in remote location (e.g. gs://bucket-name/s2_crop.tif)
save_cog(data_memory, "s2_crop.tif", descriptions=["B4","B3", "B2"])
```

## Tutorials

### Sentinel-2

- [Reading Sentinel-2 images from the public Google bucket](https://github.com/spaceml-org/georeader/blob/main/docs/read_S2_SAFE_from_bucket.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/georeader/blob/main/docs/read_S2_SAFE_from_bucket.ipynb)
- [Tiling and stitching predictions of an AI model](https://github.com/spaceml-org/georeader/blob/main/docs/advanced/tiling_and_stitching.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/georeader/blob/main/docs/advanced/tiling_and_stitching.ipynb)
- [Explore metadata of Sentinel-2 object](https://github.com/spaceml-org/georeader/blob/main/docs/Sentinel-2/explore_metadata_s2.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/georeader/blob/main/docs/Sentinel-2/explore_metadata_s2.ipynb)
- [Query Sentinel-2 images over a location and time span, mosaic and plot them](https://github.com/spaceml-org/georeader/blob/main/docs/Sentinel-2/query_mosaic_s2_images.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/georeader/blob/main/docs/Sentinel-2/query_mosaic_s2_images.ipynb)
- [Sentinel-2 images from GEE and CloudSEN12 cloud detection](https://github.com/spaceml-org/georeader/blob/main/docs/Sentinel-2/run_in_gee_image.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/georeader/blob/main/docs/Sentinel-2/run_in_gee_image.ipynb)

### Read rasters from different satellites

 * [Tutorial to read overlapping tiles from a GeoTIFF and a Sentinel-2 image](https://github.com/spaceml-org/georeader/blob/main/docs/reading_overlapping_sentinel2_aviris.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/georeader/blob/main/docs/reading_overlapping_sentinel2_aviris.ipynb)
 * [Example of reading a Proba-V image overlapping with Sentinel-2 forcing same resolution](https://github.com/spaceml-org/georeader/blob/main/docs/read_overlapping_probav_and_sentinel2.ipynb)
 * [Work with EMIT images](https://github.com/spaceml-org/georeader/blob/main/docs/emit_explore.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spaceml-org/georeader/blob/main/docs/emit_explore.ipynb)
 * [Read overlapping images of PRISMA and EMIT](https://github.com/spaceml-org/georeader/blob/main/docs/simultaneous_prisma_emit.ipynb) 
 * [Read EnMAP images, integrate them to Sentinel-2 bands, convert radiance to TOA reflectance and run CloudSEN12 cloud detection model](https://github.com/spaceml-org/georeader/blob/main/docs/enmap_with_cloudsen12.ipynb) 

### Used in other projects

 * [georeader with ml4floods to automatically download and produce flood extent maps: the Kherson Dam Break example](https://spaceml-org.github.io/ml4floods/content/ml4ops/HOWTO_postprocess_inference.html)
 * [georeader with STARCOP to simulate Sentinel-2 from AVIRIS images](https://github.com/spaceml-org/STARCOP/blob/main/notebooks/simulate_aviris_2_sentinel2.ipynb)
 * [georeader with STARCOP to run plume detection in EMIT images](https://github.com/spaceml-org/STARCOP/blob/main/notebooks/inference_on_raw_EMIT_nc_file.ipynb)
 * [georeader with CloudSEN12 to run cloud detection in Sentinel-2 images](https://github.com/IPL-UV/cloudsen12_models/blob/main/notebooks/run_in_gee_image.ipynb)


## Citation

If you find this code useful please cite:
```
@article{portales-julia_global_2023,
	title = {Global flood extent segmentation in optical satellite images},
	volume = {13},
	issn = {2045-2322},
	doi = {10.1038/s41598-023-47595-7},
	number = {1},
	urldate = {2023-11-30},
	journal = {Scientific Reports},
	author = {Portalés-Julià, Enrique and Mateo-García, Gonzalo and Purcell, Cormac and Gómez-Chova, Luis},
	month = nov,
	year = {2023},
	pages = {20316},
}
@article{ruzicka_starcop_2023,
	title = {Semantic segmentation of methane plumes with hyperspectral machine learning models},
	volume = {13},
	issn = {2045-2322},
	url = {https://www.nature.com/articles/s41598-023-44918-6},
	doi = {10.1038/s41598-023-44918-6},
	number = {1},
	journal = {Scientific Reports},
	author = {Růžička, Vít and Mateo-Garcia, Gonzalo and Gómez-Chova, Luis and Vaughan, Anna, and Guanter, Luis and Markham, Andrew},
	month = nov,
	year = {2023},
	pages = {19999},
}
```

## Acknowledgments
This research has been supported by the DEEPCLOUD project (PID2019-109026RB-I00) funded by the Spanish Ministry of Science and Innovation (MCIN/AEI/10.13039/501100011033) and the European Union (NextGenerationEU).

<img width="300" title="DEEPCLOUD project (PID2019-109026RB-I00, University of Valencia) funded by MCIN/AEI/10.13039/501100011033." alt="DEEPCLOUD project (PID2019-109026RB-I00, University of Valencia) funded by MCIN/AEI/10.13039/501100011033." src="https://www.uv.es/chovago/logos/logoMICIN.jpg">
