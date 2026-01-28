# georeader.reflectance

This module provides functions to convert between radiance and top-of-atmosphere (ToA) reflectance for satellite data, as well as to integrate hyperspectral data to multispectral bands using spectral response functions (SRFs).

These conversions are essential for harmonizing data from different sensors and for scientific analysis. The functions here are used in several tutorials and workflows, including:

- **EMIT, PRISMA, and EnMAP: Convert radiance to ToA reflectance**
  - [EMIT: Work with EMIT images and convert to reflectance](../emit_explore.ipynb) ([EMIT Reader docs](./readers_module.md#emit-reader))
  - [PRISMA & EnMAP: Integrate hyperspectral bands and convert to reflectance](../enmap_with_cloudsen12.ipynb), [PRISMA cloud detection](../prisma_with_cloudsen12.ipynb) ([PRISMA Reader docs](./readers_module.md#prisma-reader), [EnMAP Reader docs](./readers_module.md#enmap-reader))

- **Radiance to ToA reflectance plus integration of hyperspectral bands**
  - [PRISMA and EnMAP with CloudSEN12: Integrate and convert to reflectance](../enmap_with_cloudsen12.ipynb), [PRISMA cloud detection](../prisma_with_cloudsen12.ipynb)

- **Sentinel-2: Convert ToA reflectance to radiance**
  - [Sentinel-2: Convert ToA reflectance to radiance](../Sentinel-2/convert_to_radiance.ipynb)

---

::: georeader.reflectance