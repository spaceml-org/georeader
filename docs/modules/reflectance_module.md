# georeader.reflectance

This module provides functions to convert between radiance and top-of-atmosphere (ToA) reflectance for satellite data, as well as to integrate hyperspectral data to multispectral bands using spectral response functions (SRFs).

## Overview

These conversions are essential for:

- Harmonizing data from different sensors
- Scientific analysis requiring physical units
- Cross-sensor comparisons and data fusion

## Quick Start

```python
from georeader import reflectance
import numpy as np

# Convert radiance to ToA reflectance
# Requires sun-earth distance and solar zenith angle
toa_reflectance = reflectance.radiance_to_reflectance(
    radiance_data,
    solar_irradiance=1360.8,  # W/m²
    sun_earth_distance=1.0,    # AU
    solar_zenith_angle=30.0    # degrees
)

# Integrate hyperspectral bands to multispectral using SRF
# Useful for comparing hyperspectral (EMIT, PRISMA, EnMAP) with Sentinel-2
integrated = reflectance.integrate_srf(
    hyperspectral_data,
    wavelengths,
    srf_wavelengths,
    srf_response
)
```

## Related Tutorials

- **EMIT, PRISMA, and EnMAP: Convert radiance to ToA reflectance**
  - [EMIT: Work with EMIT images and convert to reflectance](../emit_explore.ipynb) ([EMIT Reader docs](./readers_module.md#emit-reader))
  - [PRISMA & EnMAP: Integrate hyperspectral bands and convert to reflectance](../enmap_with_cloudsen12.ipynb), [PRISMA cloud detection](../prisma_with_cloudsen12.ipynb) ([PRISMA Reader docs](./readers_module.md#prisma-reader), [EnMAP Reader docs](./readers_module.md#enmap-reader))

- **Radiance to ToA reflectance plus integration of hyperspectral bands**
  - [PRISMA and EnMAP with CloudSEN12: Integrate and convert to reflectance](../enmap_with_cloudsen12.ipynb), [PRISMA cloud detection](../prisma_with_cloudsen12.ipynb)

- **Sentinel-2: Convert ToA reflectance to radiance**
  - [Sentinel-2: Convert ToA reflectance to radiance](../Sentinel-2/convert_to_radiance.ipynb)

---

::: georeader.reflectance