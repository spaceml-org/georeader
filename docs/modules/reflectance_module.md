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
from datetime import date

# Convert radiance to ToA reflectance
# Requires per-band solar_irradiance plus either:
#   - date_of_acquisition and center_coords, or
#   - observation_date_corr_factor
toa_reflectance = reflectance.radiance_to_reflectance(
    radiance_data,
    solar_irradiance=np.array([1360.8]),  # per-band W/m² (example with a single band)
    date_of_acquisition=date(2023, 7, 10),
    center_coords=(12.34, 56.78),  # (longitude, latitude) of the scene center
)

# Integrate hyperspectral bands to multispectral using an SRF
# Useful for comparing hyperspectral (EMIT, PRISMA, EnMAP) with Sentinel-2
s2_srf = reflectance.srf(
    srf_wavelengths,
    srf_response
)
integrated = reflectance.transform_to_srf(
    hyperspectral_data,
    wavelengths,
    s2_srf
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