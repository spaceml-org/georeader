# Satellite Data Readers

This module provides specialized readers for various optical satellite missions. All these readers implement the [GeoData protocol](../modules/read_module.md), which means they provide a consistent interface for spatial operations, data access, and manipulation.

These readers make it easy to work with official data formats from different Earth observation missions, and they can be used with all the functions available in the `georeader.read` module.

Readers available:

* [Sentinel-2](#sentinel-2-reader)
* [Proba-V](#proba-v-reader)
* [SpotVGT](#spot-vgt-reader)
* [EMIT](#emit-reader)
* [PRISMA](#prisma-reader)
* [EnMAP](#enmap-reader)

## Sentinel-2 Reader

The Sentinel-2 reader provides functionality for reading Sentinel-2 L1C and L2A products in SAFE format. It supports:

- Direct reading from local files or cloud storage (Google Cloud Storage)
- Windowed reading for efficient memory usage
- Conversion from digital numbers to radiance
- Access to metadata, including viewing geometry and solar angles

**Tutorial examples:**

- [Reading from the public Google bucket](../read_S2_SAFE_from_bucket.ipynb)
- [Exploring image metadata](../Sentinel-2/explore_metadata_s2.ipynb)
- [Creating mosaics from multiple images](../Sentinel-2/query_mosaic_s2_images.ipynb)
- [Converting TOA reflectance to radiance](../Sentinel-2/convert_to_radiance.ipynb)

### API Reference

::: georeader.readers.S2_SAFE_reader
    options:
      members:
        - S2Image
        - S2ImageL1C
        - S2ImageL2A
        - s2loader
        - s2_public_bucket_path
        - read_srf

## Proba-V Reader

The Proba-V reader enables access to Proba-V Level 2A and Level 3 products. It handles:

- Reading TOA reflectance from HDF5 files
- Mask handling for clouds, shadows, and invalid pixels
- Extraction of metadata and acquisition parameters

**Tutorial example:**

- [Reading overlapping Proba-V and Sentinel-2 images](../read_overlapping_probav_and_sentinel2.ipynb)

### API Reference

::: georeader.readers.probav_image_operational
    options:
      members:
        - ProbaV
        - ProbaVRadiometry
        - ProbaVSM

## SPOT-VGT Reader

The SPOT-VGT reader provides functionality for reading SPOT-VGT products. Features include:

- HDF4 file format support
- Handling of radiometry and quality layers
- Cloud and shadow mask extraction

**Note:** See the Proba-V tutorial for similar processing workflows as both sensors share similar data structures.

### API Reference

::: georeader.readers.spotvgt_image_operational
    options:
      members:
        - SpotVGT

## PRISMA Reader

The PRISMA reader handles data from the Italian Space Agency's hyperspectral mission, specifically working with Level 1B radiance data (not atmospherically corrected). PRISMA provides hyperspectral imaging in the 400-2500 nm spectral range, with a spectral resolution of ~12 nm.

Key features:

- Reading L1B hyperspectral radiance data from HDF5 format files
- Handling separate VNIR (400-1000 nm) and SWIR (1000-2500 nm) spectral ranges
- Georeferencing functionality for non-orthorectified data using provided latitude/longitude coordinates
- On-demand conversion from radiance (mW/m²/sr/nm) to top-of-atmosphere reflectance
- Spectral response function integration for accurate band simulation
- Extraction of RGB previews from specific wavelengths
- Access to satellite and solar geometry information for radiometric calculations

**Tutorial examples:**

- [Reading overlapping PRISMA and EMIT images](../simultaneous_prisma_emit.ipynb)
- [Cloud detection in PRISMA images](../prisma_with_cloudsen12.ipynb)

### API Reference

::: georeader.readers.prisma
    options:
      members:
        - PRISMA

## EMIT Reader

The EMIT (Earth Surface Mineral Dust Source Investigation) reader provides access to NASA's imaging spectrometer data from the International Space Station. This reader works with Level 1B calibrated radiance data (not atmospherically corrected).

Key features:

- Reading L1B hyperspectral radiance data from NetCDF4 format files
- Working with the 380-2500 nm spectral range with 7.4 nm sampling
- Irregular grid georeferencing through GLT (Geographic Lookup Table)
- Support for the observation geometry information (solar and viewing angles)
- Integration with L2A mask products for cloud and shadow detection
- Quality-aware analysis with cloud, cirrus, and spacecraft flag masks
- Conversion from radiance (μW/cm²/sr/nm) to top-of-atmosphere reflectance
- Support for downloading data from NASA DAAC portals
- Automatic detection and use of appropriate UTM projection

**Tutorial example:**

- [Working with EMIT images](../emit_explore.ipynb)

### API Reference

::: georeader.readers.emit
    options:
      members:
        - EMITImage
        - download_product
        - get_radiance_link
        - get_obs_link
        - get_ch4enhancement_link
        - get_l2amask_link
        - valid_mask

## EnMAP Reader

The EnMAP (Environmental Mapping and Analysis Program) reader processes data from the German hyperspectral satellite mission. This reader works with Level 1B radiometrically calibrated data (not atmospherically corrected) that contains radiance values in physical units.

Key features:

- Reading L1B hyperspectral radiance data from GeoTIFF format with accompanying XML metadata
- Working with separate VNIR (420-1000 nm) and SWIR (900-2450 nm) spectral ranges
- Support for 228 spectral channels with 6.5 nm (VNIR) and 10 nm (SWIR) sampling
- Integration with Rational Polynomial Coefficients (RPCs) for accurate geometric correction
- Conversion from radiance (mW/m²/sr/nm) to top-of-atmosphere reflectance
- Access to solar illumination and viewing geometry for radiometric calculations
- Support for quality masks

**Tutorial example:**

- [Working with EnMAP and CloudSEN12](../enmap_with_cloudsen12.ipynb)

### API Reference

::: georeader.readers.enmap
    options:
      members:
        - EnMAP
