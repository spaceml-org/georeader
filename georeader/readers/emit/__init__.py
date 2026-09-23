"""
Module to read EMIT (Earth Surface Mineral Dust Source Investigation) hyperspectral images.

EMIT is a NASA imaging spectrometer aboard the International Space Station that measures
reflected solar radiation from Earth's surface in 285 spectral bands from 380 to 2500 nm.
This module provides tools to read, georeference, and process EMIT L1B radiance data.

Data Format Overview
--------------------
EMIT data is distributed in NetCDF format with a unique storage layout:

    Raw Data Structure (NetCDF file):
    ┌─────────────────────────────────────┐
    │  radiance: (downtrack, crosstrack, bands)  │
    │  └── Shape: (~1280, ~1242, 285)            │
    │                                             │
    │  location/glt_x: (rows, cols)              │
    │  location/glt_y: (rows, cols)              │
    │  └── Geographic Lookup Table (GLT)         │
    └─────────────────────────────────────┘

The raw data is stored in *sensor coordinates* (pushbroom scan lines), NOT in 
geographic coordinates. The GLT provides a mapping from geographic (orthorectified)
coordinates back to raw sensor coordinates.

GLT Orthorectification Process
------------------------------
The GLT (Geographic Lookup Table) is key to understanding EMIT data:

    Geographic Grid (Output)          Sensor Grid (Raw Data)
    ┌─────────────────────┐           ┌─────────────────────┐
    │ (0,0)               │           │ radiance array      │
    │   ┌───┬───┬───┐     │   GLT     │ ┌───────────────┐   │
    │   │ a │ b │ c │     │ ──────→   │ │ (5,2) (5,3)   │   │
    │   ├───┼───┼───┤     │ lookup    │ │ (6,1) (6,2)   │   │
    │   │ d │ e │ f │     │           │ │ ...           │   │
    │   └───┴───┴───┘     │           │ └───────────────┘   │
    │               (H,W) │           │                     │
    └─────────────────────┘           └─────────────────────┘

    For pixel (row=1, col=2) in geographic grid:
        glt_x[1,2] = 5  →  raw_col = 5
        glt_y[1,2] = 2  →  raw_row = 2
        value = radiance[2, 5, :]  (all bands)

    GLT values of 0 indicate invalid/no-data pixels

This approach allows:
1. Efficient storage (no wasted pixels from orthorectification padding)
2. Preservation of original radiometric values (no resampling)
3. Flexible reprojection to any target CRS

Radiometric Units
-----------------
- L1B Radiance: μW/(cm²·sr·nm) - microwatts per square centimeter per steradian per nanometer
- FWHM: Full Width at Half Maximum of spectral response in nm
- Wavelengths: Center wavelengths in nm (380-2500 nm range)

Key Classes and Functions
-------------------------
- EMITImage: Main class for reading and processing EMIT data
- download_product: Download EMIT products from NASA Earthdata
- get_radiance_link, get_obs_link, get_l2amask_link, get_ch4enhancement_link: Generate download URLs
- parse_product_name: Parse EMIT product ids of either naming scheme

Product Versions
----------------
NASA LP DAAC closed the v001 collections on 2026-08-31 and now publishes v002 (L2A mask: v003).
v002 names drop the orbit/scene suffix and the L2A mask moved to its own collection with a
different band layout. The link builders and mask methods handle both::

    EMIT_L1B_RAD_001_20220827T060753_2223904_013  ->  mask in EMITL2ARFL.001, CH4ENH in EMITL2BCH4ENH.002
    EMIT_L1B_RAD_002_20260921T044051              ->  mask in EMITL2AMASK.003, no CH4ENH

Requirements
------------
Requires xarray: ``pip install xarray``

Authentication for downloads requires NASA Earthdata credentials stored in:
``~/.georeader/auth_emit.json`` with format: ``{"user": "...", "password": "..."}``

Examples
--------
Basic usage::

    from georeader.readers.emit import EMITImage, download_product
    
    # Download and open EMIT image
    link = 'https://data.lpdaac.earthdatacloud.nasa.gov/...'
    filepath = download_product(link)
    emit = EMITImage(filepath)
    
    # Reproject to UTM (recommended for analysis)
    emit_utm = emit.to_crs("UTM")
    
    # Load as reflectance (applies solar irradiance correction)
    reflectance = emit_utm.load(as_reflectance=True)
    
    # Load RGB composite
    rgb = emit_utm.load_rgb(as_reflectance=True)
    
    # Get cloud mask
    cloud_mask = emit.validmask()

References
----------
- NASA EMIT Mission: https://earth.jpl.nasa.gov/emit/
- EMIT Data Resources: https://github.com/nasa/EMIT-Data-Resources
- EMIT Utils: https://github.com/emit-sds/emit-utils/
- LP DAAC Data Access: https://lpdaac.usgs.gov/products/emitl1bradv002/

"""
# ruff: noqa: F401  (re-exports: this package keeps the namespace of the former emit.py module)
import sys
import types

# Import order follows the dependency graph: utils <- download, mask <- image.
from georeader.readers.emit import utils, download, mask, image
from georeader.readers.emit.download import (
    DAAC_URL,
    download_product,
    get_auth,
    get_ch4enhancement_link,
    get_headers,
    get_l2amask_link,
    get_obs_link,
    get_radiance_link,
)
from georeader.readers.emit.image import HAS_XARRAY, WAVELENGTHS_RGB, EMITImage
from georeader.readers.emit.mask import (
    MASK_BUFFER_FLAGS,
    MASK_INVALID_FLAGS,
    MASK_INVALID_FLAGS_IF_PRESENT,
    MASK_SPECTF_FLAGS,
    _normalise_mask_label,
    mask_band_index,
    mask_flag_indexes,
    valid_mask,
)
from georeader.readers.emit.utils import (
    EMIT_PRODUCT_RE,
    L1B_COMPANIONS,
    L1B_VERSIONS_WITH_ORBIT_SCENE,
    EMITProductID,
    _bounds_indexes_raw,
    _companion_version,
    _l1b_radiance_id,
    parse_product_name,
    product_name_from_params,
    split_product_name,
)

# Download settings that callers assign on this package (``emit.TOKEN = ...``). They live in
# ``download``, so reads and writes on the package are forwarded there.
_DOWNLOAD_SETTINGS = ("AUTH_METHOD", "TOKEN")

__all__ = [
    "AUTH_METHOD", "DAAC_URL", "EMIT_PRODUCT_RE", "EMITImage", "EMITProductID", "HAS_XARRAY",
    "L1B_COMPANIONS", "L1B_VERSIONS_WITH_ORBIT_SCENE", "MASK_BUFFER_FLAGS", "MASK_INVALID_FLAGS",
    "MASK_INVALID_FLAGS_IF_PRESENT", "MASK_SPECTF_FLAGS", "TOKEN", "WAVELENGTHS_RGB",
    "download_product", "get_auth", "get_ch4enhancement_link", "get_headers", "get_l2amask_link",
    "get_obs_link", "get_radiance_link", "mask_band_index", "mask_flag_indexes",
    "parse_product_name", "product_name_from_params", "split_product_name", "valid_mask",
]


def __getattr__(name:str):
    if name in _DOWNLOAD_SETTINGS:
        return getattr(download, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _EMITModule(types.ModuleType):
    def __setattr__(self, name:str, value) -> None:
        if name in _DOWNLOAD_SETTINGS:
            setattr(download, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _EMITModule
