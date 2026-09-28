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
    EMIT_L1B_RAD_002_20220827T060753              ->  mask in EMITL2AMASK.003, no CH4ENH

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
import os
import re
import json
from dataclasses import dataclass, replace
from typing import Tuple, Optional, Any, Union, Dict, List, Sequence
from georeader.readers.download_utils import download_product as download_product_base
import rasterio
import rasterio.windows
import numpy as np
from georeader import window_utils
from shapely.geometry import Polygon
from georeader.geotensor import GeoTensor
from georeader import reflectance
from shapely.ops import unary_union
from georeader import read
import rasterio.warp
from numpy.typing import NDArray
from datetime import datetime, timezone
from georeader.griddata import georreference
from georeader import get_utm_epsg

try:
    import xarray as xr
    from georeader.io import safe_open_netcdf
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None
    safe_open_netcdf = None

AUTH_METHOD = "auth" # "auth" or "token"
TOKEN = None
WAVELENGTHS_RGB = np.array([640, 550, 460])

def _bounds_indexes_raw(glt:NDArray, valid_glt:NDArray) -> Tuple[int, int, int, int]:
        """ Return the bounds of the raw data: (min_x, min_y, max_x, max_y) """
        min_x = np.min(glt[0, valid_glt])
        max_x = np.max(glt[0, valid_glt])
        min_y = np.min(glt[1, valid_glt])
        max_y = np.max(glt[1, valid_glt])
        return min_x, min_y, max_x, max_y

def get_auth() -> Tuple[str, str]:
    home_dir = os.path.join(os.path.expanduser('~'),".georeader")
    json_file = os.path.join(home_dir, "auth_emit.json")
    if not os.path.exists(json_file):
        os.makedirs(home_dir, exist_ok=True)
        with open(json_file, "w") as fh:
            json.dump({"user": "SET-USER", "password": "SET-PASSWORD"}, fh)

        raise FileNotFoundError(f"In order to download EMIT images add user and password to file : {json_file}")

    with open(json_file, "r") as fh:
        data = json.load(fh)
    
    if data["user"] == "SET-USER":
        raise FileNotFoundError(f"In order to download EMIT images add user and password to file : {json_file}")

    return (data["user"], data["password"])


def get_headers() -> Optional[Dict[str, str]]:
    if TOKEN is None:
        return
    
    headers = {"Authorization": f"Bearer {TOKEN}"}
    return headers


DAAC_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected"

# EMIT product ids exist in two forms:
#   v001: EMIT_L1B_RAD_001_20220827T060753_2223904_013  (acquisition, orbit, scene)
#   v002: EMIT_L1B_RAD_002_20220827T060753              (NASA dropped the orbit/scene suffix)
# The optional "V" tolerates ids such as EMIT_L2B_CH4ENH_V001_... seen in some catalogues.
EMIT_PRODUCT_RE = re.compile(
    r"^EMIT_(?P<level>L[0-9][A-Z]?)_(?P<product>[A-Z0-9]+)_V?(?P<version>\d{3})_"
    r"(?P<dt>\d{8}T\d{6})(?:_(?P<orbit>\d{7})_(?P<scene>\d{3}))?$"
)

# Companion collection versions per L1B RAD version (OBS ships in the RAD granule).
# - v001: the L2A mask file ships inside the L2A RFL v001 granule (MASK=None).
#   CH4ENH v001 was emptied by NASA in 2024-11; CH4ENH v002 covers every v001 scene.
# - v002: the mask is its own collection (L2A MASK v003) and there is no CH4ENH yet.
L1B_COMPANIONS: Dict[str, Dict[str, Optional[str]]] = {
    "001": {"RFL": "001", "MASK": None, "CH4ENH": "002"},
    "002": {"RFL": "002", "MASK": "003", "CH4ENH": None},
}

# L1B RAD versions whose granule names carry the _<orbit>_<scene> suffix.
L1B_VERSIONS_WITH_ORBIT_SCENE = ("001",)


@dataclass(frozen=True)
class EMITProductID:
    """
    Parsed EMIT product id. Works for both naming schemes and any level/product.

    Attributes:
        level (str): processing level, e.g. 'L1B', 'L2A', 'L2B'.
        product (str): product short name, e.g. 'RAD', 'OBS', 'RFL', 'MASK', 'CH4ENH'.
        version (str): three-digit collection version, e.g. '001', '002'.
        dt (str): acquisition start 'YYYYmmddTHHMMSS'. It is the only field shared by
            every product and version of the same scene.
        orbit (Optional[str]): orbit number, e.g. '2223904'. None in v002+ names.
        scene (Optional[str]): DAAC scene number, e.g. '013'. None in v002+ names.

    Example:
        >>> pid = parse_product_name('EMIT_L1B_RAD_002_20220827T060753.nc')
        >>> pid.with_product('L2A', 'MASK', '003').name
        'EMIT_L2A_MASK_003_20220827T060753'
    """
    level: str
    product: str
    version: str
    dt: str
    orbit: Optional[str] = None
    scene: Optional[str] = None

    @property
    def acquisition(self) -> datetime:
        """ Acquisition start as a timezone-aware UTC datetime. """
        return datetime.strptime(self.dt, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

    @property
    def name(self) -> str:
        """ Product id without extension, e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013'. """
        suffix = f"_{self.orbit}_{self.scene}" if self.orbit is not None else ""
        return f"EMIT_{self.level}_{self.product}_{self.version}_{self.dt}{suffix}"

    def with_product(self, level:str, product:str, version:Optional[str]=None) -> 'EMITProductID':
        """ Id of another product of the same acquisition (keeps orbit and scene). """
        return replace(self, level=level, product=product, version=version or self.version)


def parse_product_name(name:str) -> EMITProductID:
    """
    Parse an EMIT product id, filename or path of either naming scheme.

    Args:
        name (str): product id, filename or path. e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013',
            '/data/EMIT_L1B_RAD_002_20220827T060753.nc' or 'EMIT_L2A_MASK_003_20220827T060753.nc'.

    Returns:
        EMITProductID: parsed id.

    Raises:
        ValueError: if the name is not an EMIT product id.
    """
    stem = os.path.basename(str(name)).split(".")[0]
    match = EMIT_PRODUCT_RE.match(stem)
    if match is None:
        raise ValueError(f"Not an EMIT product name: {name!r}")
    return EMITProductID(**match.groupdict())


def _l1b_radiance_id(product_path:str) -> EMITProductID:
    return parse_product_name(product_path).with_product("L1B", "RAD")


def _companion_version(rad:EMITProductID, product:str) -> Optional[str]:
    if rad.version not in L1B_COMPANIONS:
        raise ValueError(f"Unknown EMIT L1B version {rad.version!r} in {rad.name}. "
                         f"Known versions: {sorted(L1B_COMPANIONS)}")
    return L1B_COMPANIONS[rad.version][product]


def product_name_from_params(scene_fid:str, orbit:Optional[str]=None,
                             daac_scene_number:Optional[str]=None,
                             version:str="001")-> str:
    """
    Return the L1B radiance product name from the scene_fid, orbit and daac_scene_number

    Args:
        scene_fid (str): scene_fid of the product. e.g. 'emit20220810t064957'
        orbit (Optional[str]): orbit of the product. e.g. '2222205'. Required for v001, ignored for
            versions whose names do not carry it (v002+).
        daac_scene_number (Optional[str]): daac_scene_number of the product. e.g. '033'. Same rule as orbit.
        version (str): L1B RAD collection version. Defaults to '001'.

    Returns:
        str: product name. e.g. 'EMIT_L1B_RAD_001_20220810T064957_2222205_033' or
            'EMIT_L1B_RAD_002_20220810T064957'
    """
    scenedate = scene_fid[4:].replace("t", "T")
    if version not in L1B_VERSIONS_WITH_ORBIT_SCENE:
        return EMITProductID("L1B", "RAD", version, scenedate).name
    if orbit is None or daac_scene_number is None:
        raise ValueError(f"EMIT L1B v{version} names need orbit and daac_scene_number")
    return EMITProductID("L1B", "RAD", version, scenedate, orbit, daac_scene_number).name


def split_product_name(product_name:str) -> Tuple[str, Optional[str], Optional[str], datetime]:
    """
    Split the product name into its components

    Args:
        product_name (str): product name. e.g. 'EMIT_L1B_RAD_001_20220810T064957_2222205_033'
            or 'EMIT_L1B_RAD_002_20220810T064957'

    Returns:
        Tuple[str, Optional[str], Optional[str], datetime]: scene_fid, orbit, daac_scene_number, datetime
            e.g. ('emit20220810t064957', '2222205', '033', datetime('2022-08-10T06:49:57')).
            orbit and daac_scene_number are None for v002+ names.
    """
    pid = parse_product_name(product_name)
    scene_fid = f"emit{pid.dt}".replace("T", "t")
    return scene_fid, pid.orbit, pid.scene, pid.acquisition


def download_product(link_down:str, filename:Optional[str]=None,
                     display_progress_bar:bool=True,
                     auth:Optional[Tuple[str, str]] = None) -> str:
    """
    Download a product from the EMIT website (https://search.earthdata.nasa.gov/search). 
    It requires that you have an account in the NASA Earthdata portal. 

    This code is based on this example: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        link_down: link to the product
        filename: filename to save the product
        display_progress_bar: display tqdm progress bar
        auth: tuple with user and password to download the product. If None, it will try to read the user and password from ~/.georeader/auth_emit.json 

    Example:
        >>> link_down = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220828T051941_2224004_006/EMIT_L1B_RAD_001_20220828T051941_2224004_006.nc'
        >>> filename = download_product(link_down)
    """
    headers = None
    if auth is None:
        if AUTH_METHOD == "auth":
            auth = get_auth()
        elif AUTH_METHOD == "token":
            assert TOKEN is not None, "You need to set the TOKEN variable to download EMIT images"
            headers = get_headers()
    
    return download_product_base(link_down, filename=filename, auth=auth,
                                 headers=headers,
                                 display_progress_bar=display_progress_bar, 
                                 verify=False)


def get_radiance_link(product_path:str) -> str:
    """
    Get the link to download a product from the EMIT website.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        product_path: path to the product or filename of the product or product name with or without extension.
            Any EMIT product of the acquisition works. e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Example:
        >>> get_radiance_link('EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc')
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220827T060753_2223904_013/EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'
        >>> get_radiance_link('EMIT_L1B_RAD_002_20220827T060753.nc')
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.002/EMIT_L1B_RAD_002_20220827T060753/EMIT_L1B_RAD_002_20220827T060753.nc'
    """
    rad = _l1b_radiance_id(product_path)
    return f"{DAAC_URL}/EMITL1BRAD.{rad.version}/{rad.name}/{rad.name}.nc"


def get_obs_link(product_path:str) -> str:
    """
    Get the link to download the observation (OBS) file, which ships in the L1B RAD granule.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    Args:
        product_path: path to the product or filename of the product with or without extension.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Example:
        >>> get_obs_link('EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc')
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.001/EMIT_L1B_RAD_001_20220827T060753_2223904_013/EMIT_L1B_OBS_001_20220827T060753_2223904_013.nc'
    """
    rad = _l1b_radiance_id(product_path)
    obs = rad.with_product("L1B", "OBS")
    return f"{DAAC_URL}/EMITL1BRAD.{rad.version}/{rad.name}/{obs.name}.nc"


def get_ch4enhancement_link(tile:str) -> Optional[str]:
    """
    Get the link to download the L2B CH4 enhancement of the acquisition.
    See: https://git.earthdata.nasa.gov/projects/LPDUR/repos/daac_data_download_python/browse

    v001 L1B scenes point at CH4ENH v002 (NASA emptied CH4ENH v001 in 2024-11). There is no CH4ENH
    product for v002 L1B scenes, so this returns None for them.

    Args:
        tile (str): path to the product or filename of the product with or without extension.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Returns:
        Optional[str]: link, or None if no CH4ENH collection exists for this L1B version.

    Example:
        >>> get_ch4enhancement_link('EMIT_L1B_RAD_001_20220810T064957_2222205_033.nc')
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2BCH4ENH.002/EMIT_L2B_CH4ENH_002_20220810T064957_2222205_033/EMIT_L2B_CH4ENH_002_20220810T064957_2222205_033.tif'
    """
    rad = _l1b_radiance_id(tile)
    version = _companion_version(rad, "CH4ENH")
    if version is None:
        return None
    ch4 = rad.with_product("L2B", "CH4ENH", version)
    return f"{DAAC_URL}/EMITL2BCH4ENH.{version}/{ch4.name}/{ch4.name}.tif"


def get_l2amask_link(tile: str) -> str:
    """
    Get the link to download the L2A mask of the acquisition (https://search.earthdata.nasa.gov/search)

    v001 L1B scenes: the mask file ships inside the L2A RFL v001 granule.
    v002 L1B scenes: the mask is its own collection, L2A MASK v003.

    Args:
        tile (str): path to the product or filename of the L1B product with or without extension.
            e.g. 'EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc'

    Returns:
        str: link to the L2A mask product

    Example:
        >>> get_l2amask_link('EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc')
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/EMIT_L2A_RFL_001_20220827T060753_2223904_013/EMIT_L2A_MASK_001_20220827T060753_2223904_013.nc'
        >>> get_l2amask_link('EMIT_L1B_RAD_002_20220827T060753.nc')
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2AMASK.003/EMIT_L2A_MASK_003_20220827T060753/EMIT_L2A_MASK_003_20220827T060753.nc'
    """
    rad = _l1b_radiance_id(tile)
    mask_version = _companion_version(rad, "MASK")
    if mask_version is None:
        rfl = rad.with_product("L2A", "RFL", _companion_version(rad, "RFL"))
        mask = rfl.with_product("L2A", "MASK")
        return f"{DAAC_URL}/EMITL2ARFL.{rfl.version}/{rfl.name}/{mask.name}.nc"
    mask = rad.with_product("L2A", "MASK", mask_version)
    return f"{DAAC_URL}/EMITL2AMASK.{mask_version}/{mask.name}/{mask.name}.nc"


# L2A mask flags, matched by normalised label (lower case, single spaces). The band layout
# differs between versions, so flags are never selected by position:
#   v001: Cloud flag, Cirrus flag, Water flag, Spacecraft Flag, Dilated Cloud Flag, AOD550,
#         H2O (g cm-2), Aggregate Flag
#   v003: Cloud Flag, Cirrus Flag, Water Flag, Dilated Cloud Flag, SpecTf-Cloud Probability,
#         SpecTf-Cloud Flag, SpecTf-Buffer Distance
MASK_INVALID_FLAGS = ("cloud flag", "cirrus flag")
MASK_INVALID_FLAGS_IF_PRESENT = ("spacecraft flag",)  # v001 only
MASK_BUFFER_FLAGS = ("dilated cloud flag",)
MASK_SPECTF_FLAGS = ("spectf-cloud flag",)  # v003 only


def _normalise_mask_label(label:Any) -> str:
    return " ".join(str(label).lower().split())


def mask_band_index(mask_bands:Sequence[str], name:str, source:Optional[str]=None) -> int:
    """
    Index of the L2A mask band called ``name``, ignoring case and repeated whitespace.

    Args:
        mask_bands (Sequence[str]): band labels of the L2A mask file (``sensor_band_parameters/mask_bands``).
        name (str): band label, e.g. 'Water flag' or 'Water Flag'.
        source (Optional[str]): file name used in the error message.

    Raises:
        ValueError: if no band has that label.
    """
    labels = [_normalise_mask_label(b) for b in mask_bands]
    wanted = _normalise_mask_label(name)
    if wanted not in labels:
        where = f" in {source}" if source else ""
        raise ValueError(f"EMIT mask band {name!r} not found{where}. Bands: {list(mask_bands)}")
    return labels.index(wanted)


def mask_flag_indexes(mask_bands:Sequence[str], with_buffer:bool=True,
                      include_spectf:bool=False, source:Optional[str]=None) -> List[int]:
    """
    Indexes of the L2A mask flags that mark a pixel as invalid, selected by label.

    1. Always: Cloud flag and Cirrus flag, plus Spacecraft flag if the file has it (v001).
    2. ``with_buffer``: Dilated Cloud Flag.
    3. ``include_spectf``: SpecTf-Cloud Flag if the file has it (v003). The continuous
       SpecTf-Cloud Probability band is never selected.

    For a v001 file this returns [0, 1, 3] or [0, 1, 3, 4], the indexes georeader has always used.

    Args:
        mask_bands (Sequence[str]): band labels of the L2A mask file.
        with_buffer (bool): add the dilated cloud flag. Defaults to True.
        include_spectf (bool): add the SpecTf ML cloud flag when present. Defaults to False.
        source (Optional[str]): file name used in the error message.

    Raises:
        ValueError: if a required flag is missing.
    """
    labels = [_normalise_mask_label(b) for b in mask_bands]
    required = MASK_INVALID_FLAGS + (MASK_BUFFER_FLAGS if with_buffer else ())
    optional = MASK_INVALID_FLAGS_IF_PRESENT + (MASK_SPECTF_FLAGS if include_spectf else ())
    missing = [flag for flag in required if flag not in labels]
    if missing:
        where = f" in {source}" if source else ""
        raise ValueError(f"EMIT mask flags {missing} not found{where}. Bands: {list(mask_bands)}")
    return sorted(labels.index(flag) for flag in required + optional if flag in labels)


class EMITImage:
    """
    Reader for EMIT L1B (Earth Surface Mineral Dust Source Investigation) hyperspectral images.
    
    This class provides comprehensive functionality to read and manipulate EMIT satellite 
    imagery products from NASA's imaging spectrometer aboard the ISS. It handles the 
    unique GLT-based (Geographic Lookup Table) storage format, supporting operations like:
    
    - Loading radiometry data with automatic orthorectification
    - Converting radiance to reflectance using solar irradiance
    - Accessing cloud and quality masks
    - Extracting viewing and solar geometry angles
    - Reprojecting to different coordinate reference systems
    
    EMIT Data Model
    ---------------
    EMIT stores data in sensor coordinates, not geographic coordinates. The GLT provides
    a lookup table mapping geographic pixels to sensor pixels:
    
        GLT Orthorectification:
        ┌────────────────────────────┐      ┌──────────────────────────┐
        │    Geographic Grid         │      │   Sensor Grid (raw)      │
        │  (orthorectified space)    │      │  (pushbroom scan)        │
        │  ┌───┬───┬───┬───┐        │      │  ┌───┬───┬───┬───┐      │
        │  │ · │ a │ b │ · │        │  GLT │  │ e │ a │ b │ · │      │
        │  ├───┼───┼───┼───┤        │  ──→ │  ├───┼───┼───┼───┤      │
        │  │ c │ d │ e │ f │        │      │  │ f │ c │ d │ · │      │
        │  └───┴───┴───┴───┘        │      │  └───┴───┴───┴───┘      │
        │  (pixels with data)        │      │  (original acquistion)   │
        └────────────────────────────┘      └──────────────────────────┘
        
        · = no data (GLT value = 0)
        
        For geographic pixel (row, col):
            raw_x = glt_x[row, col]  
            raw_y = glt_y[row, col]
            value = radiance[raw_y, raw_x, :]
    
    This approach preserves original radiometric values without interpolation artifacts.
    
    Spectral Characteristics
    ------------------------
    - Wavelength range: 380-2500 nm (VNIR + SWIR)
    - Number of bands: 285
    - Spectral sampling: ~7.4 nm
    - Spatial resolution: 60m at nadir
    
    Attributes
    ----------
    filename : str
        Path to the EMIT NetCDF file.
    nc_ds : xr.Dataset
        xarray Dataset handle for the main radiance file.
    glt : GeoTensor
        Geographic Lookup Table as a GeoTensor with shape (2, H, W).
        - glt.values[0]: x-indices into raw radiance (1-based)
        - glt.values[1]: y-indices into raw radiance (1-based)
    valid_glt : np.ndarray
        Boolean mask (H, W) indicating valid GLT entries (data coverage).
    glt_relative : GeoTensor
        GLT with indices relative to the data window (0-based).
    window_raw : rasterio.windows.Window
        Window defining the subset of raw data to read (optimizes I/O).
    real_transform : rasterio.Affine
        Affine transform for the orthorectified (geographic) grid.
    time_coverage_start : datetime
        UTC datetime of acquisition start.
    time_coverage_end : datetime
        UTC datetime of acquisition end.
    wavelengths : np.ndarray
        Center wavelengths (nm) for selected bands.
    fwhm : np.ndarray
        Full Width at Half Maximum (nm) for selected bands.
    band_selection : Union[int, Tuple[int, ...], slice]
        Current band subset selection.
    units : str
        Radiance units from file metadata (typically 'uW/(cm^2 sr nm)').
    fill_value_default : float
        No-data value for radiance data.
    dims : Tuple[str]
        Dimension names ("band", "y", "x").
    dtype : np.dtype
        Data type of radiance values.
    
    Lazy-Loaded Properties
    ----------------------
    nc_ds_obs : xr.Dataset
        Observation data (viewing/solar angles, path length, elevation).
        Auto-downloaded from NASA Earthdata if not present locally.
    nc_ds_l2amask : xr.Dataset  
        L2A quality mask data (clouds, cirrus, water, aggregate flags).
        Auto-downloaded from NASA Earthdata if not present locally.
    mean_sza : float
        Mean solar zenith angle (degrees) across the scene.
    mean_vza : float
        Mean view zenith angle (degrees) across the scene.
    observation_date_correction_factor : float
        Earth-Sun distance correction factor for the acquisition date.
    
    Examples
    --------
    Basic loading and reprojection::
    
        >>> from georeader.readers.emit import EMITImage, download_product
        >>> 
        >>> # Download from NASA Earthdata
        >>> link = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/...'
        >>> filepath = download_product(link)
        >>> 
        >>> # Open and reproject to UTM
        >>> emit = EMITImage(filepath)
        >>> emit_utm = emit.to_crs("UTM", resolution_dst_crs=60)
        >>> 
        >>> # Load as reflectance
        >>> refl = emit_utm.load(as_reflectance=True)
        >>> print(refl.shape)  # (285, H, W)
    
    Working with specific wavelengths::
    
        >>> # Select RGB-like bands (640, 550, 460 nm)
        >>> emit.set_band_selection([35, 23, 11])
        >>> print(emit.wavelengths)  # [641.2, 553.1, 462.3]
        >>> rgb = emit.load(as_reflectance=True)
        >>> 
        >>> # Or use the convenience method
        >>> rgb = emit.load_rgb(as_reflectance=True)
    
    Accessing masks and quality data::
    
        >>> # Get valid (cloud-free) mask
        >>> valid_mask = emit.validmask()
        >>> print(f"Clear pixels: {emit.percentage_clear:.1f}%")
        >>> 
        >>> # Get specific mask layers
        >>> cloud_mask = emit.mask("Cloud flag")
        >>> water_mask = emit.water_mask()
    
    Working with viewing geometry::
    
        >>> # Get solar zenith angle
        >>> sza = emit.sza()  # GeoTensor with SZA values
        >>> 
        >>> # Get mean angles for quick reference
        >>> print(f"Mean SZA: {emit.mean_sza:.1f}°")
        >>> print(f"Mean VZA: {emit.mean_vza:.1f}°")
    
    Spatial subsetting::
    
        >>> import rasterio.windows
        >>> 
        >>> # Read a spatial window
        >>> window = rasterio.windows.Window(col_off=100, row_off=200, width=500, height=500)
        >>> emit_subset = emit.read_from_window(window)
        >>> data = emit_subset.load()
    
    See Also
    --------
    georeader.readers.prisma.PRISMA : PRISMA hyperspectral reader
    georeader.readers.enmap.EnMAP : EnMAP hyperspectral reader
    georeader.reflectance : Radiometric conversion utilities
    
    References
    ----------
    - EMIT L1B Product Guide: https://lpdaac.usgs.gov/products/emitl1bradv001/
    - EMIT Data Resources: https://github.com/nasa/EMIT-Data-Resources
    - EMIT Algorithms: Green et al. (2020) doi:10.1029/2020JD033451
    """
    attributes_set_if_exists = ["_nc_ds_obs", "_mean_sza", "_mean_vza",
                                "_observation_bands", "_nc_ds_l2amask", "_mask_bands",
                                "obs_file", "l2amaskfile",
                                # Option B: opt-in radiance cache. ``_cache`` is a
                                # mutable dict shared by reference across all clones
                                # built from the same parent — that's what makes the
                                # cache visible end-to-end. ``cache_radiance`` is the
                                # opt-in flag (rebind-on-clone is fine; we don't toggle
                                # per-clone).
                                "_cache", "cache_radiance"]

    # Key under which the full-spectrum windowed radiance is stored in ``_cache``.
    _CACHE_KEY_RADIANCE = "radiance_window"

    def __init__(self, filename:str, glt:Optional[GeoTensor]=None,
                 band_selection:Optional[Union[int, Tuple[int, ...],slice]]=slice(None),
                 cache_radiance:bool=False,
                 reuse_handles_from:Optional['EMITImage']=None):
        if not HAS_XARRAY:
            raise ImportError("xarray is required to read EMIT images. Please install it with: pip install xarray")

        self.filename = filename
        if reuse_handles_from is not None:
            if reuse_handles_from.filename != self.filename:
                raise ValueError("reuse_handles_from must reference the same EMIT file")
            # Clone constructor path: reuse parent handles to avoid opening
            # throwaway datasets that would immediately be overwritten.
            self.nc_ds = reuse_handles_from.nc_ds
        else:
            self.nc_ds = safe_open_netcdf(self.filename, cache=False, load=False)
        self._nc_ds_obs = None
        self._nc_ds_l2amask = None
        self._observation_bands = None
        self._mask_bands = None
        self._sensor_band_params = None
        # Opt-in radiance cache. Default off — the dict is created either way so the
        # ``_cache is parent._cache`` invariant holds for clones even when caching
        # is disabled.
        self.cache_radiance:bool = cache_radiance
        self._cache:Dict[str, Any] = {}
        # self.real_shape = (self.nc_ds['radiance'].shape[-1],) + self.nc_ds['radiance'].shape[:-1]

        self._mean_sza = None
        self._mean_vza = None
        self.obs_file:Optional[str] = None
        self.l2amaskfile:Optional[str] = None

        geotransform = self.nc_ds.attrs['geotransform']
        self.real_transform = rasterio.Affine(geotransform[1], geotransform[2], geotransform[0],
                                              geotransform[4], geotransform[5], geotransform[3])
        
        self.time_coverage_start = datetime.strptime(self.nc_ds.attrs['time_coverage_start'], "%Y-%m-%dT%H:%M:%S%z")
        self.time_coverage_end = datetime.strptime(self.nc_ds.attrs['time_coverage_end'], "%Y-%m-%dT%H:%M:%S%z")

        self.dtype = self.nc_ds['radiance'].dtype
        self.dims = ("band", "y", "x")
        self.fill_value_default = self.nc_ds['radiance'].attrs.get('_FillValue', -9999)
        self.nodata = self.fill_value_default
        self.units = self.nc_ds["radiance"].attrs.get('units', '')

        if glt is None:
            # Open the location group to access glt_x and glt_y
            location_ds = safe_open_netcdf(self.filename, cache=False, load=False, group='location')
            glt_x = np.nan_to_num(location_ds['glt_x'].values, nan=0).astype(np.int32)
            glt_y = np.nan_to_num(location_ds['glt_y'].values, nan=0).astype(np.int32)
            location_ds.close()
            
            glt_arr = np.zeros((2,) + glt_x.shape, dtype=np.int32)
            glt_arr[0] = glt_x
            glt_arr[1] = glt_y
            # glt_arr -= 1 # account for 1-based indexing

            # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html
            self.glt = GeoTensor(glt_arr, transform=self.real_transform, 
                                 crs=rasterio.crs.CRS.from_wkt(self.nc_ds.attrs['spatial_ref']),
                                 fill_value_default=0)
        else:
            self.glt = glt
        
        self.valid_glt = np.all(self.glt.values != self.glt.fill_value_default, axis=0)
        xmin, ymin, xmax, ymax = self._bounds_indexes_raw() # values are 1-based!

        # glt has the absolute indexes of the netCDF object
        # glt_relative has the relative indexes
        self.glt_relative = self.glt.copy()
        self.glt_relative.values[0, self.valid_glt] -= xmin
        self.glt_relative.values[1, self.valid_glt] -= ymin

        self.window_raw = rasterio.windows.Window(col_off=xmin-1, row_off=ymin-1, 
                                                  width=xmax-xmin+1, height=ymax-ymin+1)

        # Load sensor_band_parameters from its group, unless we're cloning from
        # an existing instance and can reuse the already-open handle.
        if reuse_handles_from is not None:
            self._sensor_band_params = reuse_handles_from._sensor_band_params
            self.bandname_dimension = reuse_handles_from.bandname_dimension
        else:
            self._sensor_band_params = safe_open_netcdf(self.filename, cache=False, load=False, group='sensor_band_parameters')
            if "wavelengths" in self._sensor_band_params:
                self.bandname_dimension = "wavelengths"
            elif "radiance_wl" in self._sensor_band_params:
                self.bandname_dimension = "radiance_wl"
            else:
                raise ValueError(f"wavelengths or radiance_wl not found in sensor_band_parameters")
        
        self.band_selection = band_selection
        self.wavelengths = self._sensor_band_params[self.bandname_dimension].values[self.band_selection]
        self.fwhm = self._sensor_band_params['fwhm'].values[self.band_selection]
        self._observation_date_correction_factor:Optional[float] = None

    @property
    def observation_date_correction_factor(self) -> float:
        if self._observation_date_correction_factor is None:
            self._observation_date_correction_factor = reflectance.observation_date_correction_factor(date_of_acquisition=self.time_coverage_start,
                                                                                                      center_coords=self.footprint("EPSG:4326").centroid.coords[0])
        return self._observation_date_correction_factor
    
    @property
    def crs(self) -> Any:
        return self.glt.crs

    @property
    def shape(self) -> Tuple:
        try:
            n_bands = len(self.wavelengths)
            return  (n_bands,) + self.glt.shape[1:]
        except Exception:
            return self.glt.shape

    @property
    def width(self) -> int:
        return self.shape[-1]
    
    @property
    def height(self) -> int:
        return self.shape[-2]

    @property
    def transform(self) -> rasterio.Affine:
        return self.glt.transform

    @property
    def res(self) -> Tuple[float, float]:
        return self.glt.res

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.glt.bounds

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        """
        Get the footprint of the image in the given CRS. If no CRS is given, the footprint is returned in the native CRS.
        This function takes into account the valid_glt mask to compute the footprint.

        Args:
            crs (Optional[str], optional): The CRS to return the footprint in. Defaults to None. 
                If None, the footprint is returned in the native CRS.
        
        Returns:
            Polygon: The footprint of the image in the given CRS.
        """
        if not hasattr(self, '_pol'):
            from georeader.vectorize import get_polygons
            pols = get_polygons(self.valid_glt, transform=self.transform)
            self._pol = unary_union(pols)
        if crs is not None:
            pol_crs = window_utils.polygon_to_crs(self._pol, self.crs, crs)
        else:
            pol_crs = self._pol
        
        pol_glt = self.glt.footprint(crs=crs)

        return pol_crs.intersection(pol_glt)
    
    def set_band_selection(self, band_selection:Optional[Union[int, Tuple[int, ...],slice]]=None):
        """
        Set the band selection. Band selection is absolute w.r.t self.nc_ds['radiance']

        Args:
            band_selection (Optional[Union[int, Tuple[int, ...],slice]], optional): slicing or selection of the bands. Defaults to None.
        
        Example:
            >>> emit_image.set_band_selection(slice(0, 3)) # will only load the three first bands
            >>> emit_image.wavelengths # will only return the wavelengths of the three first bands
            >>> emit_image.load() # will only load the three first bands
        """
        if band_selection is None:
            band_selection = slice(None)
        self.band_selection = band_selection
        self.wavelengths = self._sensor_band_params[self.bandname_dimension].values[self.band_selection]
        self.fwhm = self._sensor_band_params['fwhm'].values[self.band_selection]
    
    @ property
    def nc_ds_obs(self, obs_file:Optional[str]=None):
        """
        Loads the observation file. In this file we have information about angles (solar and viewing),
        elevation and ilumination based on elevation and path length.

        This function downloads the observation file if it does not exist from the JPL portal.

        It caches the observation file in the object. (self.nc_ds_obs)

        Args:
            obs_file (Optional[str], optional): Path to the observation file. 
                Defaults to None. If none it will download the observation file 
                from the EMIT server.
        """
        if self._nc_ds_obs is not None:
            return self._nc_ds_obs
        
        if obs_file is None:
            link_obs_file = get_obs_link(self.filename)
            obs_file = os.path.join(os.path.dirname(self.filename), os.path.basename(link_obs_file))
            if not os.path.exists(obs_file):
                download_product(link_obs_file, obs_file)
        
        self.obs_file = obs_file
        self._nc_ds_obs = safe_open_netcdf(obs_file, cache=False, load=False)
        # Load observation_bands from sensor_band_parameters group
        sensor_params = safe_open_netcdf(obs_file, cache=False, load=False, group='sensor_band_parameters')
        self._observation_bands = sensor_params['observation_bands'].values
        sensor_params.close()
        return self._nc_ds_obs
    
    @property
    def nc_ds_l2amask(self, l2amaskfile:Optional[str]=None) -> xr.Dataset:
        """
        Loads the L2A mask file. In this file we have information about the cloud mask.

        This function downloads the L2A mask file if it does not exist from the JPL portal.

        It caches the L2A mask file in the object. (self.nc_ds_l2amask)

        See https://lpdaac.usgs.gov/products/emitl2arflv001/ (v001, mask inside the RFL granule) and
        https://lpdaac.usgs.gov/products/emitl2amaskv003/ (v003) for info about the L2A mask file.

        Args:
            l2amaskfile (Optional[str], optional): Path to the L2A mask file. 
                Defaults to None. If none it will download the L2A mask file 
                from the EMIT server.
        """
        if self._nc_ds_l2amask is not None:
            return self._nc_ds_l2amask
        
        if l2amaskfile is None:
            link_l2amaskfile = get_l2amask_link(self.filename)
            l2amaskfile = os.path.join(os.path.dirname(self.filename), os.path.basename(link_l2amaskfile))
            if not os.path.exists(l2amaskfile):
                download_product(link_l2amaskfile, l2amaskfile)
        
        self.l2amaskfile = l2amaskfile
        self._nc_ds_l2amask = safe_open_netcdf(l2amaskfile, cache=False, load=False)
        # Load mask_bands from sensor_band_parameters group
        sensor_params = safe_open_netcdf(l2amaskfile, cache=False, load=False, 
                                         group='sensor_band_parameters')
        self._mask_bands = sensor_params["mask_bands"].values
        sensor_params.close()
        return self._nc_ds_l2amask
    
    @property
    def mask_bands(self) -> np.array:
        """ Returns the mask band labels. The layout depends on the mask version:

        - v001: ['Cloud flag', 'Cirrus flag', 'Water flag', 'Spacecraft Flag', 'Dilated Cloud Flag',
          'AOD550', 'H2O (g cm-2)', 'Aggregate Flag']
        - v003: ['Cloud Flag', 'Cirrus Flag', 'Water Flag', 'Dilated Cloud Flag',
          'SpecTf-Cloud Probability', 'SpecTf-Cloud Flag', 'SpecTf-Buffer Distance']
        """
        self.nc_ds_l2amask
        return self._mask_bands
    
    def validmask(self, with_buffer:bool=True, include_spectf:bool=False) -> GeoTensor:
        """
        Return the validmask mask

        Args:
            with_buffer (bool): also mask the dilated cloud flag. Defaults to True.
            include_spectf (bool): also mask the SpecTf ML cloud flag (v003 masks only). Defaults to False.

        Returns:
            GeoTensor: bool mask. True means that the pixel is valid.
        """

        validmask = ~self.invalid_mask_raw(with_buffer=with_buffer, include_spectf=include_spectf)

        return self.georreference(validmask,
                                  fill_value_default=False)
    
    def invalid_mask_raw(self, with_buffer:bool=True, include_spectf:bool=False) -> NDArray:
        """
        Returns the non georreferenced quality mask. True means that the pixel is not valid.

        This mask is computed as the sum of the Cloud flag, Cirrus flag, Spacecraft flag (v001 only)
        and, with ``with_buffer``, the Dilated Cloud Flag. Flags are selected by label (see
        ``mask_flag_indexes``) because the band layout changed between mask versions.
        True means that the pixel is not valid.

        From: https://github.com/nasa/EMIT-Data-Resources/blob/main/python/how-tos/How_to_use_EMIT_Quality_data.ipynb
        and https://github.com/nasa/EMIT-Data-Resources/blob/main/python/modules/emit_tools.py#L277

        Args:
            with_buffer (bool): also mask the dilated cloud flag. Defaults to True.
            include_spectf (bool): also mask the SpecTf ML cloud flag (v003 masks only). Defaults to False.
        """
        band_index = mask_flag_indexes(self.mask_bands, with_buffer=with_buffer,
                                       include_spectf=include_spectf, source=self.l2amaskfile)

        slice_y, slice_x = self.window_raw.toslices()
        mask_arr = self.nc_ds_l2amask['mask'].values[slice_y, slice_x, band_index]
        mask_arr = np.sum(mask_arr, axis=-1)
        mask_arr = (mask_arr >= 1)
        return mask_arr
    
    @property
    def percentage_clear(self) -> float:
        """
        Return the percentage of clear pixels in the image

        Returns:
            float: percentage of clear pixels
        """
        
        invalids = self.invalid_mask_raw(with_buffer=False)
        return 100 * (1 - np.sum(invalids) / np.prod(invalids.shape))


    def mask(self, mask_name:str="cloud_mask") -> GeoTensor:
        """
        Return the mask layer with the given name.
        Mask shall be one of self.mask_bands; the lookup ignores case and repeated whitespace,
        so 'Water flag' works on both v001 ('Water flag') and v003 ('Water Flag') masks.

        Args:
            mask_name (str, optional): Name of the mask. Defaults to "cloud_mask".

        Returns:
            GeoTensor: mask
        """
        band_index = mask_band_index(self.mask_bands, mask_name, source=self.l2amaskfile)
        slice_y, slice_x = self.window_raw.toslices()
        mask_arr = self.nc_ds_l2amask['mask'].values[slice_y, slice_x, band_index]
        return self.georreference(mask_arr,
                                  fill_value_default=self.nc_ds_l2amask['mask'].attrs.get('_FillValue', -9999))
    
    def water_mask(self) -> GeoTensor:
        """ Returns the water mask """
        return self.mask("Water flag")
    
    @property
    def observation_bands(self) -> np.array:
        """ Returns the observation bands """
        self.nc_ds_obs
        return self._observation_bands
    
    def observation(self, name:str) -> GeoTensor:
        """ Returns the observation with the given name """
        band_index = self.observation_bands.tolist().index(name)
        slice_y, slice_x = self.window_raw.toslices()
        # The obs file stores obs data in root group, not in a subgroup
        obs_arr = self.nc_ds_obs['obs'].values[slice_y, slice_x, band_index]
        return self.georreference(obs_arr, 
                                  fill_value_default=self.nc_ds_obs['obs'].attrs.get('_FillValue', -9999))

    def sza(self) -> GeoTensor:
        """ Return the solar zenith angle as a GeoTensor """
        return self.observation('To-sun zenith (0 to 90 degrees from zenith)')
    
    def vza(self) -> GeoTensor:
        """ Return the view zenith angle as a GeoTensor """
        return self.observation('To-sensor zenith (0 to 90 degrees from zenith)')
    
    def elevation(self) -> GeoTensor:
        location_ds = safe_open_netcdf(self.filename, cache=False, load=False, group='location')
        obs_arr = location_ds["elev"]
        slice_y, slice_x = self.window_raw.toslices()
        elev_data = obs_arr.values[slice_y, slice_x]
        fill_val = obs_arr.attrs.get('_FillValue', -9999)
        location_ds.close()
        return self.georreference(elev_data, fill_value_default=fill_val)

    @property
    def mean_sza(self) -> float:
        """ Return the mean solar zenith angle """
        if self._mean_sza is not None:
            return self._mean_sza
        
        band_index = self.observation_bands.tolist().index('To-sun zenith (0 to 90 degrees from zenith)')
        sza_arr = self.nc_ds_obs['obs'].values[..., band_index]
        fill_val = self.nc_ds_obs['obs'].attrs.get('_FillValue', -9999)
        self._mean_sza = float(np.mean(sza_arr[sza_arr != fill_val]))
        return self._mean_sza
    
    @property
    def mean_vza(self) -> float:
        """ Return the mean view zenith angle """
        if self._mean_vza is not None:
            return self._mean_vza
        band_index = self.observation_bands.tolist().index('To-sensor zenith (0 to 90 degrees from zenith)')
        vza_arr = self.nc_ds_obs['obs'].values[..., band_index]
        fill_val = self.nc_ds_obs['obs'].attrs.get('_FillValue', -9999)
        self._mean_vza = float(np.mean(vza_arr[vza_arr != fill_val]))
        return self._mean_vza
        
    def __copy__(self) -> '__class__':
        out = EMITImage(
            self.filename,
            glt=self.glt.copy(),
            band_selection=self.band_selection,
            reuse_handles_from=self,
        )
        
        # copy nc_ds_obs if it exists
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, getattr(self, attrname))

        return out
    def copy(self) -> '__class__':
        return self.__copy__()
    
    def to_crs(self, crs:Any="UTM", 
               resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=60) -> '__class__':
        """
        Reproject the image to a new crs

        Args:
            crs (Any): CRS. 

        Returns:
            EmitImage: EMIT image in the new CRS
        
        Example:
            >>> emit_image = EMITImage("path/to/emit_image.nc")
            >>> emit_image_utm = emit_image.to_crs(crs="UTM")
        """
        if crs == "UTM":
            footprint = self.glt.footprint("EPSG:4326")
            crs = get_utm_epsg(footprint)

        glt = read.read_to_crs(self.glt, crs, resampling=rasterio.warp.Resampling.nearest, 
                               resolution_dst_crs=resolution_dst_crs)

        out = EMITImage(
            self.filename,
            glt=glt,
            band_selection=self.band_selection,
            reuse_handles_from=self,
        )

        # Propagate eagerly-set and lazily-loaded attributes from the parent so
        # the new instance shares the parent's NetCDF handles, sensor params,
        # observation bands, mean angles, etc. without re-opening anything.
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, getattr(self, attrname))

        # _pol is not in attributes_set_if_exists because it's CRS-dependent —
        # it must be reprojected to the new CRS.
        if hasattr(self, '_pol'):
            setattr(out, '_pol', window_utils.polygon_to_crs(self._pol, self.crs, crs))

        return out


    def read_from_window(self, window:Optional[rasterio.windows.Window]=None, boundless:bool=True) -> '__class__':
        glt_window = self.glt.read_from_window(window, boundless=boundless)
        out = EMITImage(
            self.filename,
            glt=glt_window,
            band_selection=self.band_selection,
            reuse_handles_from=self,
        )

        # Propagate eagerly-set and lazily-loaded attributes from the parent.
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, getattr(self, attrname))

        return out
    
    def read_from_bands(self, bands:Union[int, Tuple[int, ...], slice]) -> '__class__':
        copy = self.__copy__()
        copy.set_band_selection(bands)
        return copy
  
    def load(self, boundless:bool=True, as_reflectance:bool=False)-> GeoTensor:
        data = self.load_raw() # (C, H, W) or (H, W)
        if as_reflectance:
            invalids = np.isnan(data) | (data == self.fill_value_default)
            thuiller = reflectance.load_thuillier_irradiance()
            response = reflectance.srf(self.wavelengths, self.fwhm, thuiller["Nanometer"].values)
            solar_irradiance_norm = thuiller["Radiance(mW/m2/nm)"].values.dot(response) / 1_000
            data = reflectance.radiance_to_reflectance(data, solar_irradiance_norm,
                                                       units=self.units,
                                                       observation_date_corr_factor=self.observation_date_correction_factor)
            data[invalids] = self.fill_value_default
        return self.georreference(data, fill_value_default=self.fill_value_default)
    
    def load_rgb(self, as_reflectance:bool=True) -> GeoTensor:
        bands_read = np.argmin(np.abs(WAVELENGTHS_RGB[:, np.newaxis] - self.wavelengths), axis=1).tolist()
        ei_rgb = self.read_from_bands(bands_read)
        return ei_rgb.load(boundless=True, as_reflectance=as_reflectance)

    @property
    def shape_raw(self) -> Tuple[int, int, int]:
        """ Return the shape of the raw data in (C, H, W) format """
        return (len(self.wavelengths),) + rasterio.windows.shape(self.window_raw)

    def _bounds_indexes_raw(self) -> Tuple[int, int, int, int]:
        """ Return the bounds of the raw data: (min_x, min_y, max_x, max_y) """
        return _bounds_indexes_raw(self.glt.values, self.valid_glt)


    def load_raw(self, transpose:bool=True) -> np.array:
        """
        Load the raw data, without orthorectification

        Args:
            transpose (bool, optional): Transpose the data if it has 3 dimentsions to (C, H, W)
                Defaults to True. if False return (H, W, C)

        Returns:
            np.array: raw data (C, H, W) or (H, W)
        """

        slice_y, slice_x = self.window_raw.toslices()

        if self.cache_radiance:
            # Option B (opt-in): cache the full-spectrum windowed radiance so that
            # subsequent loads of band subsets become pure in-memory slices.
            # ``self._cache`` is a mutable dict shared with all clones built from
            # this instance (via ``attributes_set_if_exists``), so a single
            # decompression services every algorithm downstream.
            cached = self._cache.get(self._CACHE_KEY_RADIANCE)
            if cached is None:
                radiance = self.nc_ds['radiance']
                dims = radiance.dims
                cached = radiance.isel({dims[0]: slice_y, dims[1]: slice_x}).values
                self._cache[self._CACHE_KEY_RADIANCE] = cached
            data = cached[..., self.band_selection]
        else:
            # Default path: push the spatial (and, when possible, spectral) slice
            # into the NetCDF read via xarray .isel(). Avoids materialising the
            # full radiance variable in RAM, but re-reads from disk each call.
            radiance = self.nc_ds['radiance']
            dims = radiance.dims  # typically ('downtrack', 'crosstrack', 'bands')
            radiance = radiance.isel({dims[0]: slice_y, dims[1]: slice_x})

            if isinstance(self.band_selection, slice):
                radiance = radiance.isel({dims[2]: self.band_selection})
                data = radiance.values
            else:
                # Fancy indexing (list / array of indices) — push as far as we can
                # into the read (spatial), then numpy-slice the band axis.
                data = radiance.values[..., self.band_selection]

        # transpose to (C, H, W)
        if transpose and (len(data.shape) == 3):
            data = np.transpose(data, axes=(2, 0, 1))

        return data

    def clear_radiance_cache(self) -> None:
        """Drop the cached radiance window if present.

        After this call, the next ``load_raw()`` will re-read from disk. The
        ``_cache`` dict object itself is not replaced — clones built via
        ``__copy__`` / ``read_from_bands`` / ``to_crs`` / ``read_from_window``
        share the same dict by reference, so clearing through any clone is
        visible to all of them. Intended to be called from ``EmitProcessor.process``
        after all per-scene products are computed, to release the ~1.5 GB
        radiance array before the next scene is processed.
        """
        self._cache.pop(self._CACHE_KEY_RADIANCE, None)


    def georreference(self, data:np.array, 
                      fill_value_default:Optional[Union[int,float]]=None) -> GeoTensor:
        """
        Georreference an image in sensor coordinates to coordinates of the current 
        georreferenced object. If you do some processing with the raw data, you can 
        georreference the raw output with this function.

        Args:
            data (np.array): raw data (C, H, W) or (H, W). 

        Returns:
            GeoTensor: georreferenced version of data (C, H', W') or (H', W')
        
        Example:
            >>> emit_image = EMITImage("path/to/emit_image.nc")
            >>> emit_image_rgb = emit_image.read_from_bands([35, 23, 11])
            >>> data_rgb = emit_image_rgb.load_raw() # (3, H, W)
            >>> data_rgb_ortho = emit_image.georreference(data_rgb) # (3, H', W')
        """
        return georreference(self.glt_relative, data, self.valid_glt, 
                             fill_value_default=fill_value_default)

        
    @property
    def values(self) -> np.array:
        # return np.zeros(self.shape, dtype=self.dtype)
        raise self.load(boundless=True).values
    
    def __repr__(self)->str:
        return f""" 
         File: {self.filename}
         Transform: {self.transform}
         Shape: {self.shape}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         units: {self.units}
        """


def valid_mask(filename:str, with_buffer:bool=False, 
               dst_crs:Optional[Any]="UTM", 
               resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=60,
               include_spectf:bool=False) -> Tuple[GeoTensor, float]:
    """
    Loads the valid mask from the EMIT L2AMASK file.

    Args:
        filename (str): path to the L2AMASK file. e.g. EMIT_L2A_MASK_001_20220827T060753_2223904_013.nc
            or EMIT_L2A_MASK_003_20220827T060753.nc
        with_buffer (bool, optional): If True, the buffer band is used to compute the valid mask. Defaults to False.
        include_spectf (bool, optional): If True, the SpecTf ML cloud flag (v003 masks only) is also used.
            Defaults to False.

    Returns:
        GeoTensor: valid mask
    """
    
    if not HAS_XARRAY:
        raise ImportError("xarray is required to read EMIT images. Please install it with: pip install xarray")
    
    nc_ds = safe_open_netcdf(filename, cache=False, load=False)

    geotransform = nc_ds.attrs['geotransform']
    real_transform = rasterio.Affine(geotransform[1], geotransform[2], geotransform[0],
                                     geotransform[4], geotransform[5], geotransform[3])
    
    # Open location group to access glt data
    location_ds = safe_open_netcdf(filename, cache=False, load=False, group='location')
    glt_x = location_ds['glt_x'].values
    glt_y = location_ds['glt_y'].values
    location_ds.close()
    
    glt_arr = np.zeros((2,) + glt_x.shape, dtype=np.int32)
    glt_arr[0] = glt_x
    glt_arr[1] = glt_y
    # glt_arr -= 1 # account for 1-based indexing

    # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html
    glt = GeoTensor(glt_arr, transform=real_transform, 
                    crs=rasterio.crs.CRS.from_wkt(nc_ds.attrs['spatial_ref']),
                    fill_value_default=0)
    
    if dst_crs is not None:
        if dst_crs == "UTM":
            footprint = glt.footprint("EPSG:4326")
            dst_crs = get_utm_epsg(footprint)

        glt = read.read_to_crs(glt, dst_crs=dst_crs, 
                               resampling=rasterio.warp.Resampling.nearest, 
                               resolution_dst_crs=resolution_dst_crs)
    
    valid_glt = np.all(glt.values != glt.fill_value_default, axis=0)
    xmin, ymin, xmax, ymax = _bounds_indexes_raw(glt.values, valid_glt) # values are 1-based!

    glt_relative = glt.copy()
    glt_relative.values[0, valid_glt] -= xmin
    glt_relative.values[1, valid_glt] -= ymin
    sensor_params = safe_open_netcdf(filename, cache=False, load=False, group='sensor_band_parameters')
    mask_bands = sensor_params["mask_bands"].values
    sensor_params.close()
    band_index = mask_flag_indexes(mask_bands, with_buffer=with_buffer,
                                   include_spectf=include_spectf, source=filename)

    # Read the raw window the GLT references, so glt_relative indexes it from 0.
    mask_arr = nc_ds['mask'].values[ymin-1:ymax, xmin-1:xmax][..., band_index]
    invalidmask_raw = np.sum(mask_arr, axis=-1)
    invalidmask_raw = (invalidmask_raw >= 1)

    validmask = ~invalidmask_raw

    percentage_clear = 100 * (np.sum(validmask) / np.prod(validmask.shape))

    return georreference(glt_relative, validmask, valid_glt,
                         fill_value_default=False), percentage_clear