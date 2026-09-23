"""
EMIT product names and collection versions.

EMIT product ids exist in two forms. v001 names end with the orbit and DAAC scene number;
NASA dropped that suffix from v002 onwards::

    EMIT_L1B_RAD_001_20220827T060753_2223904_013
    EMIT_L1B_RAD_002_20260921T044051

The acquisition start (``YYYYmmddTHHMMSS``) is the only field shared by every product and
version of the same scene.
"""
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def _bounds_indexes_raw(glt:NDArray, valid_glt:NDArray) -> Tuple[int, int, int, int]:
        """ Return the bounds of the raw data: (min_x, min_y, max_x, max_y) """
        min_x = np.min(glt[0, valid_glt])
        max_x = np.max(glt[0, valid_glt])
        min_y = np.min(glt[1, valid_glt])
        max_y = np.max(glt[1, valid_glt])
        return min_x, min_y, max_x, max_y


# EMIT product ids exist in two forms:
#   v001: EMIT_L1B_RAD_001_20220827T060753_2223904_013  (acquisition, orbit, scene)
#   v002: EMIT_L1B_RAD_002_20260921T044051              (NASA dropped the orbit/scene suffix)
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
        >>> pid = parse_product_name('EMIT_L1B_RAD_002_20260921T044051.nc')
        >>> pid.with_product('L2A', 'MASK', '003').name
        'EMIT_L2A_MASK_003_20260921T044051'
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
            '/data/EMIT_L1B_RAD_002_20260921T044051.nc' or 'EMIT_L2A_MASK_003_20260921T044051.nc'.

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
