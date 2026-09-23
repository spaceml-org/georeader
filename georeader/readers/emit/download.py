"""
Download EMIT products from NASA Earthdata and build their LP DAAC links.
"""
import json
import os
from typing import Dict, Optional, Tuple

from georeader.readers.download_utils import download_product as download_product_base
from georeader.readers.emit.utils import _companion_version, _l1b_radiance_id

AUTH_METHOD = "auth" # "auth" or "token"
TOKEN = None

DAAC_URL = "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected"


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
        >>> get_radiance_link('EMIT_L1B_RAD_002_20260921T044051.nc')
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL1BRAD.002/EMIT_L1B_RAD_002_20260921T044051/EMIT_L1B_RAD_002_20260921T044051.nc'
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
        >>> get_l2amask_link('EMIT_L1B_RAD_002_20260921T044051.nc')
        'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2AMASK.003/EMIT_L2A_MASK_003_20260921T044051/EMIT_L2A_MASK_003_20260921T044051.nc'
    """
    rad = _l1b_radiance_id(tile)
    mask_version = _companion_version(rad, "MASK")
    if mask_version is None:
        rfl = rad.with_product("L2A", "RFL", _companion_version(rad, "RFL"))
        mask = rfl.with_product("L2A", "MASK")
        return f"{DAAC_URL}/EMITL2ARFL.{rfl.version}/{rfl.name}/{mask.name}.nc"
    mask = rad.with_product("L2A", "MASK", mask_version)
    return f"{DAAC_URL}/EMITL2AMASK.{mask_version}/{mask.name}/{mask.name}.nc"
