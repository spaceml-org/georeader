"""
EMIT L2A mask flags, selected by label because the band layout differs between mask versions.
"""
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import rasterio.crs
import rasterio.warp

from georeader import get_utm_epsg, read
from georeader.geotensor import GeoTensor
from georeader.griddata import georreference
from georeader.readers.emit.utils import _bounds_indexes_raw

try:
    from georeader.io import safe_open_netcdf
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    safe_open_netcdf = None

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


def valid_mask(filename:str, with_buffer:bool=False, 
               dst_crs:Optional[Any]="UTM", 
               resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=60,
               include_spectf:bool=False) -> Tuple[GeoTensor, float]:
    """
    Loads the valid mask from the EMIT L2AMASK file.

    Args:
        filename (str): path to the L2AMASK file. e.g. EMIT_L2A_MASK_001_20220827T060753_2223904_013.nc
            or EMIT_L2A_MASK_003_20260921T044051.nc
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
