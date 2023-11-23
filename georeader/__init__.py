__version__ = "1.0.5"

import math
from typing import Tuple, Any, Union
from shapely.geometry.base import BaseGeometry
from shapely import Geometry
from shapely.geometry import shape, mapping
import rasterio.warp
from rasterio.crs import CRS


def _normalize_crs(a_crs):
    a_crs = str(a_crs)
    if "+init=" in a_crs:
        a_crs = a_crs.replace("+init=","")
    return a_crs.lower()


def compare_crs(a_crs:str, b_crs:str) -> bool:
    return _normalize_crs(a_crs) == _normalize_crs(b_crs)


def get_utm_epsg(point_or_geom: Union[Tuple[float,float],Geometry], 
                 crs_point_or_geom:str="EPSG:4326") -> str:
    """
    Based on lat and lng, return best utm epsg-code. For geometries it uses the centroid.

    https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
    https://stackoverflow.com/questions/40132542/get-a-cartesian-projection-accurate-around-a-lat-lng-pair/40140326#40140326
    Args:
        point_or_geom: tuple with longitude and latitude or shapely geometry.

    Returns: string with the best utm espg-code

    """
    if isinstance(point_or_geom, BaseGeometry):
        if not compare_crs(crs_point_or_geom, "EPSG:4326"):
             point_or_geom = shape(rasterio.warp.transform_geom(crs_point_or_geom, "EPSG:4326", mapping(point_or_geom)))

        lon, lat = list(point_or_geom.centroid.coords)[0]
    else:
        lon, lat = point_or_geom

    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+ utm_band
    if lat >= 0:
        epsg_code = 'EPSG:326' + utm_band
        return epsg_code
    epsg_code = 'EPSG:327' + utm_band
    return epsg_code

def get_utm_from_mgrs(mgrs_tile:str) -> Any:
    """
    Get the UTM CRS from a MGRS tile. It only uses 
    the first three digits of the MGRS tile.

    Args:
        mgrs_tile: MGRS tile. e.g. 39T or 
    
    Returns:
        EPSG CRS object
    
    """
    
    # lat, lon = mgrs.MGRS().toLatLon(mgrs_tile)


    crs = CRS.from_dict({"proj":"utm", "zone": int(mgrs_tile[:2]), 
                         "south": mgrs_tile[2] < "N"})
    return crs

