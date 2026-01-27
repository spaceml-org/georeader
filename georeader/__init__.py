import math
from typing import Tuple, Any, Union
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape, mapping, Point
import rasterio.warp
from rasterio.crs import CRS
from rasterio import Affine
import numpy as np
from typing import Optional


def _normalize_crs(a_crs):
    a_crs = str(a_crs)
    if "+init=" in a_crs:
        a_crs = a_crs.replace("+init=","")
    return a_crs.lower()


def compare_crs(a_crs:str, b_crs:str) -> bool:
    return _normalize_crs(a_crs) == _normalize_crs(b_crs)


def get_utm_epsg(point_or_geom: Union[Tuple[float,float],BaseGeometry], 
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


    crs = CRS.from_dict({"proj":"utm", 
                         "zone": int(mgrs_tile[:2]), 
                         "south": mgrs_tile[2] < "N"})
    return crs


def rasterio_crs(crs:Union[str, CRS, int]) -> CRS:
    """
    Convert input CRS to a rasterio CRS object.

    https://rasterio.readthedocs.io/en/latest/api/rasterio.crs.html

    Args:
        crs (Union[str, CRS, int]): The input CRS, which can be a string, a rasterio CRS object, or an EPSG code.

    Raises:
        ValueError: If the input CRS is not a string, rasterio CRS object, or EPSG code.

    Returns:
        CRS: A rasterio CRS object.
    """
    if isinstance(crs, str):
        if crs.upper().startswith("EPSG:"):
            crs = CRS.from_string(crs)
        else:
            crs = CRS.from_wkt(crs)
    elif isinstance(crs, int):
        crs = CRS.from_epsg(crs)
    elif not isinstance(crs, CRS):
        raise ValueError(f"crs must be str or rasterio.crs.CRS, but it is {type(crs)}")
    
    return crs


def res(transform:rasterio.Affine) -> Tuple[float, float]:
    """
    Computes the resolution from a given transform

    Args:
        transform:

    Returns:
        resolution (tuple of floats)
    """

    z0_0 = np.array(transform * (0, 0))
    z0_1 = np.array(transform * (0, 1))
    z1_0 = np.array(transform * (1, 0))

    return float(np.sqrt(np.sum((z0_0 - z1_0) ** 2))), float(np.sqrt(np.sum((z0_0 - z0_1) ** 2)))


def distance_meters(point1:Point, point2:Point) -> float:
    """
    Get the distance in meters between two points.

    Args:
        point1 (Point): A shapely Point object representing the first location.
        point2 (Point): A shapely Point object representing the second location.

    Returns:
        float: The distance in meters between the two points.
    """

    # Get the UTM CRS for the midpoint between the two points
    mid_lon = (point1.x + point2.x) / 2
    mid_lat = (point1.y + point2.y) / 2
    utm_crs = get_utm_epsg((mid_lon, mid_lat), "EPSG:4326")

    # Convert both points to UTM
    point1_utm = shape(rasterio.warp.transform_geom("EPSG:4326", utm_crs, mapping(point1)))
    point2_utm = shape(rasterio.warp.transform_geom("EPSG:4326", utm_crs, mapping(point2)))

    # Calculate the Euclidean distance in UTM coordinates
    distance = math.sqrt((point1_utm.x - point2_utm.x) ** 2 + (point1_utm.y - point2_utm.y) ** 2)

    return distance


def pixel_size_meters(point:Point, crs_transform:Any, transform:Affine,
                      crs_point:Optional[Any]="EPSG:4326") -> Tuple[float,float]:
    """
    Get the pixel size in meters for a given point and transform.

    Args:
        point (Point): A shapely Point object representing the location.
        crs_transform (Any): The coordinate reference system of the transform.
        transform (Affine): The affine transformation of the raster.
        crs_point (Any): The coordinate reference system of the point.

    Returns:
        Tuple[float, float]: A tuple containing the pixel size in meters (pixel_width, pixel_height).
    """

    crs_transform = rasterio_crs(crs_transform)
    if crs_transform.is_projected:
        res_t = res(transform)
        _, factor = crs_transform.linear_units_factor
        return res_t[0]*factor, res_t[1]*factor

    point_crs_transform = shape(rasterio.warp.transform_geom(crs_point, crs_transform, mapping(point)))
    lon, lat = list(point_crs_transform.coords)[0]
    transform_inv = ~transform
    col, row = transform_inv * (lon, lat)
    p1_crs_transform = Point(*transform * (col + 1, row))
    p2_crs_transform = Point(*transform * (col, row + 1))

    utm_crs = get_utm_epsg((lon, lat), "EPSG:4326")

    # Convert all points to UTM
    point_utm = shape(rasterio.warp.transform_geom(crs_transform, utm_crs, mapping(point)))
    p1_utm = shape(rasterio.warp.transform_geom(crs_transform, utm_crs, mapping(p1_crs_transform)))
    p2_utm = shape(rasterio.warp.transform_geom(crs_transform, utm_crs, mapping(p2_crs_transform)))

    pixel_width = p1_utm.x - point_utm.x
    pixel_height = p2_utm.y - point_utm.y

    return abs(pixel_width), abs(pixel_height)