import math
from typing import Any, Optional, Tuple, Union

import numpy as np
import rasterio.warp
from rasterio import Affine
from rasterio.crs import CRS
from shapely.geometry import Point, mapping, shape
from shapely.geometry.base import BaseGeometry


def _normalize_crs(a_crs):
    """
    Normalize a CRS string by converting to lowercase and removing deprecated '+init=' prefix.

    This internal helper ensures consistent CRS comparison by normalizing various CRS
    string formats to a standard form.

    Args:
        a_crs: Coordinate reference system in any format (string, CRS object, etc.)
            Will be converted to string for normalization.

    Returns:
        str: Normalized lowercase CRS string without '+init=' prefix.

    Examples:
        >>> _normalize_crs("EPSG:4326")
        'epsg:4326'
        >>> _normalize_crs("+init=epsg:4326")
        'epsg:4326'
    """
    a_crs = str(a_crs)
    if "+init=" in a_crs:
        a_crs = a_crs.replace("+init=", "")
    return a_crs.lower()


def compare_crs(a_crs: str, b_crs: str) -> bool:
    """
    Compare two coordinate reference systems for equality.

    This function normalizes both CRS strings before comparison, handling different
    string formats and deprecated notation (e.g., '+init=epsg:4326' vs 'EPSG:4326').

    Args:
        a_crs (str): First coordinate reference system to compare.
        b_crs (str): Second coordinate reference system to compare.

    Returns:
        bool: True if the CRS are equivalent, False otherwise.

    Examples:
        >>> compare_crs("EPSG:4326", "epsg:4326")
        True
        >>> compare_crs("+init=epsg:4326", "EPSG:4326")
        True
        >>> compare_crs("EPSG:4326", "EPSG:32633")
        False
    """
    return _normalize_crs(a_crs) == _normalize_crs(b_crs)


def get_utm_epsg(point_or_geom: Union[Tuple[float, float], BaseGeometry], crs_point_or_geom: str = "EPSG:4326") -> str:
    """
    Determine the optimal UTM (Universal Transverse Mercator) EPSG code for a location.

    This function calculates the appropriate UTM zone based on longitude and latitude,
    selecting the correct northern (326XX) or southern (327XX) hemisphere code.
    For geometries, it uses the centroid to determine the zone.

    The UTM zone is calculated as: floor((lon + 180) / 6) % 60 + 1
    which divides the Earth into 60 zones of 6° longitude each.

    Args:
        point_or_geom (Union[Tuple[float, float], BaseGeometry]): Either a (longitude, latitude)
            tuple or a shapely geometry (Point, Polygon, etc.). If a geometry is provided,
            its centroid will be used.
        crs_point_or_geom (str, optional): Coordinate reference system of the input point
            or geometry. Defaults to "EPSG:4326" (WGS84 lat/lon).

    Returns:
        str: EPSG code string in format 'EPSG:326XX' (northern hemisphere) or
            'EPSG:327XX' (southern hemisphere), where XX is the zero-padded zone number.

    Examples:
        >>> # Point in Spain (northern hemisphere, zone 30)
        >>> get_utm_epsg((-3.7038, 40.4168))  # Madrid coordinates
        'EPSG:32630'

        >>> # Point in Australia (southern hemisphere, zone 56)
        >>> get_utm_epsg((151.2093, -33.8688))  # Sydney coordinates
        'EPSG:32756'

        >>> from shapely.geometry import Point
        >>> point = Point(-122.4194, 37.7749)  # San Francisco
        >>> get_utm_epsg(point)
        'EPSG:32610'

    References:
        - https://gis.stackexchange.com/questions/269518/auto-select-suitable-utm-zone-based-on-grid-intersection
        - https://stackoverflow.com/questions/40132542/get-a-cartesian-projection-accurate-around-a-lat-lng-pair/40140326#40140326
    """
    # Handle geometry input: extract centroid coordinates
    if isinstance(point_or_geom, BaseGeometry):
        # Transform geometry to WGS84 if needed
        if not compare_crs(crs_point_or_geom, "EPSG:4326"):
            point_or_geom = shape(rasterio.warp.transform_geom(crs_point_or_geom, "EPSG:4326", mapping(point_or_geom)))

        # Extract lon, lat from centroid
        lon, lat = list(point_or_geom.centroid.coords)[0]
    else:
        lon, lat = point_or_geom

    # Calculate UTM zone: Earth divided into 60 zones of 6° longitude each
    # Formula: floor((lon + 180) / 6) % 60 + 1
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)

    # Zero-pad single digit zones (e.g., '5' -> '05')
    if len(utm_band) == 1:
        utm_band = "0" + utm_band

    # Select northern (326XX) or southern (327XX) hemisphere
    if lat >= 0:
        epsg_code = "EPSG:326" + utm_band  # Northern hemisphere
        return epsg_code
    epsg_code = "EPSG:327" + utm_band  # Southern hemisphere
    return epsg_code


def get_utm_from_mgrs(mgrs_tile: str) -> Any:
    """
    Extract UTM coordinate reference system from an MGRS (Military Grid Reference System) tile identifier.

    MGRS tiles are identified by a zone number (2 digits), latitude band letter (1 character),
    and optionally grid square identifiers. This function uses only the first 3 characters
    (zone number + band letter) to determine the UTM CRS.

    Args:
        mgrs_tile (str): MGRS tile identifier. Examples: '39T', '31TBE', '10SGC'.
            Only the first 3 characters (2-digit zone + 1-letter band) are used.

    Returns:
        rasterio.crs.CRS: UTM CRS object with the appropriate zone and hemisphere.
            Southern hemisphere is determined when the band letter is < 'N'.

    Examples:
        >>> # Northern hemisphere tile (Spain)
        >>> crs = get_utm_from_mgrs('30TYK')
        >>> print(crs)
        PROJCS["UTM Zone 30, Northern Hemisphere",...]

        >>> # Southern hemisphere tile (band < 'N')
        >>> crs = get_utm_from_mgrs('56HLH')
        >>> print(crs.to_epsg())
        32756

    Note:
        The MGRS band letter 'N' is the boundary between northern and southern hemispheres.
        Bands C-M are southern, N-X are northern (excluding I and O).
    """

    # Alternative approach: lat, lon = mgrs.MGRS().toLatLon(mgrs_tile)
    # We use a simpler method based on the tile string itself

    # Extract zone (first 2 chars) and band letter (3rd char)
    # Example: '30TYK' -> zone=30, band='T'
    crs = CRS.from_dict({
        "proj": "utm",
        "zone": int(mgrs_tile[:2]),  # Zone number (01-60)
        "south": mgrs_tile[2] < "N",
    })  # Bands C-M are south, N-X are north
    return crs


def rasterio_crs(crs: Union[str, CRS, int]) -> CRS:
    """
    Convert various CRS representations to a rasterio CRS object.

    This utility function provides a unified interface for handling different CRS formats,
    ensuring compatibility with rasterio operations.

    Args:
        crs (Union[str, CRS, int]): The input CRS in one of the following formats:
            - str: EPSG code string (e.g., "EPSG:4326") or WKT string
            - CRS: Already a rasterio CRS object (returned as-is)
            - int: EPSG code as integer (e.g., 4326)

    Returns:
        CRS: A rasterio CRS object suitable for geospatial operations.

    Raises:
        ValueError: If the input CRS type is not supported (not str, CRS, or int).

    Examples:
        >>> # From EPSG string
        >>> crs = rasterio_crs("EPSG:4326")
        >>> print(crs.to_epsg())
        4326

        >>> # From EPSG integer
        >>> crs = rasterio_crs(32633)
        >>> print(crs)
        EPSG:32633

        >>> # From WKT string
        >>> wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",...]]'
        >>> crs = rasterio_crs(wkt)

        >>> # From existing CRS object
        >>> existing_crs = CRS.from_epsg(4326)
        >>> crs = rasterio_crs(existing_crs)  # Returns same object

    References:
        https://rasterio.readthedocs.io/en/latest/api/rasterio.crs.html
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


def res(transform: rasterio.Affine) -> Tuple[float, float]:
    """
    Compute the pixel resolution from an affine transformation matrix.

    This function calculates the physical distance (in CRS units) represented by one pixel
    in both x and y directions. For rotated or skewed transforms, this computes the
    Euclidean distance, accounting for both scaling and rotation.

    The resolution is computed using the Euclidean distance formula:
    - x_res = ||transform * (1, 0) - transform * (0, 0)||₂
    - y_res = ||transform * (0, 1) - transform * (0, 0)||₂

    Args:
        transform (rasterio.Affine): Affine transformation matrix that maps pixel
            coordinates (row, col) to georeferenced coordinates (x, y).

    Returns:
        Tuple[float, float]: Resolution as (x_resolution, y_resolution) in the units
            of the CRS (e.g., meters for UTM, degrees for WGS84).

    Examples:
        >>> # Simple north-up transform: 10m pixel size
        >>> transform = rasterio.Affine(10, 0, 0, 0, -10, 1000)
        >>> res(transform)
        (10.0, 10.0)

        >>> # Rotated transform
        >>> import math
        >>> angle = math.radians(30)  # 30 degree rotation
        >>> transform = rasterio.Affine(10*math.cos(angle), 10*math.sin(angle), 0,
        ...                              -10*math.sin(angle), -10*math.cos(angle), 1000)
        >>> res(transform)
        (10.0, 10.0)
    """
    # Calculate corner positions in georeferenced space
    # Origin (0,0) in pixel coordinates
    z0_0 = np.array(transform * (0, 0))
    # One pixel right (1,0) in pixel coordinates
    z1_0 = np.array(transform * (1, 0))
    # One pixel down (0,1) in pixel coordinates
    z0_1 = np.array(transform * (0, 1))

    # Euclidean distance gives resolution accounting for rotation/skew
    # x_res: distance in CRS units for one pixel in x direction
    # y_res: distance in CRS units for one pixel in y direction
    return float(np.sqrt(np.sum((z0_0 - z1_0) ** 2))), float(np.sqrt(np.sum((z0_0 - z0_1) ** 2)))


def distance_meters(point1: Point, point2: Point) -> float:
    """
    Calculate the distance in meters between two geographic points.

    This function assumes input points are in WGS84 (EPSG:4326) lat/lon coordinates.
    It projects both points to an appropriate UTM zone based on their midpoint,
    then calculates the Euclidean distance in meters.

    For accurate distances over long ranges, the midpoint's UTM zone is used to minimize
    projection distortion. The Euclidean distance formula in projected coordinates gives:

    d = √[(x₂ - x₁)² + (y₂ - y₁)²]  [meters]

    Args:
        point1 (Point): First location as a shapely Point in WGS84 coordinates (lon, lat).
        point2 (Point): Second location as a shapely Point in WGS84 coordinates (lon, lat).

    Returns:
        float: Distance between the two points in meters.

    Examples:
        >>> from shapely.geometry import Point
        >>> # Distance between two points ~1km apart
        >>> p1 = Point(-3.7038, 40.4168)  # Madrid
        >>> p2 = Point(-3.6938, 40.4168)  # ~1km east
        >>> dist = distance_meters(p1, p2)
        >>> print(f"{dist:.0f} meters")
        ~1000 meters

        >>> # Longer distance
        >>> madrid = Point(-3.7038, 40.4168)
        >>> barcelona = Point(2.1734, 41.3851)
        >>> dist = distance_meters(madrid, barcelona)
        >>> print(f"{dist/1000:.0f} km")
        ~504 km

    Note:
        For very long distances (>1000 km), geodesic distance calculations may be more accurate.
    """
    # Calculate midpoint to select optimal UTM zone
    # This minimizes projection distortion for the distance calculation
    mid_lon = (point1.x + point2.x) / 2
    mid_lat = (point1.y + point2.y) / 2
    utm_crs = get_utm_epsg((mid_lon, mid_lat), "EPSG:4326")

    # Project both points from WGS84 to UTM
    point1_utm = shape(rasterio.warp.transform_geom("EPSG:4326", utm_crs, mapping(point1)))
    point2_utm = shape(rasterio.warp.transform_geom("EPSG:4326", utm_crs, mapping(point2)))

    # Calculate Euclidean distance in UTM coordinates (meters)
    # d = √[(x₂ - x₁)² + (y₂ - y₁)²]
    distance = math.sqrt((point1_utm.x - point2_utm.x) ** 2 + (point1_utm.y - point2_utm.y) ** 2)

    return distance


def pixel_size_meters(
    point: Point, crs_transform: Any, transform: Affine, crs_point: Optional[Any] = "EPSG:4326"
) -> Tuple[float, float]:
    """
    Calculate the physical pixel size in meters at a specific geographic location.

    For projected CRS (e.g., UTM), this converts the native resolution to meters using the
    CRS's linear units. For geographic CRS (e.g., WGS84), it projects to UTM at the point's
    location to calculate the actual ground distance represented by one pixel.

    This is important because pixel size varies with latitude for geographic CRS:
    - At equator: 1° ≈ 111 km
    - At 60° latitude: 1° ≈ 55 km (in longitude)

    Args:
        point (Point): Location where pixel size should be calculated, as a shapely Point
            in the CRS specified by `crs_point`.
        crs_transform (Any): Coordinate reference system of the raster (CRS object, EPSG code,
            or WKT string). This is the CRS that the transform is defined in.
        transform (Affine): Affine transformation matrix of the raster that maps pixel
            coordinates to georeferenced coordinates in `crs_transform`.
        crs_point (Any, optional): Coordinate reference system of the input point.
            Defaults to "EPSG:4326" (WGS84 lat/lon).

    Returns:
        Tuple[float, float]: Pixel dimensions in meters as (width_meters, height_meters).

    Examples:
        >>> from shapely.geometry import Point
        >>> import rasterio
        >>>
        >>> # For a projected CRS (UTM): straightforward conversion
        >>> point = Point(500000, 4649776)  # UTM coordinates
        >>> transform = rasterio.Affine(10, 0, 499980, 0, -10, 4649786)
        >>> pixel_size_meters(point, "EPSG:32630", transform, crs_point="EPSG:32630")
        (10.0, 10.0)
        >>>
        >>> # For geographic CRS: varies with latitude
        >>> point_madrid = Point(-3.7038, 40.4168)  # Madrid in WGS84
        >>> transform_geo = rasterio.Affine(0.0001, 0, -4, 0, -0.0001, 41)  # ~11m at this latitude
        >>> pixel_size_meters(point_madrid, "EPSG:4326", transform_geo)
        (8.5, 11.1)  # Approximate values

    Note:
        For geographic CRS, the calculation projects neighboring pixels to UTM to determine
        the actual ground distance, accounting for latitude-dependent distortion.
    """
    # Normalize CRS to rasterio object
    crs_transform = rasterio_crs(crs_transform)

    # Fast path: for projected CRS, use the linear units factor
    if crs_transform.is_projected:
        res_t = res(transform)  # Resolution in native CRS units
        _, factor = crs_transform.linear_units_factor  # Conversion factor to meters
        return res_t[0] * factor, res_t[1] * factor

    # For geographic CRS: calculate actual ground distance at this location

    # 1. Transform point to raster's CRS if needed
    point_crs_transform = shape(rasterio.warp.transform_geom(crs_point, crs_transform, mapping(point)))
    lon, lat = list(point_crs_transform.coords)[0]

    # 2. Find pixel coordinates of this point
    transform_inv = ~transform  # Inverse transform: geo -> pixel
    col, row = transform_inv * (lon, lat)

    # 3. Calculate geographic coordinates of neighboring pixels
    # One pixel to the right (for width)
    p1_crs_transform = Point(*transform * (col + 1, row))
    # One pixel down (for height)
    p2_crs_transform = Point(*transform * (col, row + 1))

    # 4. Select appropriate UTM zone for this location
    utm_crs = get_utm_epsg((lon, lat), "EPSG:4326")

    # 5. Project all points to UTM to measure distances in meters
    point_utm = shape(rasterio.warp.transform_geom(crs_transform, utm_crs, mapping(point)))
    p1_utm = shape(rasterio.warp.transform_geom(crs_transform, utm_crs, mapping(p1_crs_transform)))
    p2_utm = shape(rasterio.warp.transform_geom(crs_transform, utm_crs, mapping(p2_crs_transform)))

    # 6. Calculate distances in meters
    pixel_width = p1_utm.x - point_utm.x  # Δx in meters
    pixel_height = p2_utm.y - point_utm.y  # Δy in meters

    return abs(pixel_width), abs(pixel_height)
