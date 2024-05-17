import mercantile
import requests
from PIL import Image
from georeader.geotensor import GeoTensor
from georeader import mosaic
from shapely.geometry import Polygon, MultiPolygon
from typing import Union, Any
from io import BytesIO
import numpy as np
import georeader
from georeader import window_utils
from georeader import read
import rasterio.windows
import rasterio.transform
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor

def read_from_tileserver(tile_server:str, geometry:Union[Polygon, MultiPolygon],
                         zoom:int=16, crs_geometry:Any="EPSG:4326") -> GeoTensor:
    """
    Queries tiles from a tile server and returns it as a GeoTensor

    Args:
        tile_server (str): e.g. https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}
        geometry (Union[Polygon, MultiPolygon]): geometry to query
        zoom (int, optional): zoom level. Defaults to 15.
        crs_geometry (Any, optional): CRS of the geometry. Defaults to "EPSG:4326".

    Returns:
        GeoTensor: GeoTensor with the tile
    """
    if not georeader.compare_crs(crs_geometry, "EPSG:4326"):
        geometry = window_utils.polygon_to_crs(geometry, crs_geometry, "EPSG:4326")

    min_lon, min_lat, max_lon, max_lat = geometry.bounds
    tiles = mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom)
    tiles = [tile for tile in tiles if box(*mercantile.bounds(tile)).intersects(geometry)]
    # geotensors = []
    # for tile in tiles:
        
    #     rsp = requests.get(tile_server.format(x=tile.x, y=tile.y, z=tile.z))
    #     img = Image.open(BytesIO(rsp.content))
    #     xmin, ymin, xmax, ymax = window_utils.normalize_bounds(mercantile.xy_bounds(tile))
    #     img_np = np.array(img).transpose(2,0,1)
       
    #     transform = rasterio.transform.from_bounds(west=xmin, south=ymin, east=xmax, north=ymax,
    #                                                width=img_np.shape[2], height=img_np.shape[1])

    #     geotensors.append(GeoTensor(img_np, transform=transform, crs="EPSG:3857"))

    def read_tile(tile):
        rsp = requests.get(tile_server.format(x=tile.x, y=tile.y, z=tile.z))
        img = Image.open(BytesIO(rsp.content))
        xmin, ymin, xmax, ymax = window_utils.normalize_bounds(mercantile.xy_bounds(tile))
        img_np = np.array(img).transpose(2,0,1)
       
        transform = rasterio.transform.from_bounds(west=xmin, south=ymin, east=xmax, north=ymax,
                                                   width=img_np.shape[2], height=img_np.shape[1])

        return GeoTensor(img_np, transform=transform, crs="EPSG:3857")

    with ThreadPoolExecutor() as executor:
        geotensors = list(executor.map(lambda tile: read_tile(tile), tiles))
    
    if len(geotensors) == 1:
        return read.read_from_polygon(geotensors[0], polygon=geometry, crs_polygon="EPSG:4326")
    
    return mosaic.spatial_mosaic(geotensors, geometry, 
                                 dst_crs="EPSG:3857", crs_polygon="EPSG:4326")