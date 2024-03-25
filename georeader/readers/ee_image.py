from georeader.geotensor import GeoTensor, concatenate
from georeader.abstract_reader import GeoData
from georeader.rasterio_reader import RasterioReader
from shapely.geometry import Polygon, MultiPolygon, mapping, box
from typing import Union, Dict, Optional, Tuple, List, Any
import ee
import rasterio.windows
from rasterio import Affine
import numpy as np
from collections import namedtuple
from georeader import read, window_utils
from io import BytesIO
import rasterio
from georeader import mosaic
from concurrent.futures import ThreadPoolExecutor

FakeGeoData=namedtuple("FakeGeoData",["crs", "transform"])


def export_image_fast(image:ee.Image, geometry:Union[ee.Geometry, Polygon, MultiPolygon],
                      cat_bands:bool=True,
                      fill_value_default:Optional[float]=0,
                      return_metadata:bool=False) -> Union[GeoTensor, Dict[str, GeoTensor],
                                                           Tuple[GeoTensor, Dict[str, str]],
                                                           Tuple[Dict[str, GeoTensor], Dict[str, str]]]:
    """
    Exports an image from the GEE as a GeoTensor. It uses `sampleRectangle` method to export which is limited to small
    arrays (i.e. <1000x1000 pixels).

    Args:
        image: ee.Image to export. Expected not an ArrayBand (see example)
        geometry: geometry to export as a ee.Geometry object or as a shapely polygon in EPSG:4326.
        cat_bands: if `True` concat the bands to return a single GeoTensor object.
        fill_value_default: Value used to fill the masked areas.
        return_metadata: if `True` it will alse return the metadata of the image (`image.clip(geometry).getInfo()`)

    Returns:
        GeoTensor object or Dict of band, GeoTensor.

    """
    if not isinstance(geometry, ee.Geometry):
        geometry = ee.Geometry(mapping(geometry))

    image_clip_info = image.clip(geometry).getInfo()

    feature_exported = image.sampleRectangle(geometry, defaultValue=fill_value_default).getInfo()

    out = {}
    band_ordered = []
    for band in image_clip_info["bands"]:
        band_id = band["id"]
        crs = band["crs"]
        transform_full_image = Affine(*band["crs_transform"])
        window = rasterio.windows.Window(col_off=band["origin"][0], row_off=band["origin"][1],
                                         width=band["dimensions"][0], height=band["dimensions"][1])
        transform = rasterio.windows.transform(window, transform_full_image)
        arr = np.array(feature_exported["properties"][band_id])
        data_tensor = GeoTensor(arr, crs=crs, transform=transform, fill_value_default=fill_value_default)
        out[band_id] = data_tensor
        band_ordered.append(band_id)

    if cat_bands:
        out = concatenate([out[b] for b in band_ordered])

    if return_metadata:
        return out, image_clip_info

    return out

def split_bounds(bounds:Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
    min_x, min_y, max_x, max_y = bounds
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    return [
        (min_x, min_y, mid_x, mid_y),  # Lower left quadrant
        (mid_x, min_y, max_x, mid_y),  # Lower right quadrant
        (min_x, mid_y, mid_x, max_y),  # Upper left quadrant
        (mid_x, mid_y, max_x, max_y),  # Upper right quadrant
    ]

def export_image_getpixels(asset_id: str,
                           geometry:Union[Polygon, MultiPolygon],
                           proj:Dict[str, Any],
                           bands_gee:List[str],
                           dtype_dst:Optional[str]=None,
                           crs_polygon:str="EPSG:4326") -> GeoTensor:
    """
    Exports an image from the GEE as a GeoTensor. It uses the `ee.data.getPixels` method to export.

    Args:
        asset_id (str): Name of the asset
        geometry (Union[Polygon, MultiPolygon]): geometry to export
        proj (Dict[str, Any]): Dict with fields:
            - crs: crs of the image
            - transform: transform of the image
        bands_gee (List[str]): List of bands to export
        crs_polygon (str, optional): crs of the geometry. Defaults to "EPSG:4326".

    Returns:
        GeoTensor: GeoTensor object
    """
    geodata = FakeGeoData(crs=proj["crs"], transform=Affine(*proj["transform"]))
    window_polygon = read.window_from_polygon(geodata, geometry, crs_polygon=crs_polygon,
                                              window_surrounding=True)
    window_polygon = window_utils.round_outer_window(window_polygon)
    transform_window = rasterio.windows.transform(window_polygon, geodata.transform)

    try:
        data_raw = ee.data.getPixels({"assetId": asset_id, 
                    'fileFormat':"GEO_TIFF", 
                    'bandIds':  bands_gee,
                    'grid': {
                        'dimensions': {
                            'height': window_polygon.height, 
                            'width': window_polygon.width
                        },
                        'affineTransform': {
                            'scaleX': transform_window.a,
                            'shearX': transform_window.b,
                            'translateX': transform_window.c,
                            'shearY': transform_window.d,
                            'scaleY': transform_window.e,
                            'translateY': transform_window.f
                        },
                        'crsCode': geodata.crs
                    }
                    })
        data = rasterio.open(BytesIO(data_raw))
        geotensor = GeoTensor(data.read(), transform=data.transform,
                             crs=data.crs, fill_value_default=data.nodata)
        if dtype_dst is not None:
            geotensor = geotensor.astype(dtype_dst)
            
    except ee.EEException as e:
        # Check if the exception starts with Total request size
        if str(e).startswith("Total request size"):
            # Split the geometry in two and call recursively
            bounds = geometry.bounds

            def process_bound(sb):
                poly = box(*sb)
                if not geometry.intersects(poly):
                    return None
                gt = export_image_getpixels(asset_id, poly, 
                                            proj, bands_gee, dtype_dst, crs_polygon)
                return gt

            with ThreadPoolExecutor() as executor:
                geotensors = list(executor.map(process_bound, split_bounds(bounds)))

            # Remove None values from the list
            geotensors = [gt for gt in geotensors if gt is not None]
            
            dst_crs = geotensors[0].crs
            aoi_dst_crs = window_utils.polygon_to_crs(geometry, 
                                                      crs_polygon=crs_polygon, 
                                                      dst_crs=dst_crs)
            
            geotensor = mosaic.spatial_mosaic(geotensors, 
                                              dtype_dst=dtype_dst,
                                              polygon=aoi_dst_crs, 
                                              dst_crs=dst_crs)
        else:
            raise e
    return geotensor







