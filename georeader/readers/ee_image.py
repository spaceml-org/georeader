from georeader.geotensor import GeoTensor, concatenate
from georeader.abstract_reader import GeoData
from georeader.rasterio_reader import RasterioReader
from shapely.geometry import Polygon, MultiPolygon, mapping, box
from typing import Union, Dict, Optional, Tuple, List, Any
import rasterio.windows
from rasterio import Affine
import numpy as np
from collections import namedtuple
from georeader import read, window_utils
from io import BytesIO
import rasterio
from georeader import mosaic
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd
from tqdm import tqdm

try:
    import ee
except ImportError:
    raise ImportError("Please install the package 'earthengine-api' to use this module: pip install earthengine-api")

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
            pols_execute = []
            for sb in split_bounds(geometry.bounds):
                pol = box(*sb)
                if not geometry.intersects(pol):
                    continue
                pols_execute.append(pol.intersection(geometry))

            def process_bound(poly):
                gt = export_image_getpixels(asset_id, poly, 
                                            proj, bands_gee, dtype_dst, crs_polygon)
                return gt

            with ThreadPoolExecutor() as executor:
                geotensors = list(executor.map(process_bound, pols_execute))

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


def export_cube(query:gpd.GeoDataFrame, geometry:Union[Polygon, MultiPolygon], 
                transform:Optional[Affine]=None, crs:Optional[str]=None,
                dtype_dst:Optional[str]=None,
                bands_gee:Optional[List[str]]=None,
                crs_polygon:str="EPSG:4326",
                display_progress:bool=True) -> Optional[GeoTensor]:
    """
    Download all images in the query that intersects the geometry. 

    Note: This function is intended for small areas. If the area is too big that there are several images per day that intesesects the geometry 
         it will not group the images by day.

    Args:
        query (gpd.GeoDataFrame): dataframe from `georeaders.readers.query`. Required columns: gee_id, collection_name, bands_gee
        geometry (Union[Polygon, MultiPolygon]): geometry to export
        transform (Optional[Affine], optional): transform of the geometry. If None it will use the transform of the first image translated to the geometry. 
            Defaults to None.
        crs (Optional[str], optional): crs of the geometry. If None it will use the crs of the first image. Defaults to None.
        dtype_dst (Optional[str], optional): dtype of the output GeoTensor. Defaults to None.
        bands_gee (Optional[List[str]], optional): List of bands to export. If None it will use the bands_gee column in the query. Defaults to None.
        crs_polygon (_type_, optional): crs of the geometry. Defaults to "EPSG:4326".
        display_progress (bool, optional): Display progress bar. Defaults to False.

    Returns:
        GeoTensor: GeoTensor object with 4 dimensions: (time, band, y, x)
    """

    # TODO group by solar_day and satellite??
    if query.shape[0] == 0:
        return None
    
    # Check required columns
    required_columns = ["gee_id", "collection_name"]
    if bands_gee is None:
        required_columns.append("bands_gee")
    if not all([col in query.columns for col in required_columns]):
        raise ValueError(f"Columns {required_columns} are required in the query dataframe")

    # Get the first image to get the crs and transform if not provided
    if crs is None:
        first_image = query.iloc[0]
        if "proj" not in first_image:
            raise ValueError("proj column is required in the query dataframe if crs is not provided")
        crs = first_image["proj"]["crs"]
    
    if transform is None:
        first_image = query.iloc[0]
        if "proj" not in first_image:
            raise ValueError("proj column is required in the query dataframe if transform is not provided")
        transform = Affine(*first_image["proj"]["transform"])
    
    proj = {
            "crs": crs,
            "transform": list(transform)[:6]
        }
    
    # geotensor_list = []
    # for i, image in query.iterrows():
    #     asset_id = f'{image["collection_name"]}/{image["gee_id"]}'
    #     geotensor = export_image_getpixels(asset_id, geometry, proj, image["bands_gee"], crs_polygon=crs_polygon, dtype_dst=dtype_dst)
    #     geotensor_list.append(geotensor)
    
    def process_query_image(tuple_row):
        _, image = tuple_row
        asset_id = f'{image["collection_name"]}/{image["gee_id"]}'
        if bands_gee is None:
            bands_gee_iter = image["bands_gee"]
        else:
            bands_gee_iter = bands_gee
        geotensor = export_image_getpixels(asset_id, geometry, proj, bands_gee_iter, 
                                           crs_polygon=crs_polygon, dtype_dst=dtype_dst)
        return geotensor

    with ThreadPoolExecutor() as executor:
        geotensor_list = list(tqdm(executor.map(process_query_image, query.iterrows()), 
                                   total=query.shape[0], disable=not display_progress))
    
    return concatenate(geotensor_list)
    


                
