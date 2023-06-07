from georeader.geotensor import GeoTensor, concatenate
from georeader.abstract_reader import GeoData
from georeader.rasterio_reader import RasterioReader
from shapely.geometry import Polygon, MultiPolygon, mapping
from typing import Union, Dict, Optional, Tuple, List
import ee
import rasterio.windows
from rasterio import Affine
import numpy as np


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

def export_image_reproject(image: ee.Image,
                           transform:rasterio.Affine,
                           crs:str,
                           dimensions:Tuple[int, int],
                           bands:Optional[List[str]]=None,
                           donwload_path:Optional[str]=None) -> GeoData:
    # TODO default if not provided
    donwload_path

    download_url = image.getDownloadURL(params={
        "name": name,
        "bands": bands,
        "crs_transform": list(transform)[:6],
        "crs": str(crs).upper(),
        "dimensions": dimensions,
        "filePerBand": False})

    import requests
    r = requests.get(download_url, stream=True)
    filenamezip = f'/home/gonzalo/Downloads/{name}.zip'
    with open(filenamezip, "wb") as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)

    # TODO unzip?
    filename = f'zip+file:///{filenamezip}!{name}.tif'
    data = RasterioReader(filename)
    return data







