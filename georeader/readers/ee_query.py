from shapely.geometry import MultiPolygon, Polygon, mapping
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from georeader.readers import query_utils
from typing import Union, List, Tuple, Dict
import ee
import geopandas as gpd
import pandas as pd

def _rename_add_properties(image:ee.Image, properties_dict:Dict[str, str]) -> ee.Image:
    dict_set = {v: image.get(k) for k, v in properties_dict.items()}
    return image.set(dict_set)

def query(area:Union[MultiPolygon,Polygon],
          date_start:datetime, date_end:datetime,
          producttype:str='S2MSI1C',filter_duplicates:bool=True,
          return_collection:bool=False)-> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, ee.ImageCollection]]:
    """

    Args:
        area: area to query images in EPSG:4326
        date_start: datetime in a given timezone. If tz not provided UTC will be assumed.
        date_end: datetime in UTC. If tz not provided UTC will be assumed.
        producttype: 'S2', "Landsat", "L8" or "L9"
        filter_duplicates: Filter S2 images that are duplicated
        return_collection: returns also the corresponding image collection

    Returns:

    """

    ee.Initialize()
    pol = ee.Geometry(mapping(area))

    if date_start.tzinfo is not None:
        tz = date_start.tzinfo
        if isinstance(tz, ZoneInfo):
            tz = tz.key

        date_start = date_start.astimezone(timezone.utc)
        date_end = date_end.astimezone(timezone.utc)
    else:
        tz = timezone.utc

    if producttype == "S2":
        image_collection_name = "COPERNICUS/S2_HARMONIZED"
        keys_query = {"PRODUCT_ID": "title", 'CLOUDY_PIXEL_PERCENTAGE': "cloudcoverpercentage"}
    elif producttype == "Landsat":
        image_collection_name = "LANDSAT/LC08/C02/T1_RT_TOA"
        keys_query = {"LANDSAT_PRODUCT_ID": "title", 'CLOUD_COVER': "cloudcoverpercentage"}
    elif producttype == "L8":
        image_collection_name = "LANDSAT/LC08/C02/T1_RT_TOA"
        keys_query = {"LANDSAT_PRODUCT_ID": "title", 'CLOUD_COVER': "cloudcoverpercentage"}
    elif producttype == "L9":
        image_collection_name = "LANDSAT/LC09/C02/T1_TOA"
        keys_query = {"LANDSAT_PRODUCT_ID": "title", 'CLOUD_COVER': "cloudcoverpercentage"}
    elif producttype == "both":
        image_collection_name = "LANDSAT/LC08/C02/T1_RT_TOA"
        keys_query = {"LANDSAT_PRODUCT_ID": "title", 'CLOUD_COVER': "cloudcoverpercentage"}
    else:
        raise NotImplementedError(f"Unknown product type {producttype}")

    img_col = ee.ImageCollection(image_collection_name).filterDate(date_start.replace(tzinfo=None),
                                                                   date_end.replace(tzinfo=None)).filterBounds(
        pol)
    if (producttype == "Landsat") or (producttype == "both"):
        img_col_l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA").filterDate(date_start.replace(tzinfo=None),
                                                                   date_end.replace(tzinfo=None)).filterBounds(
        pol)
        img_col = img_col.merge(img_col_l9)

    geodf = img_collection_to_feature_collection(img_col,
                                                 ["system:time_start"] + list(keys_query.keys()),
                                                as_geopandas=True)
    geodf.rename(keys_query, axis=1, inplace=True)

    if (producttype == "Landsat") or (producttype == "both"):
        geodf["collection_name"] = geodf["title"].apply(lambda x: "LANDSAT/LC08/C02/T1_RT_TOA" if x.startswith("LC08") else "LANDSAT/LC09/C02/T1_TOA")
    else:
        geodf["collection_name"] = image_collection_name

    img_col = img_col.map(lambda x: _rename_add_properties(x, keys_query))

    if producttype == "both":
        img_col_s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate(date_start.replace(tzinfo=None),
                                                                              date_end.replace(
                                                                                  tzinfo=None)).filterBounds(
            pol)
        keys_query_s2 = {"PRODUCT_ID": "title", 'CLOUDY_PIXEL_PERCENTAGE': "cloudcoverpercentage"}
        geodf_s2 = img_collection_to_feature_collection(img_col_s2,
                                                        ["system:time_start"] + list(keys_query_s2.keys()),
                                                        as_geopandas=True)
        geodf_s2["collection_name"] = "COPERNICUS/S2_HARMONIZED"
        geodf_s2.rename(keys_query_s2, axis=1, inplace=True)
        if geodf_s2.shape[0] > 0:
            geodf = pd.concat([geodf_s2, geodf], ignore_index=True)
            img_col_s2 = img_col_s2.map(lambda x: _rename_add_properties(x, keys_query_s2))
            img_col = img_col.merge(img_col_s2)

    if geodf.shape[0] == 0:
        if return_collection:
            return geodf, img_col
        return geodf

    geodf = _add_stuff(geodf, area, tz)

    if filter_duplicates:
        geodf = query_utils.filter_products_overlap(area, geodf,
                                                    groupkey=["solarday", "satellite"]).copy()
        # filter img_col:
        img_col = img_col.filter(ee.Filter.inList("title", ee.List(geodf.index.tolist())))

    geodf.sort_values("utcdatetime")
    img_col = img_col.sort("system:time_start")

    if return_collection:
        return geodf, img_col

    return geodf


def _add_stuff(geodf, area, tz):
    geodf["utcdatetime"] = pd.to_datetime(geodf["system:time_start"], unit='ms', utc=True)
    geodf["overlappercentage"] = geodf.geometry.apply(lambda x: x.intersection(area).area / area.area * 100)
    longitude = area.centroid.coords[0][0]
    hours_add = longitude * 12 / 180.

    geodf["solardatetime"] = geodf["utcdatetime"].apply(lambda x: x + timedelta(hours=hours_add))
    geodf["solarday"] = geodf["solardatetime"].apply(lambda x: x.strftime("%Y-%m-%d"))

    geodf["localdatetime"] = pd.to_datetime(geodf["utcdatetime"], unit='ms',
                                            utc=True).dt.tz_convert(tz)
    geodf["satellite"] = [x.split("_")[0] for x in geodf["title"]]
    geodf = geodf.set_index("title", drop=True)

    return geodf



def img_collection_to_feature_collection(img_col:ee.ImageCollection,
                                         properties:List[str],
                                         as_geopandas:bool=False) -> Union[ee.FeatureCollection, gpd.GeoDataFrame]:
    """Transforms the image collection to a feature collection """

    properties = ee.List(properties)

    def extractFeatures(img):
        values = properties.map(lambda prop: img.get(prop))
        dictio = ee.Dictionary.fromLists(properties, values)
        dictio = dictio.set("gee_id", img.id())
        return ee.Feature(img.geometry(), dictio)

    feature_collection = ee.FeatureCollection(img_col.map(extractFeatures))
    if as_geopandas:
        featcol_info = feature_collection.getInfo()
        if len(featcol_info["features"]) == 0:
            geodf = gpd.GeoDataFrame(geometry=[])
        else:
            geodf = gpd.GeoDataFrame.from_features(featcol_info, crs="EPSG:4326")

        return geodf

    return feature_collection