from shapely.geometry import MultiPolygon, Polygon, mapping
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from georeader.readers import query_utils
from typing import Union, List
import ee
import geopandas as gpd
import pandas as pd


def query(area:Union[MultiPolygon,Polygon],
          date_start:datetime, date_end:datetime,
          producttype:str='S2MSI1C',filter_duplicates:bool=True)-> gpd.GeoDataFrame:
    """

    Args:
        area: area to query images in EPSG:4326
        date_start: datetime in a given timezone. If tz not provided UTC will be assumed.
        date_end: datetime in UTC. If tz not provided UTC will be assumed.
        producttype:'S2MSI1C' or 'S2MSI2A' "Landsat", "L8" or "L9"
        filter_duplicates: Filter S2 images that are duplicated

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
    if producttype == "S2MSI1C":
        image_collection_name = "COPERNICUS/S2_HARMONIZED"
        keys_query = {"PRODUCT_ID": "title", 'CLOUDY_PIXEL_PERCENTAGE': "cloudcoverpercentage"}
    elif producttype == "S2MSI2A":
        image_collection_name = "COPERNICUS/S2_SR_HARMONIZED"
        keys_query = {"PRODUCT_ID": "title", 'CLOUDY_PIXEL_PERCENTAGE': "cloudcoverpercentage"}
    elif producttype == "Landsat":
        image_collection_name = "LANDSAT/LC08/C02/T1_TOA"
        keys_query = {"LANDSAT_PRODUCT_ID": "title", 'CLOUD_COVER': "cloudcoverpercentage"}
    elif producttype == "L8":
        image_collection_name = "LANDSAT/LC08/C02/T1_TOA"
        keys_query = {"LANDSAT_PRODUCT_ID": "title", 'CLOUD_COVER': "cloudcoverpercentage"}
    elif producttype == "L9":
        image_collection_name = "LANDSAT/LC08/C02/T1_TOA"
        keys_query = {"LANDSAT_PRODUCT_ID": "title", 'CLOUD_COVER': "cloudcoverpercentage"}
    else:
        raise NotImplementedError(f"Unknown product type {producttype}")

    img_col = ee.ImageCollection(image_collection_name).filterDate(date_start.replace(tzinfo=None),
                                                                   date_end.replace(tzinfo=None)).filterBounds(
        pol)
    if producttype == "Landsat":
        img_col_l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA").filterDate(date_start.replace(tzinfo=None),
                                                                   date_end.replace(tzinfo=None)).filterBounds(
        pol)
        img_col = img_col.merge(img_col_l9)

    geodf = img_collection_to_feature_collection(img_col,
                                                 ["system:time_start"] + list(keys_query.keys()),
                                                as_geopandas=True)

    geodf = geodf.rename(keys_query, axis=1)

    geodf = _add_stuff(geodf, area, tz)
    geodf.sort_values("utcdatetime")

    geodf = geodf.set_index("title", drop=True)

    if filter_duplicates:
        geodf = query_utils.filter_products_overlap(area, geodf).copy()

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

    return geodf



def img_collection_to_feature_collection(img_col:ee.ImageCollection,
                                         properties:List[str],
                                         as_geopandas:bool=False) -> Union[ee.FeatureCollection, gpd.GeoDataFrame]:
    """Transforms the image collection to a feature collection """

    properties = ee.List(properties)

    def extractFeatures(img):
        values = properties.map(lambda prop: img.get(prop))
        dictio = ee.Dictionary.fromLists(properties, values)
        return ee.Feature(img.geometry(), dictio)

    feature_collection = ee.FeatureCollection(img_col.map(extractFeatures))
    if as_geopandas:
        geodf = gpd.GeoDataFrame.from_features(feature_collection.getInfo(), crs="EPSG:4326")
        if "system:time_start" in geodf.columns:
            geodf["datetime"] = pd.to_datetime(geodf["system:time_start"],unit="ms")
        return geodf

    return feature_collection