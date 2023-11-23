import warnings

from shapely.geometry import MultiPolygon, Polygon, mapping
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from georeader.readers import query_utils
from typing import Union, List, Tuple, Dict, Optional
import ee
import geopandas as gpd
import pandas as pd

def _rename_add_properties(image:ee.Image, properties_dict:Dict[str, str]) -> ee.Image:
    dict_set = {v: image.get(k) for k, v in properties_dict.items()}
    return image.set(dict_set)

def query_s1(area:Union[MultiPolygon,Polygon],
             date_start:datetime, date_end:datetime,
             filter_duplicates:bool=True,
             return_collection:bool=False)-> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, ee.ImageCollection]]:
    """
    Query S1 products from the Google Earth Engine

    Args:
        area:
        date_start:
        date_end:
        filter_duplicates:
        return_collection:

    Returns:

    """
    pol = ee.Geometry(mapping(area))

    if date_start.tzinfo is not None:
        tz = date_start.tzinfo
        if isinstance(tz, ZoneInfo):
            tz = tz.key

        date_start = date_start.astimezone(timezone.utc)
        date_end = date_end.astimezone(timezone.utc)
    else:
        tz = timezone.utc

    assert date_end >= date_start, f"Date end: {date_end} prior to date start: {date_start}"

    img_col = ee.ImageCollection('COPERNICUS/S1_GRD').\
       filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')).\
       filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')).\
       filter(ee.Filter.eq('instrumentMode', 'IW')).\
       filterDate(date_start.replace(tzinfo=None),
                  date_end.replace(tzinfo=None)).\
       filterBounds(pol)

    keys_query = {"orbitProperties_pass": "orbitProperties_pass"}

    geodf = img_collection_to_feature_collection(img_col,
                                                 ["system:time_start"] + list(keys_query.keys()),
                                                 as_geopandas=True, band_crs="VV")
    geodf.rename(keys_query, axis=1, inplace=True)
    geodf["title"] = geodf["gee_id"]
    geodf["collection_name"] = "COPERNICUS/S1_GRD"
    geodf = _add_stuff(geodf, area, tz)

    if filter_duplicates:
        geodf = query_utils.filter_products_overlap(area, geodf,
                                                    groupkey=["solarday", "satellite","orbitProperties_pass"]).copy()
        # filter img_col:
        img_col = img_col.filter(ee.Filter.inList("system:index", ee.List(geodf.index.tolist())))

    geodf.sort_values("utcdatetime")
    img_col = img_col.sort("system:time_start")

    if return_collection:
        return geodf, img_col

    return geodf

def figure_out_collection_landsat(tile:str) -> str:
    if tile.startswith("LC08"):
        if tile.endswith("T1_RT") or tile.endswith("T1"):
            return "LANDSAT/LC08/C02/T1_RT_TOA"
        elif tile.endswith("T2"):
            return "LANDSAT/LC08/C02/T2_TOA"
    elif tile.startswith("LC09"):
        if tile.endswith("T1"):
            return "LANDSAT/LC09/C02/T1_TOA"
        elif tile.endswith("T2"):
            return "LANDSAT/LC09/C02/T2_TOA"
    else:
        raise ValueError(f"Tile {tile} not recognized")


def query(area:Union[MultiPolygon,Polygon],
          date_start:datetime, date_end:datetime,
          producttype:str='S2', filter_duplicates:bool=True,
          return_collection:bool=False,
          add_s2cloudless:bool=False)-> Union[gpd.GeoDataFrame, Tuple[gpd.GeoDataFrame, ee.ImageCollection]]:
    """
    Query Landsat and Sentinel-2 products from the Google Earth Engine

    Args:
        area: area to query images in EPSG:4326
        date_start: datetime in a given timezone. If tz not provided UTC will be assumed.
        date_end: datetime in UTC. If tz not provided UTC will be assumed.
        producttype: 'S2', "Landsat"-> {"L8", "L9"}, "both" -> {"S2", "L8", "L9"}, "S2_SR", "L8", "L9"
        filter_duplicates: Filter S2 images that are duplicated
        return_collection: returns also the corresponding image collection
        add_s2cloudless: Adds a column that indicates if the s2cloudless image is available (from collection
            COPERNICUS/S2_CLOUD_PROBABILITY collection)

    Returns:
        geodataframe with available products in the given area and time range
        if `return_collection` is True it also returns the `ee.ImageCollection` of available images
    """

    pol = ee.Geometry(mapping(area))

    if date_start.tzinfo is not None:
        tz = date_start.tzinfo
        if isinstance(tz, ZoneInfo):
            tz = tz.key

        date_start = date_start.astimezone(timezone.utc)
        date_end = date_end.astimezone(timezone.utc)
    else:
        tz = timezone.utc

    assert date_end >= date_start, f"Date end: {date_end} prior to date start: {date_start}"

    if producttype == "S2_SR":
        image_collection_name = "COPERNICUS/S2_SR_HARMONIZED"
        keys_query = {"PRODUCT_ID": "title", 'CLOUDY_PIXEL_PERCENTAGE': "cloudcoverpercentage"}
    elif producttype == "S2":
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
    if "T1" in image_collection_name:
        image_collection_name_t2 = image_collection_name.replace("T1_RT", "T2").replace("T1","T2")
        img_col_t1 = ee.ImageCollection(image_collection_name_t2).filterDate(date_start.replace(tzinfo=None),
                                                                     date_end.replace(tzinfo=None)).filterBounds(
            pol)
        img_col = img_col.merge(img_col_t1)
    
    if (producttype == "Landsat") or (producttype == "both"):
        img_col_l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA").filterDate(date_start.replace(tzinfo=None),
                                                                   date_end.replace(tzinfo=None)).filterBounds(
        pol)
        img_col = img_col.merge(img_col_l9)
        img_col_l9_t2 = ee.ImageCollection("LANDSAT/LC09/C02/T2_TOA").filterDate(date_start.replace(tzinfo=None),
                                                                   date_end.replace(tzinfo=None)).filterBounds(
        pol)
        img_col = img_col.merge(img_col_l9_t2)

    geodf = img_collection_to_feature_collection(img_col,
                                                 ["system:time_start"] + list(keys_query.keys()),
                                                as_geopandas=True, band_crs="B2")

    geodf.rename(keys_query, axis=1, inplace=True)

    if geodf.shape[0] > 0:
        if (producttype == "Landsat") or (producttype == "both"):
            geodf["collection_name"] = geodf["title"].apply(figure_out_collection_landsat)
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
                                                        as_geopandas=True, band_crs="B2")
        geodf_s2["collection_name"] = "COPERNICUS/S2_HARMONIZED"
        geodf_s2.rename(keys_query_s2, axis=1, inplace=True)
        if geodf_s2.shape[0] > 0:
            if geodf.shape[0] == 0:
                geodf = geodf_s2
            else:
                geodf = pd.concat([geodf_s2, geodf], ignore_index=True)
            
            img_col_s2 = img_col_s2.map(lambda x: _rename_add_properties(x, keys_query_s2))
            img_col = img_col.merge(img_col_s2)

    if geodf.shape[0] == 0:
        warnings.warn(f"Not images found of collection {producttype} between dates {date_start} and {date_end}")
        if return_collection:
            return geodf, img_col
        return geodf

    if add_s2cloudless and producttype in ["both", "S2"]:
        values_s2_idx = geodf.title.apply(lambda x: x.startswith("S2"))
        indexes_s2 = geodf.gee_id[values_s2_idx].tolist()
        img_col_cloudprob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY").filterDate(date_start.replace(tzinfo=None),
                                                                              date_end.replace(
                                                                                  tzinfo=None)).filterBounds(
            pol)
        img_col_cloudprob = img_col_cloudprob.filter(ee.Filter.inList("system:index", ee.List(indexes_s2)))
        geodf_cloudprob = img_collection_to_feature_collection(img_col_cloudprob,
                                                               ["system:time_start"],
                                                               as_geopandas=True)
        geodf["s2cloudless"] = False
        list_geeid = geodf_cloudprob.gee_id.tolist()
        geodf.loc[values_s2_idx, "s2cloudless"] = geodf.loc[values_s2_idx, "gee_id"].apply(lambda x: x in list_geeid)


    geodf = _add_stuff(geodf, area, tz)

    if filter_duplicates:
        # TODO filter prioritizing s2cloudless?
        geodf = query_utils.filter_products_overlap(area, geodf,
                                                    groupkey=["solarday", "satellite"]).copy()
        # filter img_col:
        img_col = img_col.filter(ee.Filter.inList("title", ee.List(geodf.index.tolist())))

    geodf.sort_values("utcdatetime")
    img_col = img_col.sort("system:time_start")

    if return_collection:
        return geodf, img_col

    return geodf


def images_by_query_grid(images_available_gee:gpd.GeoDataFrame, grid:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    aois_images = images_available_gee.reset_index().sjoin(grid, how="inner").reset_index(drop=True)

    # Filter duplicated images for the same gridid
    indexes_selected = []
    for (day, satellite, gridid), products_gpd_day in aois_images.groupby(["solarday", "satellite", "index_right"]):
        area = grid.loc[gridid, "geometry"]
        idx_pols_selected = query_utils.select_polygons_overlap(products_gpd_day.geometry.tolist(), area)

        indexes_selected.extend(products_gpd_day.iloc[idx_pols_selected].index.tolist())

    aois_images = aois_images.loc[indexes_selected].copy()
    return aois_images.reset_index(drop=True)

def _add_stuff(geodf, area, tz):
    geodf["utcdatetime"] = pd.to_datetime(geodf["system:time_start"], unit='ms', utc=True)
    geodf["overlappercentage"] = geodf.geometry.apply(lambda x: x.intersection(area).area / area.area * 100)
    geodf["solardatetime"] = geodf.apply(lambda x: query_utils.solar_datetime(x.geometry, x.utcdatetime), axis=1)

    geodf["solarday"] = geodf["solardatetime"].apply(lambda x: x.strftime("%Y-%m-%d"))

    geodf["localdatetime"] = pd.to_datetime(geodf["utcdatetime"], unit='ms',
                                            utc=True).dt.tz_convert(tz)
    geodf["satellite"] = [x.split("_")[0] for x in geodf["title"]]
    geodf = geodf.set_index("title", drop=True)

    return geodf


def img_collection_to_feature_collection(img_col:ee.ImageCollection,
                                         properties:List[str],
                                         as_geopandas:bool=False,
                                         band_crs:Optional[str]=None) -> Union[ee.FeatureCollection, gpd.GeoDataFrame]:
    """Transforms the image collection to a feature collection """

    properties = ee.List(properties)

    def extractFeatures(img:ee.Image) -> ee.Feature:
        values = properties.map(lambda prop: img.get(prop))
        dictio = ee.Dictionary.fromLists(properties, values)
        dictio = dictio.set("gee_id", img.id())
        if band_crs is not None:
            proj = img.select(band_crs).projection()
            dictio = dictio.set("proj", proj)

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