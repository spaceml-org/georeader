from sentinelsat.sentinel import SentinelAPI
from georeader.readers import S2_SAFE_reader
from shapely.geometry import MultiPolygon, Polygon
from typing import Union, Optional, Tuple, Dict
from datetime import datetime, timedelta, timezone
import geopandas as gpd
import os
import json
from zoneinfo import ZoneInfo
import pandas as pd


def get_api(api_url:str='https://scihub.copernicus.eu/dhus/') -> SentinelAPI:
    home_dir = os.path.join(os.path.expanduser('~'), ".georeader")
    json_file = os.path.join(home_dir, "auth_S2.json")
    if not os.path.exists(json_file):
        os.makedirs(home_dir, exist_ok=True)
        with open(json_file, "w") as fh:
            json.dump({"user": "SET-USER", "password": "SET-PASSWORD"}, fh)

        raise FileNotFoundError(f"In order to query S2 images add user and password to file : {json_file}")

    with open(json_file, "r") as fh:
        data = json.load(fh)

    if data["user"] == "SET-USER":
        raise FileNotFoundError(f"In order to query S2 images add user and password to file : {json_file}")

    return SentinelAPI(data["user"], data["password"], api_url=api_url)


def query(area:Union[MultiPolygon,Polygon], date_start:datetime, date_end:datetime,
          producttype:str='S2MSI1C',api:Optional[SentinelAPI]=None,
          api_url:str='https://scihub.copernicus.eu/dhus/',
          cloudcoverpercentage:Tuple[int,int]=(0,80)) -> gpd.GeoDataFrame:
    """

    Args:
        area: area to query images
        date_start: datetime in a given timezone. If tz not provided UTC will be assumed.
        date_end: datetime in UTC. If tz not provided UTC will be assumed.
        producttype:'S2MSI1C' or 'S2MSI2A'
        api: Optional, if not provided will load the credentials from '~/.georeader/auth_S2.json'
        api_url: 'https://scihub.copernicus.eu/dhus/' or 'https://scihub.copernicus.eu/apihub'
        cloudcoverpercentage:

    Returns:

    """
    if api is None:
        api = get_api(api_url)
        # 'https://scihub.copernicus.eu/apihub'

    if date_start.tzinfo is not None:
        tz = date_start.tzinfo
        if isinstance(tz, ZoneInfo):
            tz = tz.key

        date_start = date_start.astimezone(timezone.utc)
        date_end = date_end.astimezone(timezone.utc)
    else:
        tz = timezone.utc

    assert date_start < date_end, "Date start must be before date end"

    products = api.query(area=str(area),
                         date=(date_start, date_end),
                         platformname='Sentinel-2',
                         producttype=producttype,
                         cloudcoverpercentage=cloudcoverpercentage)

    products_gpd = api.to_geodataframe(products)
    products_gpd.explore()
    products_gpd["mgrs_tile"] = products_gpd.title.apply(lambda x: S2_SAFE_reader.s2_name_split(x)[5])
    products_gpd["overlappercentage"] = products_gpd.geometry.apply(
        lambda x: x.intersection(area).area / area.area * 100)

    longitude = area.centroid.coords[0][0]
    hours_add = longitude * 12 / 180.

    products_gpd["solardatetime"] = products_gpd["datatakesensingstart"].apply(lambda x: x + timedelta(hours=hours_add))
    products_gpd["solarday"] = products_gpd["solardatetime"].apply(lambda x: x.strftime("%Y-%m-%d"))

    products_gpd["localdatetime"] = pd.to_datetime(products_gpd["datatakesensingstart"],unit='ms', utc=True).dt.tz_convert(tz)

    return products_gpd






