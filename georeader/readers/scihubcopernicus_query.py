from sentinelsat.sentinel import SentinelAPI
from shapely.geometry import MultiPolygon, Polygon
from typing import Union, Optional, Tuple
from datetime import datetime, timedelta, timezone
from georeader.readers import query_utils
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
          cloudcoverpercentage:Tuple[int,int]=(0,80),filter_duplicates:bool=True) -> gpd.GeoDataFrame:
    """

    Args:
        area: area to query images in EPSG:4326
        date_start: datetime in a given timezone. If tz not provided UTC will be assumed.
        date_end: datetime in UTC. If tz not provided UTC will be assumed.
        producttype:'S2MSI1C' or 'S2MSI2A'
        api: Optional, if not provided will load the credentials from '~/.georeader/auth_S2.json'
        api_url: 'https://scihub.copernicus.eu/dhus/' or 'https://scihub.copernicus.eu/apihub'
        cloudcoverpercentage:
        filter_duplicates: Filter S2 images that are duplicated

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

    if isinstance(area, MultiPolygon):
        pols = area.geoms
    else:
        pols = [area]

    products_all = []
    for a in pols:
        res = api.query(area=str(a.simplify(tolerance=0.25)),
                        date=(date_start, date_end),
                        platformname='Sentinel-2',
                        producttype=producttype,
                        cloudcoverpercentage=cloudcoverpercentage)
        products_all.append(api.to_geodataframe(res))

    if len(products_all) == 1:
        products_gpd = products_all[0]
    else:
        products_gpd = pd.concat(products_all).drop_duplicates()

    products_gpd["overlappercentage"] = products_gpd.geometry.apply(
        lambda x: x.intersection(area).area / area.area * 100)

    longitude = area.centroid.coords[0][0]
    hours_add = longitude * 12 / 180.

    products_gpd["solardatetime"] = products_gpd["datatakesensingstart"].apply(lambda x: x + timedelta(hours=hours_add))
    products_gpd["solarday"] = products_gpd["solardatetime"].apply(lambda x: x.strftime("%Y-%m-%d"))

    products_gpd["utcdatetime"] = pd.to_datetime(products_gpd["datatakesensingstart"], unit='ms', utc=True)
    products_gpd["localdatetime"] = products_gpd["utcdatetime"].dt.tz_convert(tz)

    products_gpd = products_gpd.sort_values("utcdatetime")
    products_gpd = products_gpd.set_index("title", drop=True)

    if filter_duplicates:
        products_gpd = query_utils.filter_products_overlap(area, products_gpd).copy()

    return products_gpd







