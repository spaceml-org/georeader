from typing import List, Union
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
from datetime import datetime, timedelta


def select_polygons_overlap(polygons: List[Union[Polygon, MultiPolygon]], aoi: Union[Polygon, MultiPolygon]) -> List[int]:
    """
    Returns the indexes of polygons that maximally overlap the given aoi polygon

    Args:
        polygons: List of polygons (footprints of rasters)
        aoi: Polygon to figure out the maximal overlap

    Examples:
        See notebooks/Sentinel-2/query_mosaic_s2_images.ipynb for an example of use.

    Returns:
        List of indexes of polygons that cover the aoi polygon

    """

    idxs_out = []
    while (len(idxs_out) < len(polygons)) and not aoi.is_empty:
        # Select idx of polygon with bigger overlap
        idx_max = None
        value_overlap_max = 0
        for idx, pol in enumerate(polygons):
            if idx in idxs_out:
                continue

            overlap_area = pol.intersection(aoi).area / aoi.area
            if overlap_area > value_overlap_max:
                value_overlap_max = overlap_area
                idx_max = idx

        if idx_max is None:
            break

        pol_max = polygons[idx_max]
        aoi = aoi.difference(pol_max)
        idxs_out.append(idx_max)

    return idxs_out

def filter_products_overlap(area:Union[Polygon,MultiPolygon],
                            products_gpd:gpd.GeoDataFrame, groupkey:Union[str,List[str]]="solarday") -> gpd.GeoDataFrame:
    indexes_selected = []
    for day, products_gpd_day in products_gpd.groupby(groupkey):
        products_gpd_day_iter = products_gpd_day.sort_index()
        idx_pols_selected = select_polygons_overlap(products_gpd_day_iter.geometry.tolist(), area)

        indexes_selected.extend(products_gpd_day_iter.iloc[idx_pols_selected].index.tolist())

    return products_gpd.loc[indexes_selected]


def solar_datetime(area:Union[Polygon,MultiPolygon],
                   datetime_utc: datetime) -> datetime:
    longitude = area.centroid.coords[0][0]
    hours_add = longitude * 12 / 180.

    return datetime_utc + timedelta(hours=hours_add)