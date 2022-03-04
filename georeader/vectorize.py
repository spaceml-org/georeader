from rasterio import features
import rasterio
from shapely.geometry import shape, mapping, Polygon
import numpy as np
from typing import List, Optional, Union
from georeader.abstract_reader import AbstractGeoData


def get_polygons(binary_mask: Union[np.ndarray, AbstractGeoData], min_area:float=25.5,
                 polygon_buffer:int=0, tolerance:float=1., transform: Optional[rasterio.Affine]=None) -> List[Polygon]:
    """

    Args:
        binary_mask: (H, W) binary mask to rasterise
        min_area: polygons with pixel area lower than this will be filtered
        polygon_buffer: buffering of the polygons
        tolerance: to simplify the polygons
        transform: affine transformation of the binary_water_mask raster. It will be used only if binary mask is 
                  numpy array.

    Returns:
        list of rasterised polygons

    """


    if isinstance(binary_mask, np.ndarray):
        binary_mask_np = binary_mask
    else:
        binary_mask_np = binary_mask.values
        shape_ = binary_mask_np.shape
        if len(shape_) != 2:
            binary_mask_np.squeeze()

        assert transform is None, "transform only must be used if input is np.ndarray"
        transform = binary_mask.transform


    assert len(binary_mask_np.shape) == 2, f"Expected mask with 2 dim found {binary_mask_np.shape}"

    geoms_polygons = []
    polygon_generator = features.shapes(binary_mask_np.astype(np.int16),
                                        binary_mask_np)

    for polygon, value in polygon_generator:
        p = shape(polygon)
        if polygon_buffer > 0:
            p = p.buffer(polygon_buffer)
        if p.area >= min_area:
            p = p.simplify(tolerance=tolerance)
            if transform is not None:
                p = transform_polygon(p, transform) # Convert polygon to raster coordinates
            geoms_polygons.append(p)

    return geoms_polygons


def transform_polygon(polygon:Polygon, transform: rasterio.Affine) -> Polygon:
    """
    Transforms a polygon from pixel coordinates to the coordinates specified by the affine transform

    Args:
        polygon: polygon to transform
        transform: Affine transformation

    Returns:
        polygon with coordinates transformed by the affine transformation

    """
    geojson_dict = mapping(polygon)
    out_coords = []
    for pol in geojson_dict["coordinates"]:
        pol_out = []
        for coords in pol:
            pol_out.append(transform * coords)

        out_coords.append(pol_out)

    geojson_dict["coordinates"] = out_coords

    return shape(geojson_dict)