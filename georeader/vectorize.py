from rasterio import features
import rasterio
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
import numpy as np
from typing import List, Optional, Union, Tuple
from georeader.abstract_reader import GeoData


def get_polygons(binary_mask: Union[np.ndarray, GeoData], min_area:float=25.5,
                 polygon_buffer:int=0, tolerance:float=1., transform: Optional[rasterio.Affine]=None) -> List[Polygon]:
    """
    Vectorize the polygons of the provided binary_mask.

    Args:
        binary_mask: (H, W) binary mask to rasterise
        min_area: polygons with pixel area lower than this will be filtered
        polygon_buffer: buffering of the polygons
        tolerance: to simplify the polygons
        transform: affine transformation of the binary_water_mask raster. It will be used only if binary mask is 
                  numpy array.

    Returns:
        list of vectorized polygons

    """

    if isinstance(binary_mask, np.ndarray):
        binary_mask_np = binary_mask
    else:
        binary_mask_np = binary_mask.values

        assert transform is None, "transform only must be used if input is np.ndarray"
        transform = binary_mask.transform

    shape_ = binary_mask_np.shape
    if len(shape_) != 2:
        binary_mask_np.squeeze()

    assert len(binary_mask_np.shape) == 2, f"Expected mask with 2 dim found {binary_mask_np.shape}"

    geoms_polygons = []
    polygon_generator = features.shapes(binary_mask_np.astype(np.int16),
                                        binary_mask_np)

    for polygon, _ in polygon_generator:
        p = shape(polygon)
        if polygon_buffer > 0:
            p = p.buffer(polygon_buffer)
        if p.area >= min_area:
            p = p.simplify(tolerance=tolerance)
            if transform is not None:
                p = transform_polygon(p, transform) # Convert polygon to raster coordinates
            geoms_polygons.append(p)

    return geoms_polygons


def transform_polygon(polygon:Union[Polygon, MultiPolygon], 
                      transform: rasterio.Affine, relative:bool=False,
                      shape_raster:Optional[Tuple[int,int]] = None) -> Union[Polygon, MultiPolygon]:
    """
    Transforms a polygon from pixel coordinates to the coordinates specified by the affine transform

    Args:
        polygon: polygon to transform
        transform: Affine transformation
        relative: if True, the polygon is transformed to relative coordinates (from 0 to 1)
        shape_raster: shape of the raster to which the polygon belongs. It is used only if relative is True

    Returns:
        polygon with coordinates transformed by the affine transformation

    """
    if relative:
        assert shape_raster is not None, "shape_raster must be provided if relative is True"
        transform = rasterio.Affine.scale(1/shape_raster[1], 1/shape_raster[0]) * transform
    
    geojson_dict = mapping(polygon)
    if geojson_dict["type"] == "Polygon":
        geojson_dict["coordinates"] = [geojson_dict["coordinates"]]

    multipol_coords = []
    for pol in geojson_dict["coordinates"]:
        pol_coords = []
        for shell_or_holes in pol:
            pol_out = []
            for coords in shell_or_holes:
                pol_out.append(transform * coords)

            pol_coords.append(pol_out)
        
        multipol_coords.append(pol_coords)

    if geojson_dict["type"] == "Polygon":
        geojson_dict["coordinates"] = multipol_coords[0]
    else:
        geojson_dict["coordinates"] = multipol_coords

    return shape(geojson_dict)