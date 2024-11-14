import rasterio
import rasterio.windows
import rasterio.warp
import rasterio.features
import numbers
import numpy as np
from math import ceil, copysign
from typing import Tuple, Union, Optional, Dict, Any
from collections import OrderedDict
import itertools
from georeader.geotensor import GeoTensor
from georeader import window_utils
from georeader.window_utils import PIXEL_PRECISION, pad_window, round_outer_window, _is_exact_round
from georeader.abstract_reader import GeoData
from itertools import product
from shapely.geometry import Polygon, MultiPolygon
import mercantile
from shapely.geometry import box
import rasterio.transform
import rasterio.rpc
import rasterio.crs
from numpy.typing import NDArray

SIZE_DEFAULT = 256
WEB_MERCATOR_CRS = "EPSG:3857"


def _round_all(x):
    x = tuple([int(round(xi)) for xi in x])
    return x


def _transform_from_crs(center_coords:Tuple[float, float], crs_input:Union[Dict[str,str],str],
                       crs_output:Union[Dict[str,str],str]) -> Tuple[float, float]:
    """ Transforms a coordinate tuple from crs_input to crs_output """

    coords_transformed = rasterio.warp.transform(crs_input, crs_output, [center_coords[0]], [center_coords[1]])
    return coords_transformed[0][0], coords_transformed[1][0]


def window_from_polygon(data_in: Union[GeoData, rasterio.DatasetReader],
                        polygon:Union[Polygon, MultiPolygon], crs_polygon:Optional[str]=None,
                        window_surrounding:bool=False) -> rasterio.windows.Window:
    """
    Obtains the data window that surrounds the polygon

    Args:
        data_in: Reader with crs and transform attributes
        polygon: Polygon or MultiPolygon
        crs_polygon: Optional coordinate reference system of the bounds. If not provided assumes same crs as `data_in`
        window_surrounding: The window surrounds the polygon. (i.e. window.row_off + window.height will not be a vertex)

    Returns:
        Window object with location in pixel coordinates relative to `data_in` of the polygon

    """
    # convert polygon to GeoData crs
    if (crs_polygon is not None) and not window_utils.compare_crs(crs_polygon, data_in.crs):
        # https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.transform_geom
        polygon_crs_data = window_utils.polygon_to_crs(polygon, crs_polygon, data_in.crs)
    else:
        polygon_crs_data = polygon

    if isinstance(polygon_crs_data, MultiPolygon):
        polygons = polygon_crs_data.geoms
    elif isinstance(polygon_crs_data, Polygon):
        polygons = [polygon_crs_data]
    else:
        raise NotImplementedError(f"Received shape of type {type(polygon_crs_data)} different from {Polygon} or {MultiPolygon}")

    # Collect all the pixel coordinates of the exterior polygons
    coords = []
    transform_inv = ~data_in.transform
    for pol in polygons:
        for pcoord in pol.exterior.coords:
            coords.append(transform_inv * pcoord)

    # Figure out min max rows and cols to build window
    row_off = min(c[1] for c in coords)
    col_off = min(c[0] for c in coords)

    row_max = max(c[1] for c in coords)
    col_max = max(c[0] for c in coords)
    if window_surrounding:
        row_max += 1
        col_max += 1

    return rasterio.windows.Window(row_off=row_off, col_off=col_off,
                                   width=col_max-col_off,
                                   height=row_max-row_off)


def window_from_bounds(data_in: Union[GeoData, rasterio.DatasetReader], 
                       bounds:Tuple[float, float, float, float],
                       crs_bounds:Optional[str]=None) -> rasterio.windows.Window:
    """
    Compute window to read in data_in from bounds in crs_bounds. If crs_bounds is None it assumes bounds are in the
    crs of data_in

    Args:
        data_in: Reader with crs and transform attributes
        bounds: tuple with bounds to find the corresponding window
        crs_bounds: Optional coordinate reference system of the bounds. If not provided assumes same crs as `data_in`

    Returns:
        Window object with location in pixel coordinates relative to `data_in` of the bounds

    """
    if (crs_bounds is not None) and not window_utils.compare_crs(crs_bounds, data_in.crs):

        bounds_in = rasterio.warp.transform_bounds(crs_bounds,
                                                   data_in.crs, *bounds)
    else:
        bounds_in = bounds

    window_in = rasterio.windows.from_bounds(*bounds_in, transform=data_in.transform)

    return window_in


def window_from_center_coords(data_in: Union[GeoData, rasterio.DatasetReader], 
                              center_coords:Tuple[float, float],
                              shape:Tuple[int,int], crs_center_coords:Optional[Any]=None) -> rasterio.windows.Window:
    """
     Compute window to read in `data_in` from the coordinates of the center pixel. If `crs_center_coords` is None it assumes
     `center_coords` are in the crs of `data_in`.

    Args:
        data_in: Reader with crs and transform attributes
        center_coords: Tuple with center coords (x, y) format
        shape: Tuple with shape to read (H, W) format
        crs_center_coords: Optional coordinate reference system of the bounds. If not provided assumes same crs as `data_in`

    Returns:
         Window object with location in pixel coordinates relative to `data_in` of the window centered on `center_coords`
    """

    if (crs_center_coords is not None) and not window_utils.compare_crs(crs_center_coords, data_in.crs):
        center_coords = _transform_from_crs(center_coords, crs_center_coords, data_in.crs)

    # The computation of the corner coordinates from the center is the same as in utils.polygon_slices
    transform = data_in.transform

    pixel_center_coords = ~transform * tuple(center_coords)
    pixel_upper_left =  _round_all((pixel_center_coords[0] - shape[1] / 2, pixel_center_coords[1] - shape[0] / 2))

    # OLD CODE that didn't support non-rectilinear transforms
    # assert transform.is_rectilinear(), "Transform is not rectilear"
    #
    # upper_left_coords = (center_coords[0] - (transform.a * shape[1] / 2),
    #                      center_coords[1] - (transform.e * shape[0] / 2))
    # pixel_upper_left = _round_all(~transform * upper_left_coords)

    window = rasterio.windows.Window(row_off=pixel_upper_left[1], col_off=pixel_upper_left[0],
                                     width=shape[1], height=shape[0])
    return window


def window_from_tile(data_in: Union[GeoData, rasterio.DatasetReader],
                     x:int, y:int, z:int) -> rasterio.windows.Window:
    """
    Returns the window corresponding to the x,y,z tile in the data_in.

    Tiles are TMS tiles defined as: (https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)

    Args:
        data_in (Union[GeoData, rasterio.DatasetReader]):  GeoData object
        x (int): x coordinate of the tile in the TMS system.
        y (int): y coordinate of the tile in the TMS system.
        z (int): z coordinate of the tile in the TMS system.

    Returns:
        rasterio.windows.Window: window corresponding to the tile
    """
    bounds_wgs = mercantile.xy_bounds(int(x), int(y), int(z))
    polygon_crs_webmercator = box(bounds_wgs.left, bounds_wgs.bottom, bounds_wgs.right, bounds_wgs.top)
    return window_from_polygon(data_in, polygon_crs_webmercator, WEB_MERCATOR_CRS,
                               window_surrounding=True)


def read_from_window(data_in: GeoData,
                     window: rasterio.windows.Window, return_only_data: bool = False,
                     trigger_load: bool = False,
                     boundless: bool = True) -> Union[GeoData, np.ndarray, None]:
    """
    Reads a window from data_in padding with `data_in.fill_value_default` if needed 
    (output GeoData will have `window.height`, `window.width` shape if boundless is `True`).

    Args:
        data_in: GeoData with "x" and "y" coordinates
        window: window to slice the GeoData with.
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoData georreferenced object.
        trigger_load: defaults to `False`. Trigger loading the data to memory.
        boundless: if `True` data read will always have the shape of the provided window
            (padding with `fill_value_default`)

    Returns:
        GeoData object
    """

    named_shape = OrderedDict(zip(data_in.dims, data_in.shape))

    window_data = rasterio.windows.Window(col_off=0, row_off=0,
                                          width=named_shape["x"], height=named_shape["y"])

    # get transform of current window
    transform = data_in.transform

    # Case the window does not intersect the data
    if not rasterio.windows.intersect([window_data, window]):
        if not boundless:
            return None

        expected_shapes = {"x": window.width, "y": window.height}
        shape = tuple([named_shape[s] if s not in ["x", "y"] else expected_shapes[s] for s in data_in.dims])
        data = np.zeros(shape, dtype=data_in.dtype)
        fill_value_default = getattr(data_in, "fill_value_default", 0)
        if fill_value_default != 0:
            data += fill_value_default
        if return_only_data:
            return data

        return GeoTensor(data, crs=data_in.crs,
                         transform=rasterio.windows.transform(window, transform=transform),
                         fill_value_default=fill_value_default)

    # Read data directly with rasterio (handles automatically the padding)
    data_sel = data_in.read_from_window(window=window, boundless=boundless)

    if return_only_data:
        return data_sel.values

    if trigger_load:
        data_sel = data_sel.load()

    return data_sel


def read_from_center_coords(data_in: GeoData, center_coords:Tuple[float, float], shape:Tuple[int,int],
                            crs_center_coords:Optional[Any]=None,
                            return_only_data:bool=False, trigger_load:bool=False,
                            boundless:bool=True) -> Union[GeoData, np.ndarray]:
    """
    Returns a chip of `data_in` centered on `center_coords` of shape `shape`.

    Args:
        data_in: GeoData object
        center_coords: x, y tuple of coords in `data_in` crs.
        shape: shape of the window to read
        crs_center_coords: CRS of center coords. If provided will check if it needs to reproject the coords before
            computing the reading window.
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoData georreferenced object.
        trigger_load: defaults to `False`. Trigger loading the data to memory.
        boundless: if `True` data read will always have the shape of the provided window
            (padding with `fill_value_default`)

    Returns:
        GeoData or np.array sliced from `data_in` of shape `shape`.

    """

    window = window_from_center_coords(data_in, center_coords, shape, crs_center_coords)

    return read_from_window(data_in, window=window, return_only_data=return_only_data,
                            trigger_load=trigger_load, boundless=boundless)


def read_from_bounds(data_in: GeoData, bounds: Tuple[float, float, float, float],
                     crs_bounds: Optional[str] = None, pad_add:Tuple[int, int]=(0, 0),
                     return_only_data: bool = False, trigger_load: bool = False,
                     boundless: bool = True) -> Union[GeoData, np.ndarray]:
    """
    Reads a slice of data_in covering the `bounds`.

    Args:
        data_in: GeoData with geographic info (crs and geotransform).
        bounds:  bounding box to read.
        crs_bounds: if not None will transform the bounds from that crs to the `data.crs` to read the chip.
        pad_add: Tuple[int, int]. Pad in pixels to add to the `window` that is read.This is useful when this function is called for
            interpolation/CNN prediction.
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoData georreferenced object.
        trigger_load: defaults to `False`. Trigger loading the data to memory.
        boundless: if `True` data read will always have the shape of the provided window
            (padding with `fill_value_default`)

    Returns:
        sliced GeoData
    """
    window_in = window_from_bounds(data_in, bounds, crs_bounds)
    if any(p > 0 for p in pad_add):
        window_in = pad_window(window_in, pad_add)  # Add padding for bicubic int or for co-registration
    window_in = round_outer_window(window_in)

    return read_from_window(data_in, window_in, return_only_data=return_only_data, trigger_load=trigger_load,
                            boundless=boundless)

def read_from_polygon(data_in: GeoData, polygon: Union[Polygon, MultiPolygon],
                      crs_polygon: Optional[str] = None, pad_add:Tuple[int, int]=(0, 0),
                      return_only_data: bool = False, trigger_load: bool = False,
                      boundless: bool = True, window_surrounding:bool=False) -> Union[GeoData, np.ndarray]:
    """
    Reads a slice of data_in covering the `polygon`.

    Args:
        data_in: GeoData with geographic info (crs and geotransform).
        polygon: Polygon or MultiPolygon that specifies the region to read.
        crs_polygon: if not None will transform the polygon from that crs to the data.crs to read the chip.
        pad_add: pad in pixels to add to the `window` that is read.This is useful when this function is called for
            interpolation/CNN prediction.
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoData georreferenced object.
        trigger_load: defaults to `False`. Trigger loading the data to memory.
        boundless: if `True` data read will always have the shape of the provided window
            (padding with `fill_value_default`)
        window_surrounding: The window surrounds the polygon. (i.e. `window.row_off` + `window.height` will not be a vertex)

    Returns:
        sliced GeoData
    """
    window_in = window_from_polygon(data_in, polygon, crs_polygon, 
                                    window_surrounding=window_surrounding)
    if any(p > 0 for p in pad_add):
        window_in = pad_window(window_in, pad_add)  # Add padding for bicubic int or for co-registration
    window_in = round_outer_window(window_in)

    return read_from_window(data_in, window_in, return_only_data=return_only_data, 
                            trigger_load=trigger_load,
                            boundless=boundless)


def read_reproject_like(data_in: GeoData, data_like: GeoData,
                        resolution_dst:Optional[Union[float, Tuple[float, float]]]=None,
                        resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
                        dtype_dst:Any=None, return_only_data: bool = False,
                        dst_nodata: Optional[int] = None) -> Union[GeoTensor, np.ndarray]:
    """
    Reads from `data_in` and reprojects to have the same extent and resolution than `data_like`.

    Args:
        data_in: GeoData to read and reproject. Expected coords "x" and "y".
        data_like: GeoData to get the bounds and resolution to reproject `data_in`.
        resampling: specifies how data is reprojected from `rasterio.warp.Resampling`.
        resolution_dst: if not None it will overwrite the resolution of `data_like`.
        dtype_dst: if None it will be inferred
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoTensor object (georreferenced array).
        dst_nodata: dst_nodata value

    Returns:
        GeoTensor read from `data_in` with same transform, crs, shape and bounds than `data_like`.
    """

    shape_out = data_like.shape[-2:]
    if resolution_dst is not None:
        if isinstance(resolution_dst, float):
            resolution_dst = (resolution_dst, resolution_dst)
        
        resolution_data_like = data_like.res

        shape_out = int(round(shape_out[0] / resolution_dst[0] * resolution_data_like[0])), \
                    int(round(shape_out[1] / resolution_dst[1] * resolution_data_like[1]))
        
    return read_reproject(data_in, dst_crs=data_like.crs, dst_transform=data_like.transform,
                          resolution_dst_crs=resolution_dst,
                          window_out=rasterio.windows.Window(0,0, width=shape_out[-1], height=shape_out[-2]),
                          resampling=resampling,dtype_dst=dtype_dst, return_only_data=return_only_data,
                          dst_nodata=dst_nodata)


def apply_anti_aliasing(data_in:GeoData, anti_aliasing_sigma:Optional[Union[float,np.ndarray]]=None,
                        resolution_dst:Optional[Union[float, Tuple[float, float]]]=None) -> GeoTensor:
    """
    Apply anti-aliasing to `data_in` assuming it will be downsampled to `resolution_dst`.

    Args:
        data_in (GeoData): GeoData to apply anti-aliasing
        anti_aliasing_sigma (Optional[Union[float,np.ndarray]], optional): Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the downsampling factor, where s > 1. Defaults to None.
        resolution_dst (Optional[Union[float, Tuple[float, float]]], optional): spatial resolution in data_in crs. Defaults
            to None.

    Returns:
        GeoTensor: GeoTensor with anti-aliasing applied
    """
    resolution_or = data_in.res
    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))
    
    scale = np.array([resolution_dst[0] / resolution_or[0], resolution_dst[1] / resolution_or[1]])
    
    if any(s1<s2 for s1,s2 in zip(resolution_or, resolution_dst)):
        # If we are downscaling the image and requested anti_aliasing
        try:
            from scipy import ndimage as ndi
        except ImportError:
            raise ImportError("scipy is required to apply anti-aliasing")

        # Copy or load the tensor in memory
        if isinstance(data_in, GeoTensor):
            data_in = data_in.copy()
        else:
            data_in = data_in.load()

        if anti_aliasing_sigma is None:
            anti_aliasing_sigma = np.mean(np.maximum(0, (scale - 1) / 2))


        # TODO if data_in.values is a torch.Tensor use kornia gaussian filter instead of ndi

        input_shape = data_in.shape
        if len(input_shape) == 4:
            for i, j in product(range(0, input_shape[0]), range(0, input_shape[1])):
                if isinstance(anti_aliasing_sigma, numbers.Number):
                    anti_aliasing_sigma_iter = anti_aliasing_sigma
                else:
                    anti_aliasing_sigma_iter = anti_aliasing_sigma[i, j]
                data_in.values[i, j] = ndi.gaussian_filter(data_in.values[i, j],
                                                           anti_aliasing_sigma_iter, cval=0, mode="reflect")
        elif len(input_shape) == 3:
            for i in range(0, input_shape[0]):
                if isinstance(anti_aliasing_sigma, numbers.Number):
                    anti_aliasing_sigma_iter = anti_aliasing_sigma
                else:
                    anti_aliasing_sigma_iter = anti_aliasing_sigma[i]

                data_in.values[i] = ndi.gaussian_filter(data_in.values[i],
                                                        anti_aliasing_sigma_iter, cval=0, mode="reflect")
        else:
            data_in.values[...] = ndi.gaussian_filter(data_in.values,
                                                      anti_aliasing_sigma, cval=0, mode="reflect")
    
    return data_in


def resize(data_in:GeoData, resolution_dst:Union[float, Tuple[float, float]],
           window_out:Optional[rasterio.windows.Window]=None,
           anti_aliasing:bool=True, anti_aliasing_sigma:Optional[Union[float,np.ndarray]]=None,
           resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
           return_only_data: bool = False)-> Union[
    GeoTensor, np.ndarray]:
    """
    Change the spatial resolution of data_in to `resolution_dst`. This function is a wrapper of the `read_reproject` function
    that adds anti_aliasing before reprojecting.

    Args:
        data_in: GeoData to change the resolution. Expected coords "x" and "y".
        resolution_dst: spatial resolution in data_in crs
        window_out: Optional. output size of the fragment to read and reproject. Defaults to the ceiling size
        anti_aliasing: Whether to apply a Gaussian filter to smooth the image prior to downsampling
        anti_aliasing_sigma:  anti_aliasing_sigma : {float}, optional
                Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the
                downsampling factor, where s > 1
        resampling: specifies how data is reprojected from `rasterio.warp.Resampling`.
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoTensor object (georreferenced array).

    Returns:
        GeoTensor with spatial resolution `resolution_dst`

    """
    resolution_or = data_in.res
    if isinstance(resolution_dst, numbers.Number):
        resolution_dst = (abs(resolution_dst), abs(resolution_dst))
    scale = np.array([resolution_dst[0] / resolution_or[0], resolution_dst[1] / resolution_or[1]])

    if window_out is None:
        spatial_shape = data_in.shape[-2:]

        # scale < 1 => make image smaller (resolution_or < resolution_dst)
        # scale > 1 => make image larger (resolution_or > resolution_dst)
        output_shape_exact = spatial_shape[0] / scale[0], spatial_shape[1] / scale[1]
        output_shape_rounded = round(output_shape_exact[0], ndigits=3), round(output_shape_exact[1], ndigits=3)
        output_shape = ceil(output_shape_rounded[0]), ceil(output_shape_rounded[1])
        window_out = rasterio.windows.Window(col_off=0, row_off=0, width=output_shape[1], height=output_shape[0])

    if anti_aliasing:
        data_in = apply_anti_aliasing(data_in, anti_aliasing_sigma=anti_aliasing_sigma, 
                                      resolution_dst=resolution_dst)

    return read_reproject(data_in, dst_crs=data_in.crs, resolution_dst_crs=resolution_dst,
                          dst_transform=data_in.transform, window_out=window_out,
                          resampling=resampling, return_only_data=return_only_data)


def read_to_crs(data_in:GeoData, dst_crs:Any, 
                resampling:rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
                resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=None,
                return_only_data: bool = False)-> Union[GeoTensor, np.ndarray]:
    """
    Change the crs of data_in to dst_crs. This function is a wrapper of the `read_reproject` function

    Args:
        data_in (GeoData): GeoData to reproyect
        dst_crs (Any): dst crs
        return_only_data (bool, optional): Defaults to False.

    Returns:
        Union[GeoTensor, np.ndarray]: data in dst_crs
    """
    if window_utils.compare_crs(data_in.crs, dst_crs):
        return data_in

    window_data, dst_transform = calculate_transform_window(data_in, dst_crs, resolution_dst_crs)


    return read_reproject(data_in, dst_crs=dst_crs,
                          dst_transform=dst_transform,
                          window_out=window_data,
                          resampling=resampling, return_only_data=return_only_data)


def calculate_transform_window(data_in:GeoData, dst_crs:Any, 
                               resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=None) -> Tuple[rasterio.Affine, rasterio.windows.Window]:
    """
    Calculate the default transform to reproject data to dst_crs with resolution_dst_crs

    Args:
        data_in (GeoData): GeoData to reproyect
        dst_crs (Any): dst crs
        resolution_dst_crs (Optional[Union[float, Tuple[float, float]]], optional): Defaults to None.
    """

    if resolution_dst_crs is not None:
        if isinstance(resolution_dst_crs, numbers.Number):
            resolution_dst_crs = (abs(resolution_dst_crs), abs(resolution_dst_crs))
    
    in_height, in_width = data_in.shape[-2:]
    dst_transform, width, height = rasterio.warp.calculate_default_transform(data_in.crs, dst_crs, in_width, in_height, *data_in.bounds,
                                                                             resolution=resolution_dst_crs)
    window_data = rasterio.windows.Window(0,0, width=width, height=height)

    return window_data, dst_transform


def read_reproject(data_in: GeoData, dst_crs: Optional[str]=None,
                   bounds: Optional[Tuple[float, float, float, float]]=None,
                   resolution_dst_crs: Optional[Union[float, Tuple[float, float]]]=None,
                   dst_transform:Optional[rasterio.Affine]=None,
                   window_out:Optional[rasterio.windows.Window]=None,
                   resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
                   dtype_dst:Any=None, return_only_data: bool = False, dst_nodata: Optional[int] = None) -> Union[
    GeoTensor, np.ndarray]:
    """
    This function slices the data by the bounds and reprojects it to the dst_crs and resolution_dst_crs

    Args:
        data_in: GeoData to read and reproject. Expected coords "x" and "y".
        bounds: Optional. bounds in CRS specified by `dst_crs`. If not provided `window_out` must be given.
        dst_crs: CRS to reproject.
        resolution_dst_crs: resolution in the CRS specified by `dst_crs`. If not provided will use the the resolution
            intrinsic of dst_transform.
        dst_transform: Optional dest transform. If not provided the dst_transform is a rectilinear transform computed
            with the bounds and resolution_dst_crs.
        window_out: Window out to read w.r.t `dst_transform`. If not provided it is computed from the bounds.
            Window out if provided has the output width and height of the reprojected data.
        resampling: specifies how data is reprojected from `rasterio.warp.Resampling`.
        dtype_dst: if None it will be data_in.dtype
        return_only_data: defaults to `False`. If `True` it returns a np.ndarray otherwise
            returns an GeoTensor object (georreferenced array).
        dst_nodata: dst_nodata value

    Returns:
        GeoTensor reprojected to dst_crs with resolution_dst_crs

    """

    named_shape = OrderedDict(zip(data_in.dims, data_in.shape))

    # Compute output transform
    dst_transform = window_utils.figure_out_transform(transform=dst_transform, bounds=bounds,
                                                      resolution_dst=resolution_dst_crs)

    # Compute size of window in out crs
    if window_out is None:
        assert bounds is not None, "Both window_out and bounds are None. This is needed to figure out the size of the output array"
        window_out = rasterio.windows.from_bounds(*bounds,
                                                  transform=dst_transform).round_lengths(op="ceil",
                                                                                         pixel_precision=PIXEL_PRECISION)

    crs_data_in = data_in.crs
    if dst_crs is None:
        dst_crs = crs_data_in

    #  if dst_crs == data_in.crs and the resolution is the same and window is exact return read_from_window
    if window_utils.compare_crs(dst_crs, crs_data_in):
        transform_data = data_in.transform
        if (dst_transform.a == transform_data.a) and (dst_transform.b == transform_data.b) and (
                dst_transform.d == transform_data.d) and (dst_transform.e == transform_data.e):
            # find shift between the two transforms
            x_dst, y_dst = dst_transform.c, dst_transform.f
            col_off, row_off = ~transform_data * (x_dst, y_dst)
            window_in_data = rasterio.windows.Window(col_off, row_off, 
                                                     window_out.width, window_out.height)

            if _is_exact_round(window_in_data.row_off) and _is_exact_round(window_in_data.col_off):
                window_in_data = window_in_data.round_offsets(op="floor", pixel_precision=PIXEL_PRECISION)
                return read_from_window(data_in, window_in_data, return_only_data=return_only_data, trigger_load=True)

    isbool_dtypein = data_in.dtype == 'bool'
    isbool_dtypedst = False

    cast = True
    if dtype_dst is None:
        cast = False
        dtype_dst = data_in.dtype
        if isbool_dtypein:
            isbool_dtypedst = True
    elif np.dtype(dtype_dst) == 'bool':
        isbool_dtypedst = True

    # Create out array for reprojection
    dict_shape_window_out = {"x": window_out.width, "y": window_out.height}
    shape_out = tuple([named_shape[s] if s not in ["x", "y"] else dict_shape_window_out[s] for s in named_shape])
    dst_nodata = dst_nodata or data_in.fill_value_default
    if isbool_dtypedst:
        dst_nodata = bool(dst_nodata)
    
    destination = np.full(shape_out, fill_value=dst_nodata, dtype=dtype_dst)

    polygon_dst_crs = window_utils.window_polygon(window_out, dst_transform)
    
    # If the polygon does not intersect the data return a GeoTensor with nodata
    if not data_in.footprint(crs=dst_crs).intersects(polygon_dst_crs):
        return GeoTensor(destination, transform=dst_transform, crs=dst_crs,
                         fill_value_default=dst_nodata)

    if not isinstance(data_in, GeoTensor):
        # Compute real polygon that is going to be read
        # Read a padded window of the input data. This data will be then used for reprojection
        geotensor_in = read_from_polygon(data_in, polygon_dst_crs, crs_polygon=dst_crs,
                                         pad_add=(3, 3), return_only_data=False,
                                         trigger_load=True)
    else:
        geotensor_in = data_in

    # Triggering load makes that fill_value_default goes to nodata
    np_array_in = np.asanyarray(geotensor_in.values)

    if cast:
        if isbool_dtypedst:
            np_array_in = np_array_in.astype(np.float32)
        else:
            np_array_in = np_array_in.astype(dtype_dst)
    elif isbool_dtypein:
        np_array_in = np_array_in.astype(np.float32)


    index_iter = [[(ns, i) for i in range(s)] for ns, s in named_shape.items() if ns not in ["x", "y"]]
    # e.g. if named_shape = {'time': 4, 'band': 2, 'x':10, 'y': 10} index_iter ->
    # [[('time', 0), ('time', 1), ('time', 2), ('time', 3)],
    #  [('band', 0), ('band', 1)]]

    for current_select_tuple in itertools.product(*index_iter):
        # current_select_tuple = (('time', 0), ('band', 0))
        i_sel_tuple = tuple(t[1] for t in current_select_tuple)

        np_array_iter = np_array_in[i_sel_tuple]
        if isbool_dtypedst:
            dst_iter_write = destination[i_sel_tuple].astype(np.float32)
            dst_nodata_iter = float(dst_nodata)
        else:
            dst_iter_write = destination[i_sel_tuple]
            dst_nodata_iter = dst_nodata

        rasterio.warp.reproject(
            np_array_iter,
            dst_iter_write,
            src_transform=geotensor_in.transform,
            src_crs=crs_data_in,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=geotensor_in.fill_value_default,
            dst_nodata=dst_nodata_iter,
            resampling=resampling)
        
        if isbool_dtypedst:
            destination[i_sel_tuple] = (dst_iter_write > .5)

    if return_only_data:
        return destination

    return GeoTensor(destination, transform=dst_transform, crs=dst_crs,
                     fill_value_default=dst_nodata)


def read_from_tile(data:GeoData, x:int, y:int, z:int, dst_crs:Optional[Any]=WEB_MERCATOR_CRS, 
                   out_shape:Optional[Tuple[int,int]]=(SIZE_DEFAULT, SIZE_DEFAULT), 
                   resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=None,
                   assert_if_not_intersects:bool=False) -> Optional[GeoTensor]:
    """
    Read a web mercator tile from a GeoData object. Tiles are TMS tiles defined as: (https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)

    Args:
        data (GeoData): GeoData object
        x (int): x. x coordinate of the tile in the TMS system.
        y (int): y. y coordinate of the tile in the TMS system.
        z (int): z. zoom level
        dst_crs (Optional[Any], optional): output crs. Defaults to WEB_MERCATOR_CRS. If None uses the crs of data.
        out_shape (Optional[Tuple[int,int]], optional): output size. Defaults to (SIZE_DEFAULT, SIZE_DEFAULT). If None it will be the size
            of the tile in the input resolution.
        resolution_dst_crs (Optional[Union[float, Tuple[float, float]]], optional): output resolution. Defaults to None. 
            If out_shape is not None it will be ignored. If None and out_shape is None the output will be at the resolution of the input data.
        assert_if_not_intersects (bool, optional): If True it will raise an error if the tile does not intersect the data. Defaults to False.

    Returns:
        GeoTensor: GeoTensor covering the tile or None if the tile does not intersect the data.
    """
    bounds_wgs = mercantile.xy_bounds(int(x), int(y), int(z))
    polygon_crs_webmercator = box(bounds_wgs.left, bounds_wgs.bottom, bounds_wgs.right, bounds_wgs.top)

    intersects = polygon_crs_webmercator.intersects(data.footprint(crs=WEB_MERCATOR_CRS))
    
    if not intersects:
        assert not assert_if_not_intersects, "Tile does not intersect data"
    else:
        return

    if out_shape is not None and hasattr(data, "read_from_tile"):
        return data.read_from_tile(x, y, z, dst_crs=dst_crs, out_shape=out_shape)
    
    if dst_crs is None:
        dst_crs = data.crs
        
    if window_utils.compare_crs(data.crs, dst_crs) and (out_shape is None) and (resolution_dst_crs is None):
        # read from polygon handles the case where the data does not intersect the polygon
        return read_from_polygon(data, polygon_crs_webmercator, WEB_MERCATOR_CRS, window_surrounding=True).load()
    
    if out_shape is not None:
        polygon_crs_dst = window_utils.polygon_to_crs(polygon_crs_webmercator, WEB_MERCATOR_CRS, dst_crs)
        bounds_dst = polygon_crs_dst.bounds
        dst_transform = rasterio.transform.from_bounds(*bounds_dst, 
                                                       width=out_shape[1], height=out_shape[0])
        window_data = rasterio.windows.Window(0, 0, width=out_shape[1], height=out_shape[0])
    else:
        if resolution_dst_crs is not None:
            if isinstance(resolution_dst_crs, numbers.Number):
                resolution_dst_crs = (abs(resolution_dst_crs), abs(resolution_dst_crs))
        
        polygon_crs_data = window_utils.polygon_to_crs(polygon_crs_webmercator, WEB_MERCATOR_CRS, data.crs)
        bounds_crs_data = polygon_crs_data.bounds
    
        in_height, in_width = data.shape[-2:]
        dst_transform, width, height = rasterio.warp.calculate_default_transform(data.crs, dst_crs, in_width, in_height, *bounds_crs_data,
                                                                                resolution=resolution_dst_crs)
        window_data = rasterio.windows.Window(0,0, width=width, height=height)
        dst_transform, window_data = calculate_transform_window(data, dst_crs, resolution_dst_crs)

    return read_reproject(data, dst_crs=dst_crs, dst_transform=dst_transform, 
                          window_out=window_data)


def read_rpcs(input_npy:NDArray, rpcs:rasterio.rpc.RPC, 
              fill_value_default:int=0,
              dst_crs:Optional[Any]=None,
              resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=None,
              resampling: rasterio.warp.Resampling = rasterio.warp.Resampling.cubic_spline,
              return_only_data:bool=False) -> GeoTensor:
    """
    This function georreferences an array using the RPCs. 
        The RPCs are used to compute the transform from the input array to the destination crs.

        This function assumes that the RPCs are in EPSG:4326.

    Args:
        input_npy (NDArray): Array to georeference. It must have 2, 3 or 4 dimensions.
        rpcs (rasterio.rpc.RPC): RPCs to compute the transform.
        fill_value_default (int, optional): how to encode the nodata value. Defaults to 0.
        dst_crs (Optional[Any], optional): Destination crs. Defaults to None.
            If None, the dst_crs is the same as in the RPC polynomial (EPSG:4326).
        resampling (rasterio.warp.Resampling, optional): Resampling method. 
            Defaults to rasterio.warp.Resampling.cubic_spline.
        return_only_data (bool, optional): If True it returns only the data. Defaults to False.

    Returns:
        GeoTensor: GeoTensor with the georeferenced array based on the RPCs.
    """
    
    isbool_dtypedst = input_npy.dtype == 'bool'
    if isbool_dtypedst:
        fill_value_default = bool(fill_value_default)

    assert input_npy.ndim >= 2 and input_npy.ndim <= 4, "Input array must have 2, 3 or 4 dimensions"

    named_shape = OrderedDict(reversed(list(zip(["y", "x", "band", "time"], 
                                                reversed(input_npy.shape)))))

    index_iter = [[(ns, i) for i in range(s)] for ns, s in named_shape.items() if ns not in ["x", "y"]]
    # e.g. if named_shape = {'time': 4, 'band': 2, 'x':10, 'y': 10} index_iter ->
    # [[('time', 0), ('time', 1), ('time', 2), ('time', 3)],
    #  [('band', 0), ('band', 1)]]

    if dst_crs is None:
        dst_crs = rasterio.crs.CRS.from_epsg(4326)

    src_crs = rasterio.crs.CRS.from_epsg(4326)

    if resolution_dst_crs is not None:
        if isinstance(resolution_dst_crs, float):
            resolution_dst_crs = (resolution_dst_crs, resolution_dst_crs)

    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
            src_crs=None, dst_crs=dst_crs, 
            width=input_npy.shape[-1], 
            height=input_npy.shape[-2], 
            resolution=resolution_dst_crs,
            rpcs=rpcs, dst_width=None, dst_height=None)

    destination = np.full(input_npy.shape[:-2] + (dst_height, dst_width),
                          fill_value=fill_value_default,
                          dtype=input_npy.dtype)

    for current_select_tuple in itertools.product(*index_iter):
        # current_select_tuple = (('time', 0), ('band', 0))
        i_sel_tuple = tuple(t[1] for t in current_select_tuple)

        np_array_iter = input_npy[i_sel_tuple]
        if isbool_dtypedst:
            dst_iter_write = destination[i_sel_tuple].astype(np.float32)
            fill_value_default_iter = float(fill_value_default)
        else:
            dst_iter_write = destination[i_sel_tuple]
            fill_value_default_iter = fill_value_default

        rasterio.warp.reproject(
            np_array_iter,
            dst_iter_write,
            src_transform=None,
            rpcs=rpcs,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=fill_value_default_iter,
            dst_nodata=fill_value_default_iter,
            resampling=resampling)
        
        if isbool_dtypedst:
            destination[i_sel_tuple] = (dst_iter_write > .5)

    if return_only_data:
        return destination

    return GeoTensor(destination, transform=dst_transform, crs=dst_crs,
                     fill_value_default=fill_value_default)
    
    

