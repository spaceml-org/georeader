
import numpy as np
from typing import Any, Dict, Union, Tuple, Optional, List
import rasterio
import rasterio.windows
from georeader import window_utils
from georeader.window_utils import window_bounds
from numpy.typing import ArrayLike
from itertools import product
from shapely.geometry import Polygon
import numbers

try:
    import torch
    import torch.nn.functional
    Tensor = Union[torch.Tensor, np.ndarray]
    torch_installed = True
except ImportError:
    Tensor =np.ndarray
    torch_installed = False

ORDERS = {
    'nearest': 0,
    'bilinear': 1,
    'bicubic': 2,
}



class GeoTensor:
    def __init__(self, values:Tensor,
                 transform:rasterio.Affine, crs:Any,
                 fill_value_default:Optional[Union[int, float]]=0):
        """
        This class is a wrapper around a numpy or torch tensor with geospatial information.

        Args:
            values (Tensor): numpy or torch tensor
            transform (rasterio.Affine): affine geospatial transform
            crs (Any): coordinate reference system
            fill_value_default (Optional[Union[int, float]], optional): Value to fill when 
            reading out of bounds. Could be None. Defaults to 0.

        Raises:
            ValueError: when the shape of the tensor is not 2d, 3d or 4d.
        """
        self.values = values
        self.transform = transform
        self.crs = crs
        self.fill_value_default = fill_value_default
        shape = self.shape
        if (len(shape) < 2) or (len(shape) > 4):
            raise ValueError(f"Expected 2d-4d array found {shape}")

    @property
    def dims(self) -> Tuple:
        # TODO allow different ordering of dimensions?
        shape = self.shape
        if len(shape) == 2:
            dims = ("y", "x")
        elif len(shape) == 3:
            dims = ("band", "y", "x")
        elif len(shape) == 4:
            dims = ("time", "band", "y", "x")
        else:
            raise ValueError(f"Unexpected 2d-4d array found {shape}")
        
        return dims
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "transform": [self.transform.a,self.transform.b,self.transform.c, 
                          self.transform.d, self.transform.e, self.transform.f] ,
            "crs": str(self.crs),
            "fill_value_default": self.fill_value_default
        }
    
    @classmethod
    def from_json(cls, json:Dict[str, Any]) -> '__class__':
        return cls(np.array(json["values"]), 
                   rasterio.Affine(*json["transform"]),
                   json["crs"], 
                   json["fill_value_default"])

    @property
    def shape(self) -> Tuple:
        return tuple(self.values.shape)

    @property
    def res(self) -> Tuple[float, float]:
        return window_utils.res(self.transform)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def height(self) -> int:
        return self.shape[-2]

    @property
    def width(self) -> int:
        return self.shape[-1]

    @property
    def count(self) -> int:
        return self.shape[-3]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return window_bounds(rasterio.windows.Window(row_off=0, col_off=0, height=self.height, width=self.width),
                             self.transform)

    def set_dtype(self, dtype):
        # TODO implement for torch tensor
        self.values = self.values.astype(dtype=dtype)
    
    def astype(self, dtype) -> '__class__':
        return GeoTensor(self.values.astype(dtype), 
                         self.transform, self.crs, self.fill_value_default)

    @property
    def attrs(self) -> Dict[str, Any]:
        return vars(self)

    def load(self) -> '__class__':
        return self

    def __copy__(self) -> '__class__':
        return GeoTensor(self.values.copy(), self.transform, self.crs, self.fill_value_default)

    def copy(self) -> '__class__':
        return self.__copy__()
    
    def same_extent(self, other:'__class__', precision:float=1e-3) -> bool:
        """
        Check if two GeoTensors have the same georeferencing (crs and transform)

        Args:
            other (__class__ | GeoData): GeoTensor to compare with. Other GeoData object can be passed (it requires crs, transform and shape attributes)
            precision (float, optional): precision to compare the transform. Defaults to 1e-3.

        Returns:
            bool: True if both GeoTensors have the same georeferencing.
        """
        return self.transform.almost_equals(other.transform, precision=precision) and window_utils.compare_crs(self.crs, other.crs) and (self.shape[-2:] == other.shape[-2:])
    
    def __add__(self, other:Union[numbers.Number,'__class__']) -> '__class__':
        """ 
        Add two GeoTensors. The georeferencing must match.

        Args:
            other (GeoTensor): GeoTensor to add.

        Raises:
            ValueError: if the georeferencing does not match.
            TypeError: if other is not a GeoTensor.

        Returns:
            GeoTensor: GeoTensor with the result of the addition.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other =  other.values
            else:
                raise ValueError("GeoTensor georref must match for addition. "
                                 "Use `read.read_reproject_like(other, self)` to "
                                 "to reproject `other` to `self` georreferencing.")
        
        result_values = self.values + other

        return GeoTensor(result_values, transform=self.transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)
    
    def __sub__(self, other:Union[numbers.Number,'__class__']) -> '__class__':
        """
        Substract two GeoTensors. The georeferencing must match.

        Args:
            other (GeoTensor): GeoTensor to add.

        Raises:
            ValueError: if the georeferencing does not match.
            TypeError: if other is not a GeoTensor.
        
        Returns:
            GeoTensor: GeoTensor with the result of the substraction.
            
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other =  other.values
            else:
                raise ValueError("GeoTensor georref must match for substraction. "
                                 "Use `read.read_reproject_like(other, self)` to "
                                 "to reproject `other` to `self` georreferencing.")
        
        result_values = self.values - other

        return GeoTensor(result_values, transform=self.transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)
    
    def __mul__(self, other:Union[numbers.Number,'__class__']) -> '__class__':
        """
        Multiply two GeoTensors. The georeferencing must match.

        Args:
            other (GeoTensor): GeoTensor to add.

        Raises:
            ValueError: if the georeferencing does not match.
            TypeError: if other is not a GeoTensor.
        
        Returns:
            GeoTensor: GeoTensor with the result of the multiplication.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other =  other.values
            else:
                raise ValueError("GeoTensor georref must match for multiplication. "
                                 "Use `read.read_reproject_like(other, self)` to "
                                 "to reproject `other` to `self` georreferencing.")
        
        result_values = self.values * other

        return GeoTensor(result_values, transform=self.transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)
    
    def __truediv__(self, other:Union[ArrayLike,'__class__']) -> '__class__':
        """
        Divide two GeoTensors. The georeferencing must match.

        Args:
            other (GeoTensor): GeoTensor to add.

        Raises:
            ValueError: if the georeferencing does not match.
            TypeError: if other is not a GeoTensor.
        
        Returns:
            GeoTensor: GeoTensor with the result of the division.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other =  other.values
            else:
                raise ValueError("GeoTensor georref must match for division. "
                                 "Use `read.read_reproject_like(other, self)` to "
                                 "to reproject `other` to `self` georreferencing.")
        
        result_values = self.values / other

        return GeoTensor(result_values, transform=self.transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)

    def __setitem__(self, index: np.ndarray, value: Union[np.ndarray, numbers.Number]) -> None:
        """
        Set the values of the GeoTensor object using an index and a new value.

        Args:
            index (tuple or numpy.ndarray): Index or boolean mask to apply to the GeoTensor values.
            value (numpy.ndarray): New value to assign to the GeoTensor values at the specified index.

        Raises:
            ValueError: If the index is not a tuple or a boolean numpy array with the same shape as the GeoTensor values.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> boolmask = gt.values > 0.5
            >>> gt[boolmask] = 0.5
        """
        if isinstance(index, np.ndarray) and (index.dtype == bool) and (index.shape == self.values.shape):
            # If the index is a boolean numpy array with the same shape as the values,
            # use it to mask the values and assign the new values to the masked values
            self.values[index] = value
        else:
            raise ValueError(f"Unsupported index type {type(index)} {index.dtype} {index} for GeoTensor set operation.")
    
    def squeeze(self) -> '__class__':
        """
        Remove single-dimensional entries from the shape of the GeoTensor values.
        It does not squeeze the spatial dimensions (last two dimensions).

        Returns:
            GeoTensor: GeoTensor with the squeezed values.
        """

        # squeeze all but last two dimensions
        squeezed_values = np.squeeze(self.values, axis=tuple(range(self.values.ndim - 2)))

        return GeoTensor(squeezed_values, transform=self.transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)
    
    def clip(self, a_min:Optional[np.array], a_max:Optional[np.array]) -> '__class__':
        """
        Clip the GeoTensor values between the GeoTensor min and max values.

        Args:
            a_min (float): Minimum value.
            a_max (float): Maximum value.

        Returns:
            GeoTensor: GeoTensor with the clipped values.
        """
        clipped_values = np.clip(self.values, a_min, a_max)
        return GeoTensor(clipped_values, transform=self.transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)

    
    def isel(self, sel: Dict[str, Union[slice, list, int]]) -> '__class__':
        """
        Slicing with dict. It doesn't work with negative indexes!

        Args:
            sel: Dict with slice selection; i.e. `{"x": slice(10, 20), "y": slice(20, 340)}`.

        Returns:
            GeoTensor: GeoTensor with the sliced values.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> gt.isel({"x": slice(10, 20), "y": slice(20, 340)})
        """
        for k in sel:
            if k not in self.dims:
                raise NotImplementedError(f"Axis {k} not in {self.dims}")

        slice_list = self._slice_tuple(sel)

        slices_window = []
        for k in ["y", "x"]:
            if k in sel:
                if not isinstance(sel[k], slice):
                    raise NotImplementedError(f"Only slice selection supported for x, y dims, found {sel[k]}")
                slices_window.append(sel[k])
            else:
                size = self.width if (k == "x") else self.height
                slices_window.append(slice(0, size))

        window_current = rasterio.windows.Window.from_slices(*slices_window, boundless=False) # if negative it will complain

        transform_current = rasterio.windows.transform(window_current, transform=self.transform)

        return GeoTensor(self.values[slice_list], transform_current, self.crs,
                         self.fill_value_default)

    def _slice_tuple(self, sel: Dict[str, Union[slice, list, int]]) -> tuple:
        slice_list = []
        # shape_ = self.shape
        # sel_copy = sel.copy()
        for _i, k in enumerate(self.dims):
            if k in sel:
                if not isinstance(sel[k], slice) and not isinstance(sel[k], list) and not isinstance(sel[k], int):
                    raise NotImplementedError(f"Only slice selection supported for x, y dims, found {sel[k]}")
                # sel_copy[k] = slice(max(0, sel_copy[k].start), min(shape_[_i], sel_copy[k].stop))
                slice_list.append(sel[k])
            else:
                slice_list.append(slice(None))
        return tuple(slice_list)

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        """Returns the footprint of the GeoTensor as a Polygon.

        Args:
            crs (Optional[str], optional): Coordinate reference system. Defaults to None.

        Returns:
            Polygon: footprint of the GeoTensor.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> gt.footprint(crs="EPSG:4326") # returns a Polygon in WGS84
        """
        pol = window_utils.window_polygon(rasterio.windows.Window(row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]),
                                          self.transform)
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)

    def __repr__(self)->str:
        return f""" 
         Transform: {self.transform}
         Shape: {self.shape}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         fill_value_default: {self.fill_value_default}
        """

    def pad(self, pad_width:Dict[str, Tuple[int, int]], mode:str="constant",
            constant_values:Optional[Any]=None)-> '__class__':
        """
        Pad the GeoTensor.

        Args:
            pad_width (_type_, optional):  dictionary with Tuple to pad for each dimension 
                `{"x": (pad_x_0, pad_x_1), "y": (pad_y_0, pad_y_1)}`. 
            mode (str, optional): pad mode (see np.pad or torch.nn.functional.pad). Defaults to "constant".
            constant_values (Any, optional): _description_. Defaults to `self.fill_value_default`.

        Returns:
            GeoTensor: padded GeoTensor.
        
        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> gt.pad({"x": (10, 10), "y": (10, 10)})
            >>> assert gt.shape == (3, 120, 120)
        """
        if constant_values is None:
            constant_values = self.fill_value_default

        # Pad the data
        pad_torch = False
        if torch_installed:
            if isinstance(self.values, torch.Tensor):
                pad_torch = True

        if pad_torch:
            pad_list_torch = []
            for k in reversed(self.dims):
                if k in pad_width:
                    pad_list_torch.extend(list(pad_width[k]))
                else:
                    pad_list_torch.extend([0,0])
            values_new = torch.nn.functional.pad(self.values, tuple(pad_list_torch), mode=mode,
                                                 value=constant_values)
        else:
            pad_list_np = []
            for k in self.dims:
                if k in pad_width:
                    pad_list_np.append(pad_width[k])
                else:
                    pad_list_np.append((0,0))
            values_new = np.pad(self.values, tuple(pad_list_np), mode=mode,
                                constant_values=constant_values)

        # Compute the new transform
        slices_window = []
        for k in ["y", "x"]:
            size = self.width if (k == "x") else self.height
            if k in pad_width:
                slices_window.append(slice(-pad_width[k][0], size+pad_width[k][1]))
            else:
                slices_window.append(slice(0, size))

        window_current = rasterio.windows.Window.from_slices(*slices_window, boundless=True)
        transform_current = rasterio.windows.transform(window_current, transform=self.transform)
        return GeoTensor(values_new, transform_current, self.crs,
                         self.fill_value_default)

    def resize(self, output_shape:Tuple[int,int],
               anti_aliasing:bool=True, anti_aliasing_sigma:Optional[Union[float,np.ndarray]]=None,
               interpolation:Optional[str]="bilinear",
               mode_pad:str="constant")-> '__class__':
        """
        Resize the geotensor to match a certain size output_shape. This function works with GeoTensors of 2D, 3D and 4D.
        The geoinformation of the output tensor is changed accordingly.

        Args:
            output_shape: output spatial shape
            anti_aliasing: Whether to apply a Gaussian filter to smooth the image prior to downsampling
            anti_aliasing_sigma:  anti_aliasing_sigma : {float}, optional
                Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the
                downsampling factor, where s > 1
            interpolation: – algorithm used for resizing: 'nearest' | 'bilinear' | ‘bicubic’
            mode_pad: mode pad for resize function

        Returns:
             resized GeoTensor
        
        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> resized = gt.resize((50, 50))
            >>> assert resized.shape == (3, 50, 50)
            >>> assert resized.res == (2*gt.res[0], 2*gt.res[1])
        """
        input_shape = self.shape
        spatial_shape = input_shape[-2:]
        resolution_or = self.res


        assert len(output_shape) == 2, f"Expected output shape to be the spatial dimensions found: {output_shape}"
        resolution_dst =  spatial_shape[0]*resolution_or[0]/output_shape[0], \
                          spatial_shape[1]*resolution_or[1]/output_shape[1]

        # Compute output transform
        transform_scale = rasterio.Affine.scale(resolution_dst[0]/resolution_or[0], resolution_dst[1]/resolution_or[1])
        transform = self.transform * transform_scale

        resize_kornia = False
        if torch_installed:
            if isinstance(self.values, torch.Tensor):
                resize_kornia = True

        if resize_kornia:
            # TODO
            # https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.resize
            raise NotImplementedError(f"Not implemented for torch Tensors")
        else:
            from skimage.transform import resize
            # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
            output_tensor = np.ndarray(input_shape[:-2]+output_shape, dtype=self.dtype)
            if len(input_shape) == 4:
                for i,j in product(range(0,input_shape[0]), range(0, input_shape[1])):
                    if (not anti_aliasing) or (anti_aliasing_sigma is None) or isinstance(anti_aliasing_sigma, numbers.Number):
                        anti_aliasing_sigma_iter = anti_aliasing_sigma
                    else:
                        anti_aliasing_sigma_iter = anti_aliasing_sigma[i, j]
                    output_tensor[i,j] = resize(self.values[i,j], output_shape, order=ORDERS[interpolation],
                                                anti_aliasing=anti_aliasing, preserve_range=False,
                                                cval=self.fill_value_default,mode=mode_pad,
                                                anti_aliasing_sigma=anti_aliasing_sigma_iter)
            elif len(input_shape) == 3:
                for i in range(0,input_shape[0]):
                    if (not anti_aliasing) or (anti_aliasing_sigma is None) or isinstance(anti_aliasing_sigma, numbers.Number):
                        anti_aliasing_sigma_iter = anti_aliasing_sigma
                    else:
                        anti_aliasing_sigma_iter = anti_aliasing_sigma[i]
                    output_tensor[i] = resize(self.values[i], output_shape, order=ORDERS[interpolation],
                                              anti_aliasing=anti_aliasing, preserve_range=False,
                                              cval=self.fill_value_default,mode=mode_pad,
                                              anti_aliasing_sigma=anti_aliasing_sigma_iter)
            else:
                output_tensor[...] = resize(self.values, output_shape, order=ORDERS[interpolation],
                                            anti_aliasing=anti_aliasing, preserve_range=False,
                                            cval=self.fill_value_default,mode=mode_pad,
                                            anti_aliasing_sigma=anti_aliasing_sigma)

        return GeoTensor(output_tensor, transform=transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)

    def write_from_window(self, data:Tensor, window:rasterio.windows.Window):
        """
        Writes array to GeoTensor values object at the given window position. If window surpasses the bounds of this
        object it crops the data to fit the object.

        Args:
            data: Tensor to write. Expected: spatial dimensions `window.width`, `window.height`. Rest: same as `self`
            window: Window object that specifies the spatial location to write the data
        
        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> data = np.random.rand(3, 50, 50)
            >>> window = rasterio.windows.Window(col_off=7, row_off=9, width=50, height=50)
            >>> gt.write_from_window(data, window)

        """
        window_data = rasterio.windows.Window(col_off=0, row_off=0,
                                              width=self.width, height=self.height)
        if not rasterio.windows.intersect(window, window_data):
            return

        assert data.shape[-2:] == (window.width, window.height), f"window {window} has different shape than data {data.shape}"
        assert data.shape[:-2] == self.shape[:-2], f"Dimension of data in non-spatial channels found {data.shape} expected: {self.shape}"

        slice_dict, pad_width = window_utils.get_slice_pad(window_data, window)
        slice_list = self._slice_tuple(slice_dict)
        # need_pad = any(p != 0 for p in pad_width["x"] + pad_width["y"])

        slice_data_spatial_x = slice(pad_width["x"][0], None if pad_width["x"][1] == 0 else -pad_width["x"][1])
        slice_data_spatial_y = slice(pad_width["y"][0], None if pad_width["y"][1] == 0 else -pad_width["y"][1])
        slice_data = self._slice_tuple({"x": slice_data_spatial_x, "y" : slice_data_spatial_y})
        self.values[slice_list] = data[slice_data]

    def read_from_window(self, window:rasterio.windows.Window, boundless:bool=True) -> '__class__':
        """
        returns a new GeoTensor object with the spatial dimensions sliced

        Args:
            window: window to slice the current GeoTensor
            boundless: read from window in boundless mode (i.e. if the window is larger or negative it will pad
                the GeoTensor with `self.fill_value_default`)

        Returns:
            GeoTensor object with the spatial dimensions sliced

        Raises:
            rasterio.windows.WindowError if `window` does not intersect the data

        """

        window_data = rasterio.windows.Window(col_off=0, row_off=0,
                                              width=self.width, height=self.height)
        if boundless:
            slice_dict, pad_width = window_utils.get_slice_pad(window_data, window)
            need_pad = any(p != 0 for p in pad_width["x"] + pad_width["y"])
            X_sliced = self.isel(slice_dict)
            if need_pad:
                X_sliced = X_sliced.pad(pad_width=pad_width, mode="constant",
                                        constant_values=self.fill_value_default)
            return X_sliced
        else:
            window_read = rasterio.windows.intersection(window, window_data)
            slice_y, slice_x = window_read.toslices()
            slice_dict = {"x": slice_x, "y": slice_y}
            slices_ = self._slice_tuple(slice_dict)
            transform_current = rasterio.windows.transform(window_read, transform=self.transform)
            return GeoTensor(self.values[slices_], transform_current, self.crs,
                             self.fill_value_default)


def concatenate(geotensors:List[GeoTensor]) -> GeoTensor:
    """
    Concatenates a list of geotensors, assert that all of them has same shape, transform and crs.

    Args:
        geotensors: list of geotensors to concat. All with same shape, transform and crs.

    Returns:
        geotensor with extra dim at the front: (len(geotensors),) + shape
    
    Examples:
        >>> gt1 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt2 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt3 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt = concatenate([gt1, gt2, gt3])
        >>> assert gt.shape == (3, 3, 100, 100)
    """
    assert len(geotensors) > 0, "Empty list provided can't concat"

    if len(geotensors) == 1:
        gt = geotensors[0].copy()
        gt.values = gt.values[np.newaxis]
        return gt

    first_geotensor = geotensors[0]
    array_out = np.zeros((len(geotensors),) + first_geotensor.shape,
                         dtype=first_geotensor.dtype)
    array_out[0] = first_geotensor.values

    for i, geo in enumerate(geotensors[1:]):
        assert geo.crs == first_geotensor.crs, f"Different crs in concat"
        assert geo.transform == first_geotensor.transform, f"Different transform in concat"
        assert geo.shape == first_geotensor.shape, f"Different shape in concat"
        assert geo.fill_value_default == first_geotensor.fill_value_default, "Different fill_value_default in concat"
        array_out[i + 1] = geo.values

    return GeoTensor(array_out, transform=first_geotensor.transform, crs=first_geotensor.crs,
                     fill_value_default=first_geotensor.fill_value_default)






