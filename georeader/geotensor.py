
import numpy as np
from typing import Any, Dict, Union, Tuple, Optional
import rasterio
import rasterio.windows
from georeader import window_utils
from georeader.window_utils import window_bounds
from itertools import product
from shapely.geometry import Polygon

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

    @property
    def attrs(self) -> Dict[str, Any]:
        return vars(self)

    def load(self) -> '__class__':
        return self

    def __copy__(self) -> '__class__':
        return GeoTensor(self.values.copy(), self.transform, self.crs, self.fill_value_default)

    def copy(self) -> '__class__':
        return self.__copy__()

    def isel(self, sel: Dict[str, slice]) -> '__class__':
        """
        Slicing with dict. It doesn't work with negative indexes!

        Args:
            sel: Dict with slice selection; i.e. `{"x": slice(10, 20), "y": slice(20, 340)}`.

        Returns:

        """
        for k in sel:
            if k not in self.dims:
                raise NotImplementedError(f"Axis {k} not in {self.dims}")

        slice_list = self._slice_tuple(sel)

        slices_window = []
        for k in ["y", "x"]:
            if k in sel:
                slices_window.append(sel[k])
            else:
                size = self.width if (k == "x") else self.height
                slices_window.append(slice(0, size))

        window_current = rasterio.windows.Window.from_slices(*slices_window, boundless=False) # if negative it will complain

        transform_current = rasterio.windows.transform(window_current, transform=self.transform)

        return GeoTensor(self.values[slice_list], transform_current, self.crs,
                         self.fill_value_default)

    def _slice_tuple(self, sel):
        slice_list = []
        # shape_ = self.shape
        # sel_copy = sel.copy()
        for _i, k in enumerate(self.dims):
            if k in sel:
                # sel_copy[k] = slice(max(0, sel_copy[k].start), min(shape_[_i], sel_copy[k].stop))
                slice_list.append(sel[k])
            else:
                slice_list.append(slice(None))
        return tuple(slice_list)

    def footprint(self, crs:Optional[str]=None) -> Polygon:
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

    def pad(self, pad_width=Dict[str, Tuple[int, int]], mode:str="constant",
            constant_values:Any=0)-> '__class__':
        """

        Args:
            pad_width: e.g. `{"x": (pad_x_0, pad_x_1), "y": (pad_y_0, pad_y_1)}`
            mode:
            constant_values:

        Returns:

        """

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

    def resize(self, output_shape:Optional[Tuple[int,int]]=None,
               anti_aliasing:bool=True, anti_aliasing_sigma:Optional[float]=None,
               interpolation:Optional[str]="bilinear",
               mode_pad:str="constant")-> '__class__':
        """
        Resize the geotensor to match a certain size output_shape. This function works with GeoTensors of 2D, 3D and 4D.
        The geoinformation of the output tensor is changed accordingly.

        Args:
            output_shape: output spatial shape
            anti_aliasing: Whether to apply a Gaussian filter to smooth the image prior to downsampling
            anti_aliasing_sigma:  anti_aliasing_sigma : {float, tuple of floats}, optional
                Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the
                downsampling factor, where s > 1
            interpolation: – algorithm used for resizing: 'nearest' | 'bilinear' | ‘bicubic’
            mode_pad: mode pad for resize function

        Returns:
             resized GeoTensor
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
                    output_tensor[i,j] = resize(self.values[i,j], output_shape, order=ORDERS[interpolation],
                                                anti_aliasing=anti_aliasing, preserve_range=False,
                                                cval=self.fill_value_default,mode=mode_pad,
                                                anti_aliasing_sigma=anti_aliasing_sigma)
            elif len(input_shape) == 3:
                for i in range(0,input_shape[0]):
                    output_tensor[i] = resize(self.values[i], output_shape, order=ORDERS[interpolation],
                                              anti_aliasing=anti_aliasing, preserve_range=False,
                                              cval=self.fill_value_default,mode=mode_pad,
                                              anti_aliasing_sigma=anti_aliasing_sigma)
            else:
                output_tensor[...] = resize(self.values, output_shape, order=ORDERS[interpolation],
                                            anti_aliasing=anti_aliasing, preserve_range=False,
                                            cval=self.fill_value_default,mode=mode_pad,
                                            anti_aliasing_sigma=anti_aliasing_sigma)

        return GeoTensor(output_tensor, transform=transform, crs=self.crs,
                         fill_value_default=self.fill_value_default)

    def write_from_window(self, data:Tensor, window:rasterio.windows.Window):
        window_data = rasterio.windows.Window(col_off=0, row_off=0,
                                              width=self.width, height=self.height)
        if not rasterio.windows.intersect(window, window_data):
            return

        assert data.shape[-2:] == (window.width, window.height), f"window {window} has different shape than data {data.shape}"

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
            GeoTensor

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







