"""
Proba-V reader

Unnoficial Proba-V reader. This reader is based in the Proba-V user manual: 
https://publications.vito.be/2017-1333-probav-products-user-manual.pdf

Author:  Gonzalo Mateo-García
"""

import numpy as np
import h5py
from datetime import datetime
import os
import re
from datetime import timezone
from h5py import h5z
from rasterio import Affine
import rasterio
import rasterio.windows
from typing import Tuple, List, Optional, Union
from georeader import window_utils, geotensor
from numbers import Number
from shapely.geometry import Polygon
import rasterio.crs


FILTERS_HDF5 = { 'gzip': h5z.FILTER_DEFLATE,
                 'szip': h5z.FILTER_SZIP,
                 'shuffle': h5z.FILTER_SHUFFLE,
                 'lzf': h5z.FILTER_LZF,
                 'so': h5z.FILTER_SCALEOFFSET,
                 'f32': h5z.FILTER_FLETCHER32}

BAND_NAMES = ["BLUE", "RED", "NIR", "SWIR"]

def read_band_toa(dataset, band:str, slice_to_read:Tuple[slice, slice]):
    attrs = dataset[band].attrs
    if ("OFFSET" in attrs) and ("SCALE" in attrs):
        if (attrs["OFFSET"] != 0) or (attrs["SCALE"] != 1):
            return (dataset[band][slice_to_read] - attrs["OFFSET"]) / attrs["SCALE"]
    return dataset[band][slice_to_read]


def is_compression_available(dataset) -> bool:
    compression = dataset.compression
    if compression is not None:
        return h5z.filter_avail(FILTERS_HDF5[compression])


def assert_compression_available(dataset):
    assert is_compression_available(dataset), f"Compression format to read image: {dataset.compression} not available.\n Reinstall h5py with pip: \n pip install h5py --no-deps --ignore-installed"


class ProbaV:
    def __init__(self, hdf5_file:str, window:Optional[rasterio.windows.Window]=None,
                 level_name:str="LEVEL3"):
        self.hdf5_file = hdf5_file
        self.name = os.path.basename(self.hdf5_file)
        if level_name == "LEVEL2A":
            matches = re.match("PROBAV_L2A_\d{8}_\d{6}_(\d)_(\d..?M)_(V\d0\d)", self.name)
            if matches is not None:
                self.camera, self.res_name, self.version = matches.groups()
            self.toatoc = "TOA"
        elif level_name == "LEVEL3":
            matches = re.match("PROBAV_S1_(TO.)_.{6}_\d{8}_(\d..?M)_(V\d0\d)", self.name)
            if matches is not None:
                self.toatoc, self.res_name, self.version = matches.groups()
        else:
            raise NotImplementedError(f"Unknown level name {level_name}")

        try:
            with h5py.File(self.hdf5_file, "r") as input_f:
                # reference metadata: http://www.vito-eodata.be/PDF/image/PROBAV-Products_User_Manual.pdf
                valores_blue = input_f[f"{level_name}/RADIOMETRY/BLUE/{self.toatoc}"].attrs["MAPPING"][3:7].astype(np.float64)
                self.real_transform = Affine(a=valores_blue[2], b=0, c=valores_blue[0],
                                             d=0, e=-valores_blue[3], f=valores_blue[1])
                self.real_shape = input_f[f"{level_name}/RADIOMETRY/BLUE/{self.toatoc}"].shape
                # self.dtype_radiometry = input_f[f"{level_name}/RADIOMETRY/RED/{self.toatoc}"].dtype

                # Set to float because we're converting the image to TOA when reading (see read_radiometry function)
                self.dtype_radiometry = np.float32
                self.dtype_sm = input_f[f"{level_name}/QUALITY/SM"].dtype
                self.metadata = dict(input_f.attrs)
        except OSError as e:
            raise FileNotFoundError("Error opening file %s" % self.hdf5_file)

        if window is None:
            self.window_focus = rasterio.windows.Window(row_off=0, col_off=0,
                                                        width=self.real_shape[1],
                                                        height=self.real_shape[0])
        else:
            self.window_focus = rasterio.windows.Window(row_off=0, col_off=0,
                                                        width=self.real_shape[1],
                                                        height=self.real_shape[0])

        self.window_data = rasterio.windows.Window(row_off=0, col_off=0,
                                                   width=self.real_shape[1],
                                                   height=self.real_shape[0])

        if "OBSERVATION_END_DATE" in self.metadata:
            self.end_date = datetime.strptime(" ".join(self.metadata["OBSERVATION_END_DATE"].astype(str).tolist()+
                                                       self.metadata["OBSERVATION_END_TIME"].astype(str).tolist()),
                                              "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            self.start_date = datetime.strptime(" ".join(self.metadata["OBSERVATION_START_DATE"].astype(str).tolist()+
                                                         self.metadata["OBSERVATION_START_TIME"].astype(str).tolist()),
                                                "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            self.map_projection_wkt = " ".join(self.metadata["MAP_PROJECTION_WKT"].astype(str).tolist())

        # Proba-V images are lat/long
        self.crs = rasterio.crs.CRS({'init': 'epsg:4326'})

        # Proba-V images have four bands
        self.level_name = level_name

    def _get_window_pad(self, boundless:bool=True)->Tuple[rasterio.windows.Window, Optional[List]]:
        window_read = rasterio.windows.intersection(self.window_focus, self.window_data)

        if boundless:
            _, pad_width = window_utils.get_slice_pad(self.window_data, self.window_focus)
            need_pad = any(p != 0 for p in pad_width["x"] + pad_width["y"])
            if need_pad:
                pad_list_np = []
                for k in ["y", "x"]:
                    if k in pad_width:
                        pad_list_np.append(pad_width[k])
                    else:
                        pad_list_np.append((0, 0))
            else:
                pad_list_np = None
        else:
            pad_list_np = None


        return window_read, pad_list_np

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        # TODO load footprint from metadata?
        pol = window_utils.window_polygon(self.window_focus, self.transform)
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)
    
    def valid_footprint(self, crs:Optional[str]=None) -> Polygon:
        valids = self.load_mask()
        return valids.valid_footprint(crs=crs)

    def _load_bands(self, bands_names:Union[List[str],str], boundless:bool=True,
                    fill_value_default:Number=0) -> geotensor.GeoTensor:
        window_read, pad_list_np = self._get_window_pad(boundless=boundless)
        slice_ = window_read.toslices()
        if isinstance(bands_names, str):
            bands_names = [bands_names]
            flatten = True
        else:
            flatten = False

        with h5py.File(self.hdf5_file, "r") as input_f:
            bands_arrs = []
            for band in bands_names:
                data = read_band_toa(input_f, band, slice_)
                if pad_list_np is not None:
                    data = np.pad(data, tuple(pad_list_np), mode="constant",
                                  constant_values=fill_value_default)

                bands_arrs.append(data)

        if boundless:
            transform = self.transform
        else:
            transform = rasterio.windows.transform(window_read, self.real_transform)

        if flatten:
            img = bands_arrs[0]
        else:
            img = np.stack(bands_arrs, axis=0)

        return geotensor.GeoTensor(img, transform=transform, crs=self.crs,
                                   fill_value_default=fill_value_default)

    def save_bands(self, img:np.ndarray):
        """

        Args:
            img: (4, self.real_height, self.real_width, 4) tensor

        Returns:

        """
        assert img.shape[0] == 4, "Unexpected number of channels expected 4 found {}".format(img.shape)
        assert img.shape[1:] == self.real_shape, f"Unexpected shape expected {self.real_shape} found {img.shape[1:]}"

        # TODO save only window_focus?

        with h5py.File(self.hdf5_file, "r+") as input_f:
            for i, b in enumerate(BAND_NAMES):
                band_to_save = img[i]
                mask_band_2_save = np.ma.getmaskarray(img[i])
                band_to_save = np.clip(np.ma.filled(band_to_save, 0), 0, 2)
                band_name = f"{self.level_name}/RADIOMETRY/{b}/{self.toatoc}"
                attrs = input_f[band_name].attrs
                band_to_save *= attrs["SCALE"]
                band_to_save += attrs["OFFSET"]
                band_to_save = np.round(band_to_save).astype(np.int16)
                band_to_save[mask_band_2_save] = -1
                input_f[band_name][...] = band_to_save

    def load_radiometry(self, indexes:Optional[List[int]]=None, boundless:bool=True) -> geotensor.GeoTensor:
        if indexes is None:
            indexes = (0, 1, 2, 3)
        bands_names = [f"{self.level_name}/RADIOMETRY/{BAND_NAMES[i]}/{self.toatoc}" for i in indexes]
        return self._load_bands(bands_names, boundless=boundless,fill_value_default=-1/2000.)

    def load_sm(self, boundless:bool=True) -> geotensor.GeoTensor:
        """
        ## Reference of values in `SM` flags.

        From user manual http://www.vito-eodata.be/PDF/image/PROBAV-Products_User_Manual.pdf pag 67
        * Clear  ->    000
        * Shadow ->    001
        * Undefined -> 010
        * Cloud  ->    011
        * Ice    ->    100
        * `2**3` sea/land
        * `2**4` quality swir (0 bad 1 good)
        * `2**5` quality nir
        * `2**6` quality red
        * `2**7` quality blue
        * `2**8` coverage swir (0 no 1 yes)
        * `2**9` coverage nir
        * `2**10` coverage red
        * `2**11` coverage blue
        """
        return self._load_bands(f'{self.level_name}/QUALITY/SM', boundless=boundless, fill_value_default=0)

    def load_mask(self,boundless:bool=True) -> geotensor.GeoTensor:
        """
        Returns the valid mask (False if the pixel is out of swath or is invalid). This function loads the SM band

        Args:
            boundless (bool, optional): boundless option to load the SM band. Defaults to True.

        Returns:
            geotensor.GeoTensor: mask with the same shape as the image
        """
        valids = self.load_sm(boundless=boundless)
        valids.values = ~mask_only_sm(valids.values)
        valids.fill_value_default = False
        return valids
    
    def load_sm_cloud_mask(self, mask_undefined:bool=False, boundless:bool=True) -> geotensor.GeoTensor:
        sm = self.load_sm(boundless=boundless)
        cloud_mask = sm_cloud_mask(sm.values, mask_undefined=mask_undefined)
        return geotensor.GeoTensor(cloud_mask, transform=self.transform, crs=self.crs, fill_value_default=0)

    def is_recompressed_and_chunked(self) -> bool:
        original_bands = [f"{self.level_name}/RADIOMETRY/{b}/{self.toatoc}" for b in BAND_NAMES]
        original_bands.append(f"{self.level_name}/QUALITY/SM")
        with h5py.File(self.hdf5_file, "r") as input_:
            for b in original_bands:
                if input_[b].compression == "szip":
                    return False
                if (input_[b].chunks is None) or (input_[b].chunks[0] == 1):
                    return False
        return True

    def assert_can_be_read(self):
        original_bands = [f"{self.level_name}/RADIOMETRY/{b}/{self.toatoc}" for b in BAND_NAMES] + [
            f"{self.level_name}/QUALITY/SM"]
        with h5py.File(self.hdf5_file, "a") as input_:
            for name in original_bands:
                assert is_compression_available(input_[name]), f"Band {name} cannot be read. Compression: {input_[name].compression}"

    def recompress_bands(self, chunks:Tuple[int,int]=(512, 512), replace:bool=True, compression_dest:str="gzip"):
        original_bands = {b: f"{self.level_name}/RADIOMETRY/{b}/{self.toatoc}" for b in BAND_NAMES}
        original_bands.update({"SM": f"{self.level_name}/QUALITY/SM"})
        copy_bands = {k: v + "_NEW" for (k, v) in original_bands.items()}
        with h5py.File(self.hdf5_file, "a") as input_:
            for b in original_bands.keys():
                assert_compression_available(input_[original_bands[b]])
                data = input_[original_bands[b]][:]
                if copy_bands[b] in input_:
                    del input_[copy_bands[b]]

                ds = input_.create_dataset(copy_bands[b],
                                           data=data,
                                           chunks=chunks,
                                           compression=compression_dest)

                attrs_copy = input_[original_bands[b]].attrs
                for k, v in attrs_copy.items():
                    ds.attrs[k] = v

                if replace:
                    del input_[original_bands[b]]
                    input_[original_bands[b]] = input_[copy_bands[b]]
                    del input_[copy_bands[b]]

    @property
    def transform(self) -> Affine:
        return rasterio.windows.transform(self.window_focus, self.real_transform)

    @property
    def res(self) -> Tuple[float, float]:
        return window_utils.res(self.transform)

    @property
    def height(self) -> int:
        return self.window_focus.height

    @property
    def width(self) -> int:
        return self.window_focus.width

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return window_utils.window_bounds(self.window_focus, self.real_transform)

    def set_window(self, window:rasterio.windows.Window, relative:bool=True, boundless:bool=True):
        if relative:
            self.window_focus = rasterio.windows.Window(col_off=window.col_off + self.window_focus.col_off,
                                                        row_off=window.row_off + self.window_focus.row_off,
                                                        height=window.height, width=window.width)
        else:
            self.window_focus = window

        if not boundless:
            self.window_focus = rasterio.windows.intersection(self.window_data, self.window_focus)

    def __copy__(self) -> '__class__':
        return ProbaV(self.hdf5_file, window=self.window_focus, level_name=self.level_name)

    def read_from_window(self, window:Optional[rasterio.windows.Window]=None, boundless:bool=True) -> '__class__':
        copy = self.__copy__()
        copy.set_window(window=window, boundless=boundless)

        return copy

    def __repr__(self)->str:
        return f""" 
         File: {self.hdf5_file}
         Transform: {self.transform}
         Shape: {self.height}, {self.width}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         Level: {self.level_name}
         TOA/TOC: {self.toatoc}
         Resolution name : {self.res_name}
        """

# Class to interface with read functions
class ProbaVRadiometry(ProbaV):
    def __init__(self, hdf5_file:str,  window:Optional[rasterio.windows.Window]=None,
                 level_name:str="LEVEL2A", indexes:Optional[List[int]] = None):
        super().__init__(hdf5_file=hdf5_file, window=window, level_name=level_name)
        self.dims = ("band", "y", "x")

        # let read only some bands?
        if indexes is None:
            self.indexes = [0, 1, 2, 3]
        else:
            self.indexes = indexes

        self.dtype = self.dtype_radiometry

    @property
    def count(self):
        return len(self.indexes)

    def load(self, boundless:bool=True)->geotensor.GeoTensor:
        return self.load_radiometry(boundless=boundless, indexes=self.indexes)

    @property
    def shape(self) -> Tuple:
        return  self.count, self.window_focus.height, self.window_focus.width

    @property
    def values(self) -> np.ndarray:
        return self.load_radiometry(boundless=True, indexes=self.indexes).values

    def __copy__(self) -> '__class__':
        return ProbaVRadiometry(self.hdf5_file, window=self.window_focus, level_name=self.level_name,
                                indexes=self.indexes)


# Class to interface with read functions
class ProbaVSM(ProbaV):
    def __init__(self, hdf5_file: str, window: Optional[rasterio.windows.Window] = None,
                 level_name: str = "LEVEL2A"):
        super().__init__(hdf5_file=hdf5_file, window=window, level_name=level_name)
        self.dims = ("y", "x")
        self.dtype = self.dtype_sm

    def load(self, boundless: bool = True) -> geotensor.GeoTensor:
        return self.load_sm(boundless=boundless)

    @property
    def shape(self) -> Tuple:
        return self.window_focus.height, self.window_focus.width

    @property
    def values(self) -> np.ndarray:
        return self.load_sm(boundless=True).values

    def __copy__(self) -> '__class__':
        return ProbaVSM(self.hdf5_file, window=self.window_focus, level_name=self.level_name)


def sm_cloud_mask(sm:np.ndarray, mask_undefined:bool=False) -> np.ndarray:
    """
    Returns a binary cloud mask: 2 if cloud, 1 if clear, 0 if invalid

    From user manual https://publications.vito.be/2017-1333-probav-products-user-manual.pdf Pag 64
        * Clear  ->    000
        * Shadow ->    001
        * Undefined -> 010
        * Cloud  ->    011
        * Ice    ->    100

    :param sm: (H, W) sm flags as loaded from ProbaVImageOperational.load_sm() method
    :param mask_undefined: if True returns also as clouds pixels those marked as undefined
    :return:
    """
    cloud_mask = np.uint8(((sm & 1) != 0) & ((sm & 2**1) != 0) & ((sm & 2**2) == 0))
    if mask_undefined:
        undefined_mask = ((sm & 1) == 0) & ((sm & 2**1) != 0) & ((sm & 2**2) == 0)
        cloud_mask |= undefined_mask
    
    cloud_mask += 1
    invalids = mask_only_sm(sm)
    cloud_mask[invalids] = 0

    return cloud_mask


def mask_only_sm(sm:np.ndarray) -> np.ndarray:
    """
    Returns a invalid mask: True if the pixel is out of swath

    https://publications.vito.be/2017-1333-probav-products-user-manual.pdf Pag 64

    Bits 8..11: SWIR, NIR, RED, BLUE coverage flags

    (If any of those bits is 0 the pixel is not covered, set as invalid)

    Args:
        sm (np.ndarray): sm flags as loaded from ProbaVImageOperational.load_sm() method

    Returns:
        np.ndarray: mask with the same shape as the image
    """
    mascara = np.zeros(sm.shape, dtype=bool)
    for i in range(4):
        mascara |= ((sm & (2 ** (i + 8))) == 0)

    return mascara



