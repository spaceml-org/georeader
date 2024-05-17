"""
SPOT VGT reader

Unofficial reader for SPOT VGT products. The reader is based on the user manual:
https://docs.terrascope.be/DataProducts/SPOT-VGT/references/SPOT_VGT_PUM_v1.3.pdf

Authors: Dan Lopez-Puigdollers, Gonzalo Mateo-García
"""

import os
import re
from rasterio import Affine
import rasterio
import rasterio.windows
import rasterio.crs
from typing import Tuple, List, Optional, Union
from georeader import window_utils, geotensor
from numbers import Number
from shapely.geometry import Polygon
import numpy as np
import datetime as dt
from glob import glob
from pyhdf.SD import SD, SDC
import warnings


FILES = ['1BL', '1BO', 'AG', 'B0', 'B2', 'B3', 'MIR',
         'OG', 'SAA', 'SM', 'SZA', 'VAA', 'VZA', 'WVG']

BANDS_NAMES = ['B0', 'B2', 'B3', 'MIR']
BANDS_DICT = {k: v for k, v in enumerate(BANDS_NAMES)}


def read_band_toa(dataset, band: str, slice_to_read: Tuple[slice, slice]):
    """

    see https://docs.terrascope.be/DataProducts/SPOT-VGT/references/SPOT_VGT_PUM_v1.3.pdf page 45
    :param dataset:
    :param band:
    :param slice_to_read:
    :return:
    """
    hdfreader = dataset[band]
    # Sometimes the dataset is called PIXEL_DATA others PIXEL DATA
    ds = hdfreader.datasets()
    if "PIXEL DATA" in ds:
        key = "PIXEL DATA"
    elif "PIXEL_DATA" in ds:
        key = "PIXEL_DATA"
    else:
        key = list(ds.keys())[0]
        warnings.warn(f"Unexpected key in HDF dataset of SPOTVGT. Expected 'PIXEL DATA' or 'PIXEL_DATA'. We will read the values in {key}")
    
    if band in BANDS_NAMES:
        return dataset[band].select(key)[slice_to_read] * 0.0005
    return dataset[band].select(key)[slice_to_read]


class SpotVGT:
    def __init__(self, hdf4_file: str, window: Optional[rasterio.windows.Window] = None):
        """
        ## SPOT-VGT READER

        User manual https://publications.vito.be/2016-1034-spotvgt-collection-3-products-user-manual-v10.pdf
        :param hdf4_file: path to HDF4 file
        :param window:
        """

        self.hdf4_file = hdf4_file
        self.name = os.path.basename(self.hdf4_file)
        matches = re.match(r'V(\d{1})(\w{3})(\w{1})____(\d{4})(\d{2})(\d{2})F(\w{3})_V(\d{3})', self.name)
        if matches is not None:
            (self.satelliteID, self.station, self.productID, self.year,
             self.month, self.day, self.segment, self.version) = matches.groups()
        else:
            raise FileNotFoundError("SPOT-VGT product not recognized %s" % self.hdf4_file)

        try:
            self.files = sorted([f for f in glob(os.path.join(self.hdf4_file, '*'))])
            self.files_dict = {re.match(r'V\d{12}_(\w+)',
                                        os.path.basename(self.files[i])).groups()[0]: self.files[i]
                               for i in range(len(self.files))}

            with open(self.files_dict['LOG'], "r") as f:
                self.metadata = {re.split(r'\s+', y)[0]: re.split(r'\s+', y)[1] for y in [x for x in f]}

            self.real_shape = (
                int(self.metadata["IMAGE_LOWER_RIGHT_ROW"]) - int(self.metadata["IMAGE_UPPER_LEFT_ROW"]) - 1,
                int(self.metadata["IMAGE_LOWER_RIGHT_COL"]) - int(self.metadata["IMAGE_UPPER_LEFT_COL"]) - 1)

            bbox = [
                float(self.metadata['CARTO_LOWER_LEFT_X']),
                float(self.metadata['CARTO_LOWER_LEFT_Y']),
                float(self.metadata['CARTO_UPPER_RIGHT_X']),
                float(self.metadata['CARTO_UPPER_RIGHT_Y'])
            ]
            self.real_transform = rasterio.transform.from_bounds(*bbox, width=self.real_shape[1],
                                                                 height=self.real_shape[0])

            self.dtype_radiometry = np.float32

        except OSError as e:
            raise FileNotFoundError("Error reading product %s" % self.hdf4_file)

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

        year, month, day = re.match(r'(\d{4})(\d{2})(\d{2})', self.metadata['SEGM_FIRST_DATE']).groups()
        hh, mm, ss = re.match(r'(\d{2})(\d{2})(\d{2})', self.metadata['SEGM_FIRST_TIME']).groups()

        self.start_date = dt.datetime(day=int(day), month=int(month), year=int(year),
                                      hour=int(hh), minute=int(mm), second=int(ss), tzinfo=dt.timezone.utc)

        year, month, day = re.match(r'(\d{4})(\d{2})(\d{2})', self.metadata['SEGM_LAST_DATE']).groups()
        hh, mm, ss = re.match(r'(\d{2})(\d{2})(\d{2})', self.metadata['SEGM_LAST_TIME']).groups()

        self.end_date = dt.datetime(day=int(day), month=int(month), year=int(year),
                                    hour=int(hh), minute=int(mm), second=int(ss), tzinfo=dt.timezone.utc)

        # self.map_projection_wkt

        self.toatoc = "TOA"

        self.res_name = '1KM'

        # SPOT-VGT images are lat/long
        self.crs = rasterio.crs.CRS({'init': 'epsg:4326'})

        # SPOT-VGT images have four bands
        self.level_name = "LEVEL2A"

    def _get_window_pad(self, boundless: bool = True) -> Tuple[rasterio.windows.Window, Optional[List]]:
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

    def _load_bands(self, bands_names: Union[List[str], str], boundless: bool = True,
                    fill_value_default: Number = 0) -> geotensor.GeoTensor:
        window_read, pad_list_np = self._get_window_pad(boundless=boundless)
        slice_ = window_read.toslices()
        if isinstance(bands_names, str):
            bands_names = [bands_names]
            flatten = True
        else:
            flatten = False

        hdf_objs = {b: SD(self.files_dict[b], SDC.READ) for b in bands_names}
        # Read dataset
        # shapes = [hdf_objs[b].datasets()["PIXEL_DATA"][1] for b in bands_names]
        # data = [hdf_objs[b].select("PIXEL_DATA")[slice_] for b in bands_names]

        bands_arrs = []
        # Original slice int32 gives an error. Cast to int
        for band in bands_names:
            data = read_band_toa(hdf_objs, band, (slice(int(slice_[0].start), int(slice_[0].stop), None),
                                                  slice(int(slice_[1].start), int(slice_[1].stop), None)))
            if pad_list_np:
                data = np.pad(data, tuple(pad_list_np), mode="constant", constant_values=fill_value_default)

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

    def load_radiometry(self, indexes: Optional[List[int]] = None, boundless: bool = True) -> geotensor.GeoTensor:
        if indexes is None:
            indexes = (0, 1, 2, 3)
        # bands_names = [f"{self.level_name}/RADIOMETRY/{BAND_NAMES[i]}/{self.toatoc}" for i in indexes]
        bands_names = [BANDS_DICT[i] for i in indexes]
        return self._load_bands(bands_names, boundless=boundless, fill_value_default=0)

    def load_sm(self, boundless: bool = True) -> geotensor.GeoTensor:
        """
        ## Reference of values in `SM` flags.

        From user manual https://docs.terrascope.be/DataProducts/SPOT-VGT/references/SPOT_VGT_PUM_v1.3.pdf pag 46
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
        """
        return self._load_bands('SM', boundless=boundless, fill_value_default=0)

    def load_mask(self, boundless: bool = True) -> geotensor.GeoTensor:
        """
        Returns the valid mask (False if the pixel is out of swath or is invalid). This function loads the SM band

        Args:
            boundless (bool, optional): boundless option to load the SM band. Defaults to True.

        Returns:
            geotensor.GeoTensor: mask with the same shape as the image
        """
        
        sm = self.load_sm(boundless=boundless)
        valids = sm.copy()
        invalids = mask_only_sm(sm.values)
        valids.values = ~invalids
        valids.fill_value_default = False
        
        return valids
    
    def load_sm_cloud_mask(self, mask_undefined:bool=False, boundless:bool=True) -> geotensor.GeoTensor:
        sm = self.load_sm(boundless=boundless)
        cloud_mask = sm_cloud_mask(sm.values, mask_undefined=mask_undefined)
        cloud_mask+=1
        invalids = mask_only_sm(sm.values)
        
        cloud_mask[invalids] = 0
        return geotensor.GeoTensor(cloud_mask, transform=self.transform, crs=self.crs, fill_value_default=0)

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

    def set_window(self, window:rasterio.windows.Window, relative: bool = True, boundless: bool = True):
        if relative:
            self.window_focus = rasterio.windows.Window(col_off=window.col_off + self.window_focus.col_off,
                                                        row_off=window.row_off + self.window_focus.row_off,
                                                        height=window.height, width=window.width)
        else:
            self.window_focus = window

        if not boundless:
            self.window_focus = rasterio.windows.intersection(self.window_data, self.window_focus)

    def __copy__(self) -> '__class__':
        return SpotVGT(self.hdf4_file, window=self.window_focus)

    def read_from_window(self, window: Optional[rasterio.windows.Window] = None, boundless: bool = True) -> '__class__':
        copy = self.__copy__()
        copy.set_window(window=window, boundless=boundless)

        return copy

    def __repr__(self) -> str:
        return f""" 
         File: {self.hdf4_file}
         Transform: {self.transform}
         Shape: {self.height}, {self.width}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         Level: {self.level_name}
         TOA/TOC: {self.toatoc}
         Resolution name : {self.res_name}
        """


def sm_cloud_mask(sm: np.ndarray, mask_undefined: bool = False) -> np.ndarray:
    """
    Returns a binary cloud mask: 1 cloudy values 0 rest

    From user manual https://docs.terrascope.be/DataProducts/SPOT-VGT/references/SPOT_VGT_PUM_v1.3.pdf pag 46
        * Clear  ->    000
        * Shadow ->    001
        * Undefined -> 010
        * Cloud  ->    011
        * Ice    ->    100

    :param sm: (H, W) sm flags as loaded from SpotVGT.load_sm() method
    :param mask_undefined: if True returns also as clouds pixels those marked as undefined
    :return:
    """
    cloud_mask = np.uint8(((sm & 1) != 0) & ((sm & 2**1) != 0) & ((sm & 2**2) == 0))
    if mask_undefined:
        undefined_mask = ((sm & 1) == 0) & ((sm & 2**1) != 0) & ((sm & 2**2) == 0)
        cloud_mask |= undefined_mask

    return cloud_mask


def mask_only_sm(sm: np.ndarray) -> np.ndarray:
    return sm == 0
