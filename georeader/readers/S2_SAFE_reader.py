"""
Sentinel-2 reader inherited from https://github.com/IPL-UV/DL-L8S2-UV.

It has several enhancements:
* Support for S2L2A images
* It can read directly images from a GCP bucket (for example data from  [here](https://cloud.google.com/storage/docs/public-datasets/sentinel-2))
* Windowed read and read and reproject in the same function (see `load_bands_bbox`)
* Creation of the image only involves reading one metadata file (`xxx.SAFE/MTD_{self.producttype}.xml`)
* Compatible with `georeader.read` functions
* It reads from pyramid if possible


https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/document-library

"""
from rasterio import windows
from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
import datetime
from collections import OrderedDict
import numpy as np
import os
import re
from typing import List, Tuple, Union, Optional, Dict, Any
from georeader.rasterio_reader import  RasterioReader
from georeader import read
from georeader import window_utils
from georeader.geotensor import GeoTensor
import rasterio.warp
from shapely.geometry import shape
from tqdm import tqdm
import json
from georeader.save import save_cog, save_tiled_geotiff
import pandas as pd


BANDS_S2 = ["B01", "B02","B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

PUBLIC_BUCKET_SENTINEL_2 = "gcp-public-data-sentinel-2"
FULL_PATH_PUBLIC_BUCKET_SENTINEL_2 = f"gs://{PUBLIC_BUCKET_SENTINEL_2}/"


BANDS_S2_L1C = list(BANDS_S2)

# TODO ADD SLC band? AOT? WP?
BANDS_S2_L2A = ["B01", "B02","B03", "B04", "B05", "B06",
                "B07", "B08", "B8A", "B09", "B11", "B12"]

BANDS_RESOLUTION = OrderedDict({"B01": 60, "B02": 10,
                                "B03": 10, "B04": 10,
                                "B05": 20, "B06": 20,
                                "B07": 20, "B08": 10,
                                "B8A": 20, "B09": 60,
                                "B10": 60, "B11": 20,
                                "B12": 20})

DEFAULT_REQUESTER_PAYS = False


def normalize_band_names(bands:List[str]) -> List[str]:
    """ Adds zero before band name for reading """
    bands_out = []
    for b in bands:
        lb = len(b)
        if lb == 2:
            bands_out.append(f"B0{b[-1]}")
        elif lb == 3:
            bands_out.append(b)
        else:
            raise NotImplementedError(f"Unknown band {b} with different number of expected characters")

    return bands_out


def islocalpath(path:str) -> bool:
    return path.startswith("file://") or ("://" not in path)

def get_filesystem(path: str, requester_pays: Optional[bool] = None):
    """Get the filesystem from a path """
    if path.startswith(FULL_PATH_PUBLIC_BUCKET_SENTINEL_2):
        import gcsfs
        return gcsfs.GCSFileSystem(token='anon', access="read_only", default_location="EUROPE-WEST1")
    
    import fsspec
    if requester_pays is None:
        requester_pays = DEFAULT_REQUESTER_PAYS
    
    path = str(path)
    if islocalpath(path):
        return fsspec.filesystem("file")
    else:
        # use the fileystem from the protocol specified
        mode = path.split(":", 1)[0]
        if mode == "gs":
            return fsspec.filesystem(mode, requester_pays=requester_pays)
        return fsspec.filesystem(mode)

def _get_info_granules_metadata(folder) -> Optional[Dict[str, Any]]:
    granules_path = os.path.join(folder, "granules.json").replace("\\", "/")
    info_granules_metadata = None
    if islocalpath(granules_path) and os.path.exists(granules_path):
        with open(granules_path, "r") as fh:
            info_granules_metadata = json.load(fh)
        info_granules_metadata["granules"] = {k:os.path.join(folder,g) for k,g in info_granules_metadata["granules"].items()}
        info_granules_metadata["metadata_msi"] = os.path.join(folder, info_granules_metadata["metadata_msi"])
        info_granules_metadata["metadata_tl"] = os.path.join(folder, info_granules_metadata["metadata_tl"])
    return info_granules_metadata

class S2Image:
    def __init__(self, s2folder:str,
                 polygon:Optional[Polygon]=None,
                 granules: Optional[Dict[str, str]]=None,
                 out_res: int = 10,
                 window_focus:Optional[rasterio.windows.Window]=None,
                 bands:Optional[List[str]]=None,
                 metadata_msi:Optional[str]=None):
        """
        Sentinel-2 image reader class.

        Args:
            s2folder: name of the SAFE product expects name
            polygon: in CRS EPSG:4326
            granules: dictionary with granule name and path
            out_res: output resolution in meters one of 10, 20, 60 (default 10)
            window_focus: rasterio window to read. All reads will be based on this window
            bands: list of bands to read. If None all bands are read.
            metadata_msi: path to metadata file. If None it is assumed to be in the SAFE folder
        
        """
        self.mission, self.producttype, sensing_date_str, self.pdgs, self.relorbitnum, self.tile_number_field, self.product_discriminator = s2_name_split(
            s2folder)

        # Remove last trailing slash
        s2folder = s2folder[:-1] if (s2folder.endswith("/") or s2folder.endswith("\\")) else s2folder
        self.name = os.path.basename(os.path.splitext(s2folder)[0])

        self.folder = s2folder
        self.datetime = datetime.datetime.strptime(sensing_date_str, "%Y%m%dT%H%M%S").replace(
            tzinfo=datetime.timezone.utc)

        info_granules_metadata = None

        if metadata_msi is None:
            info_granules_metadata = _get_info_granules_metadata(self.folder)
            if info_granules_metadata is not None:
                self.metadata_msi = info_granules_metadata["metadata_msi"]
                if "metadata_tl" in info_granules_metadata:
                    self.metadata_tl = info_granules_metadata["metadata_tl"]
            else:
                self.metadata_msi = os.path.join(self.folder, f"MTD_{self.producttype}.xml").replace("\\", "/")

        else:
            self.metadata_msi = metadata_msi

        out_res = int(out_res)

        # TODO increase possible out_res to powers of 2 of 10 meters and 60 meters
        # rst = rasterio.open('gs://gcp-public-data-sentinel-2/tiles/49/S/GV/S2B_MSIL1C_20220527T030539_N0400_R075_T49SGV_20220527T051042.SAFE/GRANULE/L1C_T49SGV_A027271_20220527T031740/IMG_DATA/T49SGV_20220527T030539_B02.jp2')
        # rst.overviews(1) -> [2, 4, 8, 16]
        assert out_res in {10, 20, 60}, "Not valid output resolution.Choose 10, 20, 60"

        # Default resolution to read
        self.out_res = out_res

        if bands is None:
            if self.producttype == "MSIL2A":
                self.bands = list(BANDS_S2_L2A)
            else:
                self.bands = list(BANDS_S2)
        else:
            self.bands = normalize_band_names(bands)

        self.dims = ("band", "y", "x")
        self.fill_value_default = 0

        # Select the band that will be used as template when reading
        self.band_check = None
        for band in self.bands:
            if BANDS_RESOLUTION[band] == self.out_res:
                self.band_check = band
                break

        assert self.band_check is not None, f"Not band found of resolution {self.out_res} in {self.bands}"

        # This dict will be filled by the _get_reader function
        self.granule_readers: Dict[str, RasterioReader] = {}
        self.window_focus = window_focus
        self.root_metadata_msi = None
        self._radio_add_offsets = None
        self._solar_irradiance = None
        self._scale_factor_U = None
        self._quantification_value = None

        # The code below could be only triggered if required
        if not granules:
            # This is useful when copying with cache_product_to_local_dir func
            if info_granules_metadata is None:
                info_granules_metadata = _get_info_granules_metadata(self.folder)

            if info_granules_metadata is not None:
                self.granules = info_granules_metadata["granules"]

            else:
                self.load_metadata_msi()
                bands_elms = self.root_metadata_msi.findall(".//IMAGE_FILE")
                all_granules = [os.path.join(self.folder, b.text + ".jp2").replace("\\", "/")  for b in bands_elms]
                if self.producttype == "MSIL2A":
                    self.granules = {j.split("_")[-2]: j for j in all_granules}
                else:
                    self.granules = {j.split("_")[-1].replace(".jp2", ""): j for j in all_granules}
        else:
            self.granules = granules

        self._pol = polygon
        if self._pol is not None:
            self._pol_crs = window_utils.polygon_to_crs(self._pol, "EPSG:4326", self.crs)
        else:
            self._pol_crs = None

    def cache_product_to_local_dir(self, path_dest:Optional[str]=None, print_progress:bool=True,
                                   format_bands:Optional[str]=None) -> '__class__':
        """
        Copy the product to a local directory and return a new instance of the class with the new path

        Args:
            path_dest: path to the destination folder. If None, the current folder ()".") is used
            print_progress: print progress bar. Default True
            format_bands: format of the bands. Default None (keep original format). Options: "COG", "GeoTIFF"
        
        Returns:
            A new instance of the class pointing to the new path
        """
        if path_dest is None:
            path_dest = "."
        
        if format_bands is not None:
            assert format_bands in {"COG", "GeoTIFF"}, "Not valid format_bands. Choose 'COG' or 'GeoTIFF'"

        name_with_safe = f"{self.name}.SAFE"
        dest_folder = os.path.join(path_dest, name_with_safe)

        # Copy metadata
        metadata_filename = os.path.basename(self.metadata_msi)
        metadata_output_path = os.path.join(dest_folder, metadata_filename)
        if not os.path.exists(metadata_output_path):
            os.makedirs(dest_folder, exist_ok=True)
            self.load_metadata_msi()
            ET.ElementTree(self.root_metadata_msi).write(metadata_output_path)
            root_metadata_msi = self.root_metadata_msi
        else:
            root_metadata_msi = read_xml(metadata_output_path)

        bands_elms = root_metadata_msi.findall(".//IMAGE_FILE")
        if self.producttype == "MSIL2A":
            granules_name_metadata = {b.text.split("_")[-2]: b.text for b in bands_elms}
        else:
            granules_name_metadata = {b.text.split("_")[-1]: b.text for b in bands_elms}

        new_granules = {}
        with tqdm(total=len(self.bands),disable=not print_progress) as pbar:
            for b in self.bands:
                granule = self.granules[b]
                ext_origin = os.path.splitext(granule)[1]

                if format_bands is not None:
                    if ext_origin.startswith(".tif"):
                        convert = False
                    else:
                        convert = True
                    
                    ext_dst = ".tif"
                else:
                    convert = False
                    ext_dst = ext_origin
                
                namefile = os.path.splitext(granules_name_metadata[b])[0]
                new_granules[b] = namefile+ext_dst
                new_granules_path = os.path.join(dest_folder, new_granules[b])
                if not os.path.exists(new_granules_path):
                    new_granules_path_tmp = os.path.join(dest_folder, namefile+ext_origin)
                    pbar.set_description(f"Donwloading band {b} from {granule} to {new_granules_path}")
                    dir_granules_path = os.path.dirname(new_granules_path)
                    os.makedirs(dir_granules_path, exist_ok=True)
                    get_file(granule, new_granules_path_tmp)
                    if convert:
                        image = RasterioReader(new_granules_path_tmp).load().squeeze()
                        if format_bands == "COG":
                            save_cog(image, new_granules_path, descriptions=[b])
                        elif format_bands == "GeoTIFF":
                            save_tiled_geotiff(image, new_granules_path, descriptions=[b])
                        else:
                            raise NotImplementedError(f"Not implemented {format_bands}")
                        os.remove(new_granules_path_tmp)
            
                pbar.update(1)

        # Save granules for fast reading
        granules_path = os.path.join(dest_folder, "granules.json").replace("\\", "/")
        if not os.path.exists(granules_path):
            with open(granules_path, "w") as fh:
                json.dump({"granules": new_granules, "metadata_msi": metadata_filename}, fh)

        new_granules_full_path = {k: os.path.join(dest_folder,v) for k, v in new_granules.items()}

        obj = s2loader(s2folder=dest_folder, out_res=self.out_res, window_focus=self.window_focus,
                       bands=self.bands, granules=new_granules_full_path, polygon=self._pol,
                       metadata_msi=metadata_output_path)
        obj.root_metadata_msi = root_metadata_msi
        return obj

    def DN_to_radiance(self, dn_data:Optional[GeoTensor]=None) -> GeoTensor:
        return DN_to_radiance(self, dn_data)

    def load_metadata_msi(self) -> ET.Element:
        if self.root_metadata_msi is None:
            self.root_metadata_msi = read_xml(self.metadata_msi)
        return self.root_metadata_msi

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        if self._pol_crs is None:
            self.load_metadata_msi()
            footprint_txt = self.root_metadata_msi.findall(".//EXT_POS_LIST")[0].text
            coords_split = footprint_txt.split(" ")[:-1]
            self._pol = Polygon(
                [(float(lngstr), float(latstr)) for latstr, lngstr in zip(coords_split[::2], coords_split[1::2])])
            self._pol_crs = window_utils.polygon_to_crs(self._pol, "EPSG:4326", self.crs)

        pol_window = window_utils.window_polygon(self._get_reader().window_focus, self.transform)

        pol = self._pol_crs.intersection(pol_window)

        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)

    def radio_add_offsets(self) ->Dict[str,float]:
        if self._radio_add_offsets is None:
            self.load_metadata_msi()
            radio_add_offsets = self.root_metadata_msi.findall(".//RADIO_ADD_OFFSET")
            if len(radio_add_offsets) == 0:
                self._radio_add_offsets = {b : 0 for b in BANDS_S2}
            else:
                self._radio_add_offsets = {BANDS_S2[int(r.attrib["band_id"])]: int(r.text) for r in radio_add_offsets}

        return self._radio_add_offsets

    def solar_irradiance(self) -> Dict[str, float]:
        """
        Returns solar irradiance per nanometer: W/m²/nm

        Reads solar irradiance from metadata_msi:
            <SOLAR_IRRADIANCE bandId="0" unit="W/m²/µm">1874.3</SOLAR_IRRADIANCE>
        """
        if self._solar_irradiance is None:
            self.load_metadata_msi()
            sr = self.root_metadata_msi.findall(".//SOLAR_IRRADIANCE")
            self._solar_irradiance = {BANDS_S2[int(r.attrib["bandId"])]: float(r.text)/1_000 for r in sr}

        return self._solar_irradiance

    def scale_factor_U(self) -> float:
        if self._scale_factor_U is None:
            self.load_metadata_msi()
            self._scale_factor_U = float(self.root_metadata_msi.find(".//U").text)

        return self._scale_factor_U

    def quantification_value(self) -> int:
        """ Returns the quantification value stored in the metadata msi file (this is always: 10_000) """
        if self._quantification_value is None:
            self.load_metadata_msi()
            self._quantification_value = int(self.root_metadata_msi.find(".//QUANTIFICATION_VALUE").text)

        return self._quantification_value

    def get_reader(self, band_names: Union[str,List[str]], overview_level:Optional[int]=None) -> RasterioReader:
        """
        Provides a RasterioReader object to read all the bands at the same resolution

        Args:
            band_names: List of band names or band. raises assertion error if bands have different resolution.
            overview_level: level of the pyramid to read (same as in rasterio)

        Returns:
            RasterioReader

        """
        if isinstance(band_names,str):
            band_names = [band_names]

        band_names = normalize_band_names(band_names)

        assert  all(BANDS_RESOLUTION[band_names[0]]==BANDS_RESOLUTION[b] for b in band_names), f"Bands: {band_names} have different resolution"

        reader = RasterioReader([self.granules[band_name] for band_name in band_names],
                                window_focus=None, stack=False,
                                fill_value_default=self.fill_value_default,
                                overview_level=overview_level)
        window_in = read.window_from_bounds(reader, self.bounds)
        window_in_rounded = read.round_outer_window(window_in)
        reader.set_window(window_in_rounded)
        return reader

    def _get_reader(self, band_name:Optional[str]=None) -> RasterioReader:
        if band_name is None:
            band_name = self.band_check

        if band_name not in self.granule_readers:
            # TODO handle different out_res than 10, 20, 60?
            if self.out_res == BANDS_RESOLUTION[band_name]:
                overview_level = None
                has_out_res = True
            elif self.out_res == BANDS_RESOLUTION[band_name]*2:
                # out_res == 20 and BANDS_RESOLUTION[band_name]==10 -> read from first overview
                overview_level = 0
                has_out_res = True
            elif self.out_res > BANDS_RESOLUTION[band_name]:
                # out_res 60 and BANDS_RESOLUTION[band_name] == 10 or BANDS_RESOLUTION[band_name] == 20
                overview_level = 1 if BANDS_RESOLUTION[band_name] == 10 else 0
                has_out_res = False
            else:
                overview_level = None
                has_out_res = False

            # figure out which window_focus to set

            if band_name == self.band_check:
                window_focus = self.window_focus
                set_window_after = False
            elif has_out_res:
                window_focus = self.window_focus
                set_window_after = False
            else:
                set_window_after = True
                window_focus = None

            self.granule_readers[band_name] = RasterioReader(self.granules[band_name],
                                                             window_focus=window_focus,
                                                             fill_value_default=self.fill_value_default,
                                                             overview_level=overview_level)
            if set_window_after:
                window_in = read.window_from_bounds(self.granule_readers[band_name], self.bounds)
                window_in_rounded = read.round_outer_window(window_in)
                self.granule_readers[band_name].set_window(window_in_rounded)

        return self.granule_readers[band_name]

    @property
    def dtype(self):
        # This is always np.uint16
        reader_band_check = self._get_reader()
        return reader_band_check.dtype

    @property
    def shape(self):
        reader_band_check = self._get_reader()
        return (len(self.bands),) + reader_band_check.shape[-2:]

    @property
    def transform(self):
        reader_band_check = self._get_reader()
        return reader_band_check.transform

    @property
    def crs(self):
        reader_band_check = self._get_reader()
        return reader_band_check.crs

    @property
    def bounds(self):
        reader_band_check = self._get_reader()
        return reader_band_check.bounds

    @property
    def res(self) -> Tuple[float, float]:
        reader_band_check = self._get_reader()
        return reader_band_check.res

    def __str__(self):
        return self.folder

    def __repr__(self)->str:
        return f""" 
         {self.folder}
         Transform: {self.transform}
         Shape: {self.shape}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         bands: {self.bands}
         fill_value_default: {self.fill_value_default}
        """

    def read_from_band_names(self, band_names:List[str]) -> '__class__':
        """
        Read from band names

        Args:
            band_names: List of band names
        
        Returns:
            Copy of current object with band names set to band_names
        """
        s2obj =  s2loader(s2folder=self.folder, out_res=self.out_res, 
                          window_focus=self.window_focus,
                           bands=band_names, granules=self.granules, polygon=self._pol,
                           metadata_msi=self.metadata_msi)
        s2obj.root_metadata_msi = self.root_metadata_msi
        return s2obj

    def read_from_window(self, window:rasterio.windows.Window, boundless:bool=True) -> '__class__':
        # return GeoTensor(values=self.values, transform=self.transform, crs=self.crs)

        reader_ref = self._get_reader()
        rasterio_reader_ref = reader_ref.read_from_window(window=window, boundless=boundless)
        s2obj =  s2loader(s2folder=self.folder, out_res=self.out_res, 
                          window_focus=rasterio_reader_ref.window_focus,
                           bands=self.bands, granules=self.granules, polygon=self._pol,
                           metadata_msi=self.metadata_msi)
        # Set band check to avoid re-reading
        s2obj.granule_readers[self.band_check] = rasterio_reader_ref
        s2obj.band_check = self.band_check

        s2obj.root_metadata_msi = self.root_metadata_msi

        return s2obj

    def load(self, boundless:bool=True)-> GeoTensor:
        reader_ref = self._get_reader()
        geotensor_ref = reader_ref.load(boundless=boundless)

        array_out = np.full((len(self.bands),) + geotensor_ref.shape[-2:],fill_value=geotensor_ref.fill_value_default,
                            dtype=geotensor_ref.dtype)

        # Deal with NODATA values
        invalids = (geotensor_ref.values == 0) | (geotensor_ref.values == (2 ** 16) - 1)

        radio_add = self.radio_add_offsets()
        for idx, b in enumerate(self.bands):
            if b == self.band_check:

                # Avoid bug of band names without zero before
                if len(b) == 2:
                    b = f"B0{b[-1]}"

                geotensor_iter = geotensor_ref
            else:
                reader_iter = self._get_reader(b)
                if np.mean(np.abs(np.array(reader_iter.res)-np.array(geotensor_ref.res))) < 1e-6:
                    geotensor_iter = reader_iter.load(boundless=boundless)
                else:
                    geotensor_iter = read.read_reproject_like(reader_iter, geotensor_ref)


            # Important: Adds radio correction! otherwise images after 2022-01-25 shifted (PROCESSING_BASELINE '04.00' or above)
            array_out[idx] = geotensor_iter.values[0] + radio_add[b]

        array_out[:, invalids[0]] = self.fill_value_default

        return GeoTensor(values=array_out, transform=geotensor_ref.transform,crs=geotensor_ref.crs,
                         fill_value_default=self.fill_value_default)

    @property
    def values(self) -> np.ndarray:
        return self.load().values

    def load_mask(self) -> GeoTensor:
        reader_ref = self._get_reader()
        geotensor_ref = reader_ref.load(boundless=True)
        geotensor_ref.values = (geotensor_ref.values == 0) | (geotensor_ref.values == (2**16)-1)
        return geotensor_ref


class S2ImageL2A(S2Image):
    def __init__(self, s2folder:str, granules: Dict[str, str],
                 polygon:Polygon, out_res:int=10,
                 window_focus:Optional[rasterio.windows.Window]=None,
                 bands:Optional[List[str]]=None,
                 metadata_msi:Optional[str]=None):
        if bands is None:
            bands = BANDS_S2_L2A

        super(S2ImageL2A, self).__init__(s2folder=s2folder, granules=granules, polygon=polygon,
                                         out_res=out_res, bands=bands,
                                         window_focus=window_focus,
                                         metadata_msi=metadata_msi)

        assert self.producttype == "MSIL2A", f"Unexpected product type {self.producttype} in image {self.folder}"

        # see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands for a description of the granules data

        # TODO include SLC bands for clouds?
        # res_band = 20 if out_res < 20 else out_res
        # band_and_res = f"SCL_{res_band}m.jp2"
        # granules_match = [g for g in self.all_granules if g.endswith(band_and_res)]
        # self.slc_granule = granules_match[0]



class S2ImageL1C(S2Image):
    def __init__(self, s2folder, granules: Dict[str, str],
                 polygon:Polygon, out_res:int=10,
                 window_focus:Optional[rasterio.windows.Window]=None,
                 bands:Optional[List[str]]=None,
                 metadata_msi:Optional[str]=None):
        super(S2ImageL1C,self).__init__(s2folder=s2folder, granules=granules, polygon=polygon,
                                         out_res=out_res, bands=bands,
                                         window_focus=window_focus,
                                        metadata_msi=metadata_msi)

        assert self.producttype == "MSIL1C", f"Unexpected product type {self.producttype} in image {self.folder}"

        first_granule = self.granules[list(self.granules.keys())[0]]
        self.granule_folder = os.path.dirname(os.path.dirname(first_granule))
        self.msk_clouds_file = os.path.join(self.granule_folder, "MSK_CLOUDS_B00.gml").replace("\\","/")
        if not hasattr(self, "metadata_tl"):
            self.metadata_tl = os.path.join(self.granule_folder, "MTD_TL.xml").replace("\\","/")
        
        self.root_metadata_tl = None

        # Granule in L1C does not include TCI
        # Assert bands in self.granule are ordered as in BANDS_S2
        # assert all(granule[-7:-4] == bname for bname, granule in zip(BANDS_S2, self.granule)), f"some granules are not in the expected order {self.granule}"

    def read_from_window(self, window:rasterio.windows.Window, boundless:bool=True) -> '__class__':
        out = super().read_from_window(window, boundless=boundless)

        if self.root_metadata_tl is None:
            return out
        
        # copy all metadata from the original image
        for atribute in ["tileId","root_metadata_tl", "satId", "procLevel", "dimsByRes", "ulxyByRes", "tileAnglesNode", 
                         "mean_sza", "mean_saa", "mean_vza", "mean_vaa", "vaa", "vza", "saa", "sza", 
                         "anglesULXY"]:
            setattr(out, atribute, getattr(self, atribute))
        
        return out

    def cache_product_to_local_dir(self, path_dest:Optional[str]=None, print_progress:bool=True,
                                   format_bands:Optional[str]=None) -> '__class__':
        """
        Overrides the parent method to copy the MTD_TL.xml file

        Args:
            path_dest (Optional[str], optional): path to the destination folder. Defaults to None.
            print_progress (bool, optional): whether to print progress. Defaults to True.

        Returns:
            __class__: the cached object
        """
        new_obj = super().cache_product_to_local_dir(path_dest=path_dest, print_progress=print_progress,
                                                     format_bands=format_bands)

        if os.path.exists(new_obj.metadata_tl):
            # the cached product already exists. returns
            return new_obj

        if self.root_metadata_tl is not None:
            new_obj.root_metadata_tl = self.root_metadata_tl
            ET.ElementTree(new_obj.metadata_tl).write(new_obj.metadata_tl)
            # copy all metadata from the original image
            for atribute in ["tileId","root_metadata_tl", "satId", "procLevel", "dimsByRes", "ulxyByRes", "tileAnglesNode", 
                            "mean_sza", "mean_saa", "mean_vza", "mean_vaa", "vaa", "vza", "saa", "sza", 
                            "anglesULXY"]:
                if hasattr(self, atribute):
                    setattr(new_obj, atribute, getattr(self, atribute))
        else:
            get_file(self.metadata_tl, new_obj.metadata_tl)
        
        granule_folder_rel = new_obj.granule_folder.replace("\\", "/").replace(new_obj.folder.replace("\\","/")+"/", "")
        # Add metadata_tl to granules.json
        granules_path = os.path.join(new_obj.folder, "granules.json").replace("\\", "/")
        with open(granules_path, "r") as fh:
            info_granules_metadata = json.load(fh)
        info_granules_metadata["metadata_tl"] = os.path.join(granule_folder_rel, "MTD_TL.xml").replace("\\","/")
        with open(granules_path, "w") as f:
            json.dump(info_granules_metadata, f)
        
        return new_obj

    def read_metadata_tl(self):
        '''
        Read metadata TILE to parse information about the acquisition and properties of GRANULE bands
        '''
        if self.root_metadata_tl is not None:
            return

        self.root_metadata_tl = read_xml(self.metadata_tl)

        # Stoopid XML namespace prefix
        nsPrefix = self.root_metadata_tl.tag[:self.root_metadata_tl.tag.index('}') + 1]
        nsDict = {'n1': nsPrefix[1:-1]}

        self.mean_sza = float(self.root_metadata_tl.find(".//Mean_Sun_Angle/ZENITH_ANGLE").text)
        self.mean_saa = float(self.root_metadata_tl.find(".//Mean_Sun_Angle/AZIMUTH_ANGLE").text)

        generalInfoNode = self.root_metadata_tl.find('n1:General_Info', nsDict)
        # N.B. I am still not entirely convinced that this SENSING_TIME is really
        # the acquisition time, but the documentation is rubbish.
        sensingTimeNode = generalInfoNode.find('SENSING_TIME')
        sensingTimeStr = sensingTimeNode.text.strip()
        # self.datetime = datetime.datetime.strptime(sensingTimeStr, "%Y-%m-%dT%H:%M:%S.%fZ")
        tileIdNode = generalInfoNode.find('TILE_ID')
        tileIdFullStr = tileIdNode.text.strip()
        self.tileId = tileIdFullStr.split('_')[-2]
        self.satId = tileIdFullStr[:3]
        self.procLevel = tileIdFullStr[13:16]  # Not sure whether to use absolute pos or split by '_'....

        geomInfoNode = self.root_metadata_tl.find('n1:Geometric_Info', nsDict)
        geocodingNode = geomInfoNode.find('Tile_Geocoding')
        self.epsg_code = geocodingNode.find('HORIZONTAL_CS_CODE').text

        # Dimensions of images at different resolutions.
        self.dimsByRes = {}
        sizeNodeList = geocodingNode.findall('Size')
        for sizeNode in sizeNodeList:
            res = sizeNode.attrib['resolution']
            nrows = int(sizeNode.find('NROWS').text)
            ncols = int(sizeNode.find('NCOLS').text)
            self.dimsByRes[res] = (nrows, ncols)

        # Upper-left corners of images at different resolutions. As far as I can
        # work out, these coords appear to be the upper left corner of the upper left
        # pixel, i.e. equivalent to GDAL's convention. This also means that they
        # are the same for the different resolutions, which is nice.
        self.ulxyByRes = {}
        posNodeList = geocodingNode.findall('Geoposition')
        for posNode in posNodeList:
            res = posNode.attrib['resolution']
            ulx = float(posNode.find('ULX').text)
            uly = float(posNode.find('ULY').text)
            self.ulxyByRes[res] = (ulx, uly)

        # Sun and satellite angles.
        # Zenith
        self.tileAnglesNode = geomInfoNode.find('Tile_Angles')
        sunZenithNode = self.tileAnglesNode.find('Sun_Angles_Grid').find('Zenith')
        # <Zenith>
        #  <COL_STEP unit="m">5000</COL_STEP>
        #  <ROW_STEP unit="m">5000</ROW_STEP>
        angleGridXres = float(sunZenithNode.find('COL_STEP').text)
        angleGridYres = float(sunZenithNode.find('ROW_STEP').text)
        sza = self.makeValueArray(sunZenithNode.find('Values_List'))
        mask_nans = np.isnan(sza)
        if np.any(mask_nans):
            from skimage.restoration import inpaint_biharmonic
            sza = inpaint_biharmonic(sza, mask_nans)
        transform_zenith = rasterio.transform.from_origin(self.ulxyByRes[str(self.out_res)][0],
                                                          self.ulxyByRes[str(self.out_res)][1],
                                                          angleGridXres, angleGridYres)
        
        self.sza = GeoTensor(sza, transform=transform_zenith, crs=self.epsg_code)
        
        # Azimuth
        sunAzimuthNode = self.tileAnglesNode.find('Sun_Angles_Grid').find('Azimuth')
        angleGridXres = float(sunAzimuthNode.find('COL_STEP').text)
        angleGridYres = float(sunAzimuthNode.find('ROW_STEP').text)
        saa = self.makeValueArray(sunAzimuthNode.find('Values_List'))
        mask_nans = np.isnan(saa)
        if np.any(mask_nans):
            from skimage.restoration import inpaint_biharmonic
            saa = inpaint_biharmonic(saa, mask_nans)
        transform_azimuth = rasterio.transform.from_origin(self.ulxyByRes[str(self.out_res)][0],
                                                            self.ulxyByRes[str(self.out_res)][1],
                                                            angleGridXres, angleGridYres)
        self.saa = GeoTensor(saa, transform=transform_azimuth, crs=self.epsg_code)

        # Now build up the viewing angle per grid cell, from the separate layers
        # given for each detector for each band. Initially I am going to keep
        # the bands separate, just to see how that looks.
        # The names of things in the XML suggest that these are view angles,
        # but the numbers suggest that they are angles as seen from the pixel's
        # frame of reference on the ground, i.e. they are in fact what we ultimately want.
        viewingAngleNodeList = self.tileAnglesNode.findall('Viewing_Incidence_Angles_Grids')       
        vza = self.buildViewAngleArr(viewingAngleNodeList, 'Zenith')
        vaa = self.buildViewAngleArr(viewingAngleNodeList, 'Azimuth')

        self.vaa = {}
        for k, varr in vaa.items():            
            mask_nans = np.isnan(varr)
            if np.any(mask_nans):
                from skimage.restoration import inpaint_biharmonic
                varr = inpaint_biharmonic(varr, mask_nans)
            
            self.vaa[k] = GeoTensor(varr, transform=transform_azimuth, crs=self.epsg_code)
        
        self.vza = {}
        for k, varr in vza.items():
            mask_nans = np.isnan(varr)
            if np.any(mask_nans):
                from skimage.restoration import inpaint_biharmonic
                varr = inpaint_biharmonic(varr, mask_nans)
            self.vza[k] = GeoTensor(varr, transform=transform_zenith, crs=self.epsg_code)

        # Make a guess at the coordinates of the angle grids. These are not given
        # explicitly in the XML, and don't line up exactly with the other grids, so I am
        # making a rough estimate. Because the angles don't change rapidly across these
        # distances, it is not important if I am a bit wrong (although it would be nice
        # to be exactly correct!).
        (ulx, uly) = self.ulxyByRes["10"]
        self.anglesULXY = (ulx - angleGridXres / 2.0, uly + angleGridYres / 2.0)

        # Read mean viewing angles for each band.
        self.mean_vaa = {}
        self.mean_vza = {}
        for elm in self.tileAnglesNode.find("Mean_Viewing_Incidence_Angle_List"):
            band_name = BANDS_S2[int(elm.attrib["bandId"])]
            viewing_zenith_angle = float(elm.find("ZENITH_ANGLE").text)
            viewing_azimuth_angle = float(elm.find("AZIMUTH_ANGLE").text)
            self.mean_vza[band_name] = viewing_zenith_angle
            self.mean_vaa[band_name] = viewing_azimuth_angle

    def buildViewAngleArr(self, viewingAngleNodeList, angleName):
        """
        Build up the named viewing angle array from the various detector strips given as
        separate arrays. I don't really understand this, and may need to re-write it once
        I have worked it out......

        The angleName is one of 'Zenith' or 'Azimuth'.
        Returns a dictionary of 2-d arrays, keyed by the bandId string.
        """
        angleArrDict = {}
        for viewingAngleNode in viewingAngleNodeList:
            band_name = BANDS_S2[int(viewingAngleNode.attrib['bandId'])]
            detectorId = viewingAngleNode.attrib['detectorId']
            
            angleNode = viewingAngleNode.find(angleName)
            angleArr = self.makeValueArray(angleNode.find('Values_List'))
            if band_name not in angleArrDict:
                angleArrDict[band_name] = angleArr
            else:
                mask = (~np.isnan(angleArr))
                angleArrDict[band_name][mask] = angleArr[mask]
        return angleArrDict

    # def get_polygons_bqa(self):
    #     def polygon_from_coords(coords, fix_geom=False, swap=True, dims=2):
    #         """
    #         Return Shapely Polygon from coordinates.
    #         - coords: list of alterating latitude / longitude coordinates
    #         - fix_geom: automatically fix geometry
    #         """
    #         assert len(coords) % dims == 0
    #         number_of_points = int(len(coords) / dims)
    #         coords_as_array = np.array(coords)
    #         reshaped = coords_as_array.reshape(number_of_points, dims)
    #         points = [(float(i[1]), float(i[0])) if swap else ((float(i[0]), float(i[1]))) for i in reshaped.tolist()]
    #         polygon = Polygon(points).buffer(0)
    #         try:
    #             assert polygon.is_valid
    #             return polygon
    #         except AssertionError:
    #             if fix_geom:
    #                 return polygon.buffer(0)
    #             else:
    #                 raise RuntimeError("Geometry is not valid.")
    #
    #
    #     exterior_str = str("eop:extentOf/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList")
    #     interior_str = str("eop:extentOf/gml:Polygon/gml:interior/gml:LinearRing/gml:posList")
    #     root = read_xml(self.msk_clouds_file)
    #     nsmap = {k: v for k, v in root.nsmap.items() if k}
    #     try:
    #         for mask_member in root.iterfind("eop:maskMembers", namespaces=nsmap):
    #             for feature in mask_member:
    #                 type = feature.findtext("eop:maskType", namespaces=nsmap)
    #
    #                 ext_elem = feature.find(exterior_str, nsmap)
    #                 dims = int(ext_elem.attrib.get('srsDimension', '2'))
    #                 ext_pts = ext_elem.text.split()
    #                 exterior = polygon_from_coords(ext_pts, fix_geom=True, swap=False, dims=dims)
    #                 try:
    #                     interiors = [polygon_from_coords(int_pts.text.split(), fix_geom=True, swap=False, dims=dims)
    #                                  for int_pts in feature.findall(interior_str, nsmap)]
    #                 except AttributeError:
    #                     interiors = []
    #
    #                 yield dict(geometry=Polygon(exterior, interiors).buffer(0),
    #                            attributes=dict(maskType=type),
    #                            interiors=interiors)
    #
    #     except StopIteration:
    #         yield dict(geometry=Polygon(),
    #                    attributes=dict(maskType=None),
    #                    interiors=[])
    #         raise StopIteration()

    # def load_clouds_bqa(self, window=None):
    #     mask_types = ["OPAQUE", "CIRRUS"]
    #     poly_list = list(self.get_polygons_bqa())
    #
    #     nrows, ncols = self.shape
    #     transform_ = self.transform
    #
    #     def get_mask(mask_type=mask_types[0]):
    #         assert mask_type in mask_types, "mask type must be OPAQUE or CIRRUS"
    #         fill_value = {m: i+1 for i, m in enumerate(mask_types)}
    #         n_polys = np.sum([poly["attributes"]["maskType"] == mask_type for poly in poly_list])
    #         msk = np.zeros(shape=(nrows, ncols), dtype=np.float32)
    #         if n_polys > 0:
    #             # n_interiors = np.sum([len(poly) for poly in poly_list if poly["interiors"]])
    #             multi_polygon = MultiPolygon([poly["geometry"]
    #                                           for poly in poly_list
    #                                           if poly["attributes"]["maskType"] == mask_type]).buffer(0)
    #             bounds = multi_polygon.bounds
    #             bbox2read = coords.BoundingBox(*bounds)
    #             window_read = windows.from_bounds(*bbox2read, transform_)
    #             slice_read = tuple(slice(int(round(s.start)), int(round(s.stop))) for s in window_read.toslices())
    #             out_shape = tuple([s.stop - s.start for s in slice_read])
    #             transform_slice = windows.transform(window_read, transform_)
    #
    #             shapes = [({"type": "Polygon",
    #                         "coordinates": [np.stack([
    #                             p_elem["geometry"].exterior.xy[0],
    #                             p_elem["geometry"].exterior.xy[1]], axis=1).tolist()]}, fill_value[mask_type])
    #                       for p_elem in poly_list if p_elem["attributes"]['maskType'] == mask_type]
    #             sub_msk = features.rasterize(shapes=shapes, fill=0,
    #                                          out_shape=out_shape, dtype=np.float32,
    #                                          transform=transform_slice)
    #             msk[slice_read] = sub_msk
    #
    #         return msk
    #
    #     if window is None:
    #         shape = self.shape
    #         window = rasterio.windows.Window(col_off=0, row_off=0,
    #                                          width=shape[1], height=shape[0])
    #
    #     mask = self.load_mask(window=window)
    #
    #     slice_ = window.toslices()
    #
    #     msk_op_cirr = [np.ma.MaskedArray(get_mask(mask_type=m)[slice_], mask=mask) for m in mask_types]
    #     msk_clouds = np.ma.MaskedArray(np.clip(np.sum(msk_op_cirr, axis=0), 0, 1), mask=mask)
    #     return msk_clouds

    @staticmethod
    def makeValueArray(valuesListNode):
        """
        Take a <Values_List> node from the XML, and return an array of the values contained
        within it. This will be a 2-d numpy array of float32 values (should I pass the dtype in??)

        """
        valuesList = valuesListNode.findall('VALUES')
        vals = []
        for valNode in valuesList:
            text = valNode.text
            vals.append([np.float32(x) for x in text.strip().split()])

        return np.array(vals)


# Cache for the spectral response function of S2A and S2B
SRF_S2 = {}
SRF_FILE_DEFAULT = "https://sentinel.esa.int/documents/247904/685211/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.1.xlsx"


def read_srf(satellite:str, 
            srf_file:str=SRF_FILE_DEFAULT,
            cache:bool=True) -> pd.DataFrame:
    """
    Process the spectral response function file. If the file is not provided
    it downloads it from https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/document-library/-/asset_publisher/Wk0TKajiISaR/content/sentinel-2a-spectral-responses
    
    This function requires the fsspec package and pandas and openpyxl for reading excel files.

    Args:
        satellite (str): satellite name (S2A or S2B)
        srf_file (str): path to the srf file
        cache (bool): if True, the srf is cached for future calls. Default True

    Returns:
        pd.DataFrame: spectral response function for each of the bands of S2
    """
    assert satellite in ["S2A", "S2B"], "satellite must be S2A or S2B"

    if cache:
        global SRF_S2
        if satellite in SRF_S2:
            return SRF_S2[satellite]

    if srf_file == SRF_FILE_DEFAULT:
        # home_dir = os.path.join(os.path.expanduser('~'),".georeader")
        home_dir = os.path.join(os.path.expanduser('~'),".georeader")
        os.makedirs(home_dir, exist_ok=True)
        srf_file_local = os.path.join(home_dir, os.path.basename(srf_file))
        if not os.path.exists(srf_file_local):
            import fsspec
            with fsspec.open(srf_file, "rb") as f:
                with open(srf_file_local, "wb") as f2:
                    f2.write(f.read())
        srf_file = srf_file_local

    srf_s2 = pd.read_excel(srf_file,
                           sheet_name=f"Spectral Responses ({satellite})")
    
    srf_s2 = srf_s2.set_index("SR_WL")

    # remove rows with all values zero
    any_not_cero = np.any((srf_s2 > 1e-6).values, axis=1)
    srf_s2 = srf_s2.loc[any_not_cero]

    # remove the satellite name from the columns
    srf_s2.columns = [c.replace(f"{satellite}_SR_AV_","") for c in srf_s2.columns]
    srf_s2.columns = normalize_band_names(srf_s2.columns)

    if cache:
        SRF_S2[satellite] = srf_s2

    return srf_s2

def get_file(remote_path:str, local_path:str):
    if remote_path.startswith(FULL_PATH_PUBLIC_BUCKET_SENTINEL_2):
        from google.cloud import storage
        from google.cloud.storage.retry import DEFAULT_RETRY
        modified_retry = DEFAULT_RETRY.with_timeout(900.0)
        modified_retry = modified_retry.with_delay(initial=1.5, multiplier=1.5, maximum=600)

        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(PUBLIC_BUCKET_SENTINEL_2)
        blob_name = remote_path.replace(FULL_PATH_PUBLIC_BUCKET_SENTINEL_2, "")
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path, retry=modified_retry)
    
    elif "://" in remote_path:
        fs = get_filesystem(remote_path)
        fs.get(remote_path, local_path)
    else:
        raise ValueError(f"Unknown remote path {remote_path}")


def read_xml(xml_file:str) -> ET.Element:
    """Reads xml with xml package """
    if xml_file.startswith(FULL_PATH_PUBLIC_BUCKET_SENTINEL_2):
        from google.cloud import storage
        from google.cloud.storage.retry import DEFAULT_RETRY

        modified_retry = DEFAULT_RETRY.with_timeout(900.0)
        modified_retry = modified_retry.with_delay(initial=1.5, multiplier=1.5, maximum=600)

        storage_client = storage.Client.create_anonymous_client()
        bucket = storage_client.bucket(PUBLIC_BUCKET_SENTINEL_2)
        blob_name = xml_file.replace(FULL_PATH_PUBLIC_BUCKET_SENTINEL_2, "")
        # print(f"Reading {xml_file} {blob_name} from {FULL_PATH_PUBLIC_BUCKET_SENTINEL_2}")
        blob = bucket.blob(blob_name)        
        root = ET.fromstring(blob.download_as_string(retry=modified_retry))
        
    elif "://" in xml_file:
        import fsspec
        with fsspec.open(xml_file, "rb") as file_obj:
            root = ET.fromstring(file_obj.read())
    else:
        root = ET.parse(xml_file).getroot()
    return root


def DN_to_radiance(s2obj: S2ImageL1C, dn_data:Optional[GeoTensor]=None) -> GeoTensor:
    """
    Function to convert Digital numbers (DN) to radiance.

    Important: this function assumes that radio correction has been applied
     otherwise images after 2022-01-25 shifted (PROCESSING_BASELINE '04.00' or above)
     by default this is applied in the S2Image class.

     ToA formula from ESA:
     https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-1c/algorithm

     Here they say U should be in the numerator
     https://gis.stackexchange.com/questions/285996/convert-sentinel-2-1c-product-from-reflectance-to-radiance

     toaBandX = dn_dataBandX / 10000
     radianceBandX = (toaBandX * cos(SZA) * solarIrradianceBandX * U) / pi

     U should be:
        U = 1. / (1-0.01673*cos(0.0172*(t-4)))^2

     0.0172 = 360/365.256363 * np.pi/180.
     t = datenum(Y,M,D) - datenum(Y,1,1) + 1;

    Args:
        s2obj: s2obj where data has been read
        dn_data: data read from an S2Image class with digital numbers. If None it will be read from s2obj

    Returns:
        geotensor with radiances in W/m²/nm/sr
    """
    if dn_data is None:
        dn_data = s2obj.load()

    data_values_new = dn_data.values.astype(np.float32) / 10_000
    s2obj.read_metadata_tl()
    solar_irr = s2obj.solar_irradiance()
    U = s2obj.scale_factor_U()
    for i,b in enumerate(s2obj.bands):
        mask = dn_data.values[i] == dn_data.fill_value_default
        data_values_new[i] = data_values_new[i] * np.cos(s2obj.mean_sza/180*np.pi) * solar_irr[b] * U / np.pi
        data_values_new[i][mask] = dn_data.fill_value_default

    return GeoTensor(data_values_new, transform=dn_data.transform, crs=dn_data.crs,
                     fill_value_default=dn_data.fill_value_default)


def s2loader(s2folder:str, out_res:int=10,
             bands:Optional[List[str]] = None,
             window_focus:Optional[rasterio.windows.Window]=None,
             granules:Optional[Dict[str,str]]=None,
             polygon:Optional[Polygon]=None,
             metadata_msi:Optional[str]=None) -> Union[S2ImageL2A, S2ImageL1C]:
    """
    Loads a S2ImageL2A or S2ImageL1C depending on the product type

    Args:
        s2folder: .SAFE folder. Expected standard ESA naming convention (see s2_name_split fun)
        out_res: default output resolution {10, 20, 60}
        bands: Bands to read. Default to BANDS_S2 or BANDS_S2_L2A depending on the product type
        window_focus: window to read when creating the object
        granules: Dict where keys are the band names and values are paths to the band location
        polygon: polygon with the footprint of the object
        metadata_msi: path to metadata file

    Returns:
        S2Image reader
    """

    _, producttype_nos2, _, _, _, _, _ = s2_name_split(s2folder)

    if producttype_nos2 == "MSIL2A":
        return S2ImageL2A(s2folder, granules=granules, polygon=polygon, out_res=out_res,
                          bands=bands, window_focus=window_focus, metadata_msi=metadata_msi)
    elif producttype_nos2 == "MSIL1C":
        return S2ImageL1C(s2folder, granules=granules, polygon=polygon, out_res=out_res, bands=bands,
                          window_focus=window_focus, metadata_msi=metadata_msi)

    raise NotImplementedError(f"Don't know how to load {producttype_nos2} products")


def s2_load_from_feature_element84(feature:Dict[str, Any], bands:Optional[List[str]]=None) -> Union[S2ImageL2A, S2ImageL1C]:
    """
    Loads a S2 image from an element feature returned by sat-search
    (see `https://github.com/spaceml-org/georeader/blob/main/notebooks/Sentinel-2/read_s2_safe_element84_cloud.ipynb`)

    Args:
        feature: dictionary as produced by satsearch API
        bands: Bands to read. Defaults to BANDS_S2_L2A

    Returns:
        S2Image reader

    """
    granules = {}
    for k, v in feature["assets"].items():
        if v["href"].endswith(".tif"):
            granules[k] = v["href"]

    polygon = shape(feature["geometry"])

    metadata_msi = feature["assets"]["metadata"]["href"]
    s2folder = feature["properties"]["sentinel:product_id"] + ".SAFE"

    return s2loader(s2folder=s2folder, granules=granules, polygon=polygon, metadata_msi=metadata_msi,
                    bands=bands)

def s2_load_from_feature_planetary_microsoft(item:Any, bands:Optional[List[str]]=None) -> Union[S2ImageL2A, S2ImageL1C]:
    """
    Loads a S2 image from an element feature returned by Microsoft Planetary Computer
    (see [example](https://github.com/microsoft/PlanetaryComputerExamples/blob/main/datasets/sentinel-2-l2a/sentinel-2-l2a-example.ipynb`))

    Args:
        item: (pystac.item.Item) dictionary as produced by pystac_client API
        bands: Bands to read. Defaults to `BANDS_S2_L2A`

    Returns:
        S2Image reader

    """

    metadata_msi = item.assets['product-metadata'].href
    s2_folder = item.properties['s2:product_uri']
    polygon = shape(item.geometry)

    bands_available = ['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP'] 
    granules = {}
    for k, v in item.assets.items():
        if k in bands_available:
            granules[k] = v.href
    return s2loader(s2folder=s2_folder, granules=granules, polygon=polygon, metadata_msi=metadata_msi,
                    bands=bands)


def s2_public_bucket_path(s2file:str, check_exists:bool=False, mode:str="gcp") -> str:
    """
    Returns the expected patch in the public bucket of the S2 file

    Args:
        s2file: safe file (e.g.  S2B_MSIL1C_20220527T030539_N0400_R075_T49SGV_20220527T051042.SAFE)
        check_exists: check if the file exists in the bucket, This will not work if GOOGLE_APPLICATION_CREDENTIALS and/or GS_USER_PROJECT 
            env variables are not set. Default to False
        mode: "gcp" or "rest"

    Returns:
        full path to the file (e.g. gs://gcp-public-data-sentinel-2/tiles/49/S/GV/S2B_MSIL1C_20220527T030539_N0400_R075_T49SGV_20220527T051042.SAFE)
    """
    mission, producttype, sensing_date_str, pdgs, relorbitnum, tile_number_field, product_discriminator = s2_name_split(s2file)
    s2file = s2file[:-1] if s2file.endswith("/") else s2file

    if not s2file.endswith(".SAFE"):
        s2file += ".SAFE"

    basename = os.path.basename(s2file)
    if mode == "gcp":
        s2folder = f"{FULL_PATH_PUBLIC_BUCKET_SENTINEL_2}tiles/{tile_number_field[:2]}/{tile_number_field[2]}/{tile_number_field[3:]}/{basename}"
    elif mode == "rest":
        s2folder = f"https://storage.googleapis.com/gcp-public-data-sentinel-2/tiles/{tile_number_field[:2]}/{tile_number_field[2]}/{tile_number_field[3:]}/{basename}"
    else:
        raise NotImplementedError(f"Mode {mode} unknown")

    if check_exists and (mode == "gcp"):
        fs = get_filesystem(s2folder)

        if not fs.exists(s2folder):
            raise FileNotFoundError(f"Sentinel-2 file not found in {s2folder}")

    return s2folder


NEW_FORMAT = "(S2\w{1})_(MSIL\w{2})_(\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})_(\w{5})_(\w{4})_T(\w{5})_(\w{15})"
OLD_FORMAT = "(S2\w{1})_(\w{4})_(\w{3}_\w{6})_(\w{4})_(\d{8}T\d{6})_(\w{4})_V(\d{4}\d{2}\d{2}T\d{6})_(\d{4}\d{2}\d{2}T\d{6})"


def s2_name_split(s2file:str) -> Optional[Tuple[str, str, str, str, str, str, str]]:
    """
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention

    ```
    s2l1c = "S2A_MSIL1C_20151218T182802_N0201_R127_T11SPD_20151218T182756.SAFE"
    mission, producttype, sensing_date_str, pdgs, relorbitnum, tile_number_field, product_discriminator = s2_name_split(s2l1c)
    >> 'S2A', 'MSIL1C', '20151218T182802', 'N0201', 'R127', '11SPD', '20151218T182756'
    ```

    S2A_MSIL1C_20151218T182802_N0201_R127_T11SPD_20151218T182756.SAFE
    MMM_MSIXXX_YYYYMMDDTHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>.SAFE
    MMM: is the mission ID(S2A/S2B)
    MSIXXX: MSIL1C denotes the Level-1C product level/ MSIL2A denotes the Level-2A product level
    YYYYMMDDHHMMSS: the datatake sensing start time
    Nxxyy: the PDGS Processing Baseline number (e.g. N0204)
    ROOO: Relative Orbit number (R001 - R143)
    Txxxxx: Tile Number field
    SAFE: Product Format (Standard Archive Format for Europe)

    Args:
        s2file: name or path to the Sentinel-2 SAFE file

    Returns:
        mission, producttype, sensing_date_str, pdgs, relorbitnum, tile_number_field, product_discriminator

    """
    s2file = s2file[:-1] if s2file.endswith("/") else s2file
    basename = os.path.basename(os.path.splitext(s2file)[0])
    matches = re.match(NEW_FORMAT, basename)
    if matches is not None:
        return matches.groups()


def s2_old_format_name_split(s2file:str) -> Optional[Tuple[str, str, str, str, str, str, str, str]]:
    """
    https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/naming-convention

    ```
    s2l1c = "S2A_OPER_PRD_MSIL1C_PDMC_20151206T093912_R090_V20151206T043239_20151206T043239.SAFE"
    mission, opertortest, filetype, sitecenter,  creation_date_str, relorbitnum, sensing_time_start, sensing_time_stop = s2_old_format_name_split(s2l1c)
    ```

    Args:
        s2file: name or path to the Sentinel-2 SAFE file

    Returns:
        mission, opertortest, filetype, sitecenter,  creation_date_str, relorbitnum, sensing_time_start, sensing_time_stop
    """

    s2file = s2file[:-1] if s2file.endswith("/") else s2file
    basename = os.path.basename(os.path.splitext(s2file)[0])
    matches = re.match(OLD_FORMAT, basename)
    if matches is not None:
        return matches.groups()
