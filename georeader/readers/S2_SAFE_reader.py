"""
Sentinel-2 reader inherited from https://github.com/IPL-UV/DL-L8S2-UV.

It has several enhancements:
* Support for S2L2A images
* It can read directly images from a GCP bucket (for example data from  [here](https://cloud.google.com/storage/docs/public-datasets/sentinel-2))
* Windowed read and read and reproject in the same function (see `load_bands_bbox`)
* Creation of the image only involves reading one metadata file (`xxx.SAFE/MTD_{self.producttype}.xml`)
* Compatible with georeader.read functions
* Read from pyramid if possible


https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/document-library

"""
from rasterio import windows
from shapely.geometry import Polygon, MultiPolygon, box
import xml.etree.ElementTree as ET
import rasterio
import datetime
from collections import OrderedDict
import numpy as np
import os
import re
from typing import List, Tuple, Union, Optional, Dict
from georeader.rasterio_reader import  RasterioReader
from georeader import read
from georeader.geotensor import GeoTensor
import rasterio.warp


BANDS_S2 = ["B01", "B02","B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

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

BANDS_S2_NO_ZERO = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]


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


class S2Image:
    def __init__(self, s2_folder:str,
                 polygon:Optional[Polygon]=None,
                 all_granules: Optional[List[str]]=None,
                 out_res: int = 10,
                 window_focus:Optional[rasterio.windows.Window]=None,
                 bands:Optional[List[str]]=None):
        mission, self.producttype, sensing_date_str, pdgs, relorbitnum, tile_number_field, product_discriminator = s2_name_split(
            s2_folder)

        # Remove last trailing slash
        s2_folder = s2_folder[:-1] if (s2_folder.endswith("/") or s2_folder.endswith("\\")) else s2_folder
        self.name = os.path.basename(os.path.splitext(s2_folder)[0])

        self.folder = s2_folder
        self.datetime = datetime.datetime.strptime(sensing_date_str, "%Y%m%dT%H%M%S").replace(
            tzinfo=datetime.timezone.utc)
        self.metadata_msi = os.path.join(self.folder, f"MTD_{self.producttype}.xml").replace("\\", "/")

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
        if all_granules is None:
            self.load_metadata_msi()
            bands_elms = self.root_metadata_msi.findall(".//IMAGE_FILE")
            all_granules = [os.path.join(self.folder, b.text + ".jp2").replace("\\", "/")  for b in bands_elms]

        self.all_granules:List[str] = all_granules
        self._pol = polygon

        self.granule_folder = os.path.dirname(os.path.dirname(self.all_granules[0])).replace(f"{self.folder}/", "")
        self.granule: Dict[str, str] = {}
        if self.producttype == "MSIL2A":
            self.granule = {j.split("_")[-2]: j for j in self.all_granules}
        else:
            self.granule = {j.split("_")[-1].replace(".jp2", ""): j for j in self.all_granules}

    def load_metadata_msi(self):
        if self.root_metadata_msi is None:
            self.root_metadata_msi = read_xml(self.metadata_msi)
        return self.root_metadata_msi

    @property
    def polygon(self) -> Union[MultiPolygon, Polygon]:
        """
        Polygon in longlat (EPSG:4326) this takes into account the selected window. This is the intersection of the
        footprint with the window.
        """
        if self._pol is None:
            self.load_metadata_msi()
            footprint_txt = self.root_metadata_msi.findall(".//EXT_POS_LIST")[0].text
            coords_split = footprint_txt.split(" ")[:-1]
            self._pol = Polygon(
                [(float(lngstr), float(latstr)) for latstr, lngstr in zip(coords_split[::2], coords_split[1::2])])

        pol_bounds_latlng = box(*rasterio.warp.transform_bounds(self.crs, "EPSG:4326", *self.bounds))

        return self._pol.intersection(pol_bounds_latlng)

    def radio_add_offsets(self) ->Dict[str,float]:
        if self._radio_add_offsets is None:
            self.load_metadata_msi()
            radio_add_offsets = self.root_metadata_msi.findall(".//RADIO_ADD_OFFSET")
            if len(radio_add_offsets) == 0:
                self._radio_add_offsets = {b : 0 for b in BANDS_S2}

            else:
                self._radio_add_offsets = {BANDS_S2[int(r.attrib["band_id"])]: float(r.text) for r in radio_add_offsets}

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

        reader = RasterioReader([self.granule[band_name] for band_name in band_names],
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

            self.granule_readers[band_name] = RasterioReader(self.granule[band_name],
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

    def read_from_window(self, window:rasterio.windows.Window, boundless:bool) -> '__class__':
        # return GeoTensor(values=self.values, transform=self.transform, crs=self.crs)

        reader_ref = self._get_reader()
        rasterio_reader_ref = reader_ref.read_from_window(window=window, boundless=boundless)
        s2obj =  __class__(s2_folder=self.folder, out_res=self.out_res, window_focus=rasterio_reader_ref.window_focus,
                           bands=self.bands, all_granules=self.all_granules, polygon=self.polygon)

        s2obj.root_metadata_msi = self.root_metadata_msi

        return s2obj

    def load(self, boundless:bool=True)-> GeoTensor:
        reader_ref = self._get_reader()
        geotensor_ref = reader_ref.load(boundless=boundless)

        array_out = np.full((len(self.bands),) + geotensor_ref.shape[-2:],fill_value=geotensor_ref.fill_value_default,
                            dtype=geotensor_ref.dtype)

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

            # TODO deal with NODATA values
            # Important: Adds radio correction! otherwise images after 2022-01-25 shifted (PROCESSING_BASELINE '04.00' or above)
            array_out[idx] = geotensor_iter.values[0] + radio_add[b]

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
    def __init__(self, s2_folder:str, all_granules: List[str],
                 polygon:Polygon, out_res:int=10,
                 window_focus:Optional[rasterio.windows.Window]=None,
                 bands:Optional[List[str]]=None):
        if bands is None:
            bands = BANDS_S2_L2A

        super(S2ImageL2A, self).__init__(s2_folder=s2_folder, all_granules=all_granules, polygon=polygon,
                                         out_res=out_res, bands=bands,
                                         window_focus=window_focus)

        assert self.producttype == "MSIL2A", f"Unexpected product type {self.producttype} in image {self.folder}"

        # see https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR#bands for a description of the granules data

        # TODO include SLC bands for clouds?
        # res_band = 20 if out_res < 20 else out_res
        # band_and_res = f"SCL_{res_band}m.jp2"
        # granules_match = [g for g in self.all_granules if g.endswith(band_and_res)]
        # self.slc_granule = granules_match[0]



class S2ImageL1C(S2Image):
    def __init__(self, s2_folder, all_granules: List[str],
                 polygon:Polygon, out_res:int=10,
                 window_focus:Optional[rasterio.windows.Window]=None,
                 bands:Optional[List[str]]=None):
        super(S2ImageL1C,self).__init__(s2_folder=s2_folder, all_granules=all_granules, polygon=polygon,
                                         out_res=out_res, bands=bands,
                                         window_focus=window_focus)

        assert self.producttype == "MSIL1C", f"Unexpected product type {self.producttype} in image {self.folder}"

        self.msk_clouds_file = os.path.join(self.folder, self.granule_folder, "MSK_CLOUDS_B00.gml").replace("\\","/")
        self.metadata_tl = os.path.join(self.folder, self.granule_folder, "MTD_TL.xml").replace("\\","/")
        self.root_metadata_tl = None

        # Granule in L1C does not include TCI
        # Assert bands in self.granule are ordered as in BANDS_S2
        # assert all(granule[-7:-4] == bname for bname, granule in zip(BANDS_S2, self.granule)), f"some granules are not in the expected order {self.granule}"


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
        epsgNode = geocodingNode.find('HORIZONTAL_CS_CODE')

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
        tileAnglesNode = geomInfoNode.find('Tile_Angles')
        sunZenithNode = tileAnglesNode.find('Sun_Angles_Grid').find('Zenith')
        self.angleGridXres = float(sunZenithNode.find('COL_STEP').text)
        self.angleGridYres = float(sunZenithNode.find('ROW_STEP').text)
        self.sunZenithGrid = self.makeValueArray(sunZenithNode.find('Values_List'))
        sunAzimuthNode = tileAnglesNode.find('Sun_Angles_Grid').find('Azimuth')
        self.sunAzimuthGrid = self.makeValueArray(sunAzimuthNode.find('Values_List'))
        self.anglesGridShape = self.sunAzimuthGrid.shape

        # Now build up the viewing angle per grid cell, from the separate layers
        # given for each detector for each band. Initially I am going to keep
        # the bands separate, just to see how that looks.
        # The names of things in the XML suggest that these are view angles,
        # but the numbers suggest that they are angles as seen from the pixel's
        # frame of reference on the ground, i.e. they are in fact what we ultimately want.
        viewingAngleNodeList = tileAnglesNode.findall('Viewing_Incidence_Angles_Grids')
        self.viewZenithDict = self.buildViewAngleArr(viewingAngleNodeList, 'Zenith')
        self.viewAzimuthDict = self.buildViewAngleArr(viewingAngleNodeList, 'Azimuth')

        # Make a guess at the coordinates of the angle grids. These are not given
        # explicitly in the XML, and don't line up exactly with the other grids, so I am
        # making a rough estimate. Because the angles don't change rapidly across these
        # distances, it is not important if I am a bit wrong (although it would be nice
        # to be exactly correct!).
        (ulx, uly) = self.ulxyByRes["10"]
        self.anglesULXY = (ulx - self.angleGridXres / 2.0, uly + self.angleGridYres / 2.0)

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
            bandId = viewingAngleNode.attrib['bandId']
            angleNode = viewingAngleNode.find(angleName)
            angleArr = self.makeValueArray(angleNode.find('Values_List'))
            if bandId not in angleArrDict:
                angleArrDict[bandId] = angleArr
            else:
                mask = (~np.isnan(angleArr))
                angleArrDict[bandId][mask] = angleArr[mask]
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


def read_xml(xml_file:str) -> ET.Element:
    """Reads xml with xml package """
    if xml_file.startswith("gs://"):
        import fsspec
        fs = fsspec.filesystem("gs", requester_pays=True)
        with fs.open(xml_file, "rb") as file_obj:
            root = ET.fromstring(file_obj.read())
    else:
        root = ET.parse(xml_file).getroot()
    return root


def DN_to_radiance(dn_data:GeoTensor, s2file: S2ImageL1C) -> GeoTensor:
    """
    Function to convert Digital numbers (DN) to radiance.

    Important: this function assumes that radio correction has been applied
     otherwise images after 2022-01-25 shifted (PROCESSING_BASELINE '04.00' or above)
     by default this is applied in the S2Image class.

     Here they say U should be in the numerator
     https://gis.stackexchange.com/questions/285996/convert-sentinel-2-1c-product-from-reflectance-to-radiance

     toaBandX = dn_dataBandX / 10000
     radianceBandX = (toaBandX * cos(SZA) * solarIrradianceBandX * U) / pi

     U should be:
        U = 1. / (1-0.01673*cos(0.0172*(t-4)))^2

     0.0172 = 360/365.256363 * np.pi/180.
     t = datenum(Y,M,D) - datenum(Y,1,1) + 1;

    Args:
        dn_data: data read from an S2Image class with digital numbers
        s2file: s2file where data has been read

    Returns:
        geotensor with radiances
    """
    data_values_new = dn_data.values.astype(np.float32) / 10_000
    s2file.read_metadata_tl()
    solar_irr = s2file.solar_irradiance()
    U = s2file.scale_factor_U()
    for i,b in enumerate(s2file.bands):
        mask = dn_data.values[i] == dn_data.fill_value_default
        data_values_new[i] = data_values_new[i] * np.cos(s2file.mean_sza/180*np.pi) * solar_irr[b] * U / np.pi
        data_values_new[i][mask] = dn_data.fill_value_default

    return GeoTensor(data_values_new, transform=dn_data.transform, crs=dn_data.crs,
                     fill_value_default=dn_data.fill_value_default)


def s2loader(s2folder:str, out_res:int=10,
             bands:Optional[List[str]] = None,
             window_focus:Optional[rasterio.windows.Window]=None,
             all_granules:Optional[List[str]]=None,
             polygon:Optional[Polygon]=None) -> Union[S2ImageL2A, S2ImageL1C]:
    """
    Loads a S2ImageL2A or S2ImageL1C depending on the product type

    Args:
        s2folder: .SAFE folder. Expected standard ESA naming convention (see s2_name_split fun)
        out_res: default output resolution {10, 20, 60}
        bands: Bands to read. Default to BANDS_S2 or BANDS_S2_L2A depending of the product type
        window_focus: window to read when creating the object
        all_granules:
        polygon:

    Returns:
        S2Image reader
    """

    _, producttype_nos2, _, _, _, _, _ = s2_name_split(s2folder)

    if producttype_nos2 == "MSIL2A":
        return S2ImageL2A(s2folder, all_granules=all_granules, polygon=polygon, out_res=out_res,
                          bands=bands, window_focus=window_focus)
    elif producttype_nos2 == "MSIL1C":
        return S2ImageL1C(s2folder, all_granules=all_granules, polygon=polygon, out_res=out_res, bands=bands,
                          window_focus=window_focus)

    raise NotImplementedError(f"Don't know how to load {producttype_nos2} products")


def s2_public_bucket_path(s2file:str, check_exists:bool=False) -> str:
    """
    Returns the expected patch in the public bucket of the S2 file

    Args:
        s2file: safe file (e.g.  S2B_MSIL1C_20220527T030539_N0400_R075_T49SGV_20220527T051042.SAFE)
        check_exists: check if the file exists in the bucket

    Returns:
        full path to the file (gs://gcp-public-data-sentinel-2/tiles/49/S/GV/S2B_MSIL1C_20220527T030539_N0400_R075_T49SGV_20220527T051042.SAFE)
    """
    mission, producttype, sensing_date_str, pdgs, relorbitnum, tile_number_field, product_discriminator = s2_name_split(s2file)
    s2file = s2file[:-1] if s2file.endswith("/") else s2file
    basename = os.path.basename(s2file)
    s2folder = f"gs://gcp-public-data-sentinel-2/tiles/{tile_number_field[:2]}/{tile_number_field[2]}/{tile_number_field[3:]}/{basename}"
    if check_exists:
        import fsspec
        fs = fsspec.filesystem("gs", requester_pays=True)
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
        s2file:

    Returns:
        mission, opertortest, filetype, sitecenter,  creation_date_str, relorbitnum, sensing_time_start, sensing_time_stop
    """

    s2file = s2file[:-1] if s2file.endswith("/") else s2file
    basename = os.path.basename(os.path.splitext(s2file)[0])
    matches = re.match(OLD_FORMAT, basename)
    if matches is not None:
        return matches.groups()