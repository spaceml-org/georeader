"""
Module to read EnMAP (Environmental Mapping and Analysis Program) hyperspectral images.

EnMAP is a German hyperspectral satellite mission operated by DLR (German Aerospace Center),
launched in 2022. It provides high-spectral-resolution data in 224 bands from 420 to 2450 nm
with a 30m spatial resolution and 30km swath width.

Data Format Overview
--------------------
EnMAP data is distributed as separate GeoTIFF files with an XML metadata file:

    EnMAP Product Structure:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ENMAP01-____L1C-DT0000000000_20220501T101523Z_001_V010110_...     │
    │  ├── *-METADATA.XML           ← Main metadata file (input)         │
    │  ├── *-SPECTRAL_IMAGE_VNIR.TIF   420-1000 nm, ~88 bands            │
    │  ├── *-SPECTRAL_IMAGE_SWIR.TIF   900-2450 nm, ~136 bands           │
    │  ├── *-QL_QUALITY_CLOUD.TIF      Cloud mask                        │
    │  ├── *-QL_QUALITY_CIRRUS.TIF     Cirrus mask                       │
    │  ├── *-QL_QUALITY_SNOW.TIF       Snow mask                         │
    │  ├── *-QL_QUALITY_HAZE.TIF       Haze mask                         │
    │  └── *-QL_PIXELMASK_*.TIF        Per-sensor pixel masks            │
    └─────────────────────────────────────────────────────────────────────┘

Unlike EMIT and PRISMA, EnMAP L1C data is already orthorectified (map-projected) with
Rational Polynomial Coefficients (RPCs) stored in the metadata for refined geolocation.

Dual-Sensor Architecture
------------------------
EnMAP uses two pushbroom sensors with overlapping spectral coverage:

    VNIR Detector                        SWIR Detector
    ┌────────────────────┐               ┌────────────────────┐
    │ 420 - 1000 nm      │               │ 900 - 2450 nm      │
    │ ~88 bands          │               │ ~136 bands         │
    │ 6.5 nm sampling    │               │ 10 nm sampling     │
    │ Si CCD             │               │ HgCdTe             │
    └────────────────────┘               └────────────────────┘
              │                                    │
              └──────────── Overlap ───────────────┘
                         900-1000 nm

The spectral overlap enables cross-calibration between the two detectors.

Radiometric Processing
----------------------
EnMAP L1B/L1C data requires conversion from Digital Numbers (DN) to radiance:

    L_λ = DN × GAIN + OFFSET   [W/(m²·sr·nm)]
    
    Note: DLR provides gains between 2000-10000 (multiplicative, not divisive)
    The reader applies: L = (GAIN × DN + OFFSET) × 1000 to get mW/(m²·sr·nm)

Rational Polynomial Coefficients (RPCs)
---------------------------------------
EnMAP includes RPCs for precise geolocation refinement:

    Pixel (col, row) ──→ RPC Transform ──→ Geographic (lon, lat)
    
    RPCs model:
    - Satellite orbit and attitude
    - Sensor geometry  
    - Terrain elevation effects (when height_off is set appropriately)
    
The reader can apply RPCs during loading for refined geolocation.

Product Levels
--------------
- L1B: At-sensor radiance, sensor geometry (unrectified)
- L1C: At-sensor radiance, orthorectified (map-projected)
- L2A: Surface reflectance, atmospheric correction applied

This reader is designed for L1B and L1C products.

Examples
--------
Basic usage::

    from georeader.readers.enmap import EnMAP
    
    # Load from metadata XML file
    enmap = EnMAP('/path/to/*-METADATA.XML')
    
    # Load specific wavelengths as reflectance
    bands = enmap.load_wavelengths([665, 865, 1600], as_reflectance=True)
    
    # Load RGB with RPC-refined geolocation
    rgb = enmap.load_rgb(as_reflectance=True, apply_rpcs=True)
    
    # Load quality masks
    cloud_mask = enmap.load_product('QL_QUALITY_CLOUD')

See Also
--------
georeader.readers.emit : EMIT hyperspectral reader
georeader.readers.prisma : PRISMA hyperspectral reader
georeader.rasterio_reader : Base reader for GeoTIFF files

References
----------
- DLR EnMAP Mission: https://www.enmap.org/
- EnMAP Product Specification: https://www.enmap.org/data_access/
- GFZ enpt Package: https://github.com/GFZ/enpt (metadata parsing reference)

"""
import numpy as np
import xml.etree.ElementTree as ET
from georeader.rasterio_reader import RasterioReader
from georeader.geotensor import GeoTensor
from georeader import reflectance
from georeader import read
from rasterio.windows import Window
from typing import Optional, Union, Any, List, Tuple
from numbers import Number
from numpy.typing import NDArray
from datetime import datetime, timezone
import os
import fsspec
import rasterio.rpc
from collections import OrderedDict


SC_COEFF = 1.0e3

WAVELENGTHS_RGB = np.array([640, 550, 460])

PRODUCT_FOLDERS = {
    "SPECTRAL_IMAGE_SWIR": "swir",
    "QL_QUALITY_CLOUDSHADOW": "ql/cloudshadow",
    "QL_PIXELMASK_SWIR": "ql/pixelmask_swir",
    "QL_QUALITY_CIRRUS": "ql/cirrus",
    "QL_QUALITY_SNOW": "ql/snow",
    "QL_PIXELMASK_VNIR": "ql/pixelmask_vnir",
    "QL_QUALITY_TESTFLAGS_SWIR": "ql/testflags_swir",
    "QL_QUALITY_HAZE": "ql/haze",
    "QL_QUALITY_TESTFLAGS_VNIR": "ql/testflags_vnir",
    "QL_QUALITY_CLASSES": "ql/classes",
    "QL_SWIR": "ql/swir",
    "QL_QUALITY_CLOUD": "ql/cloud",
    "QL_VNIR": "ql/vnir",
    "SPECTRAL_IMAGE_VNIR": "vnir",
    "METADATA": "metadata",
}


# Finds the RPCs in EnMAP's metadata file
def _find_metadata_rpcs(tree, lbl="swir", wvl=2400):
    """Reads the RPC coefficients for a wavelength"""
    # Open the metadata file
    xml = tree.getroot()
    # Initialize the rpc variable
    rpc_coeffs = OrderedDict()
    # Number of wavelengths
    startband = (
        0 if lbl == "vnir" else int(xml.find("product/image/vnir/channels").text)
    )
    nwvl = int(xml.find("product/image/%s/channels" % lbl).text)
    subset = slice(startband, startband + nwvl)
    # Read RPC coefficients
    for bID in xml.findall("product/navigation/RPC/bandID")[subset]:
        # Collect the band ID
        bN = "band_%d" % np.int64(bID.attrib["number"])
        # Keys to look for in the metadata
        keys2combine = ("row_num", "row_den", "col_num", "col_den")
        # Temporary variable having the coefficients
        tmp = OrderedDict(
            [(ele.tag.lower(), float(ele.text)) for ele in bID.findall("./")]
        )
        # Reformat the coefficients
        rpc_coeffs[bN] = {
            k: v for k, v in tmp.items() if not k.startswith(keys2combine)
        }
        for n in keys2combine:
            rpc_coeffs[bN]["%s_coeffs" % n.lower()] = np.array(
                [v for k, v in tmp.items() if k.startswith(n)]
            )
    # If only one band is requested
    if wvl is not None:
        # Collect the central wavelengths
        bi = "specific/bandCharacterisation/bandID/"
        wvl_center = np.array(
            [
                float(ele.text)
                for ele in xml.findall(bi + "wavelengthCenterOfBand")[subset]
            ]
        )
        # Find the closest band name
        band_index = np.argmin(np.abs(wvl_center - wvl))
        band_name = [band_name_i for band_name_i in rpc_coeffs.keys()][band_index]
        # Select the RPCs
        rpc_out = rpc_coeffs[band_name]
    # Otherwise, return all
    else:
        rpc_out = rpc_coeffs
    # Return
    return rpc_out


# Build an RPC Rasterio object
def _rasterio_build_rpcs(rpcs) -> rasterio.rpc.RPC:
    """Creates an RPC Rasterio object"""
    # Setting the height offset to zero
    rpcs["height_off"] = 0
    """
    Setting the height offset to 0 tricks the RPCs
    to generate lat/lon coordinates at the averge
    height of the scene (whatever that height might
    be according to the DEM). This improves EnMAPs
    georeferencing without conducting a terrain
    orthorectification.
    """
    # Build an rpc object
    rpcs_rio = rasterio.rpc.RPC(
        rpcs["height_off"],
        rpcs["height_scale"],
        rpcs["lat_off"],
        rpcs["lat_scale"],
        list(rpcs["row_den_coeffs"]),
        list(rpcs["row_num_coeffs"]),
        rpcs["row_off"],
        rpcs["row_scale"],
        rpcs["long_off"],
        rpcs["long_scale"],
        list(rpcs["col_den_coeffs"]),
        list(rpcs["col_num_coeffs"]),
        rpcs["col_off"],
        rpcs["col_scale"],
        err_bias=None,
        err_rand=None,
    )
    # Return the rasterio RPC object
    return rpcs_rio


def read_ang(tree, lab_ang):

    for fact in tree.iter(tag=lab_ang):
        try:
            ang1 = np.float64(fact.find("upper_left").text)
            ang2 = np.float64(fact.find("upper_right").text)
            ang3 = np.float64(fact.find("lower_left").text)
            ang4 = np.float64(fact.find("lower_right").text)
        except:
            pass

    return np.mean([ang1, ang2, ang3, ang4])


# Metadata reader from GFZ postdam here-> https://github.com/GFZ/enpt/blob/main/enpt/model/metadata/metadata_sensorgeo.py
def read_xml(file_xml):

    tree = ET.parse(file_xml)  # read in the XML

    rpcs_swir = _rasterio_build_rpcs(_find_metadata_rpcs(tree, lbl="swir", wvl=2400))
    rpcs_vnir = _rasterio_build_rpcs(_find_metadata_rpcs(tree, lbl="vnir", wvl=350))

    wl_center = []
    wl_fwhm = []
    gain_arr = []
    offs_arr = []
    for fact in tree.iter(tag="bandID"):
        try:
            wvl = float(fact.find("wavelengthCenterOfBand").text)
            fwhm = float(fact.find("FWHMOfBand").text)
            gain = float(fact.find("GainOfBand").text)
            offset = float(fact.find("OffsetOfBand").text)

            wl_center.append(wvl)
            wl_fwhm.append(fwhm)
            gain_arr.append(gain)
            offs_arr.append(offset)
        except:
            pass

    wl_center = np.array(wl_center)
    wl_fwhm = np.array(wl_fwhm)
    gain_arr = np.array(gain_arr)
    offs_arr = np.array(offs_arr)

    wl_center_product = {}
    wl_fwhm_product = {}
    gain_arr_product = {}
    offs_arr_product = {}

    for lab_str_m in ["swir", "vnir"]:
        num_bd_sp = None
        for fact in tree.iter(tag=lab_str_m + "ProductQuality"):
            try:
                num_bd_sp = (np.array(fact.find("numChannelsExpected").text)).astype(
                    int
                )
            except:
                pass

        if num_bd_sp is None:
            raise ValueError(f"Could not find numChannelsExpected for {lab_str_m}")

        num_bd = len(wl_center)

        if lab_str_m == "swir":
            wl_center_product[lab_str_m] = wl_center[num_bd - num_bd_sp :]
            wl_fwhm_product[lab_str_m] = wl_fwhm[num_bd - num_bd_sp :]
            gain_arr_product[lab_str_m] = gain_arr[num_bd - num_bd_sp :]
            offs_arr_product[lab_str_m] = offs_arr[num_bd - num_bd_sp :]
        else:
            wl_center_product[lab_str_m] = wl_center[0:num_bd_sp]
            wl_fwhm_product[lab_str_m] = wl_fwhm[0:num_bd_sp]
            gain_arr_product[lab_str_m] = gain_arr[0:num_bd_sp]
            offs_arr_product[lab_str_m] = offs_arr[0:num_bd_sp]

    # Read angles and mean ground elevation
    sza = 90.0 - read_ang(tree, "sunElevationAngle")
    saa = read_ang(tree, "sunAzimuthAngle")
    vaa = read_ang(tree, "sceneAzimuthAngle")
    vza = read_ang(tree, "acrossOffNadirAngle")

    startTime = datetime.strptime(
        tree.find("base/temporalCoverage/startTime").text, "%Y-%m-%dT%H:%M:%S.%fZ"
    ).replace(tzinfo=timezone.utc)
    stopTime = datetime.strptime(
        tree.find("base/temporalCoverage/stopTime").text, "%Y-%m-%dT%H:%M:%S.%fZ"
    ).replace(tzinfo=timezone.utc)

    for fact in tree.iter(tag="specific"):
        try:
            hsf = np.float64(fact.find("meanGroundElevation").text)
        except:
            pass

    return (
        wl_center_product,
        wl_fwhm_product,
        hsf,
        sza,
        saa,
        vaa,
        vza,
        gain_arr_product,
        offs_arr_product,
        startTime,
        stopTime,
        rpcs_vnir,
        rpcs_swir,
    )


class EnMAP:
    """
    Reader for EnMAP (Environmental Mapping and Analysis Program) hyperspectral images.

    This class provides comprehensive functionality to read and manipulate EnMAP satellite
    imagery products from DLR. It handles the multi-file product structure (separate VNIR/SWIR
    GeoTIFFs with XML metadata), supporting operations like:
    
    - Loading radiance or reflectance data at specific wavelengths
    - Automatic handling of VNIR/SWIR sensor selection based on wavelength
    - Converting DN to radiance using gain/offset from metadata
    - Converting radiance to reflectance using solar irradiance
    - Applying Rational Polynomial Coefficients (RPCs) for refined geolocation
    - Loading quality masks (cloud, cirrus, snow, haze)
    
    EnMAP Data Model
    ----------------
    EnMAP L1C products are orthorectified (map-projected) GeoTIFFs with separate files
    for VNIR and SWIR bands:
    
        File Structure:
        ┌────────────────────────────────────────────────────┐
        │  METADATA.XML  ──→  wavelengths, FWHM, angles,    │
        │                      gain/offset, RPCs             │
        │                                                    │
        │  SPECTRAL_IMAGE_VNIR.TIF  ──→  (88, H, W) bands   │
        │  SPECTRAL_IMAGE_SWIR.TIF  ──→  (136, H, W) bands  │
        │                                                    │
        │  QL_QUALITY_*.TIF  ──→  quality masks              │
        └────────────────────────────────────────────────────┘
    
    Radiometric Conversion
    ----------------------
    DN to radiance conversion is automatic::
    
        L_λ = (GAIN × DN + OFFSET) × 1000   [mW/(m²·sr·nm)]
        
        Note: DLR gains are multiplicative (not divisive as in some sensors)

    Spectral Configuration
    ----------------------
    EnMAP has two detectors with overlapping coverage:
    
        Wavelength: 420nm ──── 1000nm ──── 2450nm
                    ├── VNIR ────┤
                              ├──── SWIR ──────────┤
                              └ overlap┘
                              900-1000nm
    
    - VNIR: Silicon CCD, ~88 bands, 6.5nm sampling, SNR >500:1
    - SWIR: HgCdTe, ~136 bands, 10nm sampling, SNR >150:1

    Attributes
    ----------
    xml_file : str
        Path to the EnMAP XML metadata file.
    by_folder : bool
        Whether files are organized by folder structure (alternative naming convention).
    swir_file : str
        Path to the SWIR GeoTIFF file (derived from xml_file).
    fs : fsspec.AbstractFileSystem
        Filesystem for file access (local or cloud storage).
    vnir : RasterioReader
        Reader for VNIR spectral image.
    swir : RasterioReader
        Reader for SWIR spectral image.
    wl_center : Dict[str, np.ndarray]
        Center wavelengths per sensor: {'vnir': [...], 'swir': [...]}.
    wl_fwhm : Dict[str, np.ndarray]
        FWHM per sensor: {'vnir': [...], 'swir': [...]}.
    gain_arr : Dict[str, np.ndarray]
        Radiometric gains per sensor for DN→radiance conversion.
    offs_arr : Dict[str, np.ndarray]
        Radiometric offsets per sensor for DN→radiance conversion.
    vnir_range : Tuple[float, float]
        VNIR wavelength range (min, max) including FWHM margins.
    swir_range : Tuple[float, float]
        SWIR wavelength range (min, max) including FWHM margins.
    hsf : float
        Mean ground elevation (m) from scene metadata.
    sza : float
        Solar zenith angle (degrees).
    saa : float
        Solar azimuth angle (degrees).
    vza : float
        View zenith angle (across-track off-nadir angle, degrees).
    vaa : float
        View azimuth angle (scene azimuth, degrees).
    rpcs_vnir : rasterio.rpc.RPC
        Rational Polynomial Coefficients for VNIR refined geolocation.
    rpcs_swir : rasterio.rpc.RPC
        Rational Polynomial Coefficients for SWIR refined geolocation.
    time_coverage_start : datetime
        UTC datetime of acquisition start.
    time_coverage_end : datetime
        UTC datetime of acquisition end.
    units : str
        Radiance units: 'mW/m2/sr/nm'.
    
    Properties (from underlying readers)
    ------------------------------------
    shape : Tuple[int, int, int]
        Full shape (total_bands, height, width).
    transform : rasterio.Affine
        Affine geotransform from SWIR file.
    crs : rasterio.crs.CRS
        Coordinate reference system from SWIR file.
    bounds : Tuple[float, float, float, float]
        Geographic bounds (xmin, ymin, xmax, ymax).
    res : Tuple[float, float]
        Pixel resolution (x, y).

    Examples
    --------
    Basic loading::
    
        >>> from georeader.readers.enmap import EnMAP
        >>> 
        >>> enmap = EnMAP('/data/ENMAP01-...-METADATA.XML')
        >>> print(enmap)  # View metadata summary
    
    Loading specific wavelengths::
    
        >>> # Load NDVI bands as reflectance
        >>> bands = enmap.load_wavelengths([665, 865], as_reflectance=True)
        >>> print(bands.shape)  # (2, H, W)
        >>> 
        >>> # Compute NDVI
        >>> red, nir = bands.values[0], bands.values[1]
        >>> ndvi = (nir - red) / (nir + red + 1e-10)
    
    Loading RGB with RPC refinement::
    
        >>> # Apply RPCs for better geolocation (recommended)
        >>> rgb = enmap.load_rgb(as_reflectance=True, apply_rpcs=True)
        >>> 
        >>> # Without RPCs (uses original map projection)
        >>> rgb = enmap.load_rgb(as_reflectance=True, apply_rpcs=False)
    
    Loading quality masks::
    
        >>> # Load cloud mask
        >>> cloud = enmap.load_product('QL_QUALITY_CLOUD')
        >>> 
        >>> # Available products: 
        >>> # 'QL_QUALITY_CLOUD', 'QL_QUALITY_CIRRUS', 'QL_QUALITY_SNOW',
        >>> # 'QL_QUALITY_HAZE', 'QL_PIXELMASK_VNIR', 'QL_PIXELMASK_SWIR'
    
    Spatial subsetting with window_focus::
    
        >>> from rasterio.windows import Window
        >>> 
        >>> # Focus on a specific region
        >>> window = Window(col_off=100, row_off=200, width=500, height=500)
        >>> enmap_subset = EnMAP('/path/to/METADATA.XML', window_focus=window)
    
    Cloud storage access::
    
        >>> import gcsfs
        >>> 
        >>> fs = gcsfs.GCSFileSystem()
        >>> enmap = EnMAP('gs://bucket/ENMAP-METADATA.XML', fs=fs)

    See Also
    --------
    georeader.readers.emit.EMITImage : EMIT hyperspectral reader
    georeader.readers.prisma.PRISMA : PRISMA hyperspectral reader
    georeader.rasterio_reader.RasterioReader : Base reader for GeoTIFF
    georeader.read.read_rpcs : Apply RPC transformations
    
    References
    ----------
    - DLR EnMAP Mission: https://www.enmap.org/
    - GFZ enpt Package: https://github.com/GFZ/enpt (metadata parser reference)
    - EnMAP Product Specification Document
    """

    def __init__(
        self,
        xml_file: str,
        by_folder: bool = False,
        window_focus: Optional[Window] = None,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        self.xml_file = xml_file
        self.by_folder = by_folder
        if not self.xml_file.endswith(".xml") and not self.xml_file.endswith(".XML"):
            raise ValueError(
                f"Invalid SWIR file path {self.xml_file} must be a XML file"
            )

        if self.by_folder:
            assert (
                PRODUCT_FOLDERS["METADATA"] in self.xml_file
            ), f"Invalid SWIR file path {self.xml_file} must contain {PRODUCT_FOLDERS['METADATA']} if by folder"
            self.swir_file = (
                self.xml_file.replace(
                    PRODUCT_FOLDERS["METADATA"], PRODUCT_FOLDERS["SPECTRAL_IMAGE_SWIR"]
                )
                .replace(".XML", ".TIF")
                .replace(".xml", ".tif")
            )
        else:
            assert (
                "METADATA" in self.xml_file
            ), f"Invalid SWIR file path {self.xml_file} must contain METADATA if not by folder"
            self.swir_file = (
                self.xml_file.replace("METADATA", "SPECTRAL_IMAGE_SWIR")
                .replace(".XML", ".TIF")
                .replace(".xml", ".tif")
            )

        if not self.swir_file.endswith(".tif") and not self.swir_file.endswith(".TIF"):
            raise ValueError(
                f"Invalid SWIR file path {self.swir_file} must be a TIF file"
            )

        if self.xml_file.startswith("gs://") or self.xml_file.startswith("az://"):
            assert fs is not None, "Filesystem must be provided if using cloud storage"
            self.fs = fs
            assert fs.exists(self.xml_file), f"File {self.xml_file} does not exist"
            assert fs.exists(self.swir_file), f"File {self.swir_file} does not exist"
        else:
            self.fs = fs or fsspec.filesystem("file")
            assert os.path.exists(self.xml_file), f"File {self.xml_file} does not exist"
            assert os.path.exists(
                self.swir_file
            ), f"File {self.swir_file} does not exist"

        self.swir = RasterioReader(self.swir_file, window_focus=window_focus)

        if self.by_folder:
            self.vnir = RasterioReader(
                self.swir_file.replace(
                    PRODUCT_FOLDERS["SPECTRAL_IMAGE_SWIR"],
                    PRODUCT_FOLDERS["SPECTRAL_IMAGE_VNIR"],
                ),
                window_focus=window_focus,
            )
        else:
            self.vnir = RasterioReader(
                self.swir_file.replace("SPECTRAL_IMAGE_SWIR", "SPECTRAL_IMAGE_VNIR"),
                window_focus=window_focus,
            )

        with self.fs.open(self.xml_file) as fh:
            (
                self.wl_center,
                self.wl_fwhm,
                self.hsf,
                self.sza,
                self.saa,
                self.vaa,
                self.vza,
                self.gain_arr,
                self.offs_arr,
                startTime,
                endTime,
                self.rpcs_vnir,
                self.rpcs_swir,
            ) = read_xml(fh)

        self.swir_range = (
            self.wl_center["swir"][0] - self.wl_fwhm["swir"][0],
            self.wl_center["swir"][-1] + self.wl_fwhm["swir"][-1],
        )
        self.vnir_range = (
            self.wl_center["vnir"][0] - self.wl_fwhm["vnir"][0],
            self.wl_center["vnir"][-1] + self.wl_fwhm["vnir"][-1],
        )

        self.units = "mW/m2/sr/nm"  # == W/m^2/SR/um
        self.time_coverage_start = startTime
        self.time_coverage_end = endTime
        self._observation_date_correction_factor: Optional[float] = None

    @property
    def observation_date_correction_factor(self) -> float:
        if self._observation_date_correction_factor is None:
            self._observation_date_correction_factor = (
                reflectance.observation_date_correction_factor(
                    date_of_acquisition=self.time_coverage_start,
                    center_coords=self.footprint("EPSG:4326").centroid.coords[0],
                )
            )
        return self._observation_date_correction_factor

    @property
    def window_focus(self) -> Optional[Window]:
        return self.swir.window_focus

    @property
    def shape(self) -> tuple:
        return (
            len(self.wl_center["vnir"]) + len(self.wl_center["swir"]),
        ) + self.swir.shape[-2:]

    @property
    def transform(self):
        return self.swir.transform

    @property
    def crs(self):
        return self.swir.crs

    @property
    def res(self):
        return self.swir.res

    @property
    def width(self):
        return self.window_focus.width

    @property
    def height(self):
        return self.window_focus.height

    @property
    def bounds(self):
        return self.swir.bounds

    @property
    def fill_value_default(self):
        return self.swir.fill_value_default

    def footprint(self, crs: Optional[Any] = None) -> Any:
        return self.swir.footprint(crs=crs)

    def load_product(self, product_name: str) -> GeoTensor:
        if product_name not in PRODUCT_FOLDERS:
            raise ValueError(f"Invalid product name: {product_name}")

        if self.by_folder:
            folder = PRODUCT_FOLDERS[product_name]
            product_path = self.swir_file.replace(
                PRODUCT_FOLDERS["SPECTRAL_IMAGE_SWIR"], folder
            )

            raster_product = RasterioReader(
                product_path, window_focus=self.window_focus
            ).load()
        else:
            product_path = self.swir_file.replace("SPECTRAL_IMAGE_SWIR", product_name)
            raster_product = RasterioReader(
                product_path, window_focus=self.window_focus
            ).load()

        # Convert to radiance if SPECTRAL_IMAGE_SWIR or SPECRTAL_IMAGE_VNIR
        if product_name == "SPECTRAL_IMAGE_SWIR":
            name_coef = "swir"
        elif product_name == "SPECTRAL_IMAGE_VNIR":
            name_coef = "vnir"
        else:
            name_coef = None

        # https://github.com/GFZ/enpt/blob/main/enpt/model/images/images_sensorgeo.py#L327
        # Lλ = QCAL * GAIN + OFFSET
        # NOTE: - DLR provides gains between 2000 and 10000, so we have to DEVIDE by gains
        #       - DLR gains / offsets are provided in W/m2/sr/nm, so we have to multiply by 1000 to get
        #         mW/m2/sr/nm as needed later
        if name_coef is not None:
            gain = self.gain_arr[name_coef]
            offset = self.offs_arr[name_coef]
            invalids = raster_product.values == raster_product.fill_value_default
            raster_product.values = (
                gain[:, np.newaxis, np.newaxis] * raster_product.values
                + offset[:, np.newaxis, np.newaxis]
            ) * SC_COEFF
            raster_product.values[invalids] = self.fill_value_default

        return raster_product

    def load_wavelengths(
        self,
        wavelengths: Union[float, List[float], NDArray],
        as_reflectance: bool = True,
    ) -> Union[GeoTensor, NDArray]:
        """
        Load the reflectance of the given wavelengths

        Args:
            wavelengths (Union[float, List[float], NDArray]): List of wavelengths to load
            as_reflectance (bool, optional): return the values as reflectance rather than radiance.
                Defaults to True. If False values will have units of W/m^2/SR/um == mW/m2/sr/nm (`self.units`)

        Returns:
            Union[GeoTensor, NDArray]: GeoTensor with the values in reflectance or radiance units.

        Raises:
            ValueError: If any wavelength is outside the sensor's range.
        """
        if isinstance(wavelengths, Number):
            wavelengths = np.array([wavelengths])
        else:
            wavelengths = np.array(wavelengths)

        # Check all wavelengths are within the range of the sensor
        if any(
            [
                wvl < self.vnir_range[0] or wvl > self.swir_range[1]
                for wvl in wavelengths
            ]
        ):
            raise ValueError(
                f"Invalid wavelength range, must be between {self.vnir_range[0]} and {self.swir_range[1]}"
            )

        wavelengths_loaded = []
        fwhm = []
        ltoa_img = []
        for b in range(len(wavelengths)):
            if (
                wavelengths[b] >= self.swir_range[0]
                and wavelengths[b] < self.swir_range[1]
            ):
                index_band = np.argmin(np.abs(wavelengths[b] - self.wl_center["swir"]))
                fwhm.append(self.wl_fwhm["swir"][index_band])
                wavelengths_loaded.append(self.wl_center["swir"][index_band])
                rst = self.swir.isel({"band": [index_band]}).load().squeeze()
                invalids = (rst.values == rst.fill_value_default) | np.isnan(rst.values)

                # Convert to radiance
                gain = self.gain_arr["swir"][index_band]
                offset = self.offs_arr["swir"][index_band]
                img = (gain * rst.values + offset) * SC_COEFF
                img[invalids] = self.fill_value_default
            else:
                index_band = np.argmin(np.abs(wavelengths[b] - self.wl_center["vnir"]))
                fwhm.append(self.wl_fwhm["vnir"][index_band])
                wavelengths_loaded.append(self.wl_center["vnir"][index_band])
                rst = self.vnir.isel({"band": [index_band]}).load().squeeze()
                invalids = (rst.values == rst.fill_value_default) | np.isnan(rst.values)

                # Convert to radiance
                gain = self.gain_arr["vnir"][index_band]
                offset = self.offs_arr["vnir"][index_band]
                img = (gain * rst.values + offset) * SC_COEFF
                img[invalids] = self.fill_value_default

            ltoa_img.append(img)

        ltoa_img = GeoTensor(
            np.stack(ltoa_img, axis=0),
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
        )

        if as_reflectance:
            thuiller = reflectance.load_thuillier_irradiance()
            response = reflectance.srf(
                wavelengths_loaded, fwhm, thuiller["Nanometer"].values
            )

            solar_irradiance_norm = thuiller["Radiance(mW/m2/nm)"].values.dot(
                response
            )  # mW/m$^2$/SR/nm
            solar_irradiance_norm /= 1_000  # W/m$^2$/nm

            # Divide by 10 to convert from mW/m^2/SR/nm to µW /cm²/SR/nm
            ltoa_img = reflectance.radiance_to_reflectance(
                ltoa_img,
                solar_irradiance_norm,
                units=self.units,
                observation_date_corr_factor=self.observation_date_correction_factor,
            )

        return ltoa_img

    def load_rgb(
        self,
        as_reflectance: bool = True,
        apply_rpcs: bool = True,
        dst_crs: str = "EPSG:4326",
        resolution_dst_crs: Optional[Union[float, Tuple[float, float]]] = None,
    ) -> GeoTensor:
        """
        Load RGB image from VNIR bands. Converts radiance to TOA reflectance if as_reflectance is True
        otherwise it will return the radiance values in W/m^2/SR/um == mW/m2/sr/nm (`self.units`)

        Args:
            as_reflectance (bool, optional): Convert radiance to TOA reflectance. Defaults to True.
            apply_rpcs (bool, optional): Apply RPCs to the image. Defaults to True.
            dst_crs (str, optional): Destination CRS. Defaults to "EPSG:4326".
            resolution_dst_crs (Optional[Union[float, Tuple[float, float]]], optional):
                Resolution of the destination CRS. Defaults to None.
        Returns:
            GeoTensor: with the RGB image
        """
        rgb = self.load_wavelengths(WAVELENGTHS_RGB, as_reflectance=as_reflectance)
        if apply_rpcs:
            return read.read_rpcs(
                rgb.values,
                rpcs=self.rpcs_vnir,
                dst_crs=dst_crs,
                resolution_dst_crs=resolution_dst_crs,
                fill_value_default=rgb.fill_value_default,
            )
        elif dst_crs is not None:
            return read.read_to_crs(
                rgb, resolution_dst_crs=resolution_dst_crs, dst_crs=dst_crs
            )

        return rgb

    def load(self) -> GeoTensor:
        swir = self.load_product("SPECTRAL_IMAGE_SWIR")
        # vnir = self.load_product('SPECTRAL_IMAGE_VNIR')

        return swir

    def __repr__(self) -> str:
        return f"""
        File: {self.xml_file}
        Bounds: {self.bounds}
        Time: {self.time_coverage_start}
        Spatial shape (height, width): {self.height, self.width}
        VNIR Range: {self.vnir_range} nbands: {len(self.wl_center['vnir'])} 
        SWIR Range: {self.swir_range} nbands: {len(self.wl_center['swir'])}
        """
