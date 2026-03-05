"""
Module to read PRISMA (PRecursore IperSpettrale della Missione Applicativa) hyperspectral images.

PRISMA is an Italian Space Agency (ASI) Earth observation satellite launched in 2019,
carrying a hyperspectral imaging spectrometer that captures data in 239 spectral bands
from 400 to 2500 nm with a 30m spatial resolution.

Data Format Overview
--------------------
PRISMA data is distributed in HDF5 format (HE5 extension) with a specific structure:

    PRISMA HDF5 File Structure:
    ┌─────────────────────────────────────────────────────────┐
    │  /HDFEOS/SWATHS/PRS_L1_HCO/                             │
    │  ├── Data Fields/                                        │
    │  │   ├── VNIR_Cube: (bands, crosstrack, downtrack)      │
    │  │   │   └── 400-1010 nm, ~66 bands                     │
    │  │   └── SWIR_Cube: (bands, crosstrack, downtrack)      │
    │  │       └── 920-2500 nm, ~173 bands                    │
    │  ├── Geolocation Fields/                                 │
    │  │   ├── Latitude_SWIR, Longitude_SWIR                  │
    │  │   └── Latitude_VNIR, Longitude_VNIR                  │
    │  └── Attributes (solar/view angles, timing, etc.)       │
    │                                                          │
    │  /KDP_AUX/                                               │
    │  ├── Cw_Vnir_Matrix, Cw_Swir_Matrix (wavelengths)       │
    │  └── Fwhm_Vnir_Matrix, Fwhm_Swir_Matrix                 │
    └─────────────────────────────────────────────────────────┘

Unlike EMIT, PRISMA data is NOT orthorectified. The geolocation arrays provide
lat/lon coordinates for each pixel, requiring gridding for visualization.

Dual-Sensor Configuration
-------------------------
PRISMA uses two separate sensors for VNIR and SWIR:

    VNIR Sensor                          SWIR Sensor
    ┌────────────────────┐               ┌────────────────────┐
    │ 400 - 1010 nm      │               │ 920 - 2500 nm      │
    │ ~66 bands          │               │ ~173 bands         │
    │ ~10 nm sampling    │               │ ~10 nm sampling    │
    │                    │               │                    │
    │ Shared 30m GSD     │               │ Shared 30m GSD     │
    └────────────────────┘               └────────────────────┘
              │                                    │
              └──────────── Overlap ───────────────┘
                         920-1010 nm
                         
The VNIR and SWIR sensors have overlapping wavelength coverage in the 920-1010 nm
region, which can be used for cross-calibration.

Radiometric Units
-----------------
- L1 Radiance: mW/(m²·sr·nm) - milliwatts per square meter per steradian per nanometer
  (equivalent to W/(m²·sr·μm))
- Scale factors and offsets are applied during loading to convert from DN to radiance

Spectral Characteristics
------------------------
- Total bands: ~239 (66 VNIR + 173 SWIR, minus flagged bands)
- Spectral sampling: ~10 nm (varies slightly)
- FWHM: ~10-12 nm
- SNR: >200 for VNIR, >100 for SWIR

Examples
--------
Basic usage::

    from georeader.readers.prisma import PRISMA
    
    # Load PRISMA image
    prisma = PRISMA('/path/to/PRS_L1_STD_*.he5')
    
    # Load specific wavelengths as reflectance
    bands = prisma.load_wavelengths([850, 1600, 2200], as_reflectance=True)
    
    # Load RGB composite
    rgb = prisma.load_rgb(as_reflectance=True)
    
    # Get georeferenced output (reprojected to UTM)
    rgb_geo = prisma.load_rgb(as_reflectance=True, raw=False)

See Also
--------
georeader.readers.emit : EMIT hyperspectral reader
georeader.readers.enmap : EnMAP hyperspectral reader
georeader.griddata : Utilities for gridding non-orthorectified data

References
----------
- ASI PRISMA Mission: https://www.asi.it/en/earth-science/prisma/
- PRISMA User Guide: https://prisma.asi.it/

"""
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, List, Any, Tuple
from georeader.geotensor import GeoTensor
from datetime import datetime, timezone
from georeader import griddata, reflectance, window_utils
from georeader import compare_crs
import h5py
import os
from numbers import Number

WAVELENGTHS_RGB = np.array([640, 550, 460])

SWIR_FLAG = {
    "swir_cube_dat": {
        True: "/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube",
        False: "/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube",
    },
    "swir_lab": {True: "Swir", False: "Vnir"},
}
HE5_COORDS = {
    "swir_lat": "/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_SWIR",
    "swir_lon": "/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_SWIR",
    "vnir_lon": "/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_VNIR",
    "vnir_lat": "/HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR",
}

# VNIR_WAVELENGTH_RANGE = (406.01318, 976.60223)
# SWIR_WAVELENGTH_RANGE = (976.60223, 2496.7605)


class PRISMA:
    """
    Reader for PRISMA (PRecursore IperSpettrale della Missione Applicativa) hyperspectral images.

    This class provides comprehensive functionality to read and manipulate PRISMA satellite 
    imagery products from the Italian Space Agency (ASI). It handles the dual-sensor
    (VNIR + SWIR) data format, supporting operations like:
    
    - Loading radiance or reflectance data at specific wavelengths
    - Automatic handling of VNIR/SWIR sensor selection based on wavelength
    - Converting radiance to reflectance using solar irradiance
    - Georeferencing raw data to projected coordinate systems
    
    PRISMA Data Model
    -----------------
    PRISMA stores data in sensor coordinates with separate lat/lon arrays for geolocation.
    Unlike EMIT's GLT approach, PRISMA requires gridding/interpolation for orthorectification:
    
        Sensor Grid (raw)                  Geographic Grid (output)
        ┌─────────────────────┐            ┌─────────────────────┐
        │ pushbroom scan      │            │ regular grid        │
        │ ┌───┬───┬───┬───┐  │  gridding  │ ┌───┬───┬───┬───┐  │
        │ │ a │ b │ c │ d │  │  ───────→  │ │ a'│ b'│ c'│ d'│  │
        │ ├───┼───┼───┼───┤  │            │ ├───┼───┼───┼───┤  │
        │ │ e │ f │ g │ h │  │            │ │ e'│ f'│ g'│ h'│  │
        │ └───┴───┴───┴───┘  │            │ └───┴───┴───┴───┘  │
        │ + lat/lon per pixel│            │ + affine transform  │
        └─────────────────────┘            └─────────────────────┘
        
    Raw methods (raw=True) return sensor coordinates; georeferenced methods
    (raw=False) apply gridding to regular geographic coordinates.
    
    Dual Sensor Architecture
    ------------------------
    PRISMA has separate VNIR and SWIR sensors with overlapping coverage:
    
        Wavelength Range:
        ├──────────────────────────────────────────────────────────────┤
        400nm              1000nm                                 2500nm
        ├───────── VNIR ──────────┤
                          ├────────────────── SWIR ───────────────────┤
                          └─ overlap ─┘
                          920-1010nm
    
    The class automatically selects the appropriate sensor based on requested wavelengths.

    Attributes
    ----------
    filename : str
        Path to the PRISMA HE5 file.
    lats : np.ndarray
        Latitude values (H, W) for each pixel in sensor coordinates.
    lons : np.ndarray
        Longitude values (H, W) for each pixel in sensor coordinates.
    attributes_prisma : Dict
        Dictionary of PRISMA metadata attributes from HDF5 root.
    nbands_vnir : int
        Number of valid VNIR bands (excluding flagged bands).
    vnir_range : Tuple[float, float]
        Wavelength range (min, max) of VNIR sensor in nm.
    nbands_swir : int
        Number of valid SWIR bands (excluding flagged bands).
    swir_range : Tuple[float, float]
        Wavelength range (min, max) of SWIR sensor in nm.
    time_coverage_start : datetime
        UTC datetime of acquisition start.
    time_coverage_end : datetime
        UTC datetime of acquisition end.
    units : str
        Radiance units: 'mW/m2/sr/nm'.
    sza_swir : float
        Solar zenith angle (degrees) for SWIR sensor.
    sza_vnir : float
        Solar zenith angle (degrees) for VNIR sensor.
    vza_swir : float
        View zenith angle (degrees) for SWIR sensor.
    vza_vnir : float
        View zenith angle (degrees) for VNIR sensor.
    
    Lazy-Loaded Attributes
    ----------------------
    ltoa_swir : np.ndarray
        SWIR radiance data (H, W, B), loaded by `load_raw(swir_flag=True)`.
    ltoa_vnir : np.ndarray
        VNIR radiance data (H, W, B), loaded by `load_raw(swir_flag=False)`.
    wavelength_swir : np.ndarray
        SWIR wavelengths (H, B) - varies slightly across track.
    wavelength_vnir : np.ndarray
        VNIR wavelengths (H, B) - varies slightly across track.
    fwhm_swir : np.ndarray
        SWIR FWHM values (H, B) - varies slightly across track.
    fwhm_vnir : np.ndarray
        VNIR FWHM values (H, B) - varies slightly across track.
    
    Examples
    --------
    Basic loading::
    
        >>> from georeader.readers.prisma import PRISMA
        >>> 
        >>> prisma = PRISMA('/path/to/PRS_L1_STD_*.he5')
        >>> print(prisma)  # View metadata summary
        >>> print(f"VNIR: {prisma.vnir_range}, SWIR: {prisma.swir_range}")
    
    Loading specific wavelengths::
    
        >>> # Load NDVI bands (Red at 665nm, NIR at 865nm)
        >>> bands = prisma.load_wavelengths([665, 865], as_reflectance=True)
        >>> print(bands.shape)  # (2, H, W) in sensor coordinates
        >>> 
        >>> # Load and georeference to UTM
        >>> bands_geo = prisma.load_wavelengths([665, 865], as_reflectance=True, 
        ...                                       raw=False, resolution_dst=30)
        >>> print(type(bands_geo))  # GeoTensor with transform and CRS
    
    Loading RGB composite::
    
        >>> # Raw sensor coordinates
        >>> rgb_raw = prisma.load_rgb(as_reflectance=True, raw=True)
        >>> 
        >>> # Georeferenced output  
        >>> rgb_geo = prisma.load_rgb(as_reflectance=True, raw=False)
        >>> plt.imshow(np.clip(rgb_geo.values.transpose(1,2,0), 0, 0.3) / 0.3)
    
    Working with raw data::
    
        >>> # Load all SWIR bands
        >>> prisma.load_raw(swir_flag=True)
        >>> print(prisma.ltoa_swir.shape)  # (H, W, ~173)
        >>> print(prisma.wavelength_swir.shape)  # (H, ~173) - wavelengths vary across track
        >>> 
        >>> # Load all VNIR bands
        >>> prisma.load_raw(swir_flag=False)
        >>> print(prisma.ltoa_vnir.shape)  # (H, W, ~66)

    See Also
    --------
    georeader.readers.emit.EMITImage : EMIT hyperspectral reader
    georeader.readers.enmap.EnMAP : EnMAP hyperspectral reader
    georeader.griddata : Gridding utilities for non-orthorectified data
    georeader.reflectance : Radiometric conversion utilities
    
    References
    ----------
    - ASI PRISMA Mission: https://www.asi.it/en/earth-science/prisma/
    - PRISMA Data Products: https://prisma.asi.it/
    """

    def __init__(self, filename: str) -> None:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        self.filename = filename
        self.swir_cube_dat = SWIR_FLAG["swir_cube_dat"][True]
        self.vni_cube_dat = SWIR_FLAG["swir_cube_dat"][False]

        with h5py.File(filename, mode="r") as f:
            dset = f[HE5_COORDS["swir_lat"]]
            self.lats = np.flip(dset[:, :], axis=0)
            dset = f[HE5_COORDS["swir_lon"]]
            self.lons = np.flip(dset[:, :], axis=0)
            self.attributes_prisma = dict(f.attrs)
            sza = f.attrs["Sun_zenith_angle"]

        arr = self.attributes_prisma["List_Cw_Vnir"][
            self.attributes_prisma["List_Cw_Vnir"] > 0
        ]
        self.nbands_vnir = len(arr)
        self.vnir_range = arr.min(), arr.max()
        arr = self.attributes_prisma["List_Cw_Swir"][
            self.attributes_prisma["List_Cw_Swir"] > 0
        ]
        self.swir_range = arr.min(), arr.max()
        self.nbands_swir = len(arr)

        self.ltoa_swir: Optional[NDArray] = None
        self.ltoa_vnir: Optional[NDArray] = None
        self.wavelength_swir: Optional[NDArray] = None
        self.fwhm_swir: Optional[NDArray] = None
        self.wavelength_vnir: Optional[NDArray] = None
        self.fwhm_vnir: Optional[NDArray] = None
        self.vza_swir: float = 0
        self.vza_vnir: float = 0
        self.sza_swir: float = sza
        self.sza_vnir: float = sza

        # self.time_coverage_start = self.attributes_prisma['Product_StartTime']
        self.time_coverage_start = datetime.fromisoformat(
            self.attributes_prisma["Product_StartTime"].decode("utf-8")
        ).replace(tzinfo=timezone.utc)
        self.time_coverage_end = datetime.fromisoformat(
            self.attributes_prisma["Product_StopTime"].decode("utf-8")
        ).replace(tzinfo=timezone.utc)
        self.units = "mW/m2/sr/nm"  # same as W/m^2/SR/um

        self._footprint = griddata.footprint(self.lons, self.lats)
        self._observation_date_correction_factor: Optional[float] = None

    def footprint(self, crs: Optional[str] = None) -> GeoTensor:
        if (crs is None) or compare_crs("EPSG:4326", crs):
            return self._footprint

        return window_utils.polygon_to_crs(
            self._footprint, crs_polygon="EPSG:4326", crs_dst=crs
        )

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
    def bounds(self) -> Tuple[float, float, float, float]:
        return self._footprint.bounds

    def load_raw(self, swir_flag: bool) -> NDArray:
        """
        Load the all the data from all the wavelengths for the VNIR or SWIR range.
        This function caches the data, wavelegths and FWHM in the attributes of the class:
            * `ltoa_swir`, `wavelength_swir`, `fwhm_swir`, `vza_swir`, `sza_swir` if `swir_flag` is True
            * `ltoa_vnir`, `wavelength_vnir`, `fwhm_vnir`, `vza_vnir`, `sza_vnir` if `swir_flag` is False

        Args:
            swir_flag (bool): if True it will load the SWIR range, otherwise it will load the VNIR range

        Returns:
            NDArray: 3D array with the reflectance values (H, W, B)
                where N and M are the dimensions of the image and B is the number of bands.
        """

        if swir_flag:
            if all(
                x is not None
                for x in [
                    self.ltoa_swir,
                    self.wavelength_swir,
                    self.fwhm_swir,
                    self.vza_swir,
                    self.sza_swir,
                ]
            ):
                return self.ltoa_swir
        else:
            if all(
                x is not None
                for x in [
                    self.ltoa_vnir,
                    self.wavelength_vnir,
                    self.fwhm_vnir,
                    self.vza_vnir,
                    self.sza_vnir,
                ]
            ):
                return self.ltoa_vnir

        swir_cube_dat = SWIR_FLAG["swir_cube_dat"][swir_flag]
        swir_lab = SWIR_FLAG["swir_lab"][swir_flag]  # True: "Swir", False: "Vnir"

        with h5py.File(self.filename, "r") as f:
            dset = f[swir_cube_dat]

            ltoa_img = np.flip(np.transpose(dset[:, :, :], axes=[0, 2, 1]), axis=0)

            dset = f["/KDP_AUX/Cw_" + swir_lab + "_Matrix"]
            wvl_mat_ini = dset[:, :]

            dset = f["/KDP_AUX/Fwhm_" + swir_lab + "_Matrix"]
            fwhm_mat_ini = dset[:, :]
            
            wvl_cntr = f.attrs["List_Cw_" + swir_lab]
            wvl_flag = f.attrs["List_Cw_" + swir_lab + "_Flags"]
            
            sc_fac = f.attrs["ScaleFactor_" + swir_lab]

            of_fac = f.attrs["Offset_" + swir_lab]

            vza = 0.0
            sza = f.attrs["Sun_zenith_angle"]

            ltoa_img = ltoa_img / sc_fac - of_fac

        # Lambda
        wvl_mat_ini = np.flip(wvl_mat_ini, axis=1)
        li_no0 = np.where(wvl_mat_ini[100, :] > 0)[0]
        wvl_mat = np.copy(wvl_mat_ini[:, li_no0])
        wl_center_ini = np.mean(wvl_mat, axis=0)

        # FWHM
        fwhm_mat_ini = np.flip(fwhm_mat_ini, axis=1)
        fwhm_mat = np.copy(fwhm_mat_ini[:, li_no0])

        M, N, B_tot = ltoa_img.shape

        if swir_flag:
            if B_tot == len(wl_center_ini):
                ltoa_img = np.flip(ltoa_img, axis=2)
            else:
                # ltoa_img = np.flip(ltoa_img[:, :, :-2], axis=2)
                non0_bands = np.where(wvl_flag == 1)[0]
                ltoa_img = np.flip(ltoa_img[:, :, non0_bands], axis=2)

        else:
            if B_tot == len(wl_center_ini):
                ltoa_img = np.flip(ltoa_img, axis=2)
            else:
                # ltoa_img = np.flip(ltoa_img[:, :, 3:], axis=2)  # Revisar esto(not sure)
                non0_bands = np.where(wvl_flag == 1)[0]
                ltoa_img = np.flip(ltoa_img[:, :, non0_bands], axis=2)

        ltoa_img = np.transpose(ltoa_img, (1, 0, 2))
        if swir_flag:
            self.ltoa_swir = ltoa_img
            self.wavelength_swir = wvl_mat
            self.fwhm_swir = fwhm_mat
            self.vza_swir = vza
            self.sza_swir = sza
        else:
            self.ltoa_vnir = ltoa_img
            self.wavelength_vnir = wvl_mat
            self.fwhm_vnir = fwhm_mat
            self.vza_vnir = vza
            self.sza_vnir = sza

        return ltoa_img

    def load_wavelengths(
        self,
        wavelengths: Union[float, List[float], NDArray],
        as_reflectance: bool = True,
        raw: bool = True,
        resolution_dst=30,
        dst_crs: Optional[Any] = None,
        fill_value_default: float = -1,
    ) -> Union[GeoTensor, NDArray]:
        """
        Load the reflectance of the given wavelengths

        Args:
            wavelengths (Union[float, List[float], NDArray]): List of wavelengths to load
            as_reflectance (bool, optional): return the values as reflectance rather than radiance. Defaults to True.
                If False values will have units of W/m^2/SR/um (`self.units`)
            raw (bool, optional): if True it will return the raw values,
                if False it will return the values reprojected to the specified CRS and resolution. Defaults to True.
            resolution_dst (int, optional): if raw is False, it will reproject the values to this resolution. Defaults to 30.
            dst_crs (Optional[Any], optional): if None it will use the corresponding UTM zone.
            fill_value_default (float, optional): fill value. Defaults to -1.

        Returns:
            Union[GeoTensor, NDArray]: if raw is True it will return a NDArray with the values, otherwise it will return a GeoTensor
                with the reprojected values in its `.values` attribute.
        """

        if isinstance(wavelengths, Number):
            wavelengths = np.array([wavelengths])
        else:
            wavelengths = np.array(wavelengths)

        load_swir = any(
            [
                wvl >= self.swir_range[0] and wvl < self.swir_range[1]
                for wvl in wavelengths
            ]
        )
        load_vnir = any(
            [
                wvl >= self.vnir_range[0] and wvl < self.vnir_range[1]
                for wvl in wavelengths
            ]
        )
        if load_swir:
            self.load_raw(swir_flag=True)
            wavelength_swir_mean = np.mean(self.wavelength_swir, axis=0)
            fwhm_swir_mean = np.mean(self.fwhm_swir, axis=0)
        if load_vnir:
            self.load_raw(swir_flag=False)
            wavelength_vnir_mean = np.mean(self.wavelength_vnir, axis=0)
            fwhm_vnir_mean = np.mean(self.fwhm_vnir, axis=0)

        ltoa_img = []
        fwhm = []
        for b in range(len(wavelengths)):
            if (
                wavelengths[b] >= self.swir_range[0]
                and wavelengths[b] < self.swir_range[1]
            ):
                index_band = np.argmin(np.abs(wavelengths[b] - wavelength_swir_mean))
                fwhm.append(fwhm_swir_mean[index_band])
                img = self.ltoa_swir[..., index_band]
            else:
                index_band = np.argmin(np.abs(wavelengths[b] - wavelength_vnir_mean))
                fwhm.append(fwhm_vnir_mean[index_band])
                img = self.ltoa_vnir[..., index_band]

            ltoa_img.append(img)

        # Transpose to row major
        ltoa_img = np.transpose(np.stack(ltoa_img, axis=0), (0, 2, 1))

        if as_reflectance:
            thuiller = reflectance.load_thuillier_irradiance()
            response = reflectance.srf(wavelengths, fwhm, thuiller["Nanometer"].values)

            solar_irradiance_norm = thuiller["Radiance(mW/m2/nm)"].values.dot(
                response
            )  # mW/m$^2$/nm
            solar_irradiance_norm /= 1_000  # W/m$^2$/nm

            ltoa_img = reflectance.radiance_to_reflectance(
                ltoa_img,
                solar_irradiance_norm,
                units=self.units,
                observation_date_corr_factor=self.observation_date_correction_factor,
            )

        if raw:
            return ltoa_img

        return griddata.read_to_crs(
            np.transpose(ltoa_img, (1, 2, 0)),
            lons=self.lons,
            lats=self.lats,
            resolution_dst=resolution_dst,
            dst_crs=dst_crs,
            fill_value_default=fill_value_default,
        )

    def load_rgb(
        self, as_reflectance: bool = True, raw: bool = True
    ) -> Union[GeoTensor, NDArray]:
        return self.load_wavelengths(
            wavelengths=WAVELENGTHS_RGB, as_reflectance=as_reflectance, raw=raw
        )

    def __repr__(self) -> str:
        return f"""
        File: {self.filename}
        Bounds: {self.bounds}
        Time: {self.time_coverage_start}
        VNIR Range: {self.vnir_range} {self.nbands_vnir} bands
        SWIR Range: {self.swir_range} {self.nbands_swir} bands
        """
