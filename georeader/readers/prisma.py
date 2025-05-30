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
    PRISMA (PRecursore IperSpettrale della Missione Applicativa) image reader.

    This class provides functionality to read and manipulate PRISMA satellite imagery products.
    It handles the specific format and metadata of PRISMA HDF5 files, supporting operations
    like loading specific wavelengths, RGB composites, and converting to reflectance.

    Args:
        filename (str): Path to the PRISMA HDF5 file.

    Attributes:
        filename (str): Path to the PRISMA file.
        swir_cube_dat (str): Path to SWIR cube data in HDF5 file.
        vni_cube_dat (str): Path to VNIR cube data in HDF5 file.
        lats (numpy.ndarray): Latitude values for each pixel.
        lons (numpy.ndarray): Longitude values for each pixel.
        attributes_prisma (Dict): Dictionary of PRISMA metadata attributes.
        nbands_vnir (int): Number of VNIR bands.
        vnir_range (Tuple[float, float]): Wavelength range of VNIR bands (min, max).
        nbands_swir (int): Number of SWIR bands.
        swir_range (Tuple[float, float]): Wavelength range of SWIR bands (min, max).
        ltoa_swir (Optional[NDArray]): SWIR radiance data, lazily loaded.
        ltoa_vnir (Optional[NDArray]): VNIR radiance data, lazily loaded.
        wavelength_swir (Optional[NDArray]): SWIR wavelengths, lazily loaded.
        fwhm_swir (Optional[NDArray]): SWIR FWHM (Full Width at Half Maximum), lazily loaded.
        wavelength_vnir (Optional[NDArray]): VNIR wavelengths, lazily loaded.
        fwhm_vnir (Optional[NDArray]): VNIR FWHM, lazily loaded.
        vza_swir (float): Viewing zenith angle for SWIR.
        vza_vnir (float): Viewing zenith angle for VNIR.
        sza_swir (float): Solar zenith angle for SWIR.
        sza_vnir (float): Solar zenith angle for VNIR.
        time_coverage_start (datetime): Start time of acquisition.
        time_coverage_end (datetime): End time of acquisition.
        units (str): Units of radiance data (mW/m2/sr/nm).

    Examples:
        >>> # Initialize the PRISMA reader with a data path
        >>> prisma = PRISMA('/path/to/prisma_file.he5')
        >>> # Load RGB bands
        >>> rgb = prisma.load_rgb(as_reflectance=True)
        >>> # Load specific wavelengths
        >>> bands = prisma.load_wavelengths([850, 1600, 2200], as_reflectance=True)
        >>> # Get image metadata
        >>> print(prisma)
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

    # def target_spectrum(self, swir_flag:bool) -> NDArray:

    #     if swir_flag:
    #         vza = self.vza_swir
    #         sza = self.sza_swir
    #         band_array = self.wavelength_swir
    #         fwhm_array = self.fwhm_swir
    #         N, M, B = self.ltoa_swir.shape
    #     else:
    #         vza = self.vza_vnir
    #         sza = self.sza_vnir
    #         band_array = self.wavelength_vnir
    #         fwhm_array = self.fwhm_vnir
    #         N, M, B = self.ltoa_vnir.shape

    #     amf = 1.0 / np.cos(vza * np.pi / 180) + 1.0 / np.cos(sza * np.pi / 180)
    #     parent_dir = os.path.dirname(os.path.dirname(__file__))
    #     file_lut_gas = os.path.join(parent_dir, LUT_FILE["name"])

    #     wvl_mod, t_gas_arr, gas_sc_arr, mr_gas_arr = read_luts(
    #         file_lut=file_lut_gas,
    #         t_arr_str=LUT_FILE["t_arr_variable"],
    #         sc_arr_str=LUT_FILE["sc_arr_variable"],
    #         mr_arr_str=LUT_FILE["mr_arr_variable"],
    #         amf=amf,
    #     )

    #     n_wvl = len(wvl_mod)
    #     mr_gas_arr = mr_gas_arr / 1000.0
    #     delta_mr_ref = 1.0

    #     k_spectre = calc_jac_rad(mr_gas_arr, n_wvl, t_gas_arr, delta_mr_ref)
    #     k_array = np.zeros((M, B))
    #     for i in range(0, M):
    #         s = generate_filter(wvl_mod, band_array[i], fwhm_array[i])
    #         k = np.dot(k_spectre, s)
    #         k_array[i] = k

    #     return k_array

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
