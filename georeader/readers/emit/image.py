"""
EMITImage: reader for EMIT L1B radiance with GLT orthorectification. See ``georeader.readers.emit``.
"""
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows
from numpy.typing import NDArray
from shapely.geometry import Polygon
from shapely.ops import unary_union

from georeader import get_utm_epsg, read, reflectance, window_utils
from georeader.geotensor import GeoTensor
from georeader.griddata import georreference
from georeader.readers.emit.download import download_product, get_l2amask_link, get_obs_link
from georeader.readers.emit.mask import mask_band_index, mask_flag_indexes
from georeader.readers.emit.utils import _bounds_indexes_raw

try:
    import xarray as xr
    from georeader.io import safe_open_netcdf
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    xr = None
    safe_open_netcdf = None

WAVELENGTHS_RGB = np.array([640, 550, 460])


class EMITImage:
    """
    Reader for EMIT L1B (Earth Surface Mineral Dust Source Investigation) hyperspectral images.
    
    This class provides comprehensive functionality to read and manipulate EMIT satellite 
    imagery products from NASA's imaging spectrometer aboard the ISS. It handles the 
    unique GLT-based (Geographic Lookup Table) storage format, supporting operations like:
    
    - Loading radiometry data with automatic orthorectification
    - Converting radiance to reflectance using solar irradiance
    - Accessing cloud and quality masks
    - Extracting viewing and solar geometry angles
    - Reprojecting to different coordinate reference systems
    
    EMIT Data Model
    ---------------
    EMIT stores data in sensor coordinates, not geographic coordinates. The GLT provides
    a lookup table mapping geographic pixels to sensor pixels:
    
        GLT Orthorectification:
        ┌────────────────────────────┐      ┌──────────────────────────┐
        │    Geographic Grid         │      │   Sensor Grid (raw)      │
        │  (orthorectified space)    │      │  (pushbroom scan)        │
        │  ┌───┬───┬───┬───┐        │      │  ┌───┬───┬───┬───┐      │
        │  │ · │ a │ b │ · │        │  GLT │  │ e │ a │ b │ · │      │
        │  ├───┼───┼───┼───┤        │  ──→ │  ├───┼───┼───┼───┤      │
        │  │ c │ d │ e │ f │        │      │  │ f │ c │ d │ · │      │
        │  └───┴───┴───┴───┘        │      │  └───┴───┴───┴───┘      │
        │  (pixels with data)        │      │  (original acquistion)   │
        └────────────────────────────┘      └──────────────────────────┘
        
        · = no data (GLT value = 0)
        
        For geographic pixel (row, col):
            raw_x = glt_x[row, col]  
            raw_y = glt_y[row, col]
            value = radiance[raw_y, raw_x, :]
    
    This approach preserves original radiometric values without interpolation artifacts.
    
    Spectral Characteristics
    ------------------------
    - Wavelength range: 380-2500 nm (VNIR + SWIR)
    - Number of bands: 285
    - Spectral sampling: ~7.4 nm
    - Spatial resolution: 60m at nadir
    
    Attributes
    ----------
    filename : str
        Path to the EMIT NetCDF file.
    nc_ds : xr.Dataset
        xarray Dataset handle for the main radiance file.
    glt : GeoTensor
        Geographic Lookup Table as a GeoTensor with shape (2, H, W).
        - glt.values[0]: x-indices into raw radiance (1-based)
        - glt.values[1]: y-indices into raw radiance (1-based)
    valid_glt : np.ndarray
        Boolean mask (H, W) indicating valid GLT entries (data coverage).
    glt_relative : GeoTensor
        GLT with indices relative to the data window (0-based).
    window_raw : rasterio.windows.Window
        Window defining the subset of raw data to read (optimizes I/O).
    real_transform : rasterio.Affine
        Affine transform for the orthorectified (geographic) grid.
    time_coverage_start : datetime
        UTC datetime of acquisition start.
    time_coverage_end : datetime
        UTC datetime of acquisition end.
    wavelengths : np.ndarray
        Center wavelengths (nm) for selected bands.
    fwhm : np.ndarray
        Full Width at Half Maximum (nm) for selected bands.
    band_selection : Union[int, Tuple[int, ...], slice]
        Current band subset selection.
    units : str
        Radiance units from file metadata (typically 'uW/(cm^2 sr nm)').
    fill_value_default : float
        No-data value for radiance data.
    dims : Tuple[str]
        Dimension names ("band", "y", "x").
    dtype : np.dtype
        Data type of radiance values.
    
    Lazy-Loaded Properties
    ----------------------
    nc_ds_obs : xr.Dataset
        Observation data (viewing/solar angles, path length, elevation).
        Auto-downloaded from NASA Earthdata if not present locally.
    nc_ds_l2amask : xr.Dataset  
        L2A quality mask data (clouds, cirrus, water, aggregate flags).
        Auto-downloaded from NASA Earthdata if not present locally.
    mean_sza : float
        Mean solar zenith angle (degrees) across the scene.
    mean_vza : float
        Mean view zenith angle (degrees) across the scene.
    observation_date_correction_factor : float
        Earth-Sun distance correction factor for the acquisition date.
    
    Examples
    --------
    Basic loading and reprojection::
    
        >>> from georeader.readers.emit import EMITImage, download_product
        >>> 
        >>> # Download from NASA Earthdata
        >>> link = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/...'
        >>> filepath = download_product(link)
        >>> 
        >>> # Open and reproject to UTM
        >>> emit = EMITImage(filepath)
        >>> emit_utm = emit.to_crs("UTM", resolution_dst_crs=60)
        >>> 
        >>> # Load as reflectance
        >>> refl = emit_utm.load(as_reflectance=True)
        >>> print(refl.shape)  # (285, H, W)
    
    Working with specific wavelengths::
    
        >>> # Select RGB-like bands (640, 550, 460 nm)
        >>> emit.set_band_selection([35, 23, 11])
        >>> print(emit.wavelengths)  # [641.2, 553.1, 462.3]
        >>> rgb = emit.load(as_reflectance=True)
        >>> 
        >>> # Or use the convenience method
        >>> rgb = emit.load_rgb(as_reflectance=True)
    
    Accessing masks and quality data::
    
        >>> # Get valid (cloud-free) mask
        >>> valid_mask = emit.validmask()
        >>> print(f"Clear pixels: {emit.percentage_clear:.1f}%")
        >>> 
        >>> # Get specific mask layers
        >>> cloud_mask = emit.mask("Cloud flag")
        >>> water_mask = emit.water_mask()
    
    Working with viewing geometry::
    
        >>> # Get solar zenith angle
        >>> sza = emit.sza()  # GeoTensor with SZA values
        >>> 
        >>> # Get mean angles for quick reference
        >>> print(f"Mean SZA: {emit.mean_sza:.1f}°")
        >>> print(f"Mean VZA: {emit.mean_vza:.1f}°")
    
    Spatial subsetting::
    
        >>> import rasterio.windows
        >>> 
        >>> # Read a spatial window
        >>> window = rasterio.windows.Window(col_off=100, row_off=200, width=500, height=500)
        >>> emit_subset = emit.read_from_window(window)
        >>> data = emit_subset.load()
    
    See Also
    --------
    georeader.readers.prisma.PRISMA : PRISMA hyperspectral reader
    georeader.readers.enmap.EnMAP : EnMAP hyperspectral reader
    georeader.reflectance : Radiometric conversion utilities
    
    References
    ----------
    - EMIT L1B Product Guide: https://lpdaac.usgs.gov/products/emitl1bradv001/
    - EMIT Data Resources: https://github.com/nasa/EMIT-Data-Resources
    - EMIT Algorithms: Green et al. (2020) doi:10.1029/2020JD033451
    """
    attributes_set_if_exists = ["_nc_ds_obs", "_mean_sza", "_mean_vza",
                                "_observation_bands", "_nc_ds_l2amask", "_mask_bands",
                                "obs_file", "l2amaskfile",
                                # Option B: opt-in radiance cache. ``_cache`` is a
                                # mutable dict shared by reference across all clones
                                # built from the same parent — that's what makes the
                                # cache visible end-to-end. ``cache_radiance`` is the
                                # opt-in flag (rebind-on-clone is fine; we don't toggle
                                # per-clone).
                                "_cache", "cache_radiance"]

    # Key under which the full-spectrum windowed radiance is stored in ``_cache``.
    _CACHE_KEY_RADIANCE = "radiance_window"

    def __init__(self, filename:str, glt:Optional[GeoTensor]=None,
                 band_selection:Optional[Union[int, Tuple[int, ...],slice]]=slice(None),
                 cache_radiance:bool=False,
                 reuse_handles_from:Optional['EMITImage']=None):
        if not HAS_XARRAY:
            raise ImportError("xarray is required to read EMIT images. Please install it with: pip install xarray")

        self.filename = filename
        if reuse_handles_from is not None:
            if reuse_handles_from.filename != self.filename:
                raise ValueError("reuse_handles_from must reference the same EMIT file")
            # Clone constructor path: reuse parent handles to avoid opening
            # throwaway datasets that would immediately be overwritten.
            self.nc_ds = reuse_handles_from.nc_ds
        else:
            self.nc_ds = safe_open_netcdf(self.filename, cache=False, load=False)
        self._nc_ds_obs = None
        self._nc_ds_l2amask = None
        self._observation_bands = None
        self._mask_bands = None
        self._sensor_band_params = None
        # Opt-in radiance cache. Default off — the dict is created either way so the
        # ``_cache is parent._cache`` invariant holds for clones even when caching
        # is disabled.
        self.cache_radiance:bool = cache_radiance
        self._cache:Dict[str, Any] = {}
        # self.real_shape = (self.nc_ds['radiance'].shape[-1],) + self.nc_ds['radiance'].shape[:-1]

        self._mean_sza = None
        self._mean_vza = None
        self.obs_file:Optional[str] = None
        self.l2amaskfile:Optional[str] = None

        geotransform = self.nc_ds.attrs['geotransform']
        self.real_transform = rasterio.Affine(geotransform[1], geotransform[2], geotransform[0],
                                              geotransform[4], geotransform[5], geotransform[3])
        
        self.time_coverage_start = datetime.strptime(self.nc_ds.attrs['time_coverage_start'], "%Y-%m-%dT%H:%M:%S%z")
        self.time_coverage_end = datetime.strptime(self.nc_ds.attrs['time_coverage_end'], "%Y-%m-%dT%H:%M:%S%z")

        self.dtype = self.nc_ds['radiance'].dtype
        self.dims = ("band", "y", "x")
        self.fill_value_default = self.nc_ds['radiance'].attrs.get('_FillValue', -9999)
        self.nodata = self.fill_value_default
        self.units = self.nc_ds["radiance"].attrs.get('units', '')

        if glt is None:
            # Open the location group to access glt_x and glt_y
            location_ds = safe_open_netcdf(self.filename, cache=False, load=False, group='location')
            glt_x = np.nan_to_num(location_ds['glt_x'].values, nan=0).astype(np.int32)
            glt_y = np.nan_to_num(location_ds['glt_y'].values, nan=0).astype(np.int32)
            location_ds.close()
            
            glt_arr = np.zeros((2,) + glt_x.shape, dtype=np.int32)
            glt_arr[0] = glt_x
            glt_arr[1] = glt_y
            # glt_arr -= 1 # account for 1-based indexing

            # https://rasterio.readthedocs.io/en/stable/api/rasterio.crs.html
            self.glt = GeoTensor(glt_arr, transform=self.real_transform, 
                                 crs=rasterio.crs.CRS.from_wkt(self.nc_ds.attrs['spatial_ref']),
                                 fill_value_default=0)
        else:
            self.glt = glt
        
        self.valid_glt = np.all(self.glt.values != self.glt.fill_value_default, axis=0)
        xmin, ymin, xmax, ymax = self._bounds_indexes_raw() # values are 1-based!

        # glt has the absolute indexes of the netCDF object
        # glt_relative has the relative indexes
        self.glt_relative = self.glt.copy()
        self.glt_relative.values[0, self.valid_glt] -= xmin
        self.glt_relative.values[1, self.valid_glt] -= ymin

        self.window_raw = rasterio.windows.Window(col_off=xmin-1, row_off=ymin-1, 
                                                  width=xmax-xmin+1, height=ymax-ymin+1)

        # Load sensor_band_parameters from its group, unless we're cloning from
        # an existing instance and can reuse the already-open handle.
        if reuse_handles_from is not None:
            self._sensor_band_params = reuse_handles_from._sensor_band_params
            self.bandname_dimension = reuse_handles_from.bandname_dimension
        else:
            self._sensor_band_params = safe_open_netcdf(self.filename, cache=False, load=False, group='sensor_band_parameters')
            if "wavelengths" in self._sensor_band_params:
                self.bandname_dimension = "wavelengths"
            elif "radiance_wl" in self._sensor_band_params:
                self.bandname_dimension = "radiance_wl"
            else:
                raise ValueError("wavelengths or radiance_wl not found in sensor_band_parameters")
        
        self.band_selection = band_selection
        self.wavelengths = self._sensor_band_params[self.bandname_dimension].values[self.band_selection]
        self.fwhm = self._sensor_band_params['fwhm'].values[self.band_selection]
        self._observation_date_correction_factor:Optional[float] = None

    @property
    def observation_date_correction_factor(self) -> float:
        if self._observation_date_correction_factor is None:
            self._observation_date_correction_factor = reflectance.observation_date_correction_factor(date_of_acquisition=self.time_coverage_start,
                                                                                                      center_coords=self.footprint("EPSG:4326").centroid.coords[0])
        return self._observation_date_correction_factor
    
    @property
    def crs(self) -> Any:
        return self.glt.crs

    @property
    def shape(self) -> Tuple:
        try:
            n_bands = len(self.wavelengths)
            return  (n_bands,) + self.glt.shape[1:]
        except Exception:
            return self.glt.shape

    @property
    def width(self) -> int:
        return self.shape[-1]
    
    @property
    def height(self) -> int:
        return self.shape[-2]

    @property
    def transform(self) -> rasterio.Affine:
        return self.glt.transform

    @property
    def res(self) -> Tuple[float, float]:
        return self.glt.res

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.glt.bounds

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        """
        Get the footprint of the image in the given CRS. If no CRS is given, the footprint is returned in the native CRS.
        This function takes into account the valid_glt mask to compute the footprint.

        Args:
            crs (Optional[str], optional): The CRS to return the footprint in. Defaults to None. 
                If None, the footprint is returned in the native CRS.
        
        Returns:
            Polygon: The footprint of the image in the given CRS.
        """
        if not hasattr(self, '_pol'):
            from georeader.vectorize import get_polygons
            pols = get_polygons(self.valid_glt, transform=self.transform)
            self._pol = unary_union(pols)
        if crs is not None:
            pol_crs = window_utils.polygon_to_crs(self._pol, self.crs, crs)
        else:
            pol_crs = self._pol
        
        pol_glt = self.glt.footprint(crs=crs)

        return pol_crs.intersection(pol_glt)
    
    def set_band_selection(self, band_selection:Optional[Union[int, Tuple[int, ...],slice]]=None):
        """
        Set the band selection. Band selection is absolute w.r.t self.nc_ds['radiance']

        Args:
            band_selection (Optional[Union[int, Tuple[int, ...],slice]], optional): slicing or selection of the bands. Defaults to None.
        
        Example:
            >>> emit_image.set_band_selection(slice(0, 3)) # will only load the three first bands
            >>> emit_image.wavelengths # will only return the wavelengths of the three first bands
            >>> emit_image.load() # will only load the three first bands
        """
        if band_selection is None:
            band_selection = slice(None)
        self.band_selection = band_selection
        self.wavelengths = self._sensor_band_params[self.bandname_dimension].values[self.band_selection]
        self.fwhm = self._sensor_band_params['fwhm'].values[self.band_selection]
    
    @ property
    def nc_ds_obs(self, obs_file:Optional[str]=None):
        """
        Loads the observation file. In this file we have information about angles (solar and viewing),
        elevation and ilumination based on elevation and path length.

        This function downloads the observation file if it does not exist from the JPL portal.

        It caches the observation file in the object. (self.nc_ds_obs)

        Args:
            obs_file (Optional[str], optional): Path to the observation file. 
                Defaults to None. If none it will download the observation file 
                from the EMIT server.
        """
        if self._nc_ds_obs is not None:
            return self._nc_ds_obs
        
        if obs_file is None:
            link_obs_file = get_obs_link(self.filename)
            obs_file = os.path.join(os.path.dirname(self.filename), os.path.basename(link_obs_file))
            if not os.path.exists(obs_file):
                download_product(link_obs_file, obs_file)
        
        self.obs_file = obs_file
        self._nc_ds_obs = safe_open_netcdf(obs_file, cache=False, load=False)
        # Load observation_bands from sensor_band_parameters group
        sensor_params = safe_open_netcdf(obs_file, cache=False, load=False, group='sensor_band_parameters')
        self._observation_bands = sensor_params['observation_bands'].values
        sensor_params.close()
        return self._nc_ds_obs
    
    @property
    def nc_ds_l2amask(self, l2amaskfile:Optional[str]=None) -> xr.Dataset:
        """
        Loads the L2A mask file. In this file we have information about the cloud mask.

        This function downloads the L2A mask file if it does not exist from the JPL portal.

        It caches the L2A mask file in the object. (self.nc_ds_l2amask)

        See https://lpdaac.usgs.gov/products/emitl2arflv001/ (v001, mask inside the RFL granule) and
        https://lpdaac.usgs.gov/products/emitl2amaskv003/ (v003) for info about the L2A mask file.

        Args:
            l2amaskfile (Optional[str], optional): Path to the L2A mask file. 
                Defaults to None. If none it will download the L2A mask file 
                from the EMIT server.
        """
        if self._nc_ds_l2amask is not None:
            return self._nc_ds_l2amask
        
        if l2amaskfile is None:
            link_l2amaskfile = get_l2amask_link(self.filename)
            l2amaskfile = os.path.join(os.path.dirname(self.filename), os.path.basename(link_l2amaskfile))
            if not os.path.exists(l2amaskfile):
                download_product(link_l2amaskfile, l2amaskfile)
        
        self.l2amaskfile = l2amaskfile
        self._nc_ds_l2amask = safe_open_netcdf(l2amaskfile, cache=False, load=False)
        # Load mask_bands from sensor_band_parameters group
        sensor_params = safe_open_netcdf(l2amaskfile, cache=False, load=False, 
                                         group='sensor_band_parameters')
        self._mask_bands = sensor_params["mask_bands"].values
        sensor_params.close()
        return self._nc_ds_l2amask
    
    @property
    def mask_bands(self) -> np.array:
        """ Returns the mask band labels. The layout depends on the mask version:

        - v001: ['Cloud flag', 'Cirrus flag', 'Water flag', 'Spacecraft Flag', 'Dilated Cloud Flag',
          'AOD550', 'H2O (g cm-2)', 'Aggregate Flag']
        - v003: ['Cloud Flag', 'Cirrus Flag', 'Water Flag', 'Dilated Cloud Flag',
          'SpecTf-Cloud Probability', 'SpecTf-Cloud Flag', 'SpecTf-Buffer Distance']
        """
        self.nc_ds_l2amask
        return self._mask_bands
    
    def validmask(self, with_buffer:bool=True, include_spectf:bool=False) -> GeoTensor:
        """
        Return the validmask mask

        Args:
            with_buffer (bool): also mask the dilated cloud flag. Defaults to True.
            include_spectf (bool): also mask the SpecTf ML cloud flag (v003 masks only). Defaults to False.

        Returns:
            GeoTensor: bool mask. True means that the pixel is valid.
        """

        validmask = ~self.invalid_mask_raw(with_buffer=with_buffer, include_spectf=include_spectf)

        return self.georreference(validmask,
                                  fill_value_default=False)
    
    def invalid_mask_raw(self, with_buffer:bool=True, include_spectf:bool=False) -> NDArray:
        """
        Returns the non georreferenced quality mask. True means that the pixel is not valid.

        This mask is computed as the sum of the Cloud flag, Cirrus flag, Spacecraft flag (v001 only)
        and, with ``with_buffer``, the Dilated Cloud Flag. Flags are selected by label (see
        ``mask_flag_indexes``) because the band layout changed between mask versions.
        True means that the pixel is not valid.

        From: https://github.com/nasa/EMIT-Data-Resources/blob/main/python/how-tos/How_to_use_EMIT_Quality_data.ipynb
        and https://github.com/nasa/EMIT-Data-Resources/blob/main/python/modules/emit_tools.py#L277

        Args:
            with_buffer (bool): also mask the dilated cloud flag. Defaults to True.
            include_spectf (bool): also mask the SpecTf ML cloud flag (v003 masks only). Defaults to False.
        """
        band_index = mask_flag_indexes(self.mask_bands, with_buffer=with_buffer,
                                       include_spectf=include_spectf, source=self.l2amaskfile)

        slice_y, slice_x = self.window_raw.toslices()
        mask_arr = self.nc_ds_l2amask['mask'].values[slice_y, slice_x, band_index]
        mask_arr = np.sum(mask_arr, axis=-1)
        mask_arr = (mask_arr >= 1)
        return mask_arr
    
    @property
    def percentage_clear(self) -> float:
        """
        Return the percentage of clear pixels in the image

        Returns:
            float: percentage of clear pixels
        """
        
        invalids = self.invalid_mask_raw(with_buffer=False)
        return 100 * (1 - np.sum(invalids) / np.prod(invalids.shape))


    def mask(self, mask_name:str="cloud_mask") -> GeoTensor:
        """
        Return the mask layer with the given name.
        Mask shall be one of self.mask_bands; the lookup ignores case and repeated whitespace,
        so 'Water flag' works on both v001 ('Water flag') and v003 ('Water Flag') masks.

        Args:
            mask_name (str, optional): Name of the mask. Defaults to "cloud_mask".

        Returns:
            GeoTensor: mask
        """
        band_index = mask_band_index(self.mask_bands, mask_name, source=self.l2amaskfile)
        slice_y, slice_x = self.window_raw.toslices()
        mask_arr = self.nc_ds_l2amask['mask'].values[slice_y, slice_x, band_index]
        return self.georreference(mask_arr,
                                  fill_value_default=self.nc_ds_l2amask['mask'].attrs.get('_FillValue', -9999))
    
    def water_mask(self) -> GeoTensor:
        """ Returns the water mask """
        return self.mask("Water flag")
    
    @property
    def observation_bands(self) -> np.array:
        """ Returns the observation bands """
        self.nc_ds_obs
        return self._observation_bands
    
    def observation(self, name:str) -> GeoTensor:
        """ Returns the observation with the given name """
        band_index = self.observation_bands.tolist().index(name)
        slice_y, slice_x = self.window_raw.toslices()
        # The obs file stores obs data in root group, not in a subgroup
        obs_arr = self.nc_ds_obs['obs'].values[slice_y, slice_x, band_index]
        return self.georreference(obs_arr, 
                                  fill_value_default=self.nc_ds_obs['obs'].attrs.get('_FillValue', -9999))

    def sza(self) -> GeoTensor:
        """ Return the solar zenith angle as a GeoTensor """
        return self.observation('To-sun zenith (0 to 90 degrees from zenith)')
    
    def vza(self) -> GeoTensor:
        """ Return the view zenith angle as a GeoTensor """
        return self.observation('To-sensor zenith (0 to 90 degrees from zenith)')
    
    def elevation(self) -> GeoTensor:
        location_ds = safe_open_netcdf(self.filename, cache=False, load=False, group='location')
        obs_arr = location_ds["elev"]
        slice_y, slice_x = self.window_raw.toslices()
        elev_data = obs_arr.values[slice_y, slice_x]
        fill_val = obs_arr.attrs.get('_FillValue', -9999)
        location_ds.close()
        return self.georreference(elev_data, fill_value_default=fill_val)

    @property
    def mean_sza(self) -> float:
        """ Return the mean solar zenith angle """
        if self._mean_sza is not None:
            return self._mean_sza
        
        band_index = self.observation_bands.tolist().index('To-sun zenith (0 to 90 degrees from zenith)')
        sza_arr = self.nc_ds_obs['obs'].values[..., band_index]
        fill_val = self.nc_ds_obs['obs'].attrs.get('_FillValue', -9999)
        self._mean_sza = float(np.mean(sza_arr[sza_arr != fill_val]))
        return self._mean_sza
    
    @property
    def mean_vza(self) -> float:
        """ Return the mean view zenith angle """
        if self._mean_vza is not None:
            return self._mean_vza
        band_index = self.observation_bands.tolist().index('To-sensor zenith (0 to 90 degrees from zenith)')
        vza_arr = self.nc_ds_obs['obs'].values[..., band_index]
        fill_val = self.nc_ds_obs['obs'].attrs.get('_FillValue', -9999)
        self._mean_vza = float(np.mean(vza_arr[vza_arr != fill_val]))
        return self._mean_vza
        
    def __copy__(self) -> '__class__':
        out = EMITImage(
            self.filename,
            glt=self.glt.copy(),
            band_selection=self.band_selection,
            reuse_handles_from=self,
        )
        
        # copy nc_ds_obs if it exists
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, getattr(self, attrname))

        return out
    def copy(self) -> '__class__':
        return self.__copy__()
    
    def to_crs(self, crs:Any="UTM", 
               resolution_dst_crs:Optional[Union[float, Tuple[float, float]]]=60) -> '__class__':
        """
        Reproject the image to a new crs

        Args:
            crs (Any): CRS. 

        Returns:
            EmitImage: EMIT image in the new CRS
        
        Example:
            >>> emit_image = EMITImage("path/to/emit_image.nc")
            >>> emit_image_utm = emit_image.to_crs(crs="UTM")
        """
        if crs == "UTM":
            footprint = self.glt.footprint("EPSG:4326")
            crs = get_utm_epsg(footprint)

        glt = read.read_to_crs(self.glt, crs, resampling=rasterio.warp.Resampling.nearest, 
                               resolution_dst_crs=resolution_dst_crs)

        out = EMITImage(
            self.filename,
            glt=glt,
            band_selection=self.band_selection,
            reuse_handles_from=self,
        )

        # Propagate eagerly-set and lazily-loaded attributes from the parent so
        # the new instance shares the parent's NetCDF handles, sensor params,
        # observation bands, mean angles, etc. without re-opening anything.
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, getattr(self, attrname))

        # _pol is not in attributes_set_if_exists because it's CRS-dependent —
        # it must be reprojected to the new CRS.
        if hasattr(self, '_pol'):
            setattr(out, '_pol', window_utils.polygon_to_crs(self._pol, self.crs, crs))

        return out


    def read_from_window(self, window:Optional[rasterio.windows.Window]=None, boundless:bool=True) -> '__class__':
        glt_window = self.glt.read_from_window(window, boundless=boundless)
        out = EMITImage(
            self.filename,
            glt=glt_window,
            band_selection=self.band_selection,
            reuse_handles_from=self,
        )

        # Propagate eagerly-set and lazily-loaded attributes from the parent.
        for attrname in self.attributes_set_if_exists:
            if hasattr(self, attrname):
                setattr(out, attrname, getattr(self, attrname))

        return out
    
    def read_from_bands(self, bands:Union[int, Tuple[int, ...], slice]) -> '__class__':
        copy = self.__copy__()
        copy.set_band_selection(bands)
        return copy
  
    def load(self, boundless:bool=True, as_reflectance:bool=False)-> GeoTensor:
        data = self.load_raw() # (C, H, W) or (H, W)
        if as_reflectance:
            invalids = np.isnan(data) | (data == self.fill_value_default)
            thuiller = reflectance.load_thuillier_irradiance()
            response = reflectance.srf(self.wavelengths, self.fwhm, thuiller["Nanometer"].values)
            solar_irradiance_norm = thuiller["Radiance(mW/m2/nm)"].values.dot(response) / 1_000
            data = reflectance.radiance_to_reflectance(data, solar_irradiance_norm,
                                                       units=self.units,
                                                       observation_date_corr_factor=self.observation_date_correction_factor)
            data[invalids] = self.fill_value_default
        return self.georreference(data, fill_value_default=self.fill_value_default)
    
    def load_rgb(self, as_reflectance:bool=True) -> GeoTensor:
        bands_read = np.argmin(np.abs(WAVELENGTHS_RGB[:, np.newaxis] - self.wavelengths), axis=1).tolist()
        ei_rgb = self.read_from_bands(bands_read)
        return ei_rgb.load(boundless=True, as_reflectance=as_reflectance)

    @property
    def shape_raw(self) -> Tuple[int, int, int]:
        """ Return the shape of the raw data in (C, H, W) format """
        return (len(self.wavelengths),) + rasterio.windows.shape(self.window_raw)

    def _bounds_indexes_raw(self) -> Tuple[int, int, int, int]:
        """ Return the bounds of the raw data: (min_x, min_y, max_x, max_y) """
        return _bounds_indexes_raw(self.glt.values, self.valid_glt)


    def load_raw(self, transpose:bool=True) -> np.array:
        """
        Load the raw data, without orthorectification

        Args:
            transpose (bool, optional): Transpose the data if it has 3 dimentsions to (C, H, W)
                Defaults to True. if False return (H, W, C)

        Returns:
            np.array: raw data (C, H, W) or (H, W)
        """

        slice_y, slice_x = self.window_raw.toslices()

        if self.cache_radiance:
            # Option B (opt-in): cache the full-spectrum windowed radiance so that
            # subsequent loads of band subsets become pure in-memory slices.
            # ``self._cache`` is a mutable dict shared with all clones built from
            # this instance (via ``attributes_set_if_exists``), so a single
            # decompression services every algorithm downstream.
            cached = self._cache.get(self._CACHE_KEY_RADIANCE)
            if cached is None:
                radiance = self.nc_ds['radiance']
                dims = radiance.dims
                cached = radiance.isel({dims[0]: slice_y, dims[1]: slice_x}).values
                self._cache[self._CACHE_KEY_RADIANCE] = cached
            data = cached[..., self.band_selection]
        else:
            # Default path: push the spatial (and, when possible, spectral) slice
            # into the NetCDF read via xarray .isel(). Avoids materialising the
            # full radiance variable in RAM, but re-reads from disk each call.
            radiance = self.nc_ds['radiance']
            dims = radiance.dims  # typically ('downtrack', 'crosstrack', 'bands')
            radiance = radiance.isel({dims[0]: slice_y, dims[1]: slice_x})

            if isinstance(self.band_selection, slice):
                radiance = radiance.isel({dims[2]: self.band_selection})
                data = radiance.values
            else:
                # Fancy indexing (list / array of indices) — push as far as we can
                # into the read (spatial), then numpy-slice the band axis.
                data = radiance.values[..., self.band_selection]

        # transpose to (C, H, W)
        if transpose and (len(data.shape) == 3):
            data = np.transpose(data, axes=(2, 0, 1))

        return data

    def clear_radiance_cache(self) -> None:
        """Drop the cached radiance window if present.

        After this call, the next ``load_raw()`` will re-read from disk. The
        ``_cache`` dict object itself is not replaced — clones built via
        ``__copy__`` / ``read_from_bands`` / ``to_crs`` / ``read_from_window``
        share the same dict by reference, so clearing through any clone is
        visible to all of them. Intended to be called from ``EmitProcessor.process``
        after all per-scene products are computed, to release the ~1.5 GB
        radiance array before the next scene is processed.
        """
        self._cache.pop(self._CACHE_KEY_RADIANCE, None)


    def georreference(self, data:np.array, 
                      fill_value_default:Optional[Union[int,float]]=None) -> GeoTensor:
        """
        Georreference an image in sensor coordinates to coordinates of the current 
        georreferenced object. If you do some processing with the raw data, you can 
        georreference the raw output with this function.

        Args:
            data (np.array): raw data (C, H, W) or (H, W). 

        Returns:
            GeoTensor: georreferenced version of data (C, H', W') or (H', W')
        
        Example:
            >>> emit_image = EMITImage("path/to/emit_image.nc")
            >>> emit_image_rgb = emit_image.read_from_bands([35, 23, 11])
            >>> data_rgb = emit_image_rgb.load_raw() # (3, H, W)
            >>> data_rgb_ortho = emit_image.georreference(data_rgb) # (3, H', W')
        """
        return georreference(self.glt_relative, data, self.valid_glt, 
                             fill_value_default=fill_value_default)

        
    @property
    def values(self) -> np.array:
        # return np.zeros(self.shape, dtype=self.dtype)
        raise self.load(boundless=True).values
    
    def __repr__(self)->str:
        return f""" 
         File: {self.filename}
         Transform: {self.transform}
         Shape: {self.shape}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         units: {self.units}
        """
