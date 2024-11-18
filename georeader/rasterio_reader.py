import rasterio
import rasterio.windows
import numpy as np
from typing import Tuple, Dict, List, Optional, Union, Any
import warnings
import numbers
from georeader import geotensor
from collections.abc import Iterable
from georeader import window_utils
from georeader.window_utils import window_bounds, get_slice_pad
from shapely.geometry import Polygon
from georeader.abstract_reader import same_extent, GeoData
from georeader.read import WEB_MERCATOR_CRS, SIZE_DEFAULT, window_from_tile, read_from_tile
from numpy.typing import NDArray

# https://developmentseed.org/titiler/advanced/performance_tuning/#aws-configuration
RIO_ENV_OPTIONS_DEFAULT = dict(
    GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_CACHEMAX=2_000_000_000, # GDAL raster block cache size. If its value is small (less than 100000), 
    # it is assumed to be measured in megabytes, otherwise in bytes. https://trac.osgeo.org/gdal/wiki/ConfigOptions#GDAL_CACHEMAX
    GDAL_HTTP_MULTIPLEX="YES"
)

# CPL_VSIL_CURL_NON_CACHED configuration option can be set to values like 
# /vsicurl/http://example.com/foo.tif:/vsicurl/http://example.com/some_directory, so that at file handle closing, 
# all cached content related to the mentioned file(s) is no longer cached.
# https://github.com/rasterio/rasterio/issues/1877
# VSICurlClearCache()
# https://github.com/rasterio/rasterio/blob/main/rasterio/_path.py

def _vsi_path(path:str)->str:
    """
    Function to convert a path to a VSI path. We use this function to try re-reading the image 
    disabling the VSI cache. This fixes the error when the remote file is modified from another
    program.

    Args:
        - path: path to convert to VSI path
    
    Returns:
        VSI path
    """
    if not "://" in path:
        return path
    
    protocol, remainder_path = path.split("://")
    
    if path.startswith("http"):
        return f"/vsicurl/{path}"
    elif protocol in ["s3", "gs", "az", "oss"]:
        return f"/vsi{protocol}/{remainder_path}"
    else:
        warnings.warn(f"Protocol {protocol} not recognized. Returning the original path")
        return path


class RasterioReader:
    """
    Class to read a raster or a set of rasters files (``paths``). If the path is a single file it will return a 3D np.ndarray 
    with shape (C, H, W). If `paths` is a list, the `read` method will return a 4D np.ndarray with shape (len(paths), C, H, W)

    It checks that all rasters have same CRS, transform and shape. The `read` method will open the file every time it
    is called to work in parallel processing scenario.

    Parameters
    -------------------

    - paths : `Union[List[str], str]`
        Single path or list of paths of the rasters to read.
    - allow_different_shape : `bool`
        If True, will allow different shapes to be read (still checks that all rasters have same CRS,
        transform and number of bands).
    - window_focus : `Optional[rasterio.windows.Window]`
        Window to read from. If provided, all windows in read call will be relative to this window.
    - fill_value_default : `Optional[Union[int, float]]`
        Value to fill when boundless read. It defaults to nodata if it is not None, otherwise it will be
        set to zero.
    - stack : `bool`
        If `True`, returns 4D tensors; otherwise, it returns 3D tensors concatenated over the first dim. If 
        paths is string this argument is ignored and will be set to False (3D tensor).
    - indexes : `Optional[List[int]]`
        If not None, it will read from each raster only the specified bands. This argument is 1-based as in rasterio.
    - overview_level : `Optional[int]`
        If not None, it will read from the corresponding pyramid level. This argument is 0-based as in rasterio
        (None -> default resolution and 0 is the first overview).
    - check : `bool`
        Check all paths are OK.
    - rio_env_options : `Optional[Dict[str, str]]`
        GDAL options for reading. Defaults to: `RIO_ENV_OPTIONS_DEFAULT`. If you read rasters that might change
        from a remote source, you might want to set `read_with_CPL_VSIL_CURL_NON_CACHED` to True.

    Attributes
    -------------------

    - crs : `rasterio.crs.CRS`
        Coordinate reference system.
    - transform : `rasterio.Affine`
        Transform of the rasters. If `window_focus` is provided, this transform will be relative to the window.
    - dtype : `str`
        Type of the input.
    - count : `int`
        Number of bands of the rasters.
    - nodata : `Optional[Union[int, float]]`
        Nodata value of the first raster in paths.
    - fill_value_default : `Union[int, float]`
        Value to fill when boundless read. Defaults to nodata.
    - res : `Tuple[float, float]`
        Resolution of the rasters.
    - width : `int`
        Width of the rasters. If `window_focus` is not None, this will be the width of the window.
    - height : `int`
        Height of the rasters. If `window_focus` is not None, this will be the height of the window.
    - bounds : `Tuple[float, float, float, float]`
        Bounds of the rasters. If `window_focus` is provided, these bounds will be relative to the window.
    - dims : `List[str]`
        Name of the dims (to make it compatible with xr.DataArray functions).
    - attrs : `Dict[str, Any]`
        Dictionary to store extra attributes.
    """
    def __init__(self, paths:Union[List[str], str], allow_different_shape:bool=False,
                 window_focus:Optional[rasterio.windows.Window]=None,
                 fill_value_default:Optional[Union[int, float]]=None,
                 stack:bool=True, indexes:Optional[List[int]]=None,
                 overview_level:Optional[int]=None, check:bool=True,
                 rio_env_options:Optional[Dict[str, str]]=None):

        # Syntactic sugar
        if isinstance(paths, str):
            paths = [paths]
            stack = False

        if rio_env_options is None:
            self.rio_env_options = RIO_ENV_OPTIONS_DEFAULT
        else:
            self.rio_env_options = rio_env_options

        self.paths = paths

        self.stack = stack

        # TODO keep just a global nodata of size (T,C,) and fill with these values?
        self.fill_value_default = fill_value_default
        self.overview_level = overview_level
        with rasterio.Env(**self._get_rio_options_path(paths[0])):
            with rasterio.open(paths[0], "r", overview_level=overview_level) as src:
                self.real_transform = src.transform
                self.crs = src.crs
                self.dtype = src.profile["dtype"]
                self.real_count = src.count
                self.real_indexes = list(range(1, self.real_count + 1))
                if self.stack:
                    self.real_shape = (len(self.paths), src.count,) + src.shape
                else:
                    self.real_shape = (len(self.paths) * self.real_count, ) + src.shape

                self.real_width = src.width
                self.real_height = src.height

                self.nodata = src.nodata
                if self.fill_value_default is None:
                    self.fill_value_default = self.nodata if (self.nodata is not None) else 0

                self.res = src.res

        # if (abs(self.real_transform.b) > 1e-6) or (abs(self.real_transform.d) > 1e-6):
        #     warnings.warn(f"transform of {self.paths[0]} is not rectilinear {self.real_transform}. "
        #                   f"The vast majority of the code expect rectilinear transforms. This transform "
        #                   f"could cause unexpected behaviours")

        self.attrs = {}
        self.window_focus = rasterio.windows.Window(row_off=0, col_off=0,
                                                    width=self.real_width, height=self.real_height)
        self.real_window = rasterio.windows.Window(row_off=0, col_off=0,
                                                   width=self.real_width, height=self.real_height)
        self.set_indexes(self.real_indexes, relative=False)
        self.set_window(window_focus, relative=False)

        self.allow_different_shape = allow_different_shape

        if self.stack:
            self.dims = ["time", "band", "y", "x"]
        else:
            self.dims = ["band", "y", "x"]

        self._coords = None

        # Assert all paths have same tranform and crs
        #  (checking width and height will not be needed since we're reading with boundless option but I don't see the point to ignore it)
        if check and len(self.paths) > 1:
            for p in self.paths:
                with rasterio.Env(**self._get_rio_options_path(p)):
                    with rasterio.open(p, "r", overview_level=overview_level) as src:
                        if not src.transform.almost_equals(self.real_transform, 1e-6):
                            raise ValueError(f"Different transform in {self.paths[0]} and {p}: {self.real_transform} {src.transform}")
                        if not str(src.crs).lower() == str(self.crs).lower():
                            raise ValueError(f"Different CRS in {self.paths[0]} and {p}: {self.crs} {src.crs}")
                        if self.real_count != src.count:
                            raise ValueError(f"Different number of bands in {self.paths[0]} and {p} {self.real_count} {src.count}")
                        if src.nodata != self.nodata:
                            warnings.warn(
                                f"Different nodata in {self.paths[0]} and {p}: {self.nodata} {src.nodata}. This might lead to unexpected behaviour")

                        if (self.real_width != src.width) or (self.real_height != src.height):
                            if allow_different_shape:
                                warnings.warn(f"Different shape in {self.paths[0]} and {p}: ({self.real_height}, {self.real_width}) ({src.height}, {src.width}) Might lead to unexpected behaviour")
                            else:
                                raise ValueError(f"Different shape in {self.paths[0]} and {p}: ({self.real_height}, {self.real_width}) ({src.height}, {src.width})")

        self.check = check
        if indexes is not None:
            self.set_indexes(indexes)

    def set_indexes(self, indexes:List[int], relative:bool=True)-> None:
        """
        Set the channels to read. This is useful for processing only some channels of the raster. The indexes
        passed will be relative to self.indexes
        Args:
            indexes: 1-based array to mantain rasterio convention
            relative: True means the indexes arg will be treated ad relative to the current self.indexes. If false
                     it sets self.indexes = indexes (and update the count attribute)
        Examples:
            >>> r = RasterioReader("path/to/raster.tif", indexes=[2,3,4]) # Read all bands except the first one.
            >>> r.set_indexes([2,3], relative=True) # will read bands 2 and 3 of the original raster
        """
        if relative:
            new_indexes = [self.indexes[idx - 1] for idx in indexes]
        else:
            new_indexes = indexes

        # Check if indexes are valid
        assert all((s >= 1) and (s <= self.real_count) for s in new_indexes), \
               f"Indexes (1-based) out of real bounds current: {self.indexes} asked: {new_indexes} number of bands:{self.real_count}"
        
        self.indexes = new_indexes

        assert all((s >= 1) and (s <= self.real_count) for s in
                   self.indexes), f"Indexes out of real bounds current: {self.indexes} asked: {indexes} number of bands:{self.real_count}"

        self.count = len(self.indexes)

    def set_indexes_by_name(self, names:List[str]) -> None:
        """
        Function to set the indexes by the name of the band which is stored in the descriptions attribute

        Args:
            names: List of band names to read
        
        Examples:
            >>> r = RasterioReader("path/to/raster.tif") # Read all bands except the first one.
            >>> # Assume r.descriptions = ["B1", "B2", "B3"]
            >>> r.set_indexes_by_name(["B2", "B3"])

        """
        descriptions = self.descriptions
        if len(self.paths) == 1:
            if self.stack:
                descriptions = descriptions[0]
        else:
            assert all(d == descriptions[0] for d in descriptions), "There are tiffs with different names"
            descriptions = descriptions[0]

        bands = [descriptions.index(b) + 1 for b in names]
        self.set_indexes(bands, relative=False)

    @property
    def shape(self):
        if self.stack:
            return len(self.paths), self.count, self.height, self.width
        return len(self.paths) * self.count, self.height, self.width
    
    def same_extent(self, other:Union[GeoData,'RasterioReader'], precision:float=1e-3) -> bool:
        """
        Check if two GeoData objects have the same extent

        Args:
            other: GeoData object to compare
            precision: precision to compare the bounds

        Returns:
            True if both objects have the same extent

        """
        return same_extent(self, other, precision=precision)

    def set_window(self, window_focus:Optional[rasterio.windows.Window] = None,
                   relative:bool = True, boundless:bool=True)->None:
        """
        Set window to read. This is useful for processing only some part of the raster. The windows passed as
         arguments in the read calls will be relative to this window.

        Args:
            window_focus: rasterio window. If None will be set to the full raster tile
            relative: provided window is relative to current self.window_focus
            boundless: if boundless is false the windows that do not overlap the total raster will be
                intersected.
        
        Examples:
            >>> # Read the first 1000x1000 pixels of the raster
            >>> r = RasterioReader("path/to/raster.tif")
            >>> r.set_window(rasterio.windows.Window(col_off=0, row_off=0, width=1000, height=1000))
            >>> r.load() #  returns GeoTensor with shape (1, 1, 1000, 1000)

        """
        if window_focus is None:
            self.window_focus = rasterio.windows.Window(row_off=0, col_off=0,
                                                        width=self.real_width, height=self.real_height)
        elif relative:
            self.window_focus = rasterio.windows.Window(col_off=window_focus.col_off + self.window_focus.col_off,
                                                        row_off=window_focus.row_off + self.window_focus.row_off,
                                                        height=window_focus.height, width=window_focus.width)
        else:
            self.window_focus = window_focus

        if not boundless:
            self.window_focus = rasterio.windows.intersection(self.real_window, self.window_focus)

        self.height = self.window_focus.height
        self.width = self.window_focus.width

        self.bounds = window_bounds(self.window_focus, self.real_transform)
        self.transform = rasterio.windows.transform(self.window_focus, self.real_transform)

    def tags(self) -> Union[List[Dict[str, str]], Dict[str, str]]:
        """
        Returns a list with the tags for each tiff file.
        If stack and len(self.paths) == 1 it returns just the dictionary of the tags

        """
        tags = []
        for i, p in enumerate(self.paths):
            with rasterio.Env(**self._get_rio_options_path(p)):
                with rasterio.open(p, mode="r") as src:
                    tags.append(src.tags())

        if (not self.stack) and (len(tags) == 1):
            return tags[0]

        return tags

    def _get_rio_options_path(self, path:str) -> Dict[str, str]:
        options = self.rio_env_options
        if "read_with_CPL_VSIL_CURL_NON_CACHED" in options:
            options = options.copy()
            if options["read_with_CPL_VSIL_CURL_NON_CACHED"]:
                options["CPL_VSIL_CURL_NON_CACHED"] = _vsi_path(path)
            del options["read_with_CPL_VSIL_CURL_NON_CACHED"]
        return options
    
    # This function does not work for e.g. returning the descriptions of the bands
    # @contextmanager
    # def _rio_open(self, path:str, mode:str="r", overview_level:Optional[int]=None) -> rasterio.DatasetReader:
    #     with rasterio.Env(**self._get_rio_options_path(path)):
    #         with rasterio.open(path, mode=mode, overview_level=overview_level) as src:
    #             yield src

    @property
    def descriptions(self) -> Union[List[List[str]], List[str]]:
        """
        Returns a list with the descriptions for each tiff file. (This is usually the name of the bands of the raster)


        Returns:
            If `stack` it returns the flattened list of descriptions for each tiff file. If not `stack` it returns a list of lists.
        
        Examples:
            >>> r = RasterioReader("path/to/raster.tif") # Raster with band names B1, B2, B3
            >>> r.descriptions # returns ["B1", "B2", "B3"]
        """
        descriptions_all = []
        for i, p in enumerate(self.paths):
            with rasterio.Env(**self._get_rio_options_path(p)):
                with rasterio.open(p) as src:
                    desc = src.descriptions

            if self.stack:
                descriptions_all.append([desc[i-1] for i in self.indexes])
            else:
                descriptions_all.extend([desc[i-1] for i in self.indexes])

        return descriptions_all

    def read_from_window(self, window:rasterio.windows.Window, boundless:bool=True) -> '__class__':
        """
        Returns a new reader with window focus the window `window` relative to `self.window_focus`
        
        Args:
            window: rasterio.window.Window to read
            boundless: if boundless is False if the window do not overlap the total raster  it will be
                intersected.

        Raises:
            rasterio.windows.WindowError: if bounless is False and window does not intersects self.window_focus

        Returns:
            New reader object
        """
        rst_reader = RasterioReader(list(self.paths),
                                    allow_different_shape=self.allow_different_shape,
                                    window_focus=self.window_focus, fill_value_default=self.fill_value_default,
                                    stack=self.stack, overview_level=self.overview_level,
                                    check=False)

        rst_reader.set_window(window, relative=True, boundless=boundless)
        rst_reader.set_indexes(self.indexes, relative=False)
        return rst_reader

    def isel(self, sel: Dict[str, Union[slice, List[int], int]], boundless:bool=True) -> '__class__':
        """
        Creates a copy of the current RasterioReader slicing the data with a given selection dict. This function
        mimics ``xr.DataArray.isel()`` method.

        Args:
            sel: Dict of slices to slice the current reader
            boundless: If `True` slices in "x" and "y" are boundless (i.e. negative means negative indexes rather than
                values from the other side of the array as in numpy).

        Returns:
            Copy of the current reader
        
        Examples:
            >>> r = RasterioReader(["path/to/raster1.tif", "path/to/raster2.tif"])
            >>> r.isel({"time": 0, "band": [0]}) # returns a reader with the first band of the first raster
            >>> r.isel({"time": slice(0, 1), "band": [0]}) # returns a reader with the first band of the first raster and second raster
            >>> r.isel({"x": slice(4000, 5000), "band": [0, 1]}) # returns a reader slicing the x axis from 4000 to 5000 and the first two bands
        """
        for k in sel:
            if k not in self.dims:
                raise NotImplementedError(f"Axis {k} not in dims: {self.dims}")

        stack = self.stack
        if "time" in sel: # time allowed only if self.stack (would have raised error above)
            if isinstance(sel["time"], Iterable):
                paths = [self.paths[i] for i in sel["time"]]
            elif isinstance(sel["time"], slice):
                paths = self.paths[sel["time"]]
            elif isinstance(sel["time"], numbers.Number):
                paths = [self.paths[sel["time"]]]
                stack = False
            else:
                raise NotImplementedError(f"Don't know how to slice {sel['time']} in dim time")
        else:
            paths = self.paths

        # Band slicing
        if "band" in sel:
            if not self.stack:
                # if `True` returns 4D tensors otherwise it returns 3D tensors concatenated over the first dim
                assert (len(self.paths) == 1) or (len(self.indexes) == 1), f"Dont know how to slice {self.paths} and {self.indexes}"

            if self.stack or (len(self.paths) == 1):
                if isinstance(sel["band"], Iterable):
                    indexes = [self.indexes[i] for i in sel["band"]] # indexes relative to current indexes
                elif isinstance(sel["band"], slice):
                    indexes = self.indexes[sel["band"]]
                elif isinstance(sel["band"], numbers.Number):
                    raise NotImplementedError(f"Slicing band with a single number is not supported (use a list)")
                else:
                    raise NotImplementedError(f"Don't know how to slice {sel['band']} in dim band")
            else:
                indexes = self.indexes
                # len(indexes) == 1 and not self.stack in this case band slicing correspond to paths
                if isinstance(sel["band"], Iterable):
                    paths = [self.paths[i] for i in sel["band"]]
                elif isinstance(sel["band"], slice):
                    paths = self.paths[sel["band"]]
                elif isinstance(sel["band"], numbers.Number):
                    paths = [self.paths[sel["band"]]]
                else:
                    raise NotImplementedError(f"Don't know how to slice {sel['time']} in dim time")
        else:
            indexes = self.indexes

        # Spatial slicing
        slice_ = []
        spatial_shape = (self.height, self.width)
        for _i, spatial_name in enumerate(["y", "x"]):
            if spatial_name in sel:
                if not isinstance(sel[spatial_name], slice):
                    raise NotImplementedError(f"spatial dimension {spatial_name} only accept slice objects")
                slice_.append(sel[spatial_name])
            else:
                slice_.append(slice(0, spatial_shape[_i]))

        rst_reader = RasterioReader(paths, allow_different_shape=self.allow_different_shape,
                                    window_focus=self.window_focus, fill_value_default=self.fill_value_default,
                                    stack=stack, overview_level=self.overview_level,
                                    check=False)
        window_current = rasterio.windows.Window.from_slices(*slice_, boundless=boundless,
                                                             width=self.width, height=self.height)

        # Set bands to read
        rst_reader.set_indexes(indexes=indexes, relative=False)

        # set window_current relative to self.window_focus
        rst_reader.set_window(window_current, relative=True)

        return rst_reader

    def __copy__(self) -> '__class__':
        return RasterioReader(self.paths, allow_different_shape=self.allow_different_shape,
                              window_focus=self.window_focus, 
                              fill_value_default=self.fill_value_default,
                              stack=self.stack, overview_level=self.overview_level,
                              check=False)
    
    def overviews(self, index:int=1, time_index:int=0) -> List[int]:
        """
        Returns a list of the available overview levels for the current raster.
        """
        with rasterio.Env(**self._get_rio_options_path(self.paths[time_index])):
            with rasterio.open(self.paths[time_index]) as src:
                return src.overviews(index)
    
    def reader_overview(self, overview_level:int) -> '__class__':
        if overview_level < 0:
            overview_level = len(self.overviews()) + overview_level
        
        return RasterioReader(self.paths, allow_different_shape=self.allow_different_shape,
                              window_focus=self.window_focus, 
                              fill_value_default=self.fill_value_default,
                              stack=self.stack, overview_level=overview_level,
                              check=False)
    
    def block_windows(self, bidx:int=1, time_idx:int=0) -> List[Tuple[int, rasterio.windows.Window]]:
        """
        return the block windows within the object
        (see https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader.block_windows)

        Args:
            bidx: band index to read (1-based)
            time_idx: time index to read (0-based)

        Returns:
            list of (block_idx, window)

        """
        with rasterio.Env(**self._get_rio_options_path(self.paths[time_idx])):
            with rasterio.open(self.paths[time_idx]) as src:
                windows_return = [(block_idx, rasterio.windows.intersection(window, self.window_focus)) for block_idx, window in src.block_windows(bidx) if rasterio.windows.intersect(self.window_focus, window)]

        return windows_return

    def copy(self) -> '__class__':
        return self.__copy__()

    def load(self, boundless:bool=True) -> geotensor.GeoTensor:
        """
        Load all raster in memory in an GeoTensor object

        Returns:
            GeoTensor (wrapper of numpy array with spatial information)

        """
        np_data = self.read(boundless=boundless)
        if boundless:
            transform = self.transform
        else:
            # update transform, shape and coords
            window = self.window_focus
            start_col = max(window.col_off, 0)
            end_col = min(window.col_off + window.width, self.real_width)
            start_row = max(window.row_off, 0)
            end_row = min(window.row_off + window.height, self.real_height)
            spatial_shape = (end_row - start_row, end_col - start_col)
            assert np_data.shape[-2:] == spatial_shape, f"Different shapes {np_data.shape[-2:]} {spatial_shape}"

            window_real = rasterio.windows.Window(row_off=start_row, col_off=start_col,
                                                  width=spatial_shape[1], height=spatial_shape[0])
            transform = rasterio.windows.transform(window_real, self.real_transform)

        return geotensor.GeoTensor(np_data, transform=transform, crs=self.crs, fill_value_default=self.fill_value_default)

    @property
    def values(self) -> np.ndarray:
        """
        This property is added to be consistent with xr.DataArray. It reads the whole raster in memory and returns it

        Returns:
            np.ndarray raster loaded in memory
        """
        return self.read()

    def footprint(self, crs:Optional[str]=None) -> Polygon:
        pol = window_utils.window_polygon(rasterio.windows.Window(row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]),
                                          self.transform)
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)
    
    def meshgrid(self, dst_crs:Optional[Any]=None) -> Tuple[NDArray, NDArray]:
        from georeader import griddata
        return griddata.meshgrid(self.transform, self.width, self.height, source_crs=self.crs, dst_crs=dst_crs)
    
    def __repr__(self)->str:
        return f""" 
         Paths: {self.paths}
         Transform: {self.transform}
         Shape: {self.shape}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         nodata: {self.nodata}
         fill_value_default: {self.fill_value_default}
        """

    def read(self, **kwargs) -> np.ndarray:
        """
        Read data from the list of rasters. It reads with boundless=True by default and
        fill_value=self.fill_value_default by default.

        This function is process safe (opens and closes the rasterio object every time is called).

        For arguments see: https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader.read

        Returns:
            if self.stack:
                4D np.ndarray with shape (len(paths), C, H, W)
            if self.stack is False:
                3D np.ndarray with shape (len(paths)*C, H, W)
        """

        if ("window" in kwargs) and kwargs["window"] is not None:
            window_read = kwargs["window"]
            if isinstance(window_read, tuple):
                window_read = rasterio.windows.Window.from_slices(*window_read,
                                                                  boundless=kwargs.get("boundless", True))

            # Windows are relative to the windows_focus window.
            window = rasterio.windows.Window(col_off=window_read.col_off + self.window_focus.col_off,
                                             row_off=window_read.row_off + self.window_focus.row_off,
                                             height=window_read.height, width=window_read.width)
        else:
            window = self.window_focus

        kwargs["window"] = window

        if "boundless" not in kwargs:
            kwargs["boundless"] = True

        if not rasterio.windows.intersect([self.real_window, window]) and not kwargs["boundless"]:
            return None

        if not kwargs["boundless"]:
            window = window.intersection(self.real_window)

        if "fill_value" not in kwargs:
            kwargs["fill_value"] = self.fill_value_default

        if  kwargs.get("indexes", None) is not None:
            # Indexes are relative to the self.indexes window.
            indexes = kwargs["indexes"]
            if isinstance(indexes, numbers.Number):
                n_bands_read = 1
                kwargs["indexes"] = [self.indexes[kwargs["indexes"] - 1]]
                flat_channels = True
            else:
                n_bands_read = len(indexes)
                kwargs["indexes"] = [self.indexes[i - 1] for i in kwargs["indexes"]]
                flat_channels = False
        else:
            kwargs["indexes"] = self.indexes
            n_bands_read = self.count
            flat_channels = False

        if kwargs.get("out_shape", None) is not None:
            if len(kwargs["out_shape"]) == 2:
                kwargs["out_shape"] = (n_bands_read, ) + kwargs["out_shape"]
            elif len(kwargs["out_shape"]) == 3:
                assert kwargs["out_shape"][0] == n_bands_read, f"Expected to read {n_bands_read} but found out_shape: {kwargs['out_shape']}"
            else:
                raise NotImplementedError(f"Expected out_shape of len 2 or 3 found out_shape: {kwargs['out_shape']}")
            spatial_shape = kwargs["out_shape"][1:]
        else:
            spatial_shape = (window.height, window.width)

        shape = (len(self.paths), n_bands_read) + spatial_shape

        obj_out = np.full(shape, kwargs["fill_value"], dtype=self.dtype)
        if rasterio.windows.intersect([self.real_window, window]):
            pad = None
            if kwargs["boundless"]:
                slice_, pad = get_slice_pad(self.real_window, window)
                need_pad = any(x != 0 for x in pad["x"] + pad["y"])

                #  read and pad instead of using boundless attribute when transform is not rectilinear (otherwise rasterio fails!)
                if (abs(self.real_transform.b) > 1e-6) or (abs(self.real_transform.d) > 1e-6):
                    if need_pad:
                        assert kwargs.get("out_shape", None) is None, "out_shape not compatible with boundless and non rectilinear transform!"
                        kwargs["window"] = rasterio.windows.Window.from_slices(slice_["y"], slice_["x"])
                        kwargs["boundless"] = False
                    else:
                        kwargs["boundless"] = False
                else:
                    #  if transform is rectilinear read boundless if needed
                    kwargs["boundless"] = need_pad
                    pad = None

            for i, p in enumerate(self.paths):
                with rasterio.Env(**self._get_rio_options_path(p)):
                    with rasterio.open(p, "r", overview_level=self.overview_level) as src:
                    # rasterio.read API: https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader.read
                        read_data = src.read(**kwargs)

                        # Add pad when reading
                        if pad is not None and need_pad:
                            slice_y = slice(pad["y"][0], -pad["y"][1] if pad["y"][1] !=0 else None)
                            slice_x = slice(pad["x"][0], -pad["x"][1] if pad["x"][1] !=0 else None)
                            obj_out[i, :, slice_y, slice_x] = read_data
                        else:
                            obj_out[i] = read_data
                        # pad_list_np = _get_pad_list(pad)
                    #
                    # read_data = np.pad(read_data, tuple(pad_list_np), mode="constant",
                    #                    constant_values=self.fill_value_default)



        if flat_channels:
            obj_out = obj_out[:, 0]

        if not self.stack:
            if obj_out.shape[0] == 1:
                obj_out = obj_out[0]
            else:
                obj_out = np.concatenate([obj_out[i] for i in range(obj_out.shape[0])],
                                         axis=0)

        return obj_out
    
    def read_from_tile(self, x:int, y:int, z:int, 
                       out_shape:Tuple[int,int]=(SIZE_DEFAULT, SIZE_DEFAULT),
                       dst_crs:Optional[Any]=WEB_MERCATOR_CRS) -> geotensor.GeoTensor:
        """
        Read a web mercator tile from a raster.
        
        Tiles are TMS tiles defined as: (https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames)

        Args:
            x (int): x coordinate of the tile in the TMS system.
            y (int): y coordinate of the tile in the TMS system.
            z (int): z coordinate of the tile in the TMS system.
            out_shape (Tuple[int,int]: size of the tile to read. Defaults to (read.SIZE_DEFAULT, read.SIZE_DEFAULT).
            dst_crs (Optional[Any], optional): CRS of the output tile. Defaults to read.WEB_MERCATOR_CRS.
            
        Returns:
            geotensor.GeoTensor: geotensor with the tile data.
        """
        window = window_from_tile(self, x, y, z)
        window = window_utils.round_outer_window(window)
        data = read_out_shape(self, out_shape=out_shape, window=window)

        if window_utils.compare_crs(self.crs, dst_crs):
            return data
        
        # window = window_utils.pad_window(window, (1, 1))
        # data = read_out_shape(self, out_shape=size_out, window=window)

        return read_from_tile(data, x, y, z, dst_crs=dst_crs, out_shape=out_shape)
        

def _get_pad_list(pad_width:Dict[str,Tuple[int,int]]):
    pad_list_np = [(0, 0)]
    for k in ["y", "x"]:
        if k in pad_width:
            pad_list_np.append(pad_width[k])
        else:
            pad_list_np.append((0, 0))
    return pad_list_np


def read_out_shape(reader:Union[RasterioReader, rasterio.DatasetReader],
                   size_read:Optional[int]=None,
                   indexes:Optional[Union[List[int], int]]=None,
                   window:Optional[rasterio.windows.Window]=None,
                   out_shape:Optional[Tuple[int, int]]=None,
                   fill_value_default:int=0) -> geotensor.GeoTensor:
    """
    Reads data using the `out_shape` param of rasterio. This allows to read from the pyramids if the file is a COG.
    This function returns an xarray with the data with its geographic metadata.

    Args:
        reader: RasterioReader, rasterio.DatasetReader
        size_read: if out_shape is None it uses this to compute the size to read that maintains the aspect ratio
        indexes: 1-based channels to read
        window: window to read
        out_shape: shape of the output to be readed. Conceptually, the function resizes the output to this shape
        fill_value_default: if the object is rasterio.DatasetReader and nodata is None it will use this value for the
            corresponding GeoTensor

    Returns:
        GeoTensor with geo metadata

    """

    if window is None:
        shape = reader.shape[-2:]
    else:
        shape = window.height, window.width

    if out_shape is None:
        assert size_read is not None, f"Both out_shape and size_read are None"
        out_shape = get_out_shape(shape, size_read)
    else:
        assert len(out_shape) == 2, f"Expected 2 dimensions found {out_shape}"

    transform = reader.transform if window is None else rasterio.windows.transform(window, reader.transform)

    if (indexes is not None) and isinstance(indexes, (list, tuple)):
        if len(out_shape) == 2:
            out_shape = (len(indexes),) + out_shape
    
    input_output_factor = (shape[0] / out_shape[-2], shape[1] / out_shape[-1])    
    transform = transform * rasterio.Affine.scale(input_output_factor[1], input_output_factor[0])

    output = reader.read(indexes=indexes, out_shape=out_shape, window=window)

    return geotensor.GeoTensor(output, transform=transform,
                               crs=reader.crs, fill_value_default=getattr(reader, "fill_value_default",
                                                                          reader.nodata if reader.nodata else fill_value_default))




def get_out_shape(shape:Tuple[int, int], size_read:int) -> Tuple[int, int]:
    if (size_read >= shape[0]) and (size_read >= shape[1]):
        out_shape = None
    elif shape[0] > shape[1]:
        out_shape = (size_read, int(round(shape[1] / shape[0] * size_read)))
    else:
        out_shape = (int(round(shape[0] / shape[1] * size_read)), size_read)
    return out_shape


def needs_boundless(window_data:rasterio.windows.Window,
                    window_read:rasterio.windows.Window) -> bool:
    try:
        slice_, pad = get_slice_pad(window_data, window_read)
        return any(x != 0 for x in pad["x"]+pad["y"])

    except rasterio.windows.WindowError:
        return True
