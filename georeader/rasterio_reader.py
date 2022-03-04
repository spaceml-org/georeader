import rasterio
import rasterio.windows
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
import warnings
import numbers
from georeader import geotensor
from collections.abc import Iterable
from georeader.window_utils import normalize_bounds


class RasterioReader:
    """
    Class to read a set of rasters files (``paths``). the `read` method will return a 4D np.ndarray with
    shape (len(paths), C, H, W).

    It checks that all rasters have same CRS, transform and  shape. the `read` method will open the file every time it
    is called to work in parallel processing scenario.

    Parameters
    -------------------
    paths: single path or list of paths of the rasters to read
    allow_different_shape: if True will allow different shapes to be read (still checks that all rasters have same crs,
        transform and number of bands)
    window_focus: window to read from. If provided all windows in read call will be relative to this window.
    fill_value_default: value to fill when boundless read
    stack: if `True` returns 4D tensors otherwise it returns 3D tensors concatenated over the first dim

    Attributes
    -------------------
    crs : Coordinate reference system
    transform: rasterio.Affine transform of the rasters. If window_focus is provided this transform will be
        relative to the window.
    dtype: type of the input.
    count: number of bands of the rasters.
    nodata: rasterio nodata of the first raster in paths
    resolution: of the rasters
    width: width of the rasters. If window_focus is not None this will be the width of the window
    height: height of the rasters. If window_focus is not None this will be the height of the window
    bounds: bounds of the rasters. If window_focus is provided these bounds will be relative to the window.
    dims: name of the dims (to make it compatible with xr.DataArray functions)
    attrs: Dict to store extra attributes.

    """
    def __init__(self, paths:Union[List[str], str], allow_different_shape:bool=False,
                 window_focus:Optional[rasterio.windows.Window]=None, fill_value_default:Union[int, float]=0,
                 stack:bool=True):

        # Syntactic sugar
        if isinstance(paths, str):
            paths = [paths]
            stack = False

        self.paths = paths

        self.stack = stack

        # TODO keep just a global nodata of size (T,C,) and fill with these values?
        self.fill_value_default = fill_value_default

        with rasterio.open(self.paths[0], "r") as src:
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
            self.real_bounds = src.bounds
            self.resolution = src.res

        # TODO if transform is not rectilinear with b == 0 and d==0 reading boundless does not work
        if not self.real_transform.is_rectilinear:
            warnings.warn(f"transform of {self.paths[0]} is not rectilinear {self.real_transform}. "
                          f"The vast majority of the code expect rectilinear transforms. This transform "
                          f"could cause unexpected behaviours")

        self.attrs = {}
        self.window_focus = rasterio.windows.Window(row_off=0, col_off=0,
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
        for p in self.paths:
            with rasterio.open(p, "r") as src:
                if not src.transform == self.real_transform:
                    raise ValueError(f"Different transform in {self.paths[0]} and {p}: {self.real_transform} {src.transform}")
                if not str(src.crs).lower() == str(self.crs).lower():
                    raise ValueError(f"Different CRS in {self.paths[0]} and {p}: {self.crs} {src.crs}")
                if self.real_count != src.count:
                    raise ValueError(f"Different number of bands in {self.paths[0]} and {p} {self.real_count} {src.count}")

                if (self.real_width != src.width) or (self.real_height != src.height):
                    if allow_different_shape:
                        warnings.warn(f"Different shape in {self.paths[0]} and {p}: ({self.real_height}, {self.real_width}) ({src.height}, {src.width}) Might lead to unexpected behaviour")
                    else:
                        raise ValueError(f"Different shape in {self.paths[0]} and {p}: ({self.real_height}, {self.real_width}) ({src.height}, {src.width})")

    def set_indexes(self, indexes:List[int], relative:bool=True)-> None:
        """
        Set the channels to read. This is useful for processing only some channels of the raster. The indexes
        passed will be relative to self.indexes
        Args:
            indexes: 1-based array to mantain rasterio convention
            relative: True means the indexes arg will be treated ad relative to the current self.indexes. If false
                     it sets self.indexes = indexes (and update the count attribute)

        """
        if relative:
            self.indexes = [self.indexes[idx - 1] for idx in indexes]
        else:
            self.indexes = indexes

        assert all((s >= 1) and (s <= self.real_count) for s in
                   self.indexes), f"Indexes out of real bounds current: {self.indexes} asked: {indexes} number of bands:{self.real_count}"

        self.count = len(self.indexes)
    
    @property
    def shape(self):
        if self.stack:
            return len(self.paths), self.count, self.height, self.width
        return len(self.paths) * self.count, self.height, self.width

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
            window_real = rasterio.windows.Window(row_off=0, col_off=0,
                                                  width=self.real_width, height=self.real_height)
            self.window_focus = rasterio.windows.intersection(window_real, self.window_focus)

        self.height = self.window_focus.height
        self.width = self.window_focus.width

        self.bounds = normalize_bounds(rasterio.windows.bounds(self.window_focus, self.real_transform))
        self.transform = rasterio.windows.transform(self.window_focus, self.real_transform)

    def tags(self) -> Union[List[Dict[str, str]], Dict[str, str]]:
        """
        Returns a list with the tags for each tiff file.
        If stack and len(self.paths) == 1 it returns just the dictionary of the tags

        """
        tags = []
        for i, p in enumerate(self.paths):
            with rasterio.open(p, "r") as src:
                tags.append(src.tags())

        if (not self.stack) and (len(tags) == 1):
            return tags[0]

        return tags

    def descriptions(self) -> Union[List[List[str]], List[str]]:
        """
        Returns a list with the descriptions for each tiff file. (This is usually the name of the files)

        If stack and len(self.paths) == 1 it returns just the List with the descriptions
        """
        descriptions_all = []
        for i, p in enumerate(self.paths):
            with rasterio.open(p, "r") as src:
                desc = src.descriptions
            descriptions_all.append([desc[i-1] for i in self.indexes])

        if (not self.stack) and (len(descriptions_all) == 1):
            return descriptions_all[0]

        return descriptions_all

    def read_from_window(self, window:rasterio.windows.Window, boundless:bool=True) -> '__class__':
        """
        Returns a new reader with window focus the window `window` relative to `self.window_focus`
        Args:
            window:
            boundless:

        Returns:
            New reader object

        Raises:
            rasterio.windows.WindowError if bounless is False and window does not intersects self.window_focus

        """
        rst_reader = RasterioReader(list(self.paths),
                                    allow_different_shape=self.allow_different_shape,
                                    window_focus=self.window_focus, fill_value_default=self.fill_value_default,
                                    stack=self.stack)

        rst_reader.set_window(window, relative=True, boundless=boundless)
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
        """
        for k in sel:
            if k not in self.dims:
                raise NotImplementedError(f"Axis {k} not in dims: {self.dims}")

        stack = self.stack
        if "time" in sel:
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

        if "band" in sel:
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
                                    stack=stack)
        window_current = rasterio.windows.Window.from_slices(*slice_, boundless=boundless,
                                                             width=self.width, height=self.height)

        # Set bands to read
        rst_reader.set_indexes(indexes=indexes, relative=False)

        # set window_current relative to self.window_focus
        rst_reader.set_window(window_current, relative=True)

        return rst_reader

    def __copy__(self) -> '__class__':
        return RasterioReader(self.paths, allow_different_shape=self.allow_different_shape,
                              window_focus=self.window_focus, fill_value_default=self.fill_value_default,
                              stack=self.stack)

    def copy(self) -> '__class__':
        return self.__copy__()

    def load(self, boundless:bool=True) -> geotensor.GeoTensor:
        """
        Load all raster in memory in an xr.DataArray object

        Returns:
            xr.DataArray with geographic info

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
    
    def __repr__(self)->str:
        return f""" 
         Paths: {self.paths}
         Transform: {self.transform}
         Shape: {self.shape}
         Resolution: {self.resolution}
         Bounds: {self.bounds}
         CRS: {self.crs}
         nodata: {self.nodata}
         fill_value_default: {self.fill_value_default}
        """

    def read(self, **kwargs) -> np.ndarray:
        """
        Read data from the list of rasters. It reads with boundless=True by default and
        fill_value=self.fill_value_default by default.

        This function is process safe (opens the rasterio object every time is called).

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

        if "fill_value" not in kwargs:
            kwargs["fill_value"] = self.fill_value_default

        if ("indexes" in kwargs) and (kwargs["indexes"] is not None):
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

        if ("out_shape" in kwargs) and (kwargs["out_shape"] is not None):
            if len(kwargs["out_shape"]) == 2:
                kwargs["out_shape"] = (n_bands_read, ) + kwargs["out_shape"]
            elif len(kwargs["out_shape"]) == 3:
                assert kwargs["out_shape"][0] == n_bands_read, f"Expected to read {n_bands_read} but found out_shape: {kwargs['out_shape']}"
            else:
                raise NotImplementedError(f"Expected out_shape of len 2 or 3 found out_shape: {kwargs['out_shape']}")
            spatial_shape = kwargs["out_shape"][1:]
        else:
            if kwargs["boundless"]:
                spatial_shape = (window.height, window.width)
            else:
                start_col = max(window.col_off, 0)
                end_col = min(window.col_off+window.width, self.real_width)
                start_row = max(window.row_off, 0)
                end_row = min(window.row_off+window.height, self.real_height)
                spatial_shape = (end_row-start_row, end_col-start_col)

        shape = (len(self.paths), n_bands_read) + spatial_shape

        obj_out = np.ndarray(shape, dtype=self.dtype)

        for i, p in enumerate(self.paths):
            with rasterio.open(p, "r") as src:
                # rasterio.read API: https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.DatasetReader.read
                obj_out[i] = src.read(**kwargs)

        if flat_channels:
            obj_out = obj_out[:, 0]

        if not self.stack:
            if obj_out.shape[0] == 1:
                obj_out = obj_out[0]
            else:
                obj_out = np.concatenate([obj_out[i] for i in range(obj_out.shape[0])],
                                         axis=0)

        return obj_out


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
        xr.DataArray with geo metadata

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
    assert (transform.b < 1e-5) and (transform.d < 1e-5), f"Expected rectilinear transform found {transform}"

    if indexes is None:
        nbands = reader.count
    else:
        nbands = len(indexes)

    if out_shape is not None:
        input_output_factor = (shape[0] / out_shape[0], shape[1] / out_shape[1])
        out_shape = (nbands,) + out_shape
        transform = rasterio.Affine(transform.a * input_output_factor[1], transform.b, transform.c,
                                    transform.d, transform.e * input_output_factor[0], transform.f)


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
