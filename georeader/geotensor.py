import numbers
import warnings
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import rasterio.windows
from numpy.typing import ArrayLike, NDArray
from shapely.geometry import MultiPolygon, Polygon

from georeader import window_utils
from georeader.window_utils import window_bounds

try:
    import torch
    import torch.nn.functional

    Tensor = Union[torch.Tensor, NDArray]
    torch_installed = True
except ImportError:
    Tensor = NDArray
    torch_installed = False

ORDERS = {
    "nearest": 0,
    "bilinear": 1,
    "bicubic": 2,
}

# https://developmentseed.org/titiler/advanced/performance_tuning/#aws-configuration
RIO_ENV_OPTIONS_DEFAULT = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_CACHEMAX=2_000_000_000,  # GDAL raster block cache size. If its value is small (less than 100000),
    # it is assumed to be measured in megabytes, otherwise in bytes. https://trac.osgeo.org/gdal/wiki/ConfigOptions#GDAL_CACHEMAX
    GDAL_HTTP_MULTIPLEX="YES",
)


def _vsi_path(path: str) -> str:
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


def get_rio_options_path(options: dict, path: str) -> Dict[str, str]:
    if "read_with_CPL_VSIL_CURL_NON_CACHED" in options:
        options = options.copy()
        if options["read_with_CPL_VSIL_CURL_NON_CACHED"]:
            options["CPL_VSIL_CURL_NON_CACHED"] = _vsi_path(path)
        del options["read_with_CPL_VSIL_CURL_NON_CACHED"]
    return options


class GeoTensor:
    """
    This class is a wrapper around a numpy or torch tensor with geospatial information.
    It can store 2D, 3D or 4D tensors. The last two dimensions are the spatial dimensions.

    Args:
        values (Tensor): numpy or torch tensor (2D, 3D or 4D).
        transform (rasterio.Affine): affine geospatial transform
        crs (Any): coordinate reference system
        fill_value_default (Optional[Union[int, float]], optional): Value to fill when
            reading out of bounds. Could be None. Defaults to 0.

    Attributes:
        values (Tensor): numpy or torch tensor
        transform (rasterio.Affine): affine geospatial transform
        crs (Any): coordinate reference system
        fill_value_default (Optional[Union[int, float]], optional): Value to fill when
            reading out of bounds. Could be None. Defaults to 0.
        shape (Tuple): shape of the tensor
        res (Tuple[float, float]): resolution of the tensor
        dtype: data type of the tensor
        height (int): height of the tensor
        width (int): width of the tensor
        count (int): number of bands in the tensor
        bounds (Tuple[float, float, float, float]): bounds of the tensor
        dims (Tuple[str]): names of the dimensions
        attrs (Dict[str, Any]): dictionary with the attributes of the GeoTensor

    Examples:
        >>> import numpy as np
        >>> transform = rasterio.Affine(1, 0, 0, 0, -1, 0)
        >>> crs = "EPSG:4326"
        >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)

    """

    def __init__(
        self,
        values: Tensor,
        transform: rasterio.Affine,
        crs: Any,
        fill_value_default: Optional[Union[int, float]] = 0,
        attrs: Optional[Dict[str, Any]] = None,
    ):
        """
        This class is a wrapper around a numpy or torch tensor with geospatial information.

        Args:
            values (Tensor): numpy or torch tensor
            transform (rasterio.Affine): affine geospatial transform
            crs (Any): coordinate reference system
            fill_value_default (Optional[Union[int, float]], optional): Value to fill when
                reading out of bounds. Could be None. Defaults to 0.
            attrs (Optional[Dict[str, Any]], optional): dictionary with the attributes of the GeoTensor
                Defaults to None.

        Raises:
            ValueError: when the shape of the tensor is not 2d, 3d or 4d.
        """
        self.values = values
        self.transform = transform
        self.crs = crs
        self.fill_value_default = fill_value_default
        shape = self.shape
        if (len(shape) < 2) or (len(shape) > 4):
            raise ValueError(f"Expected 2d-4d array found {shape}")

        self.attrs = attrs if attrs is not None else {}

    @property
    def dims(self) -> Tuple[str]:
        # TODO allow different ordering of dimensions?
        shape = self.shape
        if len(shape) == 2:
            dims = ("y", "x")
        elif len(shape) == 3:
            dims = ("band", "y", "x")
        elif len(shape) == 4:
            dims = ("time", "band", "y", "x")
        else:
            raise ValueError(f"Unexpected 2d-4d array found {shape}")

        return dims

    def to_json(self) -> Dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "transform": [
                self.transform.a,
                self.transform.b,
                self.transform.c,
                self.transform.d,
                self.transform.e,
                self.transform.f,
            ],
            "crs": str(self.crs),
            "fill_value_default": self.fill_value_default,
        }

    @classmethod
    def from_json(cls, json: Dict[str, Any]) -> "__class__":
        return cls(
            np.array(json["values"]), rasterio.Affine(*json["transform"]), json["crs"], json["fill_value_default"]
        )

    @property
    def shape(self) -> Tuple:
        return tuple(self.values.shape)

    @property
    def res(self) -> Tuple[float, float]:
        return window_utils.res(self.transform)

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def height(self) -> int:
        return self.shape[-2]

    @property
    def width(self) -> int:
        return self.shape[-1]

    @property
    def count(self) -> int:
        return self.shape[-3]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return window_bounds(
            rasterio.windows.Window(row_off=0, col_off=0, height=self.height, width=self.width), self.transform
        )

    def set_dtype(self, dtype):
        # TODO implement for torch tensor
        self.values = self.values.astype(dtype=dtype)

    def astype(self, dtype) -> "__class__":
        return GeoTensor(self.values.astype(dtype), self.transform, self.crs, self.fill_value_default)

    def meshgrid(self, dst_crs: Optional[Any] = None) -> Tuple[NDArray, NDArray]:
        from georeader import griddata

        return griddata.meshgrid(self.transform, self.width, self.height, source_crs=self.crs, dst_crs=dst_crs)

    def load(self) -> "__class__":
        return self

    def __copy__(self) -> "__class__":
        return GeoTensor(self.values.copy(), self.transform, self.crs, self.fill_value_default)

    def copy(self) -> "__class__":
        return self.__copy__()

    def same_extent(self, other: "__class__", precision: float = 1e-3) -> bool:
        """
        Check if two GeoTensors have the same georeferencing (crs and transform)

        Args:
            other (__class__ | GeoData): GeoTensor to compare with. Other GeoData object can be passed (it requires crs, transform and shape attributes)
            precision (float, optional): precision to compare the transform. Defaults to 1e-3.

        Returns:
            bool: True if both GeoTensors have the same georeferencing.
        """
        return (
            self.transform.almost_equals(other.transform, precision=precision)
            and window_utils.compare_crs(self.crs, other.crs)
            and (self.shape[-2:] == other.shape[-2:])
        )

    def __add__(self, other: Union[numbers.Number, "__class__"]) -> "__class__":
        """
        Add a value or array to this GeoTensor element-wise.
        
        Supports broadcasting with scalars, numpy arrays, or other GeoTensors.
        When adding two GeoTensors, they must have the same spatial extent
        (matching transform, CRS, and spatial dimensions).
        
        Broadcasting Rules:
        - Scalar: Added to every pixel
        - Array: Must be broadcastable to self.values shape
        - GeoTensor: Must have identical georeferencing (same_extent)
        
        Args:
            other (Union[numbers.Number, np.ndarray, GeoTensor]): Value to add. Can be:
                - Scalar (int, float): Added to all pixels
                - np.ndarray: Must be broadcastable with self.values
                - GeoTensor: Must have same spatial extent
        
        Returns:
            GeoTensor: New GeoTensor with same transform, CRS, and fill_value_default.
                Shape matches self.shape (or broadcast result for arrays).
        
        Raises:
            ValueError: If other is a GeoTensor and georeferencing doesn't match.
        
        Examples:
            >>> import numpy as np
            >>> import rasterio
            >>> from georeader import GeoTensor
            >>>
            >>> # Create sample GeoTensor (3 bands, 100x100 pixels)
            >>> transform = rasterio.Affine(10, 0, 500000, 0, -10, 4650000)
            >>> data = np.random.rand(3, 100, 100)
            >>> gt = GeoTensor(data, transform, crs="EPSG:32630")
            >>>
            >>> # Add scalar to all pixels
            >>> gt_offset = gt + 0.1
            >>> print(gt_offset.shape)  # (3, 100, 100)
            >>>
            >>> # Add per-band offset using broadcasting
            >>> band_offsets = np.array([0.1, 0.2, 0.3])[:, None, None]  # Shape: (3, 1, 1)
            >>> gt_adjusted = gt + band_offsets
            >>>
            >>> # Add two GeoTensors (must have same extent)
            >>> gt2 = GeoTensor(np.random.rand(3, 100, 100), transform, crs="EPSG:32630")
            >>> gt_sum = gt + gt2
            >>>
            >>> # Error case: mismatched georeferencing
            >>> gt_different = GeoTensor(data, rasterio.Affine(20, 0, 0, 0, -20, 0), crs="EPSG:4326")
            >>> # gt + gt_different  # Raises ValueError
        
        Note:
            - Result inherits transform, CRS, and fill_value_default from self
            - For GeoTensor addition, use `read.read_reproject_like(other, self)`
              to align georeferencing before adding
        
        See Also:
            - `same_extent`: Check if two GeoTensors have matching georeferencing
            - `read.read_reproject_like`: Reproject one GeoTensor to match another
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for addition. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values + other

        return GeoTensor(
            result_values, transform=self.transform, crs=self.crs, fill_value_default=self.fill_value_default
        )

    def __sub__(self, other: Union[numbers.Number, "__class__"]) -> "__class__":
        """
        Subtract a value or array from this GeoTensor element-wise.
        
        Supports broadcasting with scalars, numpy arrays, or other GeoTensors.
        When subtracting two GeoTensors, they must have the same spatial extent.
        
        Args:
            other (Union[numbers.Number, np.ndarray, GeoTensor]): Value to subtract. Can be:
                - Scalar (int, float): Subtracted from all pixels
                - np.ndarray: Must be broadcastable with self.values
                - GeoTensor: Must have same spatial extent
        
        Returns:
            GeoTensor: New GeoTensor with result of subtraction.
        
        Raises:
            ValueError: If other is a GeoTensor and georeferencing doesn't match.
        
        Examples:
            >>> import numpy as np
            >>> import rasterio
            >>> from georeader import GeoTensor
            >>>
            >>> transform = rasterio.Affine(10, 0, 500000, 0, -10, 4650000)
            >>> data = np.random.rand(3, 100, 100)
            >>> gt = GeoTensor(data, transform, crs="EPSG:32630")
            >>>
            >>> # Remove offset from all pixels
            >>> gt_corrected = gt - 0.05
            >>>
            >>> # Compute difference between two images (e.g., change detection)
            >>> gt_before = GeoTensor(np.random.rand(3, 100, 100), transform, crs="EPSG:32630")
            >>> gt_after = GeoTensor(np.random.rand(3, 100, 100), transform, crs="EPSG:32630")
            >>> change = gt_after - gt_before  # Positive = increase, negative = decrease
            >>>
            >>> # Subtract mean per band (centering)
            >>> band_means = gt.values.mean(axis=(1, 2))[:, None, None]  # (3, 1, 1)
            >>> gt_centered = gt - band_means
        
        See Also:
            - `__add__`: Addition operation
            - `same_extent`: Check georeferencing compatibility
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for substraction. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values - other

        return GeoTensor(
            result_values, transform=self.transform, crs=self.crs, fill_value_default=self.fill_value_default
        )

    def __mul__(self, other: Union[numbers.Number, "__class__"]) -> "__class__":
        """
        Multiply this GeoTensor by a value or array element-wise.
        
        Supports broadcasting with scalars, numpy arrays, or other GeoTensors.
        Common uses include scaling, applying masks, and band math operations.
        
        Args:
            other (Union[numbers.Number, np.ndarray, GeoTensor]): Value to multiply. Can be:
                - Scalar (int, float): Scales all pixels
                - np.ndarray: Must be broadcastable with self.values
                - GeoTensor: Must have same spatial extent
        
        Returns:
            GeoTensor: New GeoTensor with result of multiplication.
        
        Raises:
            ValueError: If other is a GeoTensor and georeferencing doesn't match.
        
        Examples:
            >>> import numpy as np
            >>> import rasterio
            >>> from georeader import GeoTensor
            >>>
            >>> transform = rasterio.Affine(10, 0, 500000, 0, -10, 4650000)
            >>> data = np.random.rand(3, 100, 100)
            >>> gt = GeoTensor(data, transform, crs="EPSG:32630")
            >>>
            >>> # Scale all values (e.g., unit conversion)
            >>> gt_scaled = gt * 10000  # Reflectance [0-1] to [0-10000]
            >>>
            >>> # Apply per-band gain coefficients
            >>> gains = np.array([1.1, 1.0, 0.95])[:, None, None]  # Shape: (3, 1, 1)
            >>> gt_calibrated = gt * gains
            >>>
            >>> # Apply binary mask (mask out invalid pixels)
            >>> mask = np.random.rand(100, 100) > 0.5  # Boolean mask
            >>> gt_masked = gt * mask  # Broadcasts mask to all bands
            >>>
            >>> # Element-wise product of two rasters
            >>> gt2 = GeoTensor(np.random.rand(3, 100, 100), transform, crs="EPSG:32630")
            >>> gt_product = gt * gt2
        
        Note:
            - For masking, consider using `gt[mask] = fill_value` for in-place updates
            - Result inherits georeferencing from self
        
        See Also:
            - `__truediv__`: Division operation
            - `__setitem__`: In-place value assignment with masks
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for multiplication. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values * other

        return GeoTensor(
            result_values, transform=self.transform, crs=self.crs, fill_value_default=self.fill_value_default
        )

    def __truediv__(self, other: Union[ArrayLike, "__class__"]) -> "__class__":
        """
        Divide this GeoTensor by a value or array element-wise.
        
        Supports broadcasting with scalars, numpy arrays, or other GeoTensors.
        Common uses include normalization, ratio calculations, and index computation.
        
        Args:
            other (Union[numbers.Number, np.ndarray, GeoTensor]): Divisor. Can be:
                - Scalar (int, float): Divides all pixels
                - np.ndarray: Must be broadcastable with self.values
                - GeoTensor: Must have same spatial extent
        
        Returns:
            GeoTensor: New GeoTensor with result of division.
        
        Raises:
            ValueError: If other is a GeoTensor and georeferencing doesn't match.
        
        Note:
            Division by zero produces inf or nan (numpy behavior).
            Consider adding small epsilon for numerical stability: ``gt / (other + 1e-10)``
        
        Examples:
            >>> import numpy as np
            >>> import rasterio
            >>> from georeader import GeoTensor
            >>>
            >>> transform = rasterio.Affine(10, 0, 500000, 0, -10, 4650000)
            >>> data = np.random.rand(3, 100, 100)
            >>> gt = GeoTensor(data, transform, crs="EPSG:32630")
            >>>
            >>> # Normalize to [0, 1] range
            >>> gt_norm = gt / gt.values.max()
            >>>
            >>> # Per-band normalization
            >>> band_maxes = gt.values.max(axis=(1, 2))[:, None, None] + 1e-10
            >>> gt_normalized = gt / band_maxes
            >>>
            >>> # Compute NDVI-like ratio: (NIR - Red) / (NIR + Red)
            >>> # Assuming band 0 = Red, band 1 = NIR
            >>> red = gt.values[0]
            >>> nir = gt.values[1]
            >>> ndvi_values = (nir - red) / (nir + red + 1e-10)  # Add epsilon
            >>> ndvi = GeoTensor(ndvi_values, gt.transform, gt.crs)
            >>>
            >>> # Ratio of two rasters
            >>> gt2 = GeoTensor(np.random.rand(3, 100, 100) + 0.1, transform, crs="EPSG:32630")
            >>> ratio = gt / gt2
        
        Note:
            - Add small epsilon to divisor to avoid division by zero
            - Use np.where or masking for conditional division
            - Result may contain inf/nan values - clean with np.nan_to_num if needed
        
        See Also:
            - `__mul__`: Multiplication operation
            - `clip`: Clip values to valid range after division
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for division. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values / other

        return GeoTensor(
            result_values, transform=self.transform, crs=self.crs, fill_value_default=self.fill_value_default
        )

    def __setitem__(self, index: np.ndarray, value: Union[np.ndarray, numbers.Number]) -> None:
        """
        Set the values of the GeoTensor object using an index and a new value.

        Args:
            index (tuple or numpy.ndarray): Index or boolean mask to apply to the GeoTensor values.
            value (numpy.ndarray): New value to assign to the GeoTensor values at the specified index.

        Raises:
            ValueError: If the index is not a tuple or a boolean numpy array with the same shape as the GeoTensor values.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> boolmask = gt.values > 0.5
            >>> gt[boolmask] = 0.5
        """
        if isinstance(index, np.ndarray) and (index.dtype == bool) and (index.shape == self.values.shape):
            # If the index is a boolean numpy array with the same shape as the values,
            # use it to mask the values and assign the new values to the masked values
            self.values[index] = value
        else:
            raise ValueError(f"Unsupported index type {type(index)} {index.dtype} {index} for GeoTensor set operation.")

    def squeeze(self) -> "__class__":
        """
        Remove single-dimensional entries from the shape of the GeoTensor values.
        It does not squeeze the spatial dimensions (last two dimensions).

        Returns:
            GeoTensor: GeoTensor with the squeezed values.
        """

        # squeeze all but last two dimensions
        squeezed_values = np.squeeze(self.values, axis=tuple(range(self.values.ndim - 2)))

        return GeoTensor(
            squeezed_values, transform=self.transform, crs=self.crs, fill_value_default=self.fill_value_default
        )

    def clip(self, a_min: Optional[np.array], a_max: Optional[np.array]) -> "__class__":
        """
        Clip the GeoTensor values between the GeoTensor min and max values.

        Args:
            a_min (float): Minimum value.
            a_max (float): Maximum value.

        Returns:
            GeoTensor: GeoTensor with the clipped values.
        """
        clipped_values = np.clip(self.values, a_min, a_max)
        return GeoTensor(
            clipped_values, transform=self.transform, crs=self.crs, fill_value_default=self.fill_value_default
        )

    def isel(self, sel: Dict[str, Union[slice, list, int]]) -> "__class__":
        """
        Slicing with dict. It doesn't work with negative indexes!

        Args:
            sel: Dict with slice selection; i.e. `{"x": slice(10, 20), "y": slice(20, 340)}`.

        Returns:
            GeoTensor: GeoTensor with the sliced values.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> gt.isel({"x": slice(10, 20), "y": slice(20, 340)})
        """
        for k in sel:
            if k not in self.dims:
                raise NotImplementedError(f"Axis {k} not in {self.dims}")

        slice_list = self._slice_tuple(sel)

        slices_window = []
        for k in ["y", "x"]:
            if k in sel:
                if not isinstance(sel[k], slice):
                    raise NotImplementedError(f"Only slice selection supported for x, y dims, found {sel[k]}")
                slices_window.append(sel[k])
            else:
                size = self.width if (k == "x") else self.height
                slices_window.append(slice(0, size))

        window_current = rasterio.windows.Window.from_slices(
            *slices_window, boundless=False, height=self.height, width=self.width
        )

        transform_current = rasterio.windows.transform(window_current, transform=self.transform)

        return GeoTensor(self.values[slice_list], transform_current, self.crs, self.fill_value_default)

    def _slice_tuple(self, sel: Dict[str, Union[slice, list, int]]) -> tuple:
        slice_list = []
        # shape_ = self.shape
        # sel_copy = sel.copy()
        for _i, k in enumerate(self.dims):
            if k in sel:
                if not isinstance(sel[k], slice) and not isinstance(sel[k], list) and not isinstance(sel[k], int):
                    raise NotImplementedError(f"Only slice selection supported for x, y dims, found {sel[k]}")
                # sel_copy[k] = slice(max(0, sel_copy[k].start), min(shape_[_i], sel_copy[k].stop))
                slice_list.append(sel[k])
            else:
                slice_list.append(slice(None))
        return tuple(slice_list)

    def footprint(self, crs: Optional[str] = None) -> Polygon:
        """
        Get the geographic footprint (bounding polygon) of the GeoTensor.

        This method returns a Shapely Polygon representing the rectangular extent of the raster
        in geographic coordinates. It's essential for:
        - Spatial queries: checking if rasters overlap with areas of interest
        - Metadata generation: documenting geographic coverage
        - Tile intersection testing: determining which tiles cover an area
        - Visualization: displaying raster extents on maps
        - Dataset cataloging: indexing raster collections by location

        The footprint is computed from the raster's corner coordinates (bounding rectangle),
        not from valid data pixels. For the actual data extent (excluding nodata regions),
        use `valid_footprint()` instead.

        Algorithm:
        1. Create window covering entire raster: Window(0, 0, width, height)
        2. Convert window corners to geographic coordinates using transform
        3. Create polygon from four corner points
        4. Optionally transform polygon to target CRS

        Args:
            crs (Optional[str], optional): Target coordinate reference system for the footprint.
                If None, returns footprint in the GeoTensor's native CRS. Common formats:
                "EPSG:4326" (WGS84), "EPSG:32630" (UTM), CRS object, or WKT string.
                Use "EPSG:4326" for compatibility with web mapping and GeoJSON.
                Defaults to None.

        Returns:
            Polygon: Shapely Polygon with exactly 5 vertices (4 corners + closing point)
                representing the rectangular footprint. Coordinates are in the specified
                CRS (or native CRS if crs=None).

        Examples:
            >>> import rasterio
            >>> from georeader import GeoTensor
            >>> import numpy as np
            >>>
            >>> # Example 1: Get footprint in native CRS
            >>> transform = rasterio.Affine(10, 0, 500000, 0, -10, 4650000)  # UTM transform
            >>> data = np.random.rand(3, 100, 100)
            >>> gt = GeoTensor(data, transform, crs="EPSG:32630")
            >>> footprint_utm = gt.footprint()
            >>> print(f"Bounds (UTM): {footprint_utm.bounds}")
            >>> # (xmin, ymin, xmax, ymax) in UTM meters: (500000, 4649000, 501000, 4650000)

            >>> # Example 2: Transform footprint to WGS84 for web mapping
            >>> footprint_wgs84 = gt.footprint(crs="EPSG:4326")
            >>> print(f"Bounds (WGS84): {footprint_wgs84.bounds}")
            >>> # (lon_min, lat_min, lon_max, lat_max) in degrees
            >>> # Can be used directly in Leaflet, Google Maps, etc.

            >>> # Example 3: Check if rasters overlap
            >>> gt1 = GeoTensor.load_file('image1.tif')
            >>> gt2 = GeoTensor.load_file('image2.tif')
            >>> # Get footprints in common CRS
            >>> fp1 = gt1.footprint(crs="EPSG:4326")
            >>> fp2 = gt2.footprint(crs="EPSG:4326")
            >>> if fp1.intersects(fp2):
            ...     print("Rasters overlap!")
            ...     overlap_area = fp1.intersection(fp2).area
            ...     print(f"Overlap area: {overlap_area} square degrees")

            >>> # Example 4: Export footprint as GeoJSON
            >>> from shapely.geometry import mapping
            >>> import json
            >>> footprint = gt.footprint(crs="EPSG:4326")
            >>> geojson = {
            ...     "type": "Feature",
            ...     "geometry": mapping(footprint),
            ...     "properties": {"name": "Raster extent"}
            ... }
            >>> with open('footprint.geojson', 'w') as f:
            ...     json.dump(geojson, f)

            >>> # Example 5: Check if point is within raster extent
            >>> from shapely.geometry import Point
            >>> point_of_interest = Point(-3.7038, 40.4168)  # Madrid coordinates
            >>> footprint = gt.footprint(crs="EPSG:4326")
            >>> if footprint.contains(point_of_interest):
            ...     print("Point is within raster extent")

            >>> # Example 6: Calculate raster area in square kilometers
            >>> footprint_utm = gt.footprint()  # In UTM (meters)
            >>> area_sqm = footprint_utm.area
            >>> area_sqkm = area_sqm / 1_000_000
            >>> print(f"Raster covers {area_sqkm:.2f} km²")

        Note:
            - Footprint represents the full raster extent, including nodata regions
            - For actual data coverage (excluding nodata), use valid_footprint()
            - Polygon always has rectangular shape (4 corners) even for rotated rasters
            - CRS transformation is performed if target CRS differs from native CRS
            - Footprint is cached-free (computed on-demand from transform and shape)
        """
        pol = window_utils.window_polygon(
            rasterio.windows.Window(row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]), self.transform
        )
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)

    def valid_footprint(self, crs: Optional[str] = None, method: str = "all") -> Union[MultiPolygon, Polygon]:
        """
        vectorizes the valid values of the GeoTensor and returns the footprint as a Polygon.

        Args:
            crs (Optional[str], optional): Coordinate reference system. Defaults to None.
            method (str, optional): "all" or "any" to aggregate the channels of the image. Defaults to "all".

        Returns:
            Polygon or MultiPolygon: footprint of the GeoTensor.
        """
        valid_values = self.values != self.fill_value_default
        if len(valid_values.shape) > 2:
            if method == "all":
                valid_values = np.all(valid_values, axis=tuple(np.arange(0, len(valid_values.shape) - 2).tolist()))
            elif method == "any":
                valid_values = np.any(valid_values, axis=tuple(np.arange(0, len(valid_values.shape) - 2).tolist()))
            else:
                raise NotImplementedError(f"Method {method} to aggregate channels not implemented")

        from georeader import vectorize

        polygons = vectorize.get_polygons(valid_values, transform=self.transform)
        if len(polygons) == 0:
            raise ValueError("GeoTensor has no valid values")
        elif len(polygons) == 1:
            pol = polygons[0]
        else:
            pol = MultiPolygon(polygons)
        if crs is None:
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)

    def __repr__(self) -> str:
        return f"""
         Transform: {self.transform}
         Shape: {self.shape}
         Resolution: {self.res}
         Bounds: {self.bounds}
         CRS: {self.crs}
         fill_value_default: {self.fill_value_default}
        """

    def pad(
        self, pad_width: Dict[str, Tuple[int, int]], mode: str = "constant", constant_values: Optional[Any] = None
    ) -> "__class__":
        """
        Pad the GeoTensor.

        Args:
            pad_width (_type_, optional):  dictionary with Tuple to pad for each dimension
                `{"x": (pad_x_0, pad_x_1), "y": (pad_y_0, pad_y_1)}`.
            mode (str, optional): pad mode (see np.pad or torch.nn.functional.pad). Defaults to "constant".
            constant_values (Any, optional): _description_. Defaults to `self.fill_value_default`.

        Returns:
            GeoTensor: padded GeoTensor.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> gt.pad({"x": (10, 10), "y": (10, 10)})
            >>> assert gt.shape == (3, 120, 120)
        """
        if constant_values is None and mode == "constant":
            constant_values = self.fill_value_default

        # Pad the data
        pad_torch = False
        if torch_installed:
            if isinstance(self.values, torch.Tensor):
                pad_torch = True

        if pad_torch:
            pad_list_torch = []
            for k in reversed(self.dims):
                if k in pad_width:
                    pad_list_torch.extend(list(pad_width[k]))
                else:
                    pad_list_torch.extend([0, 0])

            kwargs_extra = {}
            if mode == "constant":
                kwargs_extra["value"] = constant_values
            values_new = torch.nn.functional.pad(self.values, tuple(pad_list_torch), mode=mode, **kwargs_extra)
        else:
            pad_list_np = []
            for k in self.dims:
                if k in pad_width:
                    pad_list_np.append(pad_width[k])
                else:
                    pad_list_np.append((0, 0))

            kwargs_extra = {}
            if mode == "constant":
                kwargs_extra["constant_values"] = constant_values
            values_new = np.pad(self.values, tuple(pad_list_np), mode=mode, **kwargs_extra)

        # Compute the new transform
        slices_window = []
        for k in ["y", "x"]:
            size = self.width if (k == "x") else self.height
            if k in pad_width:
                slices_window.append(slice(-pad_width[k][0], size + pad_width[k][1]))
            else:
                slices_window.append(slice(0, size))

        window_current = rasterio.windows.Window.from_slices(*slices_window, boundless=True)
        transform_current = rasterio.windows.transform(window_current, transform=self.transform)
        return GeoTensor(values_new, transform_current, self.crs, self.fill_value_default, attrs=self.attrs)

    def resize(
        self,
        output_shape: Optional[Tuple[int, int]] = None,
        resolution_dst: Optional[Tuple[float, float]] = None,
        anti_aliasing: bool = True,
        anti_aliasing_sigma: Optional[Union[float, np.ndarray]] = None,
        interpolation: Optional[str] = "bilinear",
        mode_pad: str = "constant",
    ) -> "__class__":
        """
        Resize the geotensor to match a certain size output_shape. This function works with GeoTensors of 2D, 3D and 4D.
        The geoinformation of the output tensor is changed accordingly.

        Args:
            output_shape: output spatial shape if None resolution_dst must be provided. If not provided,
                the output shape is computed from the resolution_dst rounding to the closest integer.
            resolution_dst: output resolution if None output_shape must be provided.
            anti_aliasing: Whether to apply a Gaussian filter to smooth the image prior to downsampling
            anti_aliasing_sigma:  anti_aliasing_sigma : {float}, optional
                Standard deviation for Gaussian filtering used when anti-aliasing.
                By default, this value is chosen as (s - 1) / 2 where s is the
                downsampling factor, where s > 1
            interpolation: Algorithm used for resizing: 'nearest' | 'bilinear' | 'bicubic'
            mode_pad: mode pad for resize function

        Returns:
             resized GeoTensor

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> resized = gt.resize((50, 50))
            >>> assert resized.shape == (3, 50, 50)
            >>> assert resized.res == (2*gt.res[0], 2*gt.res[1])
        """
        input_shape = self.shape
        spatial_shape = input_shape[-2:]
        resolution_or = self.res

        if output_shape is None:
            assert resolution_dst is not None, f"Can't have output_shape and resolution_dst as None"
            output_shape = (
                int(round(spatial_shape[0] * resolution_or[0] / resolution_dst[0])),
                int(round(spatial_shape[1] * resolution_or[1] / resolution_dst[1])),
            )
        else:
            assert resolution_dst is None, f"Both output_shape and resolution_dst can't be provided"
            assert len(output_shape) == 2, f"Expected output shape to be the spatial dimensions found: {output_shape}"
            resolution_dst = (
                spatial_shape[0] * resolution_or[0] / output_shape[0],
                spatial_shape[1] * resolution_or[1] / output_shape[1],
            )

        # Compute output transform
        transform_scale = rasterio.Affine.scale(
            resolution_dst[0] / resolution_or[0], resolution_dst[1] / resolution_or[1]
        )
        transform = self.transform * transform_scale

        resize_kornia = False
        if torch_installed:
            if isinstance(self.values, torch.Tensor):
                resize_kornia = True

        if resize_kornia:
            # TODO
            # https://kornia.readthedocs.io/en/latest/geometry.transform.html#kornia.geometry.transform.resize
            raise NotImplementedError(f"Not implemented for torch Tensors")
        else:
            from skimage.transform import resize

            # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
            output_tensor = np.ndarray(input_shape[:-2] + output_shape, dtype=self.dtype)
            if len(input_shape) == 4:
                for i, j in product(range(0, input_shape[0]), range(0, input_shape[1])):
                    if (
                        (not anti_aliasing)
                        or (anti_aliasing_sigma is None)
                        or isinstance(anti_aliasing_sigma, numbers.Number)
                    ):
                        anti_aliasing_sigma_iter = anti_aliasing_sigma
                    else:
                        anti_aliasing_sigma_iter = anti_aliasing_sigma[i, j]
                    output_tensor[i, j] = resize(
                        self.values[i, j],
                        output_shape,
                        order=ORDERS[interpolation],
                        anti_aliasing=anti_aliasing,
                        preserve_range=False,
                        cval=self.fill_value_default,
                        mode=mode_pad,
                        anti_aliasing_sigma=anti_aliasing_sigma_iter,
                    )
            elif len(input_shape) == 3:
                for i in range(0, input_shape[0]):
                    if (
                        (not anti_aliasing)
                        or (anti_aliasing_sigma is None)
                        or isinstance(anti_aliasing_sigma, numbers.Number)
                    ):
                        anti_aliasing_sigma_iter = anti_aliasing_sigma
                    else:
                        anti_aliasing_sigma_iter = anti_aliasing_sigma[i]
                    output_tensor[i] = resize(
                        self.values[i],
                        output_shape,
                        order=ORDERS[interpolation],
                        anti_aliasing=anti_aliasing,
                        preserve_range=False,
                        cval=self.fill_value_default,
                        mode=mode_pad,
                        anti_aliasing_sigma=anti_aliasing_sigma_iter,
                    )
            else:
                output_tensor[...] = resize(
                    self.values,
                    output_shape,
                    order=ORDERS[interpolation],
                    anti_aliasing=anti_aliasing,
                    preserve_range=False,
                    cval=self.fill_value_default,
                    mode=mode_pad,
                    anti_aliasing_sigma=anti_aliasing_sigma,
                )

        return GeoTensor(
            output_tensor,
            transform=transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    @classmethod
    def load_file(
        cls,
        path: str,
        fs: Optional[Any] = None,
        load_tags: bool = False,
        load_descriptions: bool = False,
        rio_env_options: Optional[Dict[str, str]] = None,
    ) -> "__class__":
        """
        Load a GeoTensor from a file. It uses rasterio to read the data. This function
        loads all the data in memory. For a lazy loading reader use `georeader.rasterio_reader`.

        Args:
            path (str): Path to the file.
            fs (Optional[Any], optional): fsspec.Filesystem object. Defaults to None.
            load_descriptions (bool, optional): If True, load the description of the image. Defaults to False.
            load_tags (bool, optional): If True, load the tags of the image. Defaults to False.
            rio_env_options (Optional[Dict[str, str]], optional): Rasterio environment options. Defaults to None.

        Returns:
            GeoTensor: GeoTensor object with the loaded data.
        """

        if fs is not None:
            if load_descriptions:
                raise NotImplementedError("""Description loading not supported with `fsspec`. This is because
            the `descriptions` attribute cannote be loaded from a byte stream. This is a limitation of `rasterio`.
            The issue is related to how `rasterio.io.MemoryFile` handles band descriptions
            compared to direct file access. This is a known limitation when working
            with in-memory file representations in GDAL (which `rasterio` uses under
            the hood). If you need to load descriptions, you can use `georeader.rasterio_reader`
            class.""")

            with fs.open(path, "rb") as fh:
                return cls.load_bytes(fh.read(), load_tags=load_tags, rio_env_options=rio_env_options)

        tags = None
        descriptions = None
        rio_env_options = RIO_ENV_OPTIONS_DEFAULT if rio_env_options is None else rio_env_options
        with rasterio.Env(**get_rio_options_path(rio_env_options, path)):
            with rasterio.open(path) as src:
                data = src.read()
                transform = src.transform
                crs = src.crs
                fill_value_default = src.nodata
                if load_tags:
                    tags = src.tags()
                if load_descriptions:
                    descriptions = tuple(src.descriptions)

        attrs = {}
        if tags is not None:
            attrs["tags"] = tags

        if descriptions is not None:
            attrs["descriptions"] = descriptions

        return cls(data, transform, crs, fill_value_default=fill_value_default, attrs=attrs)

    @classmethod
    def load_bytes(
        cls,
        bytes_read: Union[bytes, bytearray, memoryview],
        load_tags: bool = False,
        rio_env_options: Optional[Dict[str, str]] = None,
    ) -> "__class__":
        """
        Load a GeoTensor from a byte stream. It uses rasterio to read the data.


        Args:
            bytes_read (Union[bytes, bytearray, memoryview]): Byte stream to read.
            load_tags (bool, optional): if True, load the tags of the image. Defaults to False.
            rio_env_options (Optional[Dict[str, str]], optional): Rasterio environment options. Defaults to None.

        Returns:
            __class__: GeoTensor object with the loaded data.

        Note:
            The `descriptions` attribute cannote be loaded from a byte stream. This is a limitation of `rasterio`.
            The issue is related to how `rasterio.io.MemoryFile` handles band descriptions
            compared to direct file access. This is a known limitation when working
            with in-memory file representations in GDAL (which `rasterio` uses under
            the hood). If you need to load descriptions, you should use `georeader.rasterio_reader`
            class.
        """
        import rasterio.io

        tags = None
        rio_env_options = RIO_ENV_OPTIONS_DEFAULT if rio_env_options is None else rio_env_options
        with rasterio.Env(**rio_env_options):
            with rasterio.io.MemoryFile(bytes_read) as mem:
                with mem.open() as src:
                    data = src.read()
                    transform = src.transform
                    crs = src.crs
                    fill_value_default = src.nodata
                    if load_tags:
                        tags = src.tags()

        attrs = {}
        if tags is not None:
            attrs["tags"] = tags

        return cls(data, transform, crs, fill_value_default=fill_value_default, attrs=attrs)

    def write_from_window(self, data: Tensor, window: rasterio.windows.Window):
        """
        Writes array to GeoTensor values object at the given window position. If window surpasses the bounds of this
        object it crops the data to fit the object.

        Args:
            data: Tensor to write. Expected: spatial dimensions `window.width`, `window.height`. Rest: same as `self`
            window: Window object that specifies the spatial location to write the data

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> data = np.random.rand(3, 50, 50)
            >>> window = rasterio.windows.Window(col_off=7, row_off=9, width=50, height=50)
            >>> gt.write_from_window(data, window)

        """
        window_data = rasterio.windows.Window(col_off=0, row_off=0, width=self.width, height=self.height)
        if not rasterio.windows.intersect(window, window_data):
            return

        assert data.shape[-2:] == (window.height, window.width), (
            f"window {window} has different shape than data {data.shape}"
        )
        assert data.shape[:-2] == self.shape[:-2], (
            f"Dimension of data in non-spatial channels found {data.shape} expected: {self.shape}"
        )

        slice_dict, pad_width = window_utils.get_slice_pad(window_data, window)
        slice_list = self._slice_tuple(slice_dict)
        # need_pad = any(p != 0 for p in pad_width["x"] + pad_width["y"])

        slice_data_spatial_x = slice(pad_width["x"][0], None if pad_width["x"][1] == 0 else -pad_width["x"][1])
        slice_data_spatial_y = slice(pad_width["y"][0], None if pad_width["y"][1] == 0 else -pad_width["y"][1])
        slice_data = self._slice_tuple({"x": slice_data_spatial_x, "y": slice_data_spatial_y})
        self.values[slice_list] = data[slice_data]

    def read_from_window(self, window: rasterio.windows.Window, boundless: bool = True) -> "__class__":
        """
        returns a new GeoTensor object with the spatial dimensions sliced

        Args:
            window: window to slice the current GeoTensor
            boundless: read from window in boundless mode (i.e. if the window is larger or negative it will pad
                the GeoTensor with `self.fill_value_default`)

        Raises:
            rasterio.windows.WindowError: if `window` does not intersect the data

        Returns:
            GeoTensor object with the spatial dimensions sliced

        """

        window_data = rasterio.windows.Window(col_off=0, row_off=0, width=self.width, height=self.height)
        if boundless:
            slice_dict, pad_width = window_utils.get_slice_pad(window_data, window)
            need_pad = any(p != 0 for p in pad_width["x"] + pad_width["y"])
            X_sliced = self.isel(slice_dict)
            if need_pad:
                X_sliced = X_sliced.pad(pad_width=pad_width, mode="constant", constant_values=self.fill_value_default)
            return X_sliced
        else:
            window_read = rasterio.windows.intersection(window, window_data)
            slice_y, slice_x = window_read.toslices()
            slice_dict = {"x": slice_x, "y": slice_y}
            slices_ = self._slice_tuple(slice_dict)
            transform_current = rasterio.windows.transform(window_read, transform=self.transform)
            return GeoTensor(self.values[slices_], transform_current, self.crs, self.fill_value_default)


def stack(geotensors: List[GeoTensor]) -> GeoTensor:
    """
    Stacks a list of geotensors, assert that all of them has same shape, transform and crs.

    Args:
        geotensors: list of geotensors to concat. All with same shape, transform and crs.

    Returns:
        geotensor with extra dim at the front: (len(geotensors),) + shape

    Examples:
        >>> gt1 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt2 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt3 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt = stack([gt1, gt2, gt3])
        >>> assert gt.shape == (3, 3, 100, 100)
    """
    assert len(geotensors) > 0, "Empty list provided can't concat"

    if len(geotensors) == 1:
        gt = geotensors[0].copy()
        gt.values = gt.values[np.newaxis]
        return gt

    first_geotensor = geotensors[0]
    array_out = np.zeros((len(geotensors),) + first_geotensor.shape, dtype=first_geotensor.dtype)
    array_out[0] = first_geotensor.values

    for i, geo in enumerate(geotensors[1:]):
        assert geo.same_extent(first_geotensor), f"Different size in concat"
        assert geo.shape == first_geotensor.shape, f"Different shape in concat"
        assert geo.fill_value_default == first_geotensor.fill_value_default, "Different fill_value_default in concat"
        array_out[i + 1] = geo.values

    return GeoTensor(
        array_out,
        transform=first_geotensor.transform,
        crs=first_geotensor.crs,
        fill_value_default=first_geotensor.fill_value_default,
    )


def concatenate(geotensors: List[GeoTensor], axis: int = 0) -> GeoTensor:
    """
    Concatenates a list of geotensors along a given axis, assert that all of them has same shape, transform and crs.

    Args:
        geotensors: list of geotensors to concat. All with same shape, transform and crs.
        axis: axis to concatenate. Must be less than the number of dimensions of the geotensors minus 2.
            default is 0.

    Returns:
        geotensor with extra dim at the front: (len(geotensors),) + shape

    Examples:
        >>> gt1 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt2 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt3 = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
        >>> gt = concatenate([gt1, gt2, gt3], axis=0)
        >>> assert gt.shape == (9, 100, 100)
    """
    assert len(geotensors) > 0, "Empty list provided can't concat"

    if len(geotensors) == 1:
        return geotensors[0].copy()

    first_geotensor = geotensors[0]

    # Assert the axis is NOT an spatial axis
    assert axis < len(first_geotensor.shape) - 2, f"Can't concatenate along spatial axis"

    array_out = np.concatenate([gt.values for gt in geotensors], axis=axis)

    return GeoTensor(
        array_out,
        transform=first_geotensor.transform,
        crs=first_geotensor.crs,
        fill_value_default=first_geotensor.fill_value_default,
    )
