import numpy as np
from typing import Any, Dict, Union, Tuple, Optional, List
import rasterio
import rasterio.windows
from georeader import window_utils
from georeader.window_utils import window_bounds
from numpy.typing import ArrayLike
from itertools import product
from shapely.geometry import Polygon, MultiPolygon
import numbers
from numpy.typing import NDArray
import warnings
from typing_extensions import Self
from rasterio import Affine

Tensor = np.ndarray

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
        warnings.warn(
            f"Protocol {protocol} not recognized. Returning the original path"
        )
        return path


def get_rio_options_path(options: dict, path: str) -> Dict[str, str]:
    if "read_with_CPL_VSIL_CURL_NON_CACHED" in options:
        options = options.copy()
        if options["read_with_CPL_VSIL_CURL_NON_CACHED"]:
            options["CPL_VSIL_CURL_NON_CACHED"] = _vsi_path(path)
        del options["read_with_CPL_VSIL_CURL_NON_CACHED"]
    return options


class GeoTensor(np.ndarray):
    """
    This class is a wrapper around a numpy tensor with geospatial information.
    It can store 2D, 3D or 4D tensors. The last two dimensions are the spatial dimensions.

    Args:
        values (Tensor): numpy or torch tensor (2D, 3D or 4D).
        transform (rasterio.Affine): affine geospatial transform
        crs (Any): coordinate reference system
        fill_value_default (Optional[Union[int, float]], optional): Value to fill when
            reading out of bounds. Could be None. Defaults to 0.

    Attributes:
        values (NDArray): numpy or torch tensor
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

    def __new__(
        cls,
        values: NDArray,
        transform: rasterio.Affine,
        crs: Any,
        fill_value_default: Optional[Union[int, float]] = 0,
        attrs: Optional[Dict[str, Any]] = None,
    ):
        """
        This class is a wrapper around a numpy or torch tensor with geospatial information.

        Args:
            values (NDArray): numpy or torch tensor
            transform (rasterio.Affine): affine geospatial transform
            crs (Any): coordinate reference system
            fill_value_default (Optional[Union[int, float]], optional): Value to fill when
                reading out of bounds. Could be None. Defaults to 0.
            attrs (Optional[Dict[str, Any]], optional): dictionary with the attributes of the GeoTensor
                Defaults to None.

        Raises:
            ValueError: when the shape of the tensor is not 2d, 3d or 4d.
        """
        obj = np.asarray(values).view(cls)

        obj.transform = transform
        obj.crs = crs
        obj.fill_value_default = fill_value_default
        shape = obj.shape
        if (len(shape) < 2) or (len(shape) > 4):
            raise ValueError(f"Expected 2d-4d array found {shape}")

        obj.attrs = attrs if attrs is not None else {}

        return obj

    def __array_finalize__(self, obj: Optional[Union[np.ndarray, Self]]) -> None:
        """
        Initialize attributes when a new GeoTensor is created from an existing array.

        This method is called whenever a new array object is created from an existing array
        (e.g., through slicing, view casting, or copy operations).

        Args:
            obj (Optional[np.ndarray]): The array object from which the new array is created.
                                       Can be None if the array is being created from scratch.
        """
        if obj is None:
            return

        if hasattr(obj, "transform"):
            self.transform: rasterio.Affine = getattr(obj, "transform", None)
        if hasattr(obj, "crs"):
            self.crs = getattr(obj, "crs", None)
        if hasattr(obj, "fill_value_default"):
            self.fill_value_default = getattr(obj, "fill_value_default", None)
        if hasattr(obj, "attrs"):
            self.attrs = getattr(obj, "attrs", None)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle NumPy universal functions applied to this GeoTensor.

        This method is called when a NumPy universal function (ufunc) is applied to the GeoTensor.
        It converts GeoTensor inputs to NumPy arrays, applies the ufunc, and converts array results
        back to GeoTensor objects with the same geospatial metadata.

        Args:
            ufunc (np.ufunc): The NumPy universal function being applied
            method (str): The method of the ufunc ('__call__', 'reduce', etc.)
            *inputs: The input arrays to the ufunc
            **kwargs: Additional keyword arguments to the ufunc

        Returns:
            Union[GeoTensor, NDArray]: If the result is an array, returns a new GeoTensor with the same
                                  geospatial attributes. Otherwise, returns the original result.
        """
        # Normal processing for most operations
        inputs_arr = tuple(
            x.view(np.ndarray) if isinstance(x, GeoTensor) else x for x in inputs
        )
        # Handle 'out' argument if present
        out = kwargs.pop("out", None)
        out_arrays = None

        if out:
            # Convert GeoTensor outputs to regular arrays
            if isinstance(out, tuple):
                out_arrays = tuple(
                    o.view(np.ndarray) if isinstance(o, GeoTensor) else o for o in out
                )
            else:
                out_arrays = (
                    (out.view(np.ndarray),) if isinstance(out, GeoTensor) else (out,)
                )
            kwargs["out"] = out_arrays

        # Delegate to numpy's implementation
        result = super().__array_ufunc__(ufunc, method, *inputs_arr, **kwargs)

        cast_to_geotensor = self._preserved_spatial(method, **kwargs)

        # Propagate metadata to output arrays
        if out_arrays:
            for o_orig, o_new in zip(
                out if isinstance(out, tuple) else [out], out_arrays
            ):
                if (
                    cast_to_geotensor
                    and isinstance(o_orig, GeoTensor)
                    and isinstance(o_new, np.ndarray)
                ):
                    o_new = o_orig.array_as_geotensor(o_new)

        # Normal ufunc processing for other cases
        if cast_to_geotensor:
            return self.array_as_geotensor(result)

        return result

    def _preserved_spatial(self, method: str, **kwargs) -> bool:
        """Special handling for reduction operations (sum, mean, max, etc.)"""
        # Extract reduction axis (default is flattening)
        if method != "reduce":
            return True  # No reduction, preserve spatial dims

        axis = kwargs.get("axis", None)

        # Check if reduction preserves spatial structure (last 2 dims untouched)
        if axis is not None:
            if isinstance(axis, int):
                preserve_spatial = axis not in [-1, -2, self.ndim - 1, self.ndim - 2]
            else:  # tuple of axes
                preserve_spatial = all(
                    ax not in [-1, -2, self.ndim - 1, self.ndim - 2] for ax in axis
                )
        else:
            preserve_spatial = False  # Full reduction eliminates all dims

        # For full reductions or spatial dim reductions, return plain array/scalar
        return preserve_spatial

    def array_as_geotensor(
        self,
        result: Union[np.ndarray, Self],
        fill_value_default: Optional[numbers.Number] = None,
    ) -> Self:
        """
        Convert a NumPy array result back to a GeoTensor.

        Args:
            result (Union[np.ndarray, Self]): Any NumPy array or GeoTensor.
            fill_value_default: fill value for the returned GeoTensor.

        Returns:
            Self: A new GeoTensor with the same geospatial attributes as the original.
        """

        # Propagate metadata for array results
        if isinstance(result, np.ndarray):
            if result.shape[-2:] != self.shape[-2:]:
                raise ValueError("Operation altered spatial dimensions!")

            if fill_value_default is None:
                fill_value_default = self.fill_value_default

            result = GeoTensor(
                result,
                transform=self.transform,
                crs=self.crs,
                fill_value_default=fill_value_default,
                attrs=self.attrs,
            )

        return result

    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """
        Convert the GeoTensor to a standard NumPy array.

        This method is called by np.asarray() and most NumPy functions to get
        the underlying NumPy array representation of this object.

        Args:
            dtype (Optional[np.dtype]): The desired data type for the returned array.
                                       If None, the array's current dtype is preserved.

        Returns:
            np.ndarray: A NumPy array view of this GeoTensor.
        """
        return np.asarray(self.view(np.ndarray), dtype=dtype)

    @property
    def values(self):
        """Return a view of the array (memory shared with original)"""
        return self.view(np.ndarray)

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
    def from_json(cls, json: Dict[str, Any]) -> Self:
        return cls(
            np.array(json["values"]),
            rasterio.Affine(*json["transform"]),
            json["crs"],
            json["fill_value_default"],
        )

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
            rasterio.windows.Window(
                row_off=0, col_off=0, height=self.height, width=self.width
            ),
            self.transform,
        )

    def set_dtype(self, dtype):
        self.values = self.values.astype(dtype=dtype)

    # Not needed due to ufunc implementation?
    # def astype(self, dtype) -> Self:
    #     return GeoTensor(
    #         self.values.astype(dtype), self.transform, self.crs, self.fill_value_default
    #     )

    def meshgrid(self, dst_crs: Optional[Any] = None) -> Tuple[NDArray, NDArray]:
        """
        Create a meshgrid of spatial dimensions of the GeoTensor.

        Args:
            dst_crs (Optional[Any], optional): output coordinate reference system. Defaults to None.

        Returns:
            Tuple[NDArray, NDArray]: 2D arrays of xs and ys coordinates.
        """
        from georeader import griddata

        return griddata.meshgrid(
            self.transform,
            self.width,
            self.height,
            source_crs=self.crs,
            dst_crs=dst_crs,
        )

    def load(self) -> Self:
        return self

    def __copy__(self) -> Self:
        return GeoTensor(
            self.values.copy(), self.transform, self.crs, self.fill_value_default
        )

    def copy(self) -> Self:
        return self.__copy__()

    def same_extent(self, other: Self, precision: float = 1e-3) -> bool:
        """
        Check if two GeoTensors have the same georeferencing (crs, transform and spatial dimensions).

        Args:
            other (GeoTensor | GeoData): GeoTensor to compare with. Other GeoData object can be passed (it requires crs, transform and shape attributes)
            precision (float, optional): precision to compare the transform. Defaults to 1e-3.

        Returns:
            bool: True if both GeoTensors have the same georeferencing.
        """
        return (
            self.transform.almost_equals(other.transform, precision=precision)
            and window_utils.compare_crs(self.crs, other.crs)
            and (self.shape[-2:] == other.shape[-2:])
        )

    def __add__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Add two GeoTensors. The georeferencing must match.

        Args:
            other (GeoTensor): GeoTensor to add.

        Raises:
            ValueError: if the georeferencing does not match.
            TypeError: if other is not a GeoTensor.

        Returns:
            GeoTensor: GeoTensor with the result of the addition.
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
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def __sub__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Substract two GeoTensors. The georeferencing must match.

        Args:
            other (GeoTensor): GeoTensor to add.

        Raises:
            ValueError: if the georeferencing does not match.
            TypeError: if other is not a GeoTensor.

        Returns:
            GeoTensor: GeoTensor with the result of the substraction.

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
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def __mul__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Multiply two GeoTensors. The georeferencing must match.

        Args:
            other (GeoTensor): GeoTensor to add.

        Raises:
            ValueError: if the georeferencing does not match.
            TypeError: if other is not a GeoTensor.

        Returns:
            GeoTensor: GeoTensor with the result of the multiplication.
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
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def __truediv__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Divide two GeoTensors. The georeferencing must match.

        Args:
            other (GeoTensor): GeoTensor to add.

        Raises:
            ValueError: if the georeferencing does not match.
            TypeError: if other is not a GeoTensor.

        Returns:
            GeoTensor: GeoTensor with the result of the division.
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
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def __and__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Perform bitwise AND operation between two GeoTensors. The georeferencing must match.

        Args:
            other (Union[numbers.Number, NDArray, GeoTensor]): GeoTensor or array-like to AND with.

        Raises:
            ValueError: if the georeferencing does not match when other is a GeoTensor.

        Returns:
            GeoTensor: GeoTensor with the result of the bitwise AND operation.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for bitwise AND. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values & other

        return GeoTensor(
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def __or__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Perform bitwise OR operation between two GeoTensors. The georeferencing must match.

        Args:
            other (Union[numbers.Number, NDArray, GeoTensor]): GeoTensor or array-like to OR with.

        Raises:
            ValueError: if the georeferencing does not match when other is a GeoTensor.

        Returns:
            GeoTensor: GeoTensor with the result of the bitwise OR operation.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for bitwise OR. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values | other

        return GeoTensor(
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def __eq__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Element-wise equality comparison between GeoTensors or with a scalar/array.
        The georeferencing must match if comparing with another GeoTensor.

        Args:
            other (Union[numbers.Number, NDArray, GeoTensor]): Value to compare with.

        Raises:
            ValueError: If comparing with a GeoTensor and the georeferencing doesn't match.

        Returns:
            GeoTensor: GeoTensor with boolean values indicating equality.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for comparison. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values == other

        return GeoTensor(
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=False,
            attrs=self.attrs,
        )

    def __ne__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Element-wise inequality comparison between GeoTensors or with a scalar/array.
        The georeferencing must match if comparing with another GeoTensor.

        Args:
            other (Union[numbers.Number, NDArray, GeoTensor]): Value to compare with.

        Raises:
            ValueError: If comparing with a GeoTensor and the georeferencing doesn't match.

        Returns:
            GeoTensor: GeoTensor with boolean values indicating inequality.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for comparison. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values != other

        return GeoTensor(
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=False,
            attrs=self.attrs,
        )

    def __lt__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Element-wise less than comparison between GeoTensors or with a scalar/array.
        The georeferencing must match if comparing with another GeoTensor.

        Args:
            other (Union[numbers.Number, NDArray, GeoTensor]): Value to compare with.

        Raises:
            ValueError: If comparing with a GeoTensor and the georeferencing doesn't match.

        Returns:
            GeoTensor: GeoTensor with boolean values indicating less than relationship.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for comparison. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values < other

        return GeoTensor(
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=False,
            attrs=self.attrs,
        )

    def __le__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Element-wise less than or equal comparison between GeoTensors or with a scalar/array.
        The georeferencing must match if comparing with another GeoTensor.

        Args:
            other (Union[numbers.Number, NDArray, GeoTensor]): Value to compare with.

        Raises:
            ValueError: If comparing with a GeoTensor and the georeferencing doesn't match.

        Returns:
            GeoTensor: GeoTensor with boolean values indicating less than or equal relationship.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for comparison. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values <= other

        return GeoTensor(
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=False,
            attrs=self.attrs,
        )

    def __gt__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Element-wise greater than comparison between GeoTensors or with a scalar/array.
        The georeferencing must match if comparing with another GeoTensor.

        Args:
            other (Union[numbers.Number, NDArray, GeoTensor]): Value to compare with.

        Raises:
            ValueError: If comparing with a GeoTensor and the georeferencing doesn't match.

        Returns:
            GeoTensor: GeoTensor with boolean values indicating greater than relationship.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for comparison. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values > other

        return GeoTensor(
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=False,
            attrs=self.attrs,
        )

    def __ge__(self, other: Union[numbers.Number, NDArray, Self]) -> Self:
        """
        Element-wise greater than or equal comparison between GeoTensors or with a scalar/array.
        The georeferencing must match if comparing with another GeoTensor.

        Args:
            other (Union[numbers.Number, NDArray, GeoTensor]): Value to compare with.

        Raises:
            ValueError: If comparing with a GeoTensor and the georeferencing doesn't match.

        Returns:
            GeoTensor: GeoTensor with boolean values indicating greater than or equal relationship.
        """
        if isinstance(other, GeoTensor):
            if self.same_extent(other):
                other = other.values
            else:
                raise ValueError(
                    "GeoTensor georref must match for comparison. "
                    "Use `read.read_reproject_like(other, self)` to "
                    "to reproject `other` to `self` georreferencing."
                )

        result_values = self.values >= other

        return GeoTensor(
            result_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=False,
            attrs=self.attrs,
        )

    def squeeze(self, axis=None) -> Self:
        """
        Remove single-dimensional entries from the shape of the GeoTensor values.
        It does not squeeze the spatial dimensions (last two dimensions).

        Returns:
            GeoTensor: GeoTensor with the squeezed values.
        """
        if axis is None:
            axis = tuple(range(self.values.ndim - 2))
        else:
            if isinstance(axis, int):
                axis = (axis,)
            # Check if spatial dimesions will be squeezed
            if self.width == 1:
                if any(a in (-1, self.values.ndim - 1) for a in axis):
                    raise ValueError(
                        "Cannot squeeze spatial dimensions. "
                        "Use `squeeze(axis=0)` to squeeze the first dimension."
                    )
            elif self.height == 1:
                if any(a in (-2, self.values.ndim - 2) for a in axis):
                    raise ValueError(
                        "Cannot squeeze spatial dimensions. "
                        "Use `squeeze(axis=0)` to squeeze the first dimension."
                    )

        # squeeze all but last two dimensions
        squeezed_values = np.squeeze(self.values, axis=axis)

        return GeoTensor(
            squeezed_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def expand_dims(self, axis: Union[int, tuple]) -> Self:
        """
        Expand the dimensions of the GeoTensor values while preserving the spatial dimensions.

        This method ensures that no dimensions are added after or in between the spatial dimensions
        (which are always the last two dimensions).

        Args:
            axis (Union[int, tuple]): Position or positions where new axes should be inserted.
                Must be less than the number of dimensions minus 2 (to preserve spatial dims).
                Positions are counted from the first dimension.

        Returns:
            GeoTensor: GeoTensor with the expanded values.

        Raises:
            ValueError: If trying to add dimensions at or after the spatial dimensions.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> # Add a new dimension at axis 0
            >>> gt_expanded = gt.expand_dims(0)
            >>> assert gt_expanded.shape == (1, 3, 100, 100)
        """
        ndim = len(self.shape)

        # Check if axis is valid (not interfering with spatial dimensions)
        if isinstance(axis, int):
            if axis >= ndim - 2 or axis < -ndim:
                raise ValueError(
                    f"Cannot add dimension at or after spatial dimensions. "
                    f"Axis must be < {ndim - 2} or >= {-ndim}, got {axis}"
                )
            # Convert negative axis to positive
            if axis < 0:
                axis = ndim + axis
        else:  # tuple of axes
            for ax in axis:
                if ax >= ndim - 2 or ax < -ndim:
                    raise ValueError(
                        f"Cannot add dimension at or after spatial dimensions. "
                        f"All axes must be < {ndim - 2} or >= {-ndim}, got {ax}"
                    )

        # Use numpy expand_dims to add the new dimensions
        expanded_values = np.expand_dims(self.values, axis=axis)

        return GeoTensor(
            expanded_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def clip(self, a_min: Optional[np.array], a_max: Optional[np.array]) -> Self:
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
            clipped_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def __getitem__(self, key):
        """
        Get the values of the GeoTensor using the given key.

        Args:
            key: Key to index the GeoTensor.

        Returns:
            GeoTensor: GeoTensor with the selected values.
        """
        if not isinstance(key, tuple):
            key = (key,)

        sel_dict = {}
        for i, k in enumerate(self.dims):
            if i < len(key):
                if key[i] is None:
                    raise NotImplementedError(
                        f"Adding axis is not permitted to GeoTensors. Use `expand_dims`"
                    )
                elif isinstance(key[i], type(...)):
                    raise NotImplementedError(
                        f"Using elipsis is not permitted with GeoTensors. Use `values` attribute"
                    )
                sel_dict[k] = key[i]
            else:
                sel_dict[k] = slice(None)

        return self.isel(sel_dict)

    def isel(self, sel: Dict[str, Union[slice, list, int]]) -> Self:
        """
        Slicing with dict. Spatial dimensions can only be sliced with slices.

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

        slice_tuple = self._slice_tuple(sel)

        # Compùte the window to shift the transform
        slices_window = []
        for k in ["y", "x"]:
            if k in sel:
                if not isinstance(sel[k], slice):
                    raise NotImplementedError(
                        f"Only slice selection supported for x, y dims, found {sel[k]}"
                    )
                slices_window.append(sel[k])
            else:
                size = self.width if (k == "x") else self.height
                slices_window.append(slice(0, size))

        window_current = rasterio.windows.Window.from_slices(
            *slices_window, boundless=False, height=self.height, width=self.width
        )

        transform_current = rasterio.windows.transform(
            window_current, transform=self.transform
        )

        # Scale the spatial transform if the step of the slices > 1
        step_rows = slices_window[0].step
        step_cols = slices_window[1].step
        if step_rows is None:
            step_rows = 1

        if step_cols is None:
            step_cols = 1

        if (step_rows != 1) or (step_cols != 1):
            transform_current = transform_current * Affine.scale(step_cols, step_rows)

        return GeoTensor(
            self.values[slice_tuple],
            transform_current,
            self.crs,
            self.fill_value_default,
            attrs=self.attrs,
        )

    def _slice_tuple(self, sel: Dict[str, Union[slice, list, int]]) -> tuple:
        slice_list = []
        for k in self.dims:
            if k in sel:
                slice_list.append(sel[k])
            else:
                slice_list.append(slice(None))
        return tuple(slice_list)

    def footprint(self, crs: Optional[str] = None) -> Polygon:
        """Returns the footprint of the GeoTensor as a Polygon.

        Args:
            crs (Optional[str], optional): Coordinate reference system. Defaults to None.

        Returns:
            Polygon: footprint of the GeoTensor.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 100, 100), transform, crs)
            >>> gt.footprint(crs="EPSG:4326") # returns a Polygon in WGS84
        """
        pol = window_utils.window_polygon(
            rasterio.windows.Window(
                row_off=0, col_off=0, height=self.shape[-2], width=self.shape[-1]
            ),
            self.transform,
        )
        if (crs is None) or window_utils.compare_crs(self.crs, crs):
            return pol

        return window_utils.polygon_to_crs(pol, self.crs, crs)

    def valid_footprint(
        self, crs: Optional[str] = None, method: str = "all"
    ) -> Union[MultiPolygon, Polygon]:
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
                valid_values = np.all(
                    valid_values,
                    axis=tuple(np.arange(0, len(valid_values.shape) - 2).tolist()),
                )
            elif method == "any":
                valid_values = np.any(
                    valid_values,
                    axis=tuple(np.arange(0, len(valid_values.shape) - 2).tolist()),
                )
            else:
                raise NotImplementedError(
                    f"Method {method} to aggregate channels not implemented"
                )

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
        self,
        pad_width: Union[Dict[str, Tuple[int, int]], List[Tuple[int, int]]],
        mode: str = "constant",
        **kwargs,
    ):
        """
        Pad the GeoTensor.

        Args:
            pad_width (Union[Dict[str, Tuple[int, int]], List[Tuple[int, int]]]):
                dictionary with Tuple to pad for each dimension
                `{"x": (pad_x_0, pad_x_1), "y": (pad_y_0, pad_y_1)}` or list of tuples
                `[(pad_x_0, pad_x_1), (pad_y_0, pad_y_1)]`.
            mode (str, optional): pad mode (see np.pad or torch.nn.functional.pad). Defaults to "constant".
            kwargs: additional arguments for the pad function.

        Returns:
            GeoTensor: padded GeoTensor.
        """
        if isinstance(pad_width, list) or isinstance(pad_width, tuple):
            if len(pad_width) != len(self.dims):
                raise ValueError(
                    f"Expected {len(self.dims)} pad widths found {len(pad_width)}"
                )
            pad_width_dict = {}
            for i, k in enumerate(self.dims):
                pad_width_dict[k] = pad_width[i]
        else:
            pad_width_dict = pad_width
        return self.pad_array(pad_width_dict, mode=mode, **kwargs)

    def pad_array(
        self,
        pad_width: Dict[str, Tuple[int, int]],
        mode: str = "constant",
        constant_values: Optional[Any] = None,
    ) -> Self:
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
            >>> gt.pad_array({"x": (10, 10), "y": (10, 10)})
            >>> assert gt.shape == (3, 120, 120)
        """
        if constant_values is None and mode == "constant":
            if self.fill_value_default is None:
                raise ValueError(
                    f"Mode constant either requires constant_values passed or fill_value_default not None in current GeoTensor"
                )
            constant_values = self.fill_value_default

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

        window_current = rasterio.windows.Window.from_slices(
            *slices_window, boundless=True
        )
        transform_current = rasterio.windows.transform(
            window_current, transform=self.transform
        )
        return GeoTensor(
            values_new,
            transform_current,
            self.crs,
            self.fill_value_default,
            attrs=self.attrs,
        )

    def resize(
        self,
        output_shape: Optional[Tuple[int, int]] = None,
        resolution_dst: Optional[Tuple[float, float]] = None,
        anti_aliasing: bool = True,
        anti_aliasing_sigma: Optional[Union[float, np.ndarray]] = None,
        interpolation: Optional[str] = "bilinear",
        mode_pad: str = "constant",
    ) -> Self:
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
            assert (
                resolution_dst is not None
            ), f"Can't have output_shape and resolution_dst as None"
            output_shape = int(
                round(spatial_shape[0] * resolution_or[0] / resolution_dst[0])
            ), int(round(spatial_shape[1] * resolution_or[1] / resolution_dst[1]))
        else:
            assert (
                resolution_dst is None
            ), f"Both output_shape and resolution_dst can't be provided"
            assert (
                len(output_shape) == 2
            ), f"Expected output shape to be the spatial dimensions found: {output_shape}"
            resolution_dst = (
                spatial_shape[0] * resolution_or[0] / output_shape[0],
                spatial_shape[1] * resolution_or[1] / output_shape[1],
            )

        # Compute output transform
        transform_scale = rasterio.Affine.scale(
            resolution_dst[0] / resolution_or[0], resolution_dst[1] / resolution_or[1]
        )
        transform = self.transform * transform_scale

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

    def transpose(self, axes=None) -> Self:
        """
        Permute the dimensions of the GeoTensor while keeping the spatial dimensions at the end.

        Args:
            axes (tuple, optional): If specified, it must be a tuple or list of axes. The last two
                values must be the original spatial dimensions indices (ndim-2, ndim-1).
                If None, the non-spatial dimensions are reversed while spatial dimensions remain at the end.

        Returns:
            GeoTensor: A view of the array with dimensions transposed.

        Raises:
            ValueError: If the spatial dimensions are moved from their last positions.

        Examples:
            >>> gt = GeoTensor(np.random.rand(3, 4, 100, 200), transform, crs)
            >>> # Shape is (3, 4, 100, 200)
            >>> gt_t = gt.transpose()
            >>> # Shape is now (4, 3, 100, 200)
            >>>
            >>> # You can also specify axes explicitly:
            >>> gt_t = gt.transpose((1, 0, 2, 3))  # Valid: spatial dims remain at end
            >>> # But this would raise an error:
            >>> # gt.transpose((0, 2, 1, 3))  # Invalid: spatial dims must stay at end
        """
        ndim = len(self.shape)

        if ndim <= 2:
            # Nothing meaningful to transpose for arrays with only spatial dimensions
            return self.copy()

        # Original spatial dimensions indices
        y_dim = ndim - 2
        x_dim = ndim - 1

        if axes is None:
            # Reverse all dimensions except the spatial ones which stay at the end
            non_spatial_axes = list(range(ndim - 2))
            non_spatial_axes.reverse()
            axes = tuple(non_spatial_axes + [y_dim, x_dim])
        else:
            # Convert to tuple if necessary
            axes = tuple(axes)

            # Check if axes has the right length
            if len(axes) != ndim:
                raise ValueError(
                    f"axes should contain {ndim} dimensions, got {len(axes)}"
                )

            # Check if the last two values in axes are the spatial dimensions
            if axes[-2:] != (y_dim, x_dim):
                raise ValueError(
                    "Cannot change the position of spatial dimensions. "
                    f"The last two axes must be {y_dim} and {x_dim}."
                )

        # Perform the transpose
        transposed_values = np.transpose(self.values, axes)

        return GeoTensor(
            transposed_values,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=self.fill_value_default,
            attrs=self.attrs,
        )

    def validmask(self) -> Self:
        """
        Returns a mask of the valid values of the GeoTensor. The mask is a boolean array
        with the same shape as the GeoTensor values, where True indicates valid values and
        False indicates invalid values.
        The mask is created by comparing the values of the GeoTensor with the `self.fill_value_default`.

        Returns:
            Self: GeoTensor with the valid boolean mask.
        """
        if self.fill_value_default is None:
            return GeoTensor(
                np.ones(self.shape, dtype=bool),
                transform=self.transform,
                crs=self.crs,
                fill_value_default=self.fill_value_default,
                attrs=self.attrs,
            )
        return GeoTensor(
            values=self.values != self.fill_value_default,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=False,
            attrs=self.attrs,
        )

    def invalidmask(self) -> Self:
        """
        Returns a mask of the invalid values of the GeoTensor. The mask is a boolean array
        with the same shape as the GeoTensor values, where True indicates invalid values and
        False indicates valid values.
        The mask is created by comparing the values of the GeoTensor with the `self.fill_value_default`.

        Returns:
            Self: GeoTensor with the invalid boolean mask.
        """
        if self.fill_value_default is None:
            return GeoTensor(
                np.zeros(self.shape, dtype=bool),
                transform=self.transform,
                crs=self.crs,
                fill_value_default=self.fill_value_default,
            )
        return GeoTensor(
            values=self.values == self.fill_value_default,
            transform=self.transform,
            crs=self.crs,
            fill_value_default=False,
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
    ) -> Self:
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
                raise NotImplementedError(
                    """Description loading not supported with `fsspec`. This is because
            the `descriptions` attribute cannote be loaded from a byte stream. This is a limitation of `rasterio`.
            The issue is related to how `rasterio.io.MemoryFile` handles band descriptions 
            compared to direct file access. This is a known limitation when working 
            with in-memory file representations in GDAL (which `rasterio` uses under 
            the hood). If you need to load descriptions, you can use `georeader.rasterio_reader`
            class."""
                )

            with fs.open(path, "rb") as fh:
                return cls.load_bytes(
                    fh.read(), load_tags=load_tags, rio_env_options=rio_env_options
                )

        tags = None
        descriptions = None
        rio_env_options = (
            RIO_ENV_OPTIONS_DEFAULT if rio_env_options is None else rio_env_options
        )
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

        return cls(
            data, transform, crs, fill_value_default=fill_value_default, attrs=attrs
        )

    @classmethod
    def load_bytes(
        cls,
        bytes_read: Union[bytes, bytearray, memoryview],
        load_tags: bool = False,
        rio_env_options: Optional[Dict[str, str]] = None,
    ) -> Self:
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
        rio_env_options = (
            RIO_ENV_OPTIONS_DEFAULT if rio_env_options is None else rio_env_options
        )
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

        return cls(
            data, transform, crs, fill_value_default=fill_value_default, attrs=attrs
        )

    def write_from_window(self, data: np.ndarray, window: rasterio.windows.Window):
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
        window_data = rasterio.windows.Window(
            col_off=0, row_off=0, width=self.width, height=self.height
        )
        if not rasterio.windows.intersect(window, window_data):
            return

        assert data.shape[-2:] == (
            window.height,
            window.width,
        ), f"window {window} has different shape than data {data.shape}"
        assert (
            data.shape[:-2] == self.shape[:-2]
        ), f"Dimension of data in non-spatial channels found {data.shape} expected: {self.shape}"

        slice_dict, pad_width = window_utils.get_slice_pad(window_data, window)
        slice_list = self._slice_tuple(slice_dict)
        # need_pad = any(p != 0 for p in pad_width["x"] + pad_width["y"])

        slice_data_spatial_x = slice(
            pad_width["x"][0], None if pad_width["x"][1] == 0 else -pad_width["x"][1]
        )
        slice_data_spatial_y = slice(
            pad_width["y"][0], None if pad_width["y"][1] == 0 else -pad_width["y"][1]
        )
        slice_data = self._slice_tuple(
            {"x": slice_data_spatial_x, "y": slice_data_spatial_y}
        )
        self.values[slice_list] = data[slice_data]

    def read_from_window(
        self, window: rasterio.windows.Window, boundless: bool = True
    ) -> Self:
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

        window_data = rasterio.windows.Window(
            col_off=0, row_off=0, width=self.width, height=self.height
        )
        if boundless:
            slice_dict, pad_width = window_utils.get_slice_pad(window_data, window)
            need_pad = any(p != 0 for p in pad_width["x"] + pad_width["y"])
            X_sliced = self.isel(slice_dict)
            if need_pad:
                X_sliced = X_sliced.pad(
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=self.fill_value_default,
                )
            return X_sliced
        else:
            window_read = rasterio.windows.intersection(window, window_data)
            slice_y, slice_x = window_read.toslices()
            slice_dict = {"x": slice_x, "y": slice_y}
            slices_ = self._slice_tuple(slice_dict)
            transform_current = rasterio.windows.transform(
                window_read, transform=self.transform
            )
            return GeoTensor(
                self.values[slices_],
                transform_current,
                self.crs,
                self.fill_value_default,
                attrs=self.attrs,
            )

    @classmethod
    def stack(cls, geotensors: List[Self]) -> Self:
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
        array_out = np.zeros(
            (len(geotensors),) + first_geotensor.shape, dtype=first_geotensor.dtype
        )
        array_out[0] = first_geotensor.values

        for i, geo in enumerate(geotensors[1:]):
            assert geo.same_extent(first_geotensor), f"Different size in concat {i+1}"
            assert (
                geo.shape == first_geotensor.shape
            ), f"Different shape in concat {i+1}"
            assert (
                geo.fill_value_default == first_geotensor.fill_value_default
            ), "Different fill_value_default in concat"
            array_out[i + 1] = geo.values

        return cls(
            array_out,
            transform=first_geotensor.transform,
            crs=first_geotensor.crs,
            fill_value_default=first_geotensor.fill_value_default,
            attrs=first_geotensor.attrs,
        )

    @classmethod
    def concatenate(cls, geotensors: List[Self], axis: int = 0) -> Self:
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
        assert (
            axis < len(first_geotensor.shape) - 2
        ), f"Can't concatenate along spatial axis"

        for i, geo in enumerate(geotensors[1:]):
            assert geo.same_extent(first_geotensor), f"Different extent in concat {i+1}"
            assert (
                geo.shape == first_geotensor.shape
            ), f"Different shape in concat {i+1}"
            assert (
                geo.fill_value_default == first_geotensor.fill_value_default
            ), "Different fill_value_default in concat"

        array_out = np.concatenate([gt.values for gt in geotensors], axis=axis)

        return cls(
            array_out,
            transform=first_geotensor.transform,
            crs=first_geotensor.crs,
            fill_value_default=first_geotensor.fill_value_default,
            attrs=first_geotensor.attrs,
        )


concatenate = GeoTensor.concatenate
stack = GeoTensor.stack
