# GeoTensor

The `GeoTensor` class is a subclass of `np.ndarray` that stores the spatial affine transform, the coordinate reference system (`crs`) and no data value (called `fill_value_default`). When a `GeoTensor` is sliced, its `transform` attribute is shifted accordingly. Additionally its transform is also shifted if the `GeoTensor` is padded. `GeoTensor`s are restricted to be 2D, 3D or 4D arrays and their two last dimensions are assumed to be the `y` and `x` spatial axis.

`GeoTensor` implements the [GeoData protocol](../modules/read_module.md#geodata-protocol). This makes it fully compatible with all [`read_*` methods](../modules/read_module.md#read-methods) in the library, allowing for operations like reprojection, subseting from spatial coordinates, mosaicking or vectorization.

For a detailed guide on working with the NumPy API in GeoTensors, see the [GeoTensor NumPy API tutorial](../geotensor_numpy_api.ipynb) which covers slicing, operations, masking, and more practical examples.

As a subclass of `np.ndarray`, operations with `GeoTensor` objects work similar than operations with `np.ndarray`s. However, there are some restrictions that we have implemented to keep consistency with the `GeoTensor` concept. If you need to use the `numpy` implementation you can access the bare `numpy` object with the `.values` attribute. Below there's a list with restrictions on `numpy` operations:

1. Slicing a `GeoTensor` is more restrictive than a `numpy` array. It only allows to slice with `lists`, numbers or `slice`s. In particular the spatial dimensions can only be sliced with `slice`s. Slicing for inplace modification is not restricted (i.e. you can slice with boolean arrays to modify certain values of the object). See [isel](#georeader.geotensor.GeoTensor.isel) and [__getitem__](#georeader.geotensor.GeoTensor.__getitem__) methods.

2. Binary operations (such as add [__add__](#georeader.geotensor.GeoTensor.__add__), multiply [__mul__](#georeader.geotensor.GeoTensor.__mul__), [==](#georeader.geotensor.GeoTensor.__eq__), [|](#georeader.geotensor.GeoTensor.__or__) etc) check, for `GeoTensor` inputs, if they have the [same_extent](#georeader.geotensor.GeoTensor.same_extent); that is, same `transform` `crs` and spatial dimensions (`width` and `height`).

3. [squeeze](#georeader.geotensor.GeoTensor.squeeze), [expand_dims](#georeader.geotensor.GeoTensor.expand_dims) and [transpose](#georeader.geotensor.GeoTensor.transpose) make sure spatial dimensions (last two axes) are not modified and kept at the end of the array.

4. [concatenate](#georeader.geotensor.GeoTensor.concatenate) and [stack](#georeader.geotensor.GeoTensor.stack) make sure all operated `GeoTensor`s have `same_extent` and `shape`. `concatenate` does not allow to concatenate on the spatial dims.

5. Reductions (such as `np.mean` or `np.all`) return `GeoTensor` object if the spatial dimensions are preserved and `np.ndarray` or scalars otherwise. This is handled by the [__array_ufunc__](#georeader.geotensor.GeoTensor.__array_ufunc__) method.

## Additional Features

- **Masking utilities**: Methods like [validmask()](#georeader.geotensor.GeoTensor.validmask) and [invalidmask()](#georeader.geotensor.GeoTensor.invalidmask) create boolean masks based on the `fill_value_default`.
- **Window-based access**: [read_from_window()](#georeader.geotensor.GeoTensor.read_from_window) and [write_from_window()](#georeader.geotensor.GeoTensor.write_from_window) provide window-based operations using rasterio's window system.
- **File I/O**: Class methods [load_file()](#georeader.geotensor.GeoTensor.load_file) and [load_bytes()](#georeader.geotensor.GeoTensor.load_bytes) for easily loading GeoTensors from files or memory.
- **Footprint extraction**: Methods to extract the valid data footprint as vector geometries ([footprint()](#georeader.geotensor.GeoTensor.footprint) and [valid_footprint()](#georeader.geotensor.GeoTensor.valid_footprint)).
- **Coordinate transformation**: The [meshgrid()](#georeader.geotensor.GeoTensor.meshgrid) method creates coordinate arrays for the spatial dimensions.
- **Resizing**: The [resize()](#georeader.geotensor.GeoTensor.resize) method changes the spatial resolution while maintaining geospatial information correct (i.e. changing the spatial resolution of the `transform`).
- **Metadata storage**: GeoTensor includes an `attrs` dictionary for storing additional metadata like tags and band descriptions.


## API Reference

::: georeader.geotensor