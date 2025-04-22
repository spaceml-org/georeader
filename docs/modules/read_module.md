# Geospatial Data Reading and Manipulation

## Protocols

The `georeader` package uses two main protocols to define interfaces for geospatial data:

### GeoDataBase Protocol

This is the minimal interface required for geospatial operations. Any class implementing this protocol provides basic spatial information:

- `transform`: A rasterio.Affine object defining the spatial transform
- `crs`: Coordinate reference system
- `shape`: The shape of the data array
- `width`: Width of the data (shape[-1])
- `height`: Height of the data (shape[-2])

### GeoData Protocol

This extends the `GeoDataBase` protocol with methods for data access and manipulation:

- All properties from `GeoDataBase`
- `load(boundless: bool = True) -> GeoTensor`: Loads data into memory
- `read_from_window(window, boundless) -> Union[Self, GeoTensor]`: Reads data from a window
- `values`: Returns the data array
- `res`: Resolution (tuple of x and y resolution)
- `dtype`: Data type
- `dims`: Dimension names
- `fill_value_default`: Fill value for missing data
- `bounds`: Data bounds
- `footprint(crs: Optional[str] = None) -> Polygon`: Returns the footprint as a polygon

## Implementations

The library provides two main implementations of these protocols:

1. **[GeoTensor](../modules/geotensor_module.md)**: A numpy-based implementation for in-memory operations.
2. **[RasterioReader](../modules/rasterio_reader.md)**: An implementation for lazy-loading with `rasterio`.

## Window and Read Methods

The API provides two types of methods:

### Window Methods

These methods work with any object implementing the `GeoDataBase` protocol. They calculate [`rasterio.windows`](https://rasterio.readthedocs.io/en/stable/api/rasterio.windows.html) objects without reading any data:

- [`window_from_bounds`](#georeader.read.window_from_bounds): Creates a window to read from the raster from geographic bounds
- [`window_from_center_coords`](#georeader.read.window_from_center_coords): Creates a window  to read from the raster  centered on specific coordinates.
- [`window_from_polygon`](#georeader.read.window_from_polygon): Creates a window  to read from the raster that contains a polygon
- [`window_from_tile`](#georeader.read.window_from_tile): Creates a window  to read from the raster from X/Y/Z Web Mercator tiles.

### Read Methods

These methods require objects implementing the `GeoData` protocol. They load and transform data:

- [`read_from_center_coords`](#georeader.read.read_from_center_coords): Reads data centered on specific coordinates
- [`read_from_bounds`](#georeader.read.read_from_bounds): Reads data within geographic bounds
- [`read_from_polygon`](#georeader.read.read_from_polygon): Reads data within a polygon's boundaries
- [`read_from_tile`](#georeader.read.read_from_tile): Reads data from X/Y/Z Web Mercator tiles.
- [`read_to_crs`](#georeader.read.read_to_crs): Reads data and reprojects to a different coordinate reference system
- [`read_reproject_like`](#georeader.read.read_reproject_like): Reprojects data to match spatial extent and shape of another GeoData object.
- [`resize`](#georeader.read.resize): Changes the spatial resolution of the data.
- [`read_reproject`](#georeader.read.read_reproject): Low-level function for arbitrary reprojection.
- [`read_rpcs`](#georeader.read.read_rpcs): Georeferences data using rational polynomial coefficients.
- [`spatial_mosaic`](#georeader.mosaic.spatial_mosaic): Creates a spatial mosaic by combining spatially multiple GeoData objects.

## API Reference

### Read Methods

::: georeader.read
    options:
      members:
        - read_from_center_coords
        - read_from_bounds
        - read_from_polygon
        - read_from_window
        - read_from_tile
        - read_to_crs
        - read_reproject_like
        - resize
        - read_reproject
        - read_rpcs

::: georeader.mosaic
    options:
      members:
        - spatial_mosaic

### Window Methods

::: georeader.read
    options:
      members:
        - read_from_center_coords
        - window_from_bounds
        - window_from_center_coords
        - window_from_polygon
        - window_from_tile