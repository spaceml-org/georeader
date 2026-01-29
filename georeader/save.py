"""
Save Module: Export GeoTensor data to geospatial file formats.

This module provides functions to save GeoTensor and GeoData objects to
various raster file formats, with special support for Cloud Optimized GeoTIFFs
(COGs) - the standard format for cloud-native geospatial data.

Cloud Optimized GeoTIFF (COG) Overview
--------------------------------------

COGs are GeoTIFFs organized for efficient HTTP range requests::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │              CLOUD OPTIMIZED GEOTIFF STRUCTURE                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Traditional GeoTIFF            Cloud Optimized GeoTIFF (COG)           │
    │  ────────────────────           ─────────────────────────────           │
    │                                                                          │
    │  ┌─────────────────┐            ┌─────────────────┐                     │
    │  │ Header (end)    │            │ Header (start)  │ ← HTTP range 0-N   │
    │  │                 │            │ ─────────────── │                     │
    │  │                 │            │ Overview 8x     │ ← Pyramid layers   │
    │  │    Full        │            │ Overview 4x     │   for fast zoom    │
    │  │    Resolution  │            │ Overview 2x     │                     │
    │  │    Data        │            │ ─────────────── │                     │
    │  │                 │            │ ┌───┬───┬───┐  │ ← Tiled structure  │
    │  │                 │            │ │256│256│256│  │   (default 256x256)│
    │  │                 │            │ ├───┼───┼───┤  │                     │
    │  │                 │            │ │256│256│256│  │                     │
    │  └─────────────────┘            │ └───┴───┴───┘  │                     │
    │                                 └─────────────────┘                     │
    │                                                                          │
    │  Benefits:                                                               │
    │  • Read any region without downloading whole file                       │
    │  • Fast preview via overviews (pyramid layers)                          │
    │  • Efficient streaming for web mapping applications                     │
    │  • Compatible with all GeoTIFF readers                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Compression Options
-------------------

Choose compression based on data type::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    COMPRESSION RECOMMENDATIONS                           │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Compression    Best For                    Speed    Size                │
    │  ───────────    ────────                    ─────    ────                │
    │  lzw           General purpose (DEFAULT)    Fast     Medium             │
    │  deflate       Better compression ratio     Medium   Small              │
    │  zstd          Modern, fast + small         Fast     Small              │
    │  lzma          Maximum compression          Slow     Smallest           │
    │  jpeg          RGB imagery (lossy)          Fast     Tiny               │
    │  webp          RGB imagery (lossy/lossless) Fast     Tiny               │
    │  none          Already compressed data      N/A      Original           │
    │                                                                          │
    │  For scientific data: Use lzw or zstd (lossless)                        │
    │  For visualization: Consider jpeg or webp (lossy ok)                    │
    │  For integer masks: Use deflate with predictor=2                        │
    └─────────────────────────────────────────────────────────────────────────┘

Data Type Mapping
-----------------

NumPy to GeoTIFF dtype mapping::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  NumPy dtype     GeoTIFF dtype    Notes                                 │
    │  ────────────    ─────────────    ─────                                 │
    │  uint8           Byte             8-bit unsigned (0-255)                │
    │  uint16          UInt16           16-bit unsigned (0-65535)             │
    │  int16           Int16            16-bit signed                         │
    │  uint32          UInt32           32-bit unsigned                       │
    │  int32           Int32            32-bit signed                         │
    │  float32         Float32          32-bit floating point                 │
    │  float64         Float64          64-bit floating point                 │
    │  complex64       CFloat32         Complex 32-bit                        │
    │  bool            Byte             Converted to 0/1                      │
    └─────────────────────────────────────────────────────────────────────────┘

Module Functions Overview
-------------------------

Primary Functions:
    - :func:`save_cog`: Save as Cloud Optimized GeoTIFF with overviews
    - :func:`save_tiled_geotiff`: Save as tiled GeoTIFF (no overviews)

Format Detection:
    - :func:`profile_from_dtype`: Get rasterio profile for numpy dtype
    - :func:`profile_dtype_cog`: Default COG profile

Cloud Storage Support:
    - Supports gs://, s3://, az://, abfs://, oss:// paths
    - Automatically handles temp file creation for remote writes
    - Pass `fs` parameter for custom filesystem (e.g., gcsfs, s3fs)

Quick Start
-----------

Save a GeoTensor as COG::

    from georeader import save
    from georeader.geotensor import GeoTensor
    import numpy as np
    from rasterio.transform import from_bounds

    # Create example data
    data = np.random.rand(3, 256, 256).astype(np.float32)
    transform = from_bounds(-122.5, 37.0, -122.0, 37.5, 256, 256)
    gt = GeoTensor(data, transform=transform, crs="EPSG:4326")

    # Save as COG (creates overviews automatically)
    save.save_cog(gt, "output.tif")

Save with band descriptions and custom compression::

    save.save_cog(
        gt,
        "output.tif",
        descriptions=["Red", "Green", "Blue"],
        profile_arg={"compress": "zstd"},
        tags={"source": "Sentinel-2", "date": "2024-01-15"}
    )

See Also
--------
georeader.geotensor : GeoTensor class (input to save functions)
georeader.rasterio_reader : Reading saved files back
rasterio.profiles : Profile documentation

References
----------
- COG specification: https://www.cogeo.org/
- Rasterio profiles: https://rasterio.readthedocs.io/en/latest/topics/profiles.html
- GDAL COG driver: https://gdal.org/drivers/raster/cog.html
"""
import rasterio
import rasterio.rio.overview
import rasterio.shutil as rasterio_shutil
import os
import tempfile
import numpy as np
from georeader.abstract_reader import AbstractGeoData
from georeader.geotensor import GeoTensor
from typing import Optional, List, Union, Dict, Any
import time


GeoData = Union[AbstractGeoData, GeoTensor]
REMOTE_FILE_EXTENSIONS = ["gs://", "s3://", "az://", "http://", "https://", "abfs://", "oss://"]

BLOCKSIZE_DEFAULT = 256
PROFILE_TILED_GEOTIFF_DEFAULT = {
    "compress": "lzw",
    "BIGTIFF": "IF_SAFER",
    "blockxsize": BLOCKSIZE_DEFAULT,
    "blockysize": BLOCKSIZE_DEFAULT
}

def save_tiled_geotiff(data_save:GeoData, path_tiff_save:str,
                       profile_arg:Optional[Dict[str, Any]]=None,
                       descriptions:Optional[List[str]] = None,
                       tags:Optional[Dict[str, Any]]=None,
                       dir_tmpfiles:str=".",
                       blocksize:int=BLOCKSIZE_DEFAULT,
                       fs:Optional[Any]=None) -> None:
    """
    Save a GeoData object as a tiled GeoTIFF file.

    Tiled GeoTIFFs store data in blocks (tiles) rather than strips, enabling
    efficient random access to subregions. This is the standard format for
    large rasters but does NOT include overviews. For cloud-optimized files
    with overviews, use :func:`save_cog` instead.

    The function handles both local and cloud storage paths (gs://, s3://, az://).
    For remote paths, data is written to a local temp file first, then uploaded.

    Args:
        data_save (GeoData): Raster data in (C, H, W) or (H, W) format with
            geospatial metadata (crs and transform). Can be a GeoTensor or
            any object implementing the AbstractGeoData protocol.
        path_tiff_save (str): Output file path. Supports local paths and cloud
            storage URIs: gs://, s3://, az://, abfs://, oss://.
        profile_arg (Optional[Dict[str, Any]]): Rasterio profile options to
            customize output. Common options:

            - ``compress``: Compression method ('lzw', 'deflate', 'zstd', 'none').
            - ``dtype``: Output data type (auto-detected if not provided).
            - ``nodata``: NoData value (uses fill_value_default if not set).

            The crs and transform are always set from data_save.
        descriptions (Optional[List[str]]): Band names/descriptions. Length must
            match number of bands. Shows in GIS software band properties.
        tags (Optional[Dict[str, Any]]): Metadata tags stored in the TIFF.
            Example: ``{"source": "Sentinel-2", "date": "2024-01-15"}``.
        dir_tmpfiles (str): Directory for temporary files when writing to
            cloud storage. Defaults to current directory.
        blocksize (int): Tile size in pixels (both width and height).
            Defaults to 256. Common values: 256, 512. Must be power of 2.
        fs (Optional[Any]): fsspec filesystem object for cloud storage.
            Auto-detected from path prefix if not provided.

    Returns:
        None: File is written to disk/cloud.

    Raises:
        NotImplementedError: If data_save has dimensions other than 2 or 3.
        AssertionError: If descriptions length doesn't match band count.
        FileNotFoundError: If temp file creation fails for remote writes.

    Examples:
        Save a simple GeoTensor to local file:

        >>> import numpy as np
        >>> from georeader.geotensor import GeoTensor
        >>> from georeader import save
        >>> from rasterio.transform import from_bounds
        >>>
        >>> # Create sample data
        >>> data = np.random.rand(3, 512, 512).astype(np.float32)
        >>> transform = from_bounds(-122.5, 37.0, -122.0, 37.5, 512, 512)
        >>> gt = GeoTensor(data, transform=transform, crs="EPSG:4326")
        >>>
        >>> # Save with default settings
        >>> save.save_tiled_geotiff(gt, "output.tif")

        Save with custom compression and band names:

        >>> save.save_tiled_geotiff(
        ...     gt,
        ...     "output_custom.tif",
        ...     profile_arg={"compress": "zstd"},
        ...     descriptions=["Red", "Green", "Blue"],
        ...     blocksize=512
        ... )

        Save to Google Cloud Storage:

        >>> save.save_tiled_geotiff(
        ...     gt,
        ...     "gs://my-bucket/rasters/output.tif",
        ...     dir_tmpfiles="/tmp"  # Use system temp for cloud uploads
        ... )

    See Also:
        save_cog: For Cloud Optimized GeoTIFFs with overviews (recommended
            for most use cases).

    Note:
        - Tiled GeoTIFFs without overviews are fine for processing workflows
        - For visualization or web serving, use save_cog for better performance
        - Block size should match your typical read access patterns
    """
    profile = PROFILE_TILED_GEOTIFF_DEFAULT.copy()
    profile.update({"blockxsize": blocksize, "blockysize": blocksize})
    if profile_arg is not None:
        profile.update(profile_arg)
    
    if len(data_save.shape) == 3:
        out_np = np.asanyarray(data_save.values)
    elif len(data_save.shape) == 2:
        out_np = np.asanyarray(data_save.values[np.newaxis])
    else:
        raise NotImplementedError(f"Expected data with 2 or 3 dimensions found: {data_save.shape}")

    profile["crs"] = data_save.crs
    profile["transform"] = data_save.transform

    if "nodata" not in profile:
        profile["nodata"] = data_save.fill_value_default

    if descriptions is not None:
        assert len(descriptions) == out_np.shape[0], f"Unexpected band descriptions {len(descriptions)} expected {out_np.shape[0]}"

    # Set count, height, width
    for idx, c in enumerate(["count", "height", "width"]):
        if c in profile:
            assert profile[c] == out_np.shape[idx], f"Unexpected shape: {profile[c]} {out_np.shape}"
        else:
            profile[c] = out_np.shape[idx]

    if "dtype" not in profile:
        profile["dtype"] = str(out_np.dtype)

    # check blocksize
    for idx, b in enumerate(["blockysize", "blockxsize"]):
        if b in profile:
            profile[b] = min(profile[b], out_np.shape[idx + 1])

    if (out_np.shape[1] > profile["blockysize"]) or (out_np.shape[2] > profile["blockxsize"]):
        profile["tiled"] = True

    profile["driver"] = "GTiff"
    is_remote_file = any((path_tiff_save.startswith(ext) for ext in REMOTE_FILE_EXTENSIONS))

    # Create a tempfile if is a remote file
    if is_remote_file:
        with tempfile.NamedTemporaryFile(dir=dir_tmpfiles, suffix=".tif", delete=True) as fileobj:
            name_save = fileobj.name
    else:
        name_save = path_tiff_save

    with rasterio.open(name_save, "w", **profile) as rst_out:
        if tags is not None:
            rst_out.update_tags(**tags)
        rst_out.write(out_np)
        if descriptions is not None:
            for i in range(1, out_np.shape[0] + 1):
                rst_out.set_band_description(i, descriptions[i - 1])
    
    if is_remote_file:
        if fs is None:
            import fsspec
            fs = fsspec.filesystem(path_tiff_save.split(":")[0])
        
        if not os.path.exists(name_save):
            raise FileNotFoundError(f"File {name_save} have not been created")
        
        fs.put_file(name_save, path_tiff_save, overwrite=True)
        if os.path.exists(name_save):
            os.remove(name_save)


def save_cog(data_save:GeoData, path_tiff_save:str,
             profile:Optional[Dict[str, Any]]=None,
             descriptions:Optional[List[str]] = None, 
             tags:Optional[Dict[str, Any]]=None,
             dir_tmpfiles:str=".",
             fs:Optional[Any]=None) -> None:
    """
    Save a GeoData object as a Cloud Optimized GeoTIFF (COG).

    COGs are the recommended format for cloud-native geospatial workflows. They
    include:

    - **Internal tiling**: Efficient random access to subregions
    - **Overviews (pyramids)**: Fast rendering at multiple zoom levels
    - **Optimized header placement**: Enables HTTP range requests

    This function automatically generates overviews using cubic spline
    resampling, which produces smooth results for continuous data.

    Args:
        data_save (GeoData): Raster data in (C, H, W) or (H, W) format with
            geospatial metadata (crs and transform). Accepts GeoTensor or any
            AbstractGeoData implementation.
        path_tiff_save (str): Output file path. Supports local paths and cloud
            storage URIs (gs://, s3://, az://, abfs://, oss://).
        profile (Optional[Dict[str, Any]]): Rasterio profile options. Common:

            - ``compress``: Compression ('lzw', 'deflate', 'zstd'). Default: 'lzw'.
            - ``RESAMPLING``: Overview resampling method. Default: 'CUBICSPLINE'.
            - ``dtype``: Output dtype (auto-detected from data if not set).
            - ``nodata``: NoData value (uses fill_value_default if not set).

            CRS and transform are always taken from data_save.
        descriptions (Optional[List[str]]): Band names shown in GIS software.
            Length must equal number of bands in data_save.
        tags (Optional[Dict[str, Any]]): Metadata tags embedded in the TIFF.
            Example: ``{"source": "Sentinel-2", "acquisition_date": "2024-01-15"}``.
        dir_tmpfiles (str): Temporary file directory for cloud storage writes.
            Defaults to current directory.
        fs (Optional[Any]): fsspec filesystem object for cloud storage.
            Auto-detected from path prefix if not provided.

    Returns:
        None: File is written to disk/cloud storage.

    Raises:
        NotImplementedError: If data_save has dimensions other than 2 or 3.
        AssertionError: If descriptions length doesn't match band count.

    Examples:
        Basic COG creation:

        >>> import numpy as np
        >>> from georeader.geotensor import GeoTensor
        >>> from georeader import save
        >>> import rasterio
        >>>
        >>> # Create 4-band raster
        >>> img = np.random.randn(4, 256, 256).astype(np.float32)
        >>> transform = rasterio.Affine(10, 0, 799980.0, 0, -10, 1900020.0)
        >>> data = GeoTensor(img, crs="EPSG:32644", transform=transform)
        >>>
        >>> # Save as COG with band descriptions
        >>> save.save_cog(
        ...     data,
        ...     "example.tif",
        ...     descriptions=["band1", "band2", "band3", "band4"]
        ... )

        Save with high compression for archival:

        >>> save.save_cog(
        ...     data,
        ...     "archived.tif",
        ...     profile={"compress": "zstd", "ZSTD_LEVEL": 9},
        ...     tags={"project": "climate-analysis", "version": "1.0"}
        ... )

        Save classification result with nearest-neighbor overviews:

        >>> classification = GeoTensor(
        ...     np.random.randint(0, 10, (256, 256), dtype=np.uint8),
        ...     transform=transform, crs="EPSG:32644"
        ... )
        >>> save.save_cog(
        ...     classification,
        ...     "landcover.tif",
        ...     profile={"compress": "deflate", "RESAMPLING": "NEAREST"},
        ...     descriptions=["Land Cover Class"]
        ... )

    See Also:
        save_tiled_geotiff: For tiled GeoTIFF without overviews (faster write,
            smaller file, but slower visualization).

    Note:
        - COG creation is slower than save_tiled_geotiff due to overview generation
        - Use 'NEAREST' resampling for categorical/classified data
        - Use 'CUBICSPLINE' or 'LANCZOS' for continuous imagery
        - Files can be directly served via HTTP range requests (e.g., via STAC)
    """
    if profile is None:
        profile = {
            "compress": "lzw",
            "RESAMPLING": "CUBICSPLINE",  # for pyramids
        }
    if len(data_save.shape) == 3:
        np_data = np.asanyarray(data_save.values)
    elif len(data_save.shape) == 2:
        np_data = np.asanyarray(data_save.values[np.newaxis])
    else:
        raise NotImplementedError(f"Expected data with 2 or 3 dimensions found: {data_save.shape}")

    profile["crs"] = data_save.crs
    profile["transform"] = data_save.transform

    if "nodata" not in profile:
        profile["nodata"] = data_save.fill_value_default

    _save_cog(np_data,
              path_tiff_save, profile, descriptions=descriptions,
              tags=tags, dir_tmpfiles=dir_tmpfiles, fs=fs)

def _add_overviews(rst_out, tile_size, verbose=False):
    """ Add overviews to be a cog and be displayed nicely in GIS software """

    overview_level = rasterio.rio.overview.get_maximum_overview_level(*rst_out.shape, tile_size)
    overviews = [2 ** j for j in range(1, overview_level + 1)]

    if verbose:
        print(f"Adding pyramid overviews to raster {overviews}")

    # Copied from https://github.com/cogeotiff/rio-cogeo/blob/master/rio_cogeo/cogeo.py#L274
    rst_out.build_overviews(overviews, rasterio.warp.Resampling.average)
    rst_out.update_tags(ns='rio_overview', resampling='nearest')
    tags = rst_out.tags()
    tags.update(OVR_RESAMPLING_ALG="NEAREST")
    rst_out.update_tags(**tags)
    rst_out._set_all_scales([rst_out.scales[b - 1] for b in rst_out.indexes])
    rst_out._set_all_offsets([rst_out.offsets[b - 1] for b in rst_out.indexes])


def _save_cog(out_np: np.ndarray, path_tiff_save: str, profile: dict,
             descriptions:Optional[List[str]] = None,
             tags: Optional[dict] = None,
             dir_tmpfiles:str=".",
             requester_pays:bool=False,
             fs:Optional[Any]=None):
    """
    Saves `out_np` np array as a COG GeoTIFF in path_tiff_save. profile is a dict with the geospatial info to be saved
    with the TiFF.

    Args:
        out_np: 3D numpy array to save in CHW format
        path_tiff_save:
        profile: dict with profile to write geospatial info of the dataset: (crs, transform)
        descriptions: List[str]
        tags: extra dict to save as tags
        dir_tmpfiles: dir to create tempfiles if needed
        requester_pays: if True and the path is in a cloud bucket it will initialize fsspec with requester_pays=True
        fs: fsspec filesystem to save the file

    Returns:
        None

    Examples:
        >> img = np.random.randn(4,256,256)
        >> transform = rasterio.Affine(10, 0, 799980.0, 0, -10, 1900020.0)
        >> _save_cog(img, "example.tif", {"crs": {"init": "epsg:32644"}, "transform":transform})
    """

    assert len(out_np.shape) == 3, f"Expected 3d tensor found tensor with shape {out_np.shape}"
    if descriptions is not None:
        assert len(descriptions) == out_np.shape[0], f"Unexpected band descriptions {len(descriptions)} expected {out_np.shape[0]}"

    # Set count, height, width
    for idx, c in enumerate(["count", "height", "width"]):
        if c in profile:
            assert profile[c] == out_np.shape[idx], f"Unexpected shape: {profile[c]} {out_np.shape}"
        else:
            profile[c] = out_np.shape[idx]

    for field in ["crs", "transform"]:
        assert field in profile, f"{field} not in profile: {profile}. it will not write cog without geo information"

    profile["BIGTIFF"] = "IF_SAFER"
    if "dtype" not in profile:
        profile["dtype"] = str(out_np.dtype)
    
    with rasterio.Env() as env:
        cog_driver = "COG" in env.drivers()

    if "RESAMPLING" not in profile:
        profile["RESAMPLING"] = "CUBICSPLINE"  # for pyramids

    if cog_driver:
        assert ("blockxsize" not in profile) and ("blockysize" not in profile), "In COG driver blockxsize and blockysize options are BLOCKSIZE"
        # Save tiff locally and copy it to GCP with fsspec is path is a GCP path
        is_remote_file = any((path_tiff_save.startswith(ext) for ext in REMOTE_FILE_EXTENSIONS))
        if is_remote_file:
            with tempfile.NamedTemporaryFile(dir=dir_tmpfiles, suffix=".tif", delete=True) as fileobj:
                name_save = fileobj.name
        else:
            name_save = path_tiff_save
        profile["driver"] = "COG"
        with rasterio.open(name_save, "w", **profile) as rst_out:
            if tags is not None:
                rst_out.update_tags(**tags)
            rst_out.write(out_np)
            if descriptions is not None:
                for i in range(1, out_np.shape[0] + 1):
                    rst_out.set_band_description(i, descriptions[i-1])

        if is_remote_file:
            if fs is None:
                import fsspec
                fs = fsspec.filesystem(path_tiff_save.split(":")[0], 
                                       requester_pays=requester_pays)
            if not os.path.exists(name_save):
                raise FileNotFoundError(f"File {name_save} have not been created")
            fs.put_file(name_save, path_tiff_save, overwrite=True)
            # subprocess.run(["gsutil", "-m", "mv", name_save, path_tiff_save])
            if os.path.exists(name_save):
                os.remove(name_save)

        return path_tiff_save

    print("COG driver not available. Generate COG manually with GTiff driver")
    # If COG driver is not available (GDAL < 3.1) we go to copying the file using GTiff driver
    # Set blockysize, blockxsize
    for idx, b in enumerate(["blockysize", "blockxsize"]):
        if b in profile:
            assert profile[b] <= 512, f"{b} is {profile[b]} must be <=512 to be displayed in GEE "
        else:
            profile[b] = min(512, out_np.shape[idx + 1])

    if (out_np.shape[1] >= 512) or (out_np.shape[2] >= 512):
        profile["tiled"] = True

    profile["driver"] = "GTiff"
    with tempfile.NamedTemporaryFile(dir=dir_tmpfiles, suffix=".tif", delete=True) as fileobj:
        named_tempfile = fileobj.name

    with rasterio.open(named_tempfile, "w", **profile) as rst_out:
        if tags is not None:
            rst_out.update_tags(**tags)
        rst_out.write(out_np)
        if descriptions is not None:
            for i in range(1, out_np.shape[0] + 1):
                rst_out.set_band_description(i, descriptions[i - 1])
        
        _add_overviews(rst_out, tile_size=profile["blockysize"])
        print("Copying temp file")
        rasterio_shutil.copy(rst_out, path_tiff_save, copy_src_overviews=True, tiled=True,
                             blockxsize=profile["blockxsize"],
                             blockysize=profile["blockysize"],
                             driver="GTiff")

    rasterio_shutil.delete(named_tempfile)
    return path_tiff_save
