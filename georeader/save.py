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
    Save data GeoData object as tiled GeoTIFF (see `save_cog` to save as a Cloud Optimized GeoTIFF)

    Args:
        data_save: GeoData (C, H, W) format with geoinformation (crs and transform).
        path_tiff_save: path to save the GeoTIFF
        profile_arg: profile dict to save the data. crs and transform will be updated from data_save.
        descriptions: name of the bands
        profile: profile dict to save the data. crs and transform will be updated from data_save.
        tags: Dict to save as tags of the image
        dir_tmpfiles: dir to create tempfiles if needed
        blocksize: blocksize of the GeoTIFF
        fs: fsspec filesystem to save the file

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
    Save data GeoData object as cloud optimized GeoTIFF

    Args:
        data_save: GeoData (C, H, W) format with geoinformation (crs and transform).
        descriptions: name of the bands
        path_tiff_save: path to save the COG GeoTIFF
        profile: profile dict to save the data. crs and transform will be updated from data_save.
        tags: Dict to save as tags of the image
        dir_tmpfiles: dir to create tempfiles if needed
        fs: fsspec filesystem to save the file
    
        
    Examples:
        >> img = np.random.randn(4,256,256)
        >> transform = rasterio.Affine(10, 0, 799980.0, 0, -10, 1900020.0)
        >> data = GeoTensor(img, crs="EPSG:32644", transform=transform)
        >> save_cog(data, "example.tif", descriptions=["band1", "band2", "band3", "band4"])

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
