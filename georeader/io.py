"""
I/O utilities for georeader package.
"""
import logging
from typing import Optional, Union
from io import IOBase

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

logger = logging.getLogger(__name__)


def is_url(path) -> bool:
    """Check if the given path is a URL."""
    if not isinstance(path, str):
        return False
    return path.startswith(("http://", "https://", "ftp://"))


def safe_open_netcdf(
    file_path_or_object: Union[str, IOBase],
    cache: bool = False,
    load: bool = True,
    group: Optional[str] = None,
    **kwargs,
) -> "xr.Dataset":
    """
    Safely open a NetCDF file by trying multiple engines in order.
    
    This function attempts to open a NetCDF file using different xarray backends,
    cycling through them until one succeeds. This is useful for handling different
    NetCDF formats and file-like objects (e.g., from Azure Blob Storage).
    
    The order of engines tried depends on the input type:
    - For URLs (OPeNDAP): netcdf4 only (the only engine supporting remote OPeNDAP)
    - For local file paths and file-like objects: h5netcdf, scipy, netcdf4
    
    Args:
        file_path_or_object: Path to the NetCDF file, URL (OPeNDAP), or a file-like object.
        cache (bool, optional): Whether to cache the file in memory. Defaults to False.
        load (bool, optional): Whether to load the data into memory. Defaults to True.
        group (str, optional): NetCDF4 group to open. Defaults to None (root group).
        **kwargs (Any): Additional keyword arguments passed to xr.open_dataset.
    
    Returns:
        xr.Dataset: The opened xarray Dataset with data loaded into memory (if load=True).
    
    Raises:
        ImportError: If xarray is not installed.
        IOError: If all engines fail to open the file.
    
    Example:
        >>> ds = safe_open_netcdf("path/to/file.nc")
        >>> ds = safe_open_netcdf(azure_blob_file_object)
        >>> ds = safe_open_netcdf("https://opendap.server/data.nc")
        >>> ds_location = safe_open_netcdf("path/to/file.nc", group="location")
    """
    if not HAS_XARRAY:
        raise ImportError("xarray is required to use safe_open_netcdf. Please install it with: pip install xarray")
    
    # Get a string representation for logging
    if isinstance(file_path_or_object, str):
        file_description = file_path_or_object
    else:
        file_description = f"file-like object ({type(file_path_or_object).__name__})"
    
    # Check if input is a URL (OPeNDAP endpoint)
    is_remote_url = is_url(file_path_or_object)
    
    # Check if input is a file-like object (not a string path)
    is_file_like = hasattr(file_path_or_object, 'read')
    
    # Determine which engines to try based on input type
    if is_remote_url:
        # For OPeNDAP URLs, only netcdf4 engine supports remote access
        engines = ["netcdf4"]
    else:
        # Try h5netcdf first (often faster/better for cloud), then scipy (NetCDF3), 
        # then netcdf4 (comprehensive). All support local paths and file-like objects.
        engines = ["h5netcdf", "scipy", "netcdf4"]
    
    errors = []
    
    for engine in engines:
        try:
            # Reset file position if it's a file-like object
            if is_file_like and hasattr(file_path_or_object, 'seek'):
                try:
                    file_path_or_object.seek(0)
                except Exception:
                    pass  # Some file objects may not support seek
            
            ds = xr.open_dataset(file_path_or_object, cache=cache, engine=engine, group=group, **kwargs)
            if load:
                ds = ds.load()
            logger.debug(f"Successfully opened NetCDF file with engine: {engine}")
            return ds
        except Exception as e:
            error_msg = f"{engine}: {type(e).__name__}: {str(e)[:200]}"
            errors.append(error_msg)
            logger.debug(f"Failed to open with engine {engine}: {e}")
            continue
    
    # All engines failed
    error_summary = "; ".join(errors)
    raise IOError(
        f"Failed to open NetCDF file '{file_description}' with all available engines ({', '.join(engines)}). "
        f"Errors: {error_summary}"
    )
