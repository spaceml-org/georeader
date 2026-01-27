# I/O Module

The `georeader.io` module provides I/O utilities for handling various geospatial file formats,
with a focus on robust file opening that handles different backends and data sources.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              georeader.io                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Functions:                                                                 │
│  ├── safe_open_netcdf()  Multi-engine NetCDF/HDF5 opener                   │
│  └── is_url()            Check if path is a remote URL                     │
│                                                                             │
│  Supported Data Sources:                                                    │
│  ├── Local file paths                                                       │
│  ├── File-like objects (Azure Blob, S3, etc.)                              │
│  └── OPeNDAP URLs for remote data access                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why safe_open_netcdf?

NetCDF files come in different formats (NetCDF3, NetCDF4/HDF5, compressed variants), and 
not all xarray backends handle all formats equally. The `safe_open_netcdf` function tries
multiple engines in sequence until one succeeds:

```
Engine Selection Logic:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Input Type          │ Engines Tried (in order)                            │
│ ─────────────────────┼───────────────────────────────────────────────────── │
│  OPeNDAP URL         │ netcdf4 only (remote protocol support)              │
│  Local path          │ h5netcdf → scipy → netcdf4                          │
│  File-like object    │ h5netcdf → scipy → netcdf4                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Why this order?                                                            │
│  • h5netcdf: Often faster, pure-Python, good cloud storage support         │
│  • scipy: Handles classic NetCDF3 format well                              │
│  • netcdf4: Most comprehensive, but requires C library                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Opening Local NetCDF Files

```python
from georeader.io import safe_open_netcdf

# Basic usage - data loaded into memory
ds = safe_open_netcdf("path/to/data.nc")

# Access variables
temperature = ds["temperature"]
print(ds.dims)
```

### Opening Specific Groups (NetCDF4)

NetCDF4 files can contain groups (like folders). Access them with the `group` parameter:

```python
# Open root group (default)
ds_root = safe_open_netcdf("path/to/data.nc")

# Open a specific group
ds_location = safe_open_netcdf("path/to/data.nc", group="location")
ds_metadata = safe_open_netcdf("path/to/data.nc", group="metadata")
```

### Opening from Cloud Storage

The function handles file-like objects, making it easy to work with cloud storage:

```python
from azure.storage.blob import BlobServiceClient
from georeader.io import safe_open_netcdf
import io

# Azure Blob Storage
blob_client = blob_service.get_blob_client("container", "data.nc")
blob_data = blob_client.download_blob().readall()
file_obj = io.BytesIO(blob_data)

ds = safe_open_netcdf(file_obj)
```

```python
# AWS S3 with boto3
import boto3
import io
from georeader.io import safe_open_netcdf

s3 = boto3.client('s3')
response = s3.get_object(Bucket='bucket', Key='data.nc')
file_obj = io.BytesIO(response['Body'].read())

ds = safe_open_netcdf(file_obj)
```

### Opening OPeNDAP Remote Data

```python
from georeader.io import safe_open_netcdf

# OPeNDAP URL (uses netcdf4 engine for remote protocol)
opendap_url = "https://thredds.server.org/thredds/dodsC/dataset.nc"
ds = safe_open_netcdf(opendap_url)
```

## Function Reference

### safe_open_netcdf()

```python
safe_open_netcdf(
    file_path_or_object,  # str path, URL, or file-like object
    cache=False,          # Cache file in memory
    load=True,            # Load data into memory immediately
    group=None,           # NetCDF4 group to open
    **kwargs              # Additional xr.open_dataset arguments
) -> xr.Dataset
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path_or_object` | `str` or file-like | - | Path, URL, or file object |
| `cache` | `bool` | `False` | Cache file in memory |
| `load` | `bool` | `True` | Load data into memory |
| `group` | `str` or `None` | `None` | NetCDF4 group to open |

**Returns:** `xr.Dataset` with data loaded into memory (if `load=True`)

**Raises:**
- `ImportError`: If xarray is not installed
- `IOError`: If all engines fail to open the file

### is_url()

Check if a given path is a remote URL.

```python
from georeader.io import is_url

is_url("https://example.com/data.nc")  # True
is_url("http://server.org/file.nc")     # True
is_url("ftp://ftp.server.org/data.nc")  # True
is_url("/local/path/to/file.nc")        # False
is_url(file_object)                      # False
```

## Common Patterns

### Lazy Loading for Large Files

For very large files, you may want to avoid loading everything into memory:

```python
# Keep data on disk, load chunks as needed
ds = safe_open_netcdf("large_file.nc", load=False)

# Access only what you need
subset = ds["variable"].sel(time="2024-01-01").load()
```

### Error Handling

```python
from georeader.io import safe_open_netcdf
import logging

# Enable debug logging to see which engines are tried
logging.basicConfig(level=logging.DEBUG)

try:
    ds = safe_open_netcdf("data.nc")
except IOError as e:
    print(f"Failed to open file: {e}")
    # Error message includes all attempted engines and their errors
```

### Passing Additional Options

Extra keyword arguments are passed through to `xr.open_dataset`:

```python
# Decode times, set mask/scale
ds = safe_open_netcdf(
    "data.nc",
    decode_times=True,
    mask_and_scale=True,
    chunks={"time": 10}  # Enable dask chunking
)
```

## Integration with georeader

The io module is used internally by various georeader readers. For example, readers
that handle NetCDF-based formats (like some hyperspectral data products) use 
`safe_open_netcdf` for robust file opening:

```python
from georeader.io import safe_open_netcdf
from georeader.geotensor import GeoTensor
import rasterio

# Example: Custom reader for NetCDF with geographic metadata
def read_netcdf_as_geotensor(path, variable):
    ds = safe_open_netcdf(path)
    
    # Extract variable and coordinates
    data = ds[variable].values
    
    # Build transform from coordinates
    x = ds["longitude"].values
    y = ds["latitude"].values
    transform = rasterio.transform.from_bounds(
        x.min(), y.min(), x.max(), y.max(),
        len(x), len(y)
    )
    
    return GeoTensor(data, transform=transform, crs="EPSG:4326")
```

## Dependencies

The io module requires:

- **xarray**: Core dependency for dataset handling
- **NetCDF engines** (at least one):
  - `h5netcdf`: `pip install h5netcdf`
  - `scipy`: `pip install scipy`
  - `netcdf4`: `pip install netcdf4`

For OPeNDAP support, `netcdf4` is required.

## API Reference

::: georeader.io
    options:
      show_root_heading: true
      members:
        - safe_open_netcdf
        - is_url
