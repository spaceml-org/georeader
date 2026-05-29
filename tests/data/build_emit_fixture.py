"""Build a small EMIT L1B RAD test fixture by slicing a real EMIT scene.

The fixture is a 200x200 raw-sensor window of a real EMIT L1B RAD scene plus
the matching crops of the companion OBS (observation angles, path length, etc.)
and L2A MASK files. The GLT (Geographic Lookup Table) is cropped to the
geographic pixels that actually reference the raw subset, and its indices are
rewritten to be 1-based relative to the new 200x200 raw bbox.

Usage
-----
Run once to (re)build the fixture files committed under tests/data/. Requires
the original three EMIT files for scene 20220827T060753_2223904_013, available
locally or via NASA Earthdata download.

    python tests/data/build_emit_fixture.py \\
        --source-dir /path/to/local/copies \\
        --output-dir tests/data

The fixture filenames follow the EMIT naming pattern so the EMITImage class
can auto-discover the OBS and L2A MASK companion files via
``get_obs_link`` / ``get_l2amask_link`` without any download.

Provenance: 200x200 raw window centered at (downtrack=600, crosstrack=620)
of EMIT_L1B_RAD_001_20220827T060753_2223904_013. Choice of this window is
arbitrary; any other window with dense GLT coverage would work.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import netCDF4 as nc

# Raw-sensor window to keep. (downtrack_start, downtrack_stop, crosstrack_start, crosstrack_stop)
RAW_Y0, RAW_Y1 = 500, 700
RAW_X0, RAW_X1 = 520, 720

# Synthetic scene-id suffix that marks these files as test fixtures and avoids
# colliding with real EMIT product names. Suffix "9999999_999" is unambiguous.
SRC_SCENE = "20220827T060753_2223904_013"
DST_SCENE = "20220827T060753_9999999_999"


_RESERVED_ATTRS = {"_NCProperties"}  # netCDF library-managed attrs we must not set.


def _copy_root_attrs(src: h5py.File, dst: nc.Dataset, override_geotransform: tuple[float, float, float, float, float, float]) -> None:
    """Copy root-level attributes, overriding the geotransform for the cropped GLT."""
    for k in src.attrs:
        if k in _RESERVED_ATTRS:
            continue
        v = src.attrs[k]
        if isinstance(v, np.ndarray) and v.shape == ():
            v = v.item()
        elif isinstance(v, np.ndarray) and v.shape == (1,):
            v = v[0].item() if v.dtype.kind in "iuf" else v[0]
        if k == "geotransform":
            v = np.asarray(override_geotransform, dtype=np.float64)
        try:
            dst.setncattr(k, v)
        except (TypeError, ValueError, AttributeError):
            pass


def _compute_cropped_glt(
    glt_x: np.ndarray, glt_y: np.ndarray,
    raw_y0: int, raw_y1: int, raw_x0: int, raw_x1: int,
) -> tuple[np.ndarray, np.ndarray, slice, slice]:
    """Crop the GLT grid to its bbox of pixels referencing the raw window, and
    rewrite the GLT indices to be 1-based relative to the new (200x200) raw subset.

    Returns:
        new_glt_x, new_glt_y (cropped + rewritten), geo_row_slice, geo_col_slice
    """
    # EMIT GLT values are 1-based indices into raw (downtrack, crosstrack); 0 = invalid.
    in_raw = (
        (glt_x > raw_x0) & (glt_x <= raw_x1)
        & (glt_y > raw_y0) & (glt_y <= raw_y1)
    )
    rows = np.where(np.any(in_raw, axis=1))[0]
    cols = np.where(np.any(in_raw, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise RuntimeError("Empty GLT crop — picked raw window has no GLT pixels pointing into it.")
    geo_row_slice = slice(int(rows[0]), int(rows[-1]) + 1)
    geo_col_slice = slice(int(cols[0]), int(cols[-1]) + 1)

    new_x = glt_x[geo_row_slice, geo_col_slice].copy()
    new_y = glt_y[geo_row_slice, geo_col_slice].copy()
    invalid = (
        (new_x == 0) | (new_y == 0)
        | (new_x <= raw_x0) | (new_x > raw_x1)
        | (new_y <= raw_y0) | (new_y > raw_y1)
    )
    # Shift indices: glt values must point into the new raw subset [1, 200].
    new_x = new_x - raw_x0
    new_y = new_y - raw_y0
    new_x[invalid] = 0
    new_y[invalid] = 0
    return new_x.astype(np.int32), new_y.astype(np.int32), geo_row_slice, geo_col_slice


def _adjusted_geotransform(orig_gt: np.ndarray, geo_row_slice: slice, geo_col_slice: slice) -> tuple[float, ...]:
    """Translate the geotransform origin by the cropped GLT's top-left offset.

    EMIT stores geotransform as [ulx, dx_per_col, dx_per_row, uly, dy_per_col, dy_per_row]
    (i.e. GDAL/rasterio convention with origin first, then row/col deltas).
    """
    gt = np.asarray(orig_gt).flatten().astype(np.float64)
    ulx, dx_col, dx_row, uly, dy_col, dy_row = gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]
    col_off = geo_col_slice.start
    row_off = geo_row_slice.start
    new_ulx = ulx + col_off * dx_col + row_off * dx_row
    new_uly = uly + col_off * dy_col + row_off * dy_row
    return (new_ulx, dx_col, dx_row, new_uly, dy_col, dy_row)


def _build_one(
    src_path: Path, dst_path: Path,
    main_var_name: str,   # 'radiance' / 'obs' / 'mask'
    sensor_band_groups: list[str],  # group names under sensor_band_parameters
    extra_root_vars: dict[str, str],  # name → dtype label (e.g. {'packed_wavelength_bands': 'f4'})
    geo_row_slice: slice, geo_col_slice: slice,
    new_glt_x: np.ndarray, new_glt_y: np.ndarray,
    new_geotransform: tuple[float, ...],
) -> None:
    """Build one fixture file (RAD / OBS / MASK) by slicing the source."""
    with h5py.File(src_path, "r") as src:
        nbands = src[main_var_name].shape[-1]
        main = src[main_var_name][RAW_Y0:RAW_Y1, RAW_X0:RAW_X1, :]
        # location subgroup arrays for the cropped GLT geographic region
        loc_lat = src["location/lat"][RAW_Y0:RAW_Y1, RAW_X0:RAW_X1]
        loc_lon = src["location/lon"][RAW_Y0:RAW_Y1, RAW_X0:RAW_X1]
        loc_elev = src["location/elev"][RAW_Y0:RAW_Y1, RAW_X0:RAW_X1]
        # ortho_x / ortho_y describe the geographic grid; crop those too
        ortho_x = src["ortho_x"][geo_col_slice]
        ortho_y = src["ortho_y"][geo_row_slice]
        # Downtrack / crosstrack coords (per pixel)
        downtrack = src["downtrack"][RAW_Y0:RAW_Y1]
        crosstrack = src["crosstrack"][RAW_X0:RAW_X1]
        bands = src["bands"][:]
        # Sensor band parameters subgroup
        sbp = {k: src[f"sensor_band_parameters/{k}"][:] for k in sensor_band_groups}
        # Extra root-level vars (band_mask, packed_wavelength_bands)
        extra = {k: src[k][:] if src[k].ndim == 1
                 else src[k][RAW_Y0:RAW_Y1, RAW_X0:RAW_X1, :]
                 for k in extra_root_vars}

        with nc.Dataset(dst_path, "w", format="NETCDF4") as dst:
            # Create dims
            dst.createDimension("downtrack", RAW_Y1 - RAW_Y0)
            dst.createDimension("crosstrack", RAW_X1 - RAW_X0)
            dst.createDimension("bands", nbands)
            dst.createDimension("ortho_y", geo_row_slice.stop - geo_row_slice.start)
            dst.createDimension("ortho_x", geo_col_slice.stop - geo_col_slice.start)
            for k, arr in extra.items():
                if arr.ndim == 1 and k not in dst.dimensions:
                    # extra dim e.g. packed_wavelength_bands → 36
                    dim_name = f"_{k}"
                    dst.createDimension(dim_name, len(arr))

            # Root vars. _FillValue must be passed at creation time.
            src_main_attrs = dict(src[main_var_name].attrs)
            fill_value = src_main_attrs.pop("_FillValue", None)
            v = dst.createVariable(
                main_var_name, main.dtype, ("downtrack", "crosstrack", "bands"),
                zlib=True, complevel=4,
                fill_value=fill_value,
            )
            for k, val in src_main_attrs.items():
                try:
                    v.setncattr(k, val)
                except (TypeError, ValueError, AttributeError):
                    pass
            v[...] = main

            for k, arr in extra.items():
                if arr.ndim == 1:
                    var = dst.createVariable(k, arr.dtype, (f"_{k}",))
                else:
                    # 3-D root var (e.g. MASK band_mask)
                    last_dim_name = f"_{k}_last"
                    if last_dim_name not in dst.dimensions:
                        dst.createDimension(last_dim_name, arr.shape[-1])
                    var = dst.createVariable(
                        k, arr.dtype, ("downtrack", "crosstrack", last_dim_name),
                        zlib=True, complevel=4,
                    )
                var[...] = arr

            # 1-D coordinate vars
            for name, arr, dim in [
                ("downtrack", downtrack, "downtrack"),
                ("crosstrack", crosstrack, "crosstrack"),
                ("bands", bands, "bands"),
                ("ortho_x", ortho_x, "ortho_x"),
                ("ortho_y", ortho_y, "ortho_y"),
            ]:
                var = dst.createVariable(name, arr.dtype, (dim,))
                var[...] = arr

            # location subgroup
            loc = dst.createGroup("location")
            for name, arr in [("lat", loc_lat), ("lon", loc_lon), ("elev", loc_elev)]:
                src_var = src[f"location/{name}"]
                src_attrs = dict(src_var.attrs)
                fv = src_attrs.pop("_FillValue", None)
                var = loc.createVariable(
                    name, arr.dtype, ("downtrack", "crosstrack"),
                    zlib=True, complevel=4, fill_value=fv,
                )
                var[...] = arr
                for k, val in src_attrs.items():
                    try:
                        var.setncattr(k, val)
                    except (TypeError, ValueError, AttributeError):
                        pass
            for name, arr in [("glt_x", new_glt_x), ("glt_y", new_glt_y)]:
                var = loc.createVariable(name, arr.dtype, ("ortho_y", "ortho_x"), zlib=True, complevel=4)
                var[...] = arr

            # sensor_band_parameters subgroup
            sbp_group = dst.createGroup("sensor_band_parameters")
            for k, arr in sbp.items():
                if arr.dtype == object:
                    # Strings (e.g. observation_bands, mask_bands)
                    str_arr = np.array([s.decode("utf-8") if isinstance(s, bytes) else s for s in arr], dtype=str)
                    var = sbp_group.createVariable(k, str, ("bands",) if len(arr) == nbands else (f"_{k}",))
                    if len(arr) != nbands:
                        sbp_group.createDimension(f"_{k}", len(arr))
                        var = sbp_group.createVariable(k, str, (f"_{k}",))
                    var[...] = str_arr
                else:
                    if len(arr) == nbands:
                        var = sbp_group.createVariable(k, arr.dtype, ("bands",))
                    else:
                        dim_name = f"_{k}_dim"
                        sbp_group.createDimension(dim_name, len(arr))
                        var = sbp_group.createVariable(k, arr.dtype, (dim_name,))
                    var[...] = arr

            # Root attrs
            _copy_root_attrs(src, dst, new_geotransform)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", required=True, type=Path,
                   help="Directory containing the source RAD/OBS/MASK .nc files.")
    p.add_argument("--output-dir", default=Path(__file__).parent, type=Path,
                   help="Where to write the fixture files. Default: this script's dir.")
    args = p.parse_args()

    src_rad = args.source_dir / f"EMIT_L1B_RAD_001_{SRC_SCENE}.nc"
    src_obs = args.source_dir / f"EMIT_L1B_OBS_001_{SRC_SCENE}.nc"
    src_mask = args.source_dir / f"EMIT_L2A_MASK_001_{SRC_SCENE}.nc"
    for p_ in [src_rad, src_obs, src_mask]:
        if not p_.exists():
            raise FileNotFoundError(f"Missing source: {p_}")

    dst_rad = args.output_dir / f"EMIT_L1B_RAD_001_{DST_SCENE}.nc"
    dst_obs = args.output_dir / f"EMIT_L1B_OBS_001_{DST_SCENE}.nc"
    dst_mask = args.output_dir / f"EMIT_L2A_MASK_001_{DST_SCENE}.nc"

    # Build cropped GLT once from the RAD file (OBS / MASK have identical GLT).
    with h5py.File(src_rad, "r") as f:
        glt_x = f["location/glt_x"][:]
        glt_y = f["location/glt_y"][:]
        orig_gt = f.attrs["geotransform"]
    new_glt_x, new_glt_y, geo_row_slice, geo_col_slice = _compute_cropped_glt(
        glt_x, glt_y, RAW_Y0, RAW_Y1, RAW_X0, RAW_X1,
    )
    new_gt = _adjusted_geotransform(orig_gt, geo_row_slice, geo_col_slice)
    print(f"Cropped GLT: shape={new_glt_x.shape}, valid pixels={(new_glt_x != 0).sum()}")
    print(f"New geotransform: {new_gt}")

    _build_one(
        src_rad, dst_rad,
        main_var_name="radiance",
        sensor_band_groups=["wavelengths", "fwhm"],
        extra_root_vars={},
        geo_row_slice=geo_row_slice, geo_col_slice=geo_col_slice,
        new_glt_x=new_glt_x, new_glt_y=new_glt_y,
        new_geotransform=new_gt,
    )
    print(f"Wrote {dst_rad}  ({dst_rad.stat().st_size / 1e6:.1f} MB)")

    _build_one(
        src_obs, dst_obs,
        main_var_name="obs",
        sensor_band_groups=["observation_bands"],
        extra_root_vars={},
        geo_row_slice=geo_row_slice, geo_col_slice=geo_col_slice,
        new_glt_x=new_glt_x, new_glt_y=new_glt_y,
        new_geotransform=new_gt,
    )
    print(f"Wrote {dst_obs}  ({dst_obs.stat().st_size / 1e6:.1f} MB)")

    _build_one(
        src_mask, dst_mask,
        main_var_name="mask",
        sensor_band_groups=["mask_bands"],
        extra_root_vars={"band_mask": "u1", "packed_wavelength_bands": "f4"},
        geo_row_slice=geo_row_slice, geo_col_slice=geo_col_slice,
        new_glt_x=new_glt_x, new_glt_y=new_glt_y,
        new_geotransform=new_gt,
    )
    print(f"Wrote {dst_mask}  ({dst_mask.stat().st_size / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
