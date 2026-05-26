"""Tests for georeader.readers.emit.EMITImage.

These tests run against a 200x200 EMIT scene fixture committed under
tests/data/, built from a real EMIT L1B RAD scene via the
``tests/data/build_emit_fixture.py`` script. The fixture filename pattern
matches the EMIT naming convention so the companion OBS and L2A_MASK files
are auto-discovered by ``get_obs_link`` / ``get_l2amask_link`` without any
network access.

The coverage targets two consumers of EMITImage:

1. ``marsml.mars_emit.emit_processor.EmitProcessor.process``, which uses
   ``EMITImage(...).to_crs() → load_rgb / load_raw / validmask / nc_ds_l2amask
   / mean_sza / mean_vza`` and the ``read_from_bands(...).load_raw(transpose=False)``
   pattern used inside ``marshsi.emit.mag1c_emit`` and
   ``marshsi.emit.retrieval_upv_emit.AT_MF_total_EMIT``.

2. ``georeader/docs/emit_explore.ipynb``, which exercises construction,
   ``wavelengths``, ``fwhm``, ``footprint``, ``read_from_bands``, ``load``,
   ``load_raw``, ``to_crs``, ``read_from_window`` (via
   ``read.read_from_center_coords``), ``elevation``, ``mask_bands``,
   ``validmask``, ``observation_bands``, ``sza`` / ``vza`` /
   ``observation(name)``, and ``observation_date_correction_factor``.
"""

from __future__ import annotations
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import rasterio.windows
import shapely.geometry

from georeader import read
from georeader.geotensor import GeoTensor
from georeader.readers import emit


_FIXTURE_DIR = Path(__file__).parent.parent / "data"
_FIXTURE_RAD = _FIXTURE_DIR / "EMIT_L1B_RAD_001_20220827T060753_9999999_999.nc"


pytestmark = pytest.mark.skipif(
    not _FIXTURE_RAD.exists(),
    reason=(
        "EMIT fixture not built; run "
        "`python tests/data/build_emit_fixture.py --source-dir <path>` "
        "to generate it."
    ),
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def emit_image() -> emit.EMITImage:
    """Fresh EMITImage in native CRS (EPSG:4326) — re-used across tests."""
    return emit.EMITImage(str(_FIXTURE_RAD))


@pytest.fixture(scope="module")
def emit_image_utm(emit_image: emit.EMITImage) -> emit.EMITImage:
    """EMITImage reprojected to UTM. The to_crs result is what
    EmitProcessor.load_emit_image stores on emit_image.emit_image."""
    return emit_image.to_crs("UTM")


# ── Construction & basic attributes ───────────────────────────────────────


class TestConstruction:
    """API used in emit_explore.ipynb cells 5-8 (open + inspect)."""

    def test_open_returns_emit_image(self, emit_image):
        assert isinstance(emit_image, emit.EMITImage)

    def test_filename_attribute(self, emit_image):
        assert emit_image.filename == str(_FIXTURE_RAD)

    def test_nc_ds_radiance_dataset(self, emit_image):
        # emit_explore.ipynb cell 6 inspects ei.nc_ds directly
        assert "radiance" in emit_image.nc_ds.variables

    def test_wavelengths_shape_and_range(self, emit_image):
        # 285 EMIT bands spanning ~380-2500 nm
        assert emit_image.wavelengths.shape == (285,)
        assert 380 <= emit_image.wavelengths.min() <= 400
        assert 2480 <= emit_image.wavelengths.max() <= 2500

    def test_fwhm_matches_wavelengths(self, emit_image):
        assert emit_image.fwhm.shape == emit_image.wavelengths.shape
        # EMIT FWHM is roughly 7-8 nm across the spectrum
        assert np.all(emit_image.fwhm > 4)
        assert np.all(emit_image.fwhm < 15)

    def test_time_coverage_parsed(self, emit_image):
        assert emit_image.time_coverage_start.year == 2022
        assert emit_image.time_coverage_end > emit_image.time_coverage_start

    def test_fill_value_default(self, emit_image):
        # EMIT L1B RAD uses -9999 for invalid radiance
        assert emit_image.fill_value_default == -9999
        assert emit_image.nodata == emit_image.fill_value_default

    def test_units_attribute(self, emit_image):
        # microwatts per square cm per steradian per nm
        assert "cm" in emit_image.units.lower() or emit_image.units == ""

    def test_dtype_float32(self, emit_image):
        assert emit_image.dtype == np.float32

    def test_dims_band_y_x(self, emit_image):
        assert emit_image.dims == ("band", "y", "x")


# ── Geometric properties ──────────────────────────────────────────────────


class TestGeometry:
    def test_shape(self, emit_image):
        # (n_bands, H, W) where H, W are the GLT geographic dims
        assert emit_image.shape == (285, 318, 354)

    def test_width_height(self, emit_image):
        assert emit_image.width == 354
        assert emit_image.height == 318

    def test_crs_is_4326(self, emit_image):
        # Native EMIT product is in EPSG:4326
        assert emit_image.crs.to_epsg() == 4326

    def test_transform_matches_geotransform_attr(self, emit_image):
        gt = emit_image.nc_ds.attrs["geotransform"]
        # Affine(a, b, c, d, e, f) corresponds to gt[1,2,0,4,5,3]
        assert emit_image.transform.a == pytest.approx(gt[1])
        assert emit_image.transform.c == pytest.approx(gt[0])
        assert emit_image.transform.e == pytest.approx(gt[5])
        assert emit_image.transform.f == pytest.approx(gt[3])

    def test_bounds_finite(self, emit_image):
        l, b, r, t = emit_image.bounds
        assert all(np.isfinite([l, b, r, t]))
        assert l < r and b < t

    def test_res_positive(self, emit_image):
        rx, ry = emit_image.res
        assert rx > 0 and ry > 0

    def test_footprint_polygon(self, emit_image):
        # emit_explore.ipynb cell 9 plots the footprint
        fp = emit_image.footprint()
        assert isinstance(fp, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon))
        assert fp.area > 0

    def test_footprint_reprojected(self, emit_image):
        # Used by EmitProcessor.load_emit_image via get_utm_epsg(emit.footprint("EPSG:4326"))
        fp_native = emit_image.footprint()
        fp_4326 = emit_image.footprint("EPSG:4326")
        # native CRS is 4326 — areas should match
        assert fp_4326.area == pytest.approx(fp_native.area, rel=1e-6)

    def test_valid_glt_mask(self, emit_image):
        # The fixture is built so most GLT entries are valid
        assert emit_image.valid_glt.shape == (318, 354)
        assert emit_image.valid_glt.sum() > 0


# ── Clone constructors: state propagation (the side-fix scope) ───────────


class TestCloneStatePropagation:
    """All four clone pathways must share the parent's nc_ds and sensor
    band parameters. Prior to the Phase 1 side-fix, ``read_from_bands`` left
    `_nc_ds` unset (dead-code list entry), ``to_crs`` did not propagate
    `attributes_set_if_exists` at all, and ``read_from_window`` had a
    copy-paste bug. The current implementation reuses handles at construction
    time via ``reuse_handles_from`` and no longer relies on propagating handle
    attrs through ``attributes_set_if_exists``.
    """

    def test_read_from_bands_shares_nc_ds(self, emit_image):
        child = emit_image.read_from_bands([35, 23, 11])
        assert child.nc_ds is emit_image.nc_ds
        assert child._sensor_band_params is emit_image._sensor_band_params

    def test_to_crs_shares_nc_ds(self, emit_image):
        child = emit_image.to_crs("UTM")
        assert child.nc_ds is emit_image.nc_ds
        assert child._sensor_band_params is emit_image._sensor_band_params

    def test_read_from_window_shares_nc_ds(self, emit_image):
        # Pick a window inside the GLT — top-left corner of the fixture.
        win = rasterio.windows.Window(col_off=20, row_off=20, width=100, height=100)
        child = emit_image.read_from_window(win)
        assert child.nc_ds is emit_image.nc_ds
        assert child._sensor_band_params is emit_image._sensor_band_params

    def test_copy_shares_nc_ds(self, emit_image):
        child = emit_image.copy()
        assert child.nc_ds is emit_image.nc_ds
        assert child._sensor_band_params is emit_image._sensor_band_params

    def test_grandchild_shares_with_root(self, emit_image_utm):
        # The path EmitProcessor follows: EMITImage(path).to_crs().read_from_bands(...)
        grandchild = emit_image_utm.read_from_bands([0, 1, 2])
        assert grandchild.nc_ds is emit_image_utm.nc_ds

    def test_to_crs_reprojects_pol_if_present(self, emit_image):
        # _pol is CRS-dependent and must be reprojected, not propagated as-is.
        _ = emit_image.footprint()  # populates _pol
        assert hasattr(emit_image, "_pol")
        utm = emit_image.to_crs("UTM")
        assert hasattr(utm, "_pol")
        # In UTM the polygon coords are metres, not degrees → different bounds.
        native_b = emit_image._pol.bounds
        utm_b = utm._pol.bounds
        assert max(abs(b1 - b2) for b1, b2 in zip(native_b, utm_b)) > 1.0

    def test_clone_constructors_do_not_reopen_netcdf_handles(self):
        """Clone constructors should reuse parent handles and avoid new opens."""
        with mock.patch(
            "georeader.readers.emit.safe_open_netcdf",
            wraps=emit.safe_open_netcdf,
        ) as open_netcdf:
            base = emit.EMITImage(str(_FIXTURE_RAD))
            # Parent creation opens root + location + sensor_band_parameters.
            assert open_netcdf.call_count >= 3

            open_netcdf.reset_mock()

            _ = base.copy()
            _ = base.read_from_bands([0, 1, 2])
            _ = base.to_crs("UTM")
            win = rasterio.windows.Window(col_off=20, row_off=20, width=100, height=100)
            _ = base.read_from_window(win)

            assert open_netcdf.call_count == 0

    def test_attributes_list_does_not_propagate_handle_attrs(self):
        # Handle sharing is now constructor-level via reuse_handles_from.
        assert "nc_ds" not in emit.EMITImage.attributes_set_if_exists
        assert "_sensor_band_params" not in emit.EMITImage.attributes_set_if_exists

    def test_reuse_handles_requires_same_filename(self):
        base = emit.EMITImage(str(_FIXTURE_RAD))
        with pytest.raises(ValueError, match="same EMIT file"):
            emit.EMITImage(
                "different_file.nc",
                glt=base.glt.copy(),
                reuse_handles_from=base,
            )


# ── load_raw correctness ──────────────────────────────────────────────────


class TestLoadRaw:
    """load_raw is hit by every product in EmitProcessor.process."""

    def test_default_returns_c_h_w(self, emit_image):
        full = emit_image.load_raw()
        assert full.shape == (285, 200, 200)  # (bands, raw_y, raw_x) for window_raw

    def test_transpose_false_returns_h_w_c(self, emit_image):
        # mag1c_emit and AT_MF_total_EMIT both pass transpose=False
        full = emit_image.load_raw(transpose=False)
        assert full.shape == (200, 200, 285)

    def test_slice_band_selection_identity(self, emit_image):
        full = emit_image.load_raw()
        sub = emit_image.read_from_bands(slice(10, 15)).load_raw()
        np.testing.assert_array_equal(sub, full[10:15])

    def test_list_band_selection_identity(self, emit_image):
        # emit_explore.ipynb passes a list of integer indices
        full = emit_image.load_raw()
        bands = [35, 23, 11]
        sub = emit_image.read_from_bands(bands).load_raw()
        for i, b in enumerate(bands):
            np.testing.assert_array_equal(sub[i], full[b])

    def test_integer_band_selection_drops_band_axis(self, emit_image):
        full = emit_image.load_raw()
        sub = emit_image.read_from_bands(50).load_raw()
        assert sub.shape == (200, 200)
        np.testing.assert_array_equal(sub, full[50])

    def test_boolean_band_selection_identity(self, emit_image):
        # mag1c_emit passes a boolean mask derived from wavelength range
        full = emit_image.load_raw()
        mask = (emit_image.wavelengths >= 2122) & (emit_image.wavelengths <= 2488)
        sub = emit_image.read_from_bands(mask).load_raw()
        assert sub.shape[0] == int(mask.sum())
        np.testing.assert_array_equal(sub, full[mask])


# ── georreference ─────────────────────────────────────────────────────────


class TestGeorreference:
    """georreference maps a raw-sensor-frame array into the geographic grid.
    Used by compute_emit in plume_vetting to project the radiance after
    load_raw, and by mag1c_emit / AT_MF_total_EMIT to project their outputs.
    """

    def test_georreference_2d_raw(self, emit_image):
        raw_2d = np.full(emit_image.shape_raw[1:], 1.5, dtype=np.float32)
        out = emit_image.georreference(raw_2d, fill_value_default=-9999.0)
        assert isinstance(out, GeoTensor)
        assert out.shape == emit_image.shape[1:]   # (H, W)
        assert out.crs == emit_image.crs

    def test_georreference_3d_raw(self, emit_image):
        # load_raw(C,H,W) → georreference → GeoTensor with (C, H', W')
        rdn = emit_image.load_raw()
        out = emit_image.georreference(rdn, fill_value_default=emit_image.fill_value_default)
        assert out.shape == emit_image.shape

    def test_georreference_invalid_pixels_get_fill_value(self, emit_image):
        raw_2d = np.zeros(emit_image.shape_raw[1:], dtype=np.float32)
        out = emit_image.georreference(raw_2d, fill_value_default=-1.0)
        # Pixels where valid_glt is False → fill value
        assert (out.values[~emit_image.valid_glt] == -1.0).all()


# ── load + reflectance ────────────────────────────────────────────────────


class TestLoadAndReflectance:
    """The .load(as_reflectance=...) path used by emit_explore.ipynb."""

    def test_load_radiance(self, emit_image):
        rgb_view = emit_image.read_from_bands([35, 23, 11])
        rgb = rgb_view.load(as_reflectance=False)
        assert isinstance(rgb, GeoTensor)
        assert rgb.shape[0] == 3
        # Radiance values are bounded by the EMIT scale (uW/cm^2/sr/nm).
        finite = rgb.values[rgb.values != rgb.fill_value_default]
        assert finite.max() < 50.0

    def test_load_reflectance(self, emit_image):
        rgb_view = emit_image.read_from_bands([35, 23, 11])
        ref = rgb_view.load(as_reflectance=True)
        finite = ref.values[ref.values != ref.fill_value_default]
        # Reflectance should be roughly in [0, 1] (some bands can exceed slightly).
        assert finite.min() > -0.1
        assert np.median(finite) < 1.0

    def test_load_rgb_convenience(self, emit_image):
        # emit_explore.ipynb cell 11 + EmitProcessor.process_rgb both use this.
        rgb = emit_image.load_rgb(as_reflectance=False)
        assert rgb.shape[0] == 3
        # Wavelengths 640/550/460 → RGB
        rgb_via_bands_argmin = np.argmin(
            np.abs(emit.WAVELENGTHS_RGB[:, np.newaxis] - emit_image.wavelengths), axis=1,
        )
        expected = emit_image.read_from_bands(rgb_via_bands_argmin.tolist()).load(as_reflectance=False)
        np.testing.assert_array_equal(rgb.values, expected.values)

    def test_observation_date_correction_factor(self, emit_image):
        # Used implicitly by load(as_reflectance=True) and exposed in the notebook.
        f = emit_image.observation_date_correction_factor
        assert np.isfinite(f) and f > 0


# ── Masks ─────────────────────────────────────────────────────────────────


class TestMasks:
    def test_mask_bands_list(self, emit_image):
        # emit_explore.ipynb cell 30 inspects mask_bands
        names = emit_image.mask_bands.tolist()
        assert "Cloud flag" in names
        assert "Water flag" in names

    def test_validmask_returns_geotensor(self, emit_image):
        vm = emit_image.validmask()
        assert isinstance(vm, GeoTensor)
        assert vm.shape == emit_image.shape[1:]   # (H, W)
        assert vm.values.dtype == bool

    def test_percentage_clear_range(self, emit_image):
        pct = emit_image.percentage_clear
        assert 0.0 <= pct <= 100.0

    def test_invalid_mask_raw_shape(self, emit_image):
        # Used by validmask() internally; also exposed for diagnostics.
        invalid = emit_image.invalid_mask_raw()
        # Returned in raw (sensor) coordinates, windowed to valid GLT bbox.
        assert invalid.shape == emit_image.shape_raw[1:]
        assert invalid.dtype == bool

    def test_water_mask(self, emit_image):
        # Used by AT_MF_total_EMIT for water masking.
        wm = emit_image.water_mask()
        assert isinstance(wm, GeoTensor)
        assert wm.shape == emit_image.shape[1:]

    def test_named_mask(self, emit_image):
        cloud = emit_image.mask("Cloud flag")
        assert isinstance(cloud, GeoTensor)
        assert cloud.shape == emit_image.shape[1:]

    def test_nc_ds_l2amask_accessible(self, emit_image):
        # compute_emit (plume_vetting) reads emit_image.nc_ds_l2amask['mask'] directly.
        ds = emit_image.nc_ds_l2amask
        assert "mask" in ds.variables


# ── Observations (angles, path length, elevation) ────────────────────────


class TestObservations:
    def test_observation_bands_list(self, emit_image):
        # emit_explore.ipynb cell 33
        names = emit_image.observation_bands.tolist()
        # EMIT obs has 11 bands; check a couple of well-known ones.
        assert any("zenith" in n.lower() for n in names)

    def test_mean_sza_and_vza_finite(self, emit_image):
        # EmitProcessor.load_emit_image stores these on the MarsEmitImage.
        assert np.isfinite(emit_image.mean_sza)
        assert np.isfinite(emit_image.mean_vza)
        assert 0 < emit_image.mean_sza < 90
        assert 0 <= emit_image.mean_vza < 90

    def test_sza_geotensor(self, emit_image):
        # emit_explore.ipynb cell 34
        sza = emit_image.sza()
        assert isinstance(sza, GeoTensor)
        assert sza.shape == emit_image.shape[1:]

    def test_vza_geotensor(self, emit_image):
        vza = emit_image.vza()
        assert isinstance(vza, GeoTensor)
        assert vza.shape == emit_image.shape[1:]

    def test_named_observation(self, emit_image):
        # emit_explore.ipynb cell 36: 'Path length (sensor-to-ground in meters)'
        # Use the first band name available to avoid hard-coding a string.
        name = emit_image.observation_bands.tolist()[0]
        obs = emit_image.observation(name)
        assert isinstance(obs, GeoTensor)
        assert obs.shape == emit_image.shape[1:]

    def test_elevation(self, emit_image):
        # emit_explore.ipynb cell 21
        elev = emit_image.elevation()
        assert isinstance(elev, GeoTensor)
        assert elev.shape == emit_image.shape[1:]


# ── to_crs reprojection ───────────────────────────────────────────────────


def _crs_epsg(crs) -> int:
    """Robust EPSG extraction — to_crs may return a string CRS rather than a CRS object."""
    from rasterio.crs import CRS as _CRS
    return _CRS.from_user_input(crs).to_epsg()


class TestToCrs:
    def test_to_crs_utm_changes_crs(self, emit_image, emit_image_utm):
        assert _crs_epsg(emit_image_utm.crs) != _crs_epsg(emit_image.crs)
        assert _crs_epsg(emit_image_utm.crs) != 4326

    def test_to_crs_explicit_epsg(self, emit_image):
        # emit_explore.ipynb cell 18: passes a specific UTM EPSG.
        out = emit_image.to_crs("EPSG:32641")
        assert _crs_epsg(out.crs) == 32641

    def test_to_crs_default_resolution(self, emit_image):
        out = emit_image.to_crs("UTM")
        # Default resolution_dst_crs=60 metres
        rx, ry = abs(out.res[0]), abs(out.res[1])
        assert rx == pytest.approx(60, abs=1)
        assert ry == pytest.approx(60, abs=1)

    def test_to_crs_load_rgb_runs(self, emit_image_utm):
        # End-to-end notebook flow: to_crs → load_rgb → display.
        rgb = emit_image_utm.load_rgb(as_reflectance=False)
        assert rgb.shape[0] == 3


# ── read_from_window / read_from_center_coords ───────────────────────────


class TestReadFromWindow:
    def test_read_from_window_subset_shape(self, emit_image):
        # Use a window inside the GLT's valid extent.
        win = rasterio.windows.Window(col_off=50, row_off=50, width=80, height=80)
        sub = emit_image.read_from_window(win)
        assert sub.shape[1] == 80 and sub.shape[2] == 80

    def test_read_from_center_coords_runs(self, emit_image_utm):
        # emit_explore.ipynb cell 23 uses this to extract a 200x200 patch.
        # Pick a centre inside the fixture footprint.
        fp = emit_image_utm.footprint("EPSG:4326")
        cx, cy = fp.centroid.x, fp.centroid.y
        sub = read.read_from_center_coords(
            emit_image_utm, (cx, cy), shape=(50, 50), crs_center_coords="EPSG:4326",
        )
        assert sub.shape[1] == 50 and sub.shape[2] == 50


# ── Sanity smoke test mirroring the notebook end-to-end ──────────────────


def test_notebook_smoke(emit_image):
    """Mirror the bulk of emit_explore.ipynb in one pass — open, inspect,
    band-subset, reproject, reflectance, masks, observations, elevation."""
    # Open + basic inspection
    assert emit_image.wavelengths.shape == (285,)
    _ = emit_image.time_coverage_start, emit_image.time_coverage_end

    # Footprint in native + 4326
    fp_native = emit_image.footprint()
    fp_4326 = emit_image.footprint("EPSG:4326")
    assert fp_native.area > 0 and fp_4326.area > 0

    # RGB band selection + load
    bands = np.argmin(np.abs(emit.WAVELENGTHS_RGB[:, np.newaxis] - emit_image.wavelengths), axis=1).tolist()
    ei_rgb = emit_image.read_from_bands(bands)
    rgb = ei_rgb.load(as_reflectance=True)
    assert rgb.shape[0] == 3

    # Reproject + reflectance
    utm = emit_image.to_crs("UTM")
    utm_rgb = utm.read_from_bands(bands).load(as_reflectance=True)
    assert utm_rgb.shape[0] == 3

    # Masks + observations + elevation
    vm = utm.validmask()
    sza = utm.sza()
    vza = utm.vza()
    elev = utm.elevation()
    for arr in [vm, sza, vza, elev]:
        assert arr.values.shape[-2:] == utm.shape[1:]


# ── Option B: opt-in radiance cache ──────────────────────────────────────


_CACHE_KEY = emit.EMITImage._CACHE_KEY_RADIANCE


class TestRadianceCacheOff:
    """When ``cache_radiance=False`` (the default), no cache is populated and
    every ``load_raw`` re-reads from disk via the Option A path. The ``_cache``
    dict still exists (so the propagation invariant holds for clones), but stays
    empty for the radiance key."""

    def test_default_flag_off(self, emit_image):
        assert emit_image.cache_radiance is False

    def test_cache_dict_exists(self, emit_image):
        assert isinstance(emit_image._cache, dict)
        assert _CACHE_KEY not in emit_image._cache

    def test_load_raw_does_not_populate_cache(self, emit_image):
        # Use a throwaway instance so we don't pollute the module-scoped fixture.
        ei = emit.EMITImage(str(_FIXTURE_RAD), cache_radiance=False)
        ei.load_raw()
        ei.read_from_bands([0, 1, 2]).load_raw()
        assert _CACHE_KEY not in ei._cache

    def test_default_flag_propagates_through_clones(self, emit_image):
        # Pick a window inside the GLT valid extent.
        win = rasterio.windows.Window(col_off=50, row_off=50, width=80, height=80)
        for clone in (
            emit_image.read_from_bands([0]),
            emit_image.to_crs("UTM"),
            emit_image.read_from_window(win),
            emit_image.copy(),
        ):
            assert clone.cache_radiance is False


class TestRadianceCacheOn:
    """When ``cache_radiance=True``, the first ``load_raw`` decompresses the
    full-spectrum windowed radiance once; subsequent calls (any band selection,
    any clone) serve from the in-memory array."""

    @pytest.fixture
    def ei_cached(self):
        # Fresh instance per test so cache state and clone-graphs don't leak.
        return emit.EMITImage(str(_FIXTURE_RAD), cache_radiance=True)

    def test_flag_propagates_through_all_clone_paths(self, ei_cached):
        win = rasterio.windows.Window(col_off=50, row_off=50, width=80, height=80)
        for clone in (
            ei_cached.read_from_bands([0]),
            ei_cached.to_crs("UTM"),
            ei_cached.read_from_window(win),
            ei_cached.copy(),
        ):
            assert clone.cache_radiance is True

    def test_cache_dict_shared_by_reference(self, ei_cached):
        # The dict-by-reference invariant — this is what makes the cache visible
        # across the clone graph. A regression here would silently break Option B.
        utm = ei_cached.to_crs("UTM")
        child = utm.read_from_bands([0, 1, 2])
        grandchild = child.read_from_bands([0])
        assert utm._cache is ei_cached._cache
        assert child._cache is ei_cached._cache
        assert grandchild._cache is ei_cached._cache
        # Mutation through any clone is visible to all
        child.load_raw()
        assert _CACHE_KEY in ei_cached._cache
        assert _CACHE_KEY in grandchild._cache

    def test_first_load_populates_cache(self, ei_cached):
        assert _CACHE_KEY not in ei_cached._cache
        ei_cached.load_raw()
        cached = ei_cached._cache[_CACHE_KEY]
        # Cache stores full-spectrum (H, W, B) windowed-to-window_raw.
        assert cached.shape == (200, 200, 285)
        assert cached.dtype == np.float32

    def test_second_load_comes_from_cache_proof_by_poisoning(self, ei_cached):
        # Mutate the cached array in place. A subsequent load_raw must return
        # the mutated values — that's only possible if the cache was hit. If a
        # second disk read happens, it would overwrite our poison and the test
        # would see the original (unmutated) values.
        ei_cached.load_raw()
        cached = ei_cached._cache[_CACHE_KEY]
        sentinel = -123456.0
        cached[..., 0] = sentinel
        cached[..., 50] = sentinel + 1.0

        # New clone, different band selection — must serve from cache.
        single = ei_cached.read_from_bands(0).load_raw()
        assert np.all(single == sentinel)

        many = ei_cached.read_from_bands(slice(48, 53)).load_raw()
        # Band 50 row of the (C, H, W) result corresponds to cached band 50.
        assert np.all(many[2] == sentinel + 1.0)

    def test_clone_first_load_populates_parent_cache(self, ei_cached):
        # First read happens on a clone, not the parent. The shared-by-reference
        # _cache dict means the parent sees the populated entry too.
        assert _CACHE_KEY not in ei_cached._cache
        ei_cached.read_from_bands([0, 1, 2]).load_raw()
        assert _CACHE_KEY in ei_cached._cache

    @pytest.mark.parametrize("band_sel", [
        slice(None),
        slice(10, 50),
        [35, 23, 11],
        0,
        50,
    ])
    def test_cached_matches_non_cached(self, ei_cached, band_sel):
        # The cached path must return exactly the same array as the non-cached
        # path for every supported band_selection type.
        ei_nocache = emit.EMITImage(str(_FIXTURE_RAD), cache_radiance=False)

        expected = ei_nocache.read_from_bands(band_sel).load_raw()
        got = ei_cached.read_from_bands(band_sel).load_raw()
        np.testing.assert_array_equal(expected, got)

    def test_cached_matches_non_cached_boolean_mask(self, ei_cached):
        # mag1c_emit-style boolean band_selection.
        ei_nocache = emit.EMITImage(str(_FIXTURE_RAD), cache_radiance=False)
        mask = (ei_cached.wavelengths >= 2122) & (ei_cached.wavelengths <= 2488)
        expected = ei_nocache.read_from_bands(mask).load_raw()
        got = ei_cached.read_from_bands(mask).load_raw()
        np.testing.assert_array_equal(expected, got)

    def test_cached_transpose_false(self, ei_cached):
        ei_nocache = emit.EMITImage(str(_FIXTURE_RAD), cache_radiance=False)
        expected = ei_nocache.read_from_bands(slice(10, 15)).load_raw(transpose=False)
        got = ei_cached.read_from_bands(slice(10, 15)).load_raw(transpose=False)
        np.testing.assert_array_equal(expected, got)


class TestClearRadianceCache:
    @pytest.fixture
    def ei_populated(self):
        ei = emit.EMITImage(str(_FIXTURE_RAD), cache_radiance=True)
        ei.load_raw()
        assert _CACHE_KEY in ei._cache
        return ei

    def test_clear_removes_cache_entry(self, ei_populated):
        ei_populated.clear_radiance_cache()
        assert _CACHE_KEY not in ei_populated._cache

    def test_clear_through_any_clone_empties_shared_dict(self, ei_populated):
        # The cache is per-process, not per-instance — any clone can release it.
        child = ei_populated.read_from_bands([0, 1, 2])
        child.clear_radiance_cache()
        assert _CACHE_KEY not in ei_populated._cache
        assert _CACHE_KEY not in child._cache

    def test_clear_preserves_dict_identity(self, ei_populated):
        # The _cache dict object itself must NOT be replaced — clones share it
        # by reference, so rebinding would silently fork the cache.
        before = ei_populated._cache
        ei_populated.clear_radiance_cache()
        assert ei_populated._cache is before

    def test_array_collectable_after_clear(self, ei_populated):
        # RAM-budget-critical: clear_radiance_cache must release the underlying
        # ~1.5 GB numpy array (no internal handles pinning it).
        import gc
        import weakref
        arr = ei_populated._cache[_CACHE_KEY]
        ref = weakref.ref(arr)
        del arr
        ei_populated.clear_radiance_cache()
        gc.collect()
        assert ref() is None, "cached radiance array is still referenced after clear"

    def test_clear_on_unpopulated_cache_is_noop(self):
        ei = emit.EMITImage(str(_FIXTURE_RAD), cache_radiance=True)
        ei.clear_radiance_cache()   # must not raise
        assert _CACHE_KEY not in ei._cache

    def test_load_raw_after_clear_repopulates(self, ei_populated):
        ei_populated.clear_radiance_cache()
        ei_populated.load_raw()
        assert _CACHE_KEY in ei_populated._cache


class TestCacheMultiprocessing:
    """Each process must have its own EMITImage instance and its own _cache —
    there is no module-level / process-shared state. Catches a future regression
    that accidentally introduces a global cache keyed by filename.
    """

    def test_independent_caches_across_processes(self):
        import multiprocessing as mp
        # 'spawn' avoids forking xarray's file handles, which mirrors how
        # Azure ML / Linux + Python workers actually operate.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=2) as pool:
            results = pool.map(_subprocess_load_first_pixel, [str(_FIXTURE_RAD)] * 2)
        # Workers ran in distinct processes.
        assert results[0]["pid"] != results[1]["pid"]
        # Both observed the canonical first-pixel radiance value (consistency).
        assert results[0]["first_pixel_band0"] == results[1]["first_pixel_band0"]
        # Each worker's cache populated its own _cache. If a module-level cache
        # had leaked across processes via fork-style state, this would still
        # hold; the harder invariant is that each worker built the cache itself,
        # asserted by the test passing in 'spawn' mode (fresh interpreter).
        for r in results:
            assert r["cache_key_present"] is True


def _subprocess_load_first_pixel(filename: str) -> dict:
    """Worker for the multiprocessing test. Runs in a fresh process."""
    from georeader.readers import emit as emit_mod
    ei = emit_mod.EMITImage(filename, cache_radiance=True)
    rdn = ei.load_raw()
    return {
        "first_pixel_band0": float(rdn[0, 0, 0]),
        "cache_key_present": emit_mod.EMITImage._CACHE_KEY_RADIANCE in ei._cache,
        "pid": os.getpid(),
    }
