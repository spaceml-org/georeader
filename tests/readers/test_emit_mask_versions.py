"""Tests for EMIT L2A mask flag selection across mask versions and for v002 companion-file resolution.

The v003 L2A mask (companion of v002 L1B) has a different band layout from v001:

    v001: Cloud flag, Cirrus flag, Water flag, Spacecraft Flag, Dilated Cloud Flag,
          AOD550, H2O (g cm-2), Aggregate Flag
    v003: Cloud Flag, Cirrus Flag, Water Flag, Dilated Cloud Flag,
          SpecTf-Cloud Probability, SpecTf-Cloud Flag, SpecTf-Buffer Distance

so flags must be selected by label. The v002/v003 files used here are built in
``tmp_path`` from the committed v001 fixture: the RAD and OBS files are symlinked
under v002 names and a v003-layout mask is written from the v001 mask bands.
"""

from __future__ import annotations
import shutil
from pathlib import Path

import numpy as np
import pytest

from georeader.readers import emit

xr = pytest.importorskip("xarray")


_FIXTURE_DIR = Path(__file__).parent.parent / "data"
_DT = "20220827T060753"
_V001_SUFFIX = f"{_DT}_9999999_999"
_FIXTURE_RAD = _FIXTURE_DIR / f"EMIT_L1B_RAD_001_{_V001_SUFFIX}.nc"
_FIXTURE_OBS = _FIXTURE_DIR / f"EMIT_L1B_OBS_001_{_V001_SUFFIX}.nc"
_FIXTURE_MASK = _FIXTURE_DIR / f"EMIT_L2A_MASK_001_{_V001_SUFFIX}.nc"

V001_LABELS = ["Cloud flag", "Cirrus flag", "Water flag", "Spacecraft Flag", "Dilated Cloud Flag",
               "AOD550", "H2O (g cm-2)", "Aggregate Flag"]
V003_LABELS = ["Cloud Flag", "Cirrus Flag", "Water Flag", "Dilated Cloud Flag",
               "SpecTf-Cloud Probability", "SpecTf-Cloud Flag", "SpecTf-Buffer Distance"]


# ── mask_flag_indexes / mask_band_index (labels only) ─────────────────────


def test_v001_mask_flags_unchanged():
    """The positional indexes georeader 2.3.5 used for every file."""
    assert emit.mask_flag_indexes(V001_LABELS, with_buffer=False) == [0, 1, 3]
    assert emit.mask_flag_indexes(V001_LABELS, with_buffer=True) == [0, 1, 3, 4]


def test_v001_include_spectf_is_noop():
    assert emit.mask_flag_indexes(V001_LABELS, with_buffer=True, include_spectf=True) == [0, 1, 3, 4]


def test_v003_mask_flags_by_name():
    assert emit.mask_flag_indexes(V003_LABELS, with_buffer=False) == [0, 1]
    assert emit.mask_flag_indexes(V003_LABELS, with_buffer=True) == [0, 1, 3]
    assert emit.mask_flag_indexes(V003_LABELS, with_buffer=True, include_spectf=True) == [0, 1, 3, 5]


@pytest.mark.parametrize("with_buffer", [False, True])
@pytest.mark.parametrize("include_spectf", [False, True])
def test_v003_never_selects_continuous_bands(with_buffer, include_spectf):
    idx = emit.mask_flag_indexes(V003_LABELS, with_buffer=with_buffer, include_spectf=include_spectf)
    assert 4 not in idx and 6 not in idx  # SpecTf-Cloud Probability, SpecTf-Buffer Distance


def test_missing_required_flag_raises():
    labels = [b for b in V003_LABELS if b != "Cirrus Flag"]
    with pytest.raises(ValueError, match="cirrus flag.*in mask.nc"):
        emit.mask_flag_indexes(labels, with_buffer=False, source="mask.nc")


def test_missing_buffer_flag_raises_only_with_buffer():
    labels = [b for b in V003_LABELS if b != "Dilated Cloud Flag"]
    assert emit.mask_flag_indexes(labels, with_buffer=False) == [0, 1]
    with pytest.raises(ValueError, match="dilated cloud flag"):
        emit.mask_flag_indexes(labels, with_buffer=True)


@pytest.mark.parametrize("name", ["Water flag", "Water Flag", "water  FLAG"])
def test_mask_band_index_ignores_case_and_spaces(name):
    assert emit.mask_band_index(V001_LABELS, name) == 2
    assert emit.mask_band_index(V003_LABELS, name) == 2


def test_mask_band_index_unknown_raises():
    with pytest.raises(ValueError, match="'Spacecraft Flag' not found in m.nc"):
        emit.mask_band_index(V003_LABELS, "Spacecraft Flag", source="m.nc")


# ── Files: v002 RAD + v003 MASK built from the v001 fixture ───────────────


pytestmark_files = pytest.mark.skipif(
    not _FIXTURE_MASK.exists(), reason="EMIT fixture not built; see tests/data/build_emit_fixture.py",
)

# SpecTf-Cloud Flag is set on this raw block only, so its effect is easy to isolate.
_SPECTF_BLOCK = (slice(10, 30), slice(40, 70))


def _v003_mask_bands(v001_mask: np.ndarray) -> np.ndarray:
    """(downtrack, crosstrack, 7) v003-layout mask from a v001-layout one.

    The SpecTf probability is > 0 everywhere, so selecting it by mistake marks every pixel invalid.
    """
    rng = np.random.default_rng(0)
    shape = v001_mask.shape[:2]
    spectf_flag = np.zeros(shape, dtype=v001_mask.dtype)
    spectf_flag[_SPECTF_BLOCK] = 1
    return np.stack([
        v001_mask[..., 0],                                     # Cloud Flag
        v001_mask[..., 1],                                     # Cirrus Flag
        v001_mask[..., 2],                                     # Water Flag
        v001_mask[..., 4],                                     # Dilated Cloud Flag
        rng.uniform(0.2, 0.9, shape).astype(v001_mask.dtype),  # SpecTf-Cloud Probability
        spectf_flag,                                           # SpecTf-Cloud Flag
        rng.uniform(100, 5000, shape).astype(v001_mask.dtype), # SpecTf-Buffer Distance
    ], axis=-1)


def _write_v003_mask(dst: Path) -> None:
    """Write a v003-layout L2A mask with the v001 fixture's geolocation."""
    with xr.open_dataset(_FIXTURE_MASK, engine="h5netcdf") as src:
        root = xr.Dataset(
            {"mask": (("downtrack", "crosstrack", "bands"), _v003_mask_bands(src["mask"].values),
                      {"_FillValue": -9999.0})},
            attrs=dict(src.attrs),
        )
    with xr.open_dataset(_FIXTURE_MASK, engine="h5netcdf", group="location") as loc:
        location = loc[["glt_x", "glt_y"]].load()
    params = xr.Dataset({"mask_bands": (("bands",), np.array(V003_LABELS, dtype=object))})
    root.to_netcdf(dst, engine="h5netcdf", mode="w")
    location.to_netcdf(dst, engine="h5netcdf", mode="a", group="location")
    params.to_netcdf(dst, engine="h5netcdf", mode="a", group="sensor_band_parameters")


@pytest.fixture(scope="module")
def v002_dir(tmp_path_factory) -> Path:
    """Directory holding a v002 RAD and OBS (symlinks to the v001 fixture) and a v003 mask."""
    d = tmp_path_factory.mktemp("emit_v002")
    (d / f"EMIT_L1B_RAD_002_{_DT}.nc").symlink_to(_FIXTURE_RAD)
    (d / f"EMIT_L1B_OBS_002_{_DT}.nc").symlink_to(_FIXTURE_OBS)
    _write_v003_mask(d / f"EMIT_L2A_MASK_003_{_DT}.nc")
    return d


@pytest.fixture(scope="module")
def image_v001() -> emit.EMITImage:
    return emit.EMITImage(str(_FIXTURE_RAD))


@pytest.fixture(scope="module")
def image_v002(v002_dir) -> emit.EMITImage:
    return emit.EMITImage(str(v002_dir / f"EMIT_L1B_RAD_002_{_DT}.nc"))


def _raw_flags(image: emit.EMITImage, band_index) -> np.ndarray:
    slice_y, slice_x = image.window_raw.toslices()
    return image.nc_ds_l2amask["mask"].values[slice_y, slice_x][..., band_index]


@pytestmark_files
class TestV001FileUnchanged:
    @pytest.mark.parametrize("with_buffer, positions", [(False, [0, 1, 3]), (True, [0, 1, 3, 4])])
    def test_invalid_mask_raw_matches_positional(self, image_v001, with_buffer, positions):
        expected = _raw_flags(image_v001, positions).sum(axis=-1) >= 1
        np.testing.assert_array_equal(image_v001.invalid_mask_raw(with_buffer=with_buffer), expected)

    def test_module_valid_mask_matches_image(self, image_v001):
        # Crashed on every file in 2.3.5 (indexed an xarray DataArray with the GLT).
        vm, pct = emit.valid_mask(str(_FIXTURE_MASK), with_buffer=False, dst_crs=None)
        np.testing.assert_array_equal(vm.values, image_v001.validmask(with_buffer=False).values)
        assert pct == pytest.approx(image_v001.percentage_clear)


@pytestmark_files
class TestV002Image:
    def test_resolves_v003_mask_next_to_radiance(self, image_v002, v002_dir):
        assert image_v002.mask_bands.tolist() == V003_LABELS
        assert image_v002.l2amaskfile == str(v002_dir / f"EMIT_L2A_MASK_003_{_DT}.nc")

    def test_resolves_v002_obs_next_to_radiance(self, image_v002, v002_dir):
        assert np.isfinite(image_v002.mean_sza)
        assert image_v002.obs_file == str(v002_dir / f"EMIT_L1B_OBS_002_{_DT}.nc")

    def test_invalid_mask_without_buffer_is_cloud_or_cirrus(self, image_v002):
        expected = _raw_flags(image_v002, [0, 1]).sum(axis=-1) >= 1
        np.testing.assert_array_equal(image_v002.invalid_mask_raw(with_buffer=False), expected)

    def test_invalid_mask_with_buffer_adds_dilated_cloud(self, image_v002):
        expected = _raw_flags(image_v002, [0, 1, 3]).sum(axis=-1) >= 1
        np.testing.assert_array_equal(image_v002.invalid_mask_raw(with_buffer=True), expected)

    def test_probability_band_not_used(self, image_v002):
        # Selecting SpecTf-Cloud Probability (> 0 everywhere) would invalidate every pixel.
        assert image_v002.percentage_clear > 0

    def test_include_spectf_adds_ml_flag(self, image_v002):
        with_spectf = image_v002.invalid_mask_raw(with_buffer=False, include_spectf=True)
        without = image_v002.invalid_mask_raw(with_buffer=False)
        assert with_spectf[_SPECTF_BLOCK].all()
        outside = np.ones_like(without, dtype=bool)
        outside[_SPECTF_BLOCK] = False
        np.testing.assert_array_equal(with_spectf[outside], without[outside])

    def test_v001_and_v003_agree_on_shared_flags(self, image_v001, image_v002):
        """Same Cloud/Cirrus/Dilated flags in both files, so the buffered masks differ only by Spacecraft."""
        spacecraft = _raw_flags(image_v001, 3).astype(bool)
        v001 = image_v001.invalid_mask_raw(with_buffer=True)
        v003 = image_v002.invalid_mask_raw(with_buffer=True)
        np.testing.assert_array_equal(v001 & ~spacecraft, v003 & ~spacecraft)

    def test_validmask_geotensor(self, image_v002):
        vm = image_v002.validmask(with_buffer=False)
        assert vm.values.dtype == bool
        assert vm.shape == image_v002.glt.shape[1:]

    def test_water_mask_v003_label(self, image_v002):
        wm = image_v002.water_mask()
        np.testing.assert_array_equal(np.unique(wm.values[image_v002.valid_glt]),
                                      np.unique(_raw_flags(image_v002, 2)))

    def test_named_mask_case_insensitive(self, image_v002):
        np.testing.assert_array_equal(image_v002.mask("cloud flag").values,
                                      image_v002.mask("Cloud Flag").values)

    def test_module_valid_mask_matches_image(self, image_v002, v002_dir):
        vm, pct = emit.valid_mask(str(v002_dir / f"EMIT_L2A_MASK_003_{_DT}.nc"), with_buffer=False, dst_crs=None)
        np.testing.assert_array_equal(vm.values, image_v002.validmask(with_buffer=False).values)
        assert pct == pytest.approx(image_v002.percentage_clear)


@pytestmark_files
def test_v002_downloads_v003_mask_when_missing(tmp_path, v002_dir, monkeypatch):
    rad = tmp_path / f"EMIT_L1B_RAD_002_{_DT}.nc"
    rad.symlink_to(_FIXTURE_RAD)
    calls = []

    def fake_download(link, filename, *args, **kwargs):
        calls.append((link, filename))
        shutil.copy(v002_dir / f"EMIT_L2A_MASK_003_{_DT}.nc", filename)
        return filename

    monkeypatch.setattr(emit, "download_product", fake_download)
    image = emit.EMITImage(str(rad))
    assert image.mask_bands.tolist() == V003_LABELS
    assert calls == [(
        f"{emit.DAAC_URL}/EMITL2AMASK.003/EMIT_L2A_MASK_003_{_DT}/EMIT_L2A_MASK_003_{_DT}.nc",
        str(tmp_path / f"EMIT_L2A_MASK_003_{_DT}.nc"),
    )]
