import xarray as xr
from georeader import read, rasterio_reader
import rasterio
import rasterio.windows
import numpy as np
import pytest
import math
import itertools

FILE_TEST = "gs://wotus_2020/taudem/SD/dem.tif"

WINDOW_DEM = rasterio.windows.Window.from_slices(slice(73_491, 74_000, None),
                                                 slice(30_881, 31_000, None))

FILE_TEST_PLANET = "gs://wotus_2020/PlanetCOG/SD_train_0/2018-06-15.tif" # (4, 25517, 7705)

WINDOW_OUT_PLANET = rasterio.windows.Window(col_off=-10, row_off=-15, width=128, height=64)
WINDOW_OUT_PLANET_2 = rasterio.windows.Window(col_off=-10, row_off=25510, width=64, height=128)
WINDOW_OUT_PLANET_3 = rasterio.windows.Window(col_off=7706, row_off=20, width=128, height=128)  # Out of bounds
WINDOW_OUT_PLANET_4 = rasterio.windows.Window(col_off=7700, row_off=25518, width=64, height=128) # out of bounds
WINDOW_OUT_PLANET_5 =  rasterio.windows.Window(col_off=7700, row_off=25505, width=128, height=128)
WINDOW_OUT_PLANET_6 = rasterio.windows.Window(col_off=7577, row_off=20, width=128, height=128)  # Border case
WINDOW_OUT_PLANET_7 = rasterio.windows.Window(col_off=7578, row_off=20, width=128, height=256)  # Border case
WINDOW_OUT_PLANET_8 = rasterio.windows.Window(col_off=7578, row_off=25389, width=128, height=128)  # Border case
WINDOW_OUT_PLANET_9 = rasterio.windows.Window(col_off=7578, row_off=25390, width=128, height=128)  # Border case
WINDOW_OUT_PLANET_10 = rasterio.windows.Window(col_off=7578, row_off=25388, width=128, height=128)  # Border case
WINDOW_OUT_PLANET_11 = rasterio.windows.Window(col_off=-129, row_off=34, width=128, height=64) # Out of bounds
WINDOW_OUT_PLANET_12 = rasterio.windows.Window(col_off=-1, row_off=-129, width=14, height=64) # Out of bounds


@pytest.mark.parametrize("fil", [FILE_TEST, FILE_TEST_PLANET])
def test_transform_bounds(fil):
    data_array = xr.open_rasterio(fil)
    bounds, transform = xarray_utils.coords_to_bounds_transform(data_array.coords)

    with rasterio.open(fil) as src:
        exp_bounds = src.bounds
        exp_transform = src.transform
    assert (exp_bounds[0] == exp_transform.c) and (
                exp_bounds[3] == exp_transform.f), f"Sanity check, we go insane? {exp_bounds} {exp_transform}"

    assert bounds == exp_bounds, f"Different bounds {bounds} expected: {exp_bounds}"
    assert transform == exp_transform, f"Different transform {transform} expected: {exp_transform}"


TESTS_WINDOW = [(FILE_TEST, WINDOW_DEM),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_2),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_3),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_4),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_5),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_6),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_7),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_8),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_9),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_10),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_11),
                (FILE_TEST_PLANET, WINDOW_OUT_PLANET_12)]


FILE_INFO = list(itertools.product(TESTS_WINDOW, [True, False], [True, False]))


@pytest.mark.parametrize("file_info", FILE_INFO)
def test_read_window(file_info):
    (fil, window), use_xarray, trigger_load = file_info
    if use_xarray:
        data_array = xr.open_rasterio(fil)
        data_array.attrs["bounds"], data_array.attrs["transform"] = xarray_utils.coords_to_bounds_transform(data_array.coords)
    else:
        data_array = rasterio_reader.RasterioReader([fil])

    chip_out = xarray_utils.read_from_window(data_array, window, trigger_load=trigger_load)

    named_shape = dict(zip(chip_out.dims, chip_out.shape))
    assert named_shape["y"] == window.height, f"Different height found: ({named_shape['y']}, {named_shape['x']}) expected ({window.height}, {window.width})"
    assert named_shape["x"] == window.width, f"Different width found: ({named_shape['y']}, {named_shape['x']}) expected ({window.height}, {window.width})"

    # check the same data is read with rasterio
    with rasterio.open(fil) as src:
        chip_out_expected = src.read(window=window, boundless=True, fill_value=0)
        expected_transform = rasterio.windows.transform(window, src.transform)
        expected_bounds = rasterio.windows.bounds(window, src.transform)

    bounds, transform = xarray_utils.coords_to_bounds_transform(chip_out.coords)

    assert bounds == expected_bounds, f"Different bounds found: {bounds} expected: {expected_bounds}"
    assert transform == expected_transform, f"Different transform found: {transform} expected: {expected_transform}"

    assert chip_out.bounds == expected_bounds, f"Different bounds found: {bounds} expected: {expected_bounds}"
    assert chip_out.transform == expected_transform, f"Different transform found: {transform} expected: {expected_transform}"

    expected_values = chip_out.values
    if not use_xarray:
        expected_values = expected_values[0]

    assert np.allclose(chip_out_expected, expected_values), "Content of the array is different"


@pytest.mark.parametrize("file_info", FILE_INFO)
def test_read_bounds(file_info):
    (fil, window), use_xarray, trigger_load = file_info
    if use_xarray:
        data_array = xr.open_rasterio(fil)
        data_array.attrs["bounds"], data_array.attrs["transform"] = xarray_utils.coords_to_bounds_transform(
            data_array.coords)
    else:
        data_array = rasterio_reader.RasterioReader([fil])

    with rasterio.open(fil) as src:
        chip_out_expected = src.read(window=window, boundless=True, fill_value=0)
        expected_transform = rasterio.windows.transform(window, src.transform)
        bounds_read = rasterio.windows.bounds(window, src.transform)
        crs_bounds = src.crs

    chip_out = xarray_utils.read_from_bounds(data_array, bounds_read, crs_bounds=crs_bounds, trigger_load=trigger_load)

    named_shape = dict(zip(chip_out.dims, chip_out.shape))
    assert named_shape[
               "y"] == window.height, f"Different height found: ({named_shape['y']}, {named_shape['x']}) expected ({window.height}, {window.width})"
    assert named_shape[
               "x"] == window.width, f"Different width found: ({named_shape['y']}, {named_shape['x']}) expected ({window.height}, {window.width})"

    bounds, transform = xarray_utils.coords_to_bounds_transform(chip_out.coords)

    assert bounds == bounds_read, f"Different bounds found: {bounds} expected: {bounds_read}"
    assert transform == expected_transform, f"Different transform found: {transform} expected: {expected_transform}"
    assert chip_out.bounds == bounds_read, f"Different bounds found: {bounds} expected: {bounds_read}"
    assert chip_out.transform == expected_transform, f"Different transform found: {transform} expected: {expected_transform}"

    expected_values = chip_out.values
    if not use_xarray:
        expected_values = expected_values[0]

    assert np.allclose(chip_out_expected, expected_values), "Content of the array is different"


@pytest.mark.parametrize("file_info", TESTS_WINDOW)
def test_read_reproject_same(file_info):
    fil, window = file_info
    data_array = xr.open_rasterio(fil)
    data_array.attrs["bounds"], data_array.attrs["transform"] = xarray_utils.coords_to_bounds_transform(data_array.coords)

    with rasterio.open(fil) as src:
        chip_out_expected = src.read(window=window, boundless=True, fill_value=0)
        expected_transform = rasterio.windows.transform(window, src.transform)
        bounds_read = rasterio.windows.bounds(window, src.transform)
        crs_bounds = src.crs

    chip_out = xarray_utils.read_reproject(data_array, bounds_read, dst_crs=crs_bounds,
                                           resolution_dst_crs=(abs(expected_transform.a), abs(expected_transform.e)))

    named_shape = dict(zip(chip_out.dims, chip_out.shape))
    assert named_shape[
               "y"] == window.height, f"Different height found: ({named_shape['y']}, {named_shape['x']}) expected ({window.height}, {window.width})"
    assert named_shape[
               "x"] == window.width, f"Different width found: ({named_shape['y']}, {named_shape['x']}) expected ({window.height}, {window.width})"

    bounds, transform = xarray_utils.coords_to_bounds_transform(chip_out.coords)

    assert bounds == bounds_read, f"Different bounds found: {bounds} expected: {bounds_read}"
    assert transform == expected_transform, f"Different transform found: {transform} expected: {expected_transform}"

    assert np.allclose(chip_out_expected, chip_out.values), "Content of the array is different"


RESOLUTION_TEST = (0.5, 0.5)
@pytest.mark.parametrize("file_info", TESTS_WINDOW)
def test_read_reproject_other_res(file_info):
    fil, window = file_info
    data_array = xr.open_rasterio(fil)
    data_array.attrs["bounds"], data_array.attrs["transform"] = xarray_utils.coords_to_bounds_transform(data_array.coords)

    with rasterio.open(fil) as src:
        expected_transform = rasterio.windows.transform(window, src.transform)
        bounds_read = rasterio.windows.bounds(window, src.transform)
        factor_diff_shape = np.array(src.res) / np.array(RESOLUTION_TEST)
        crs_bounds = src.crs

    chip_out = xarray_utils.read_reproject(data_array, bounds_read, dst_crs=crs_bounds,
                                           resolution_dst_crs=RESOLUTION_TEST)

    named_shape = dict(zip(chip_out.dims, chip_out.shape))
    assert named_shape[
               "y"] == (window.height * factor_diff_shape[1]), f"Different height found: ({named_shape['y']}, {named_shape['x']}) expected ({window.height}, {window.width})"
    assert named_shape[
               "x"] == (window.width * factor_diff_shape[0]), f"Different width found: ({named_shape['y']}, {named_shape['x']}) expected ({window.height}, {window.width})"

    bounds, transform = xarray_utils.coords_to_bounds_transform(chip_out.coords)

    assert bounds == bounds_read, f"Different bounds found: {bounds} expected: {bounds_read}"

    resolution_signed = (math.copysign(RESOLUTION_TEST[0], expected_transform.a),
                         math.copysign(RESOLUTION_TEST[1], expected_transform.e))

    expected_transform = rasterio.Affine.translation(expected_transform.c, expected_transform.f) * rasterio.Affine.scale(*resolution_signed)

    assert transform == expected_transform, f"Different transform found: {transform} expected: {expected_transform}"


def read_after_set_window():
    window_focus = rasterio.windows.Window(col_off=75615, row_off=40643, width=2310, height=2310)
    paths = [FILE_TEST]
    rst1 = rasterio_reader.RasterioReader(paths)

    rst2 = rasterio_reader.RasterioReader(paths)
    rst2.set_window(window_focus)

    bounds_read = (604727.9999991455, 4961265.000119502, 604830.9999991446, 4961415.000119504)
    crs_bounds = {"init": "epsg:32613"}

    data1 = xarray_utils.read_from_bounds(rst1, bounds_read, crs_bounds=crs_bounds)
    data2 = xarray_utils.read_from_bounds(rst2, bounds_read, crs_bounds=crs_bounds)

    assert data1.bounds == data2.bounds, f"Different bounds found: {data1.bounds} expected: {data2.bounds}"
    assert data1.transform == data2.transform, f"Different transform found: {data1.transform} expected: {data2.transform}"

    assert np.allclose(data1.values, data2.values), "Content of the array is different"





