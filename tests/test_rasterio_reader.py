
from georeader import rasterio_reader, read
import rasterio
import rasterio.windows
import numpy as np

WINDOW_PLANET = rasterio.windows.Window(col_off=1000, row_off=1000, width=128, height=64)
# WINDOW_PLANET_OUT_1 = rasterio.windows.Window(col_off=-10, row_off=-10, width=128, height=64)

FILE_TEST_PLANET = "gs://wotus_2020/PlanetCOG/SD_train_0/2018-06-15.tif" # (4, 25517, 7705)


def test_read_indexes():
    window, file = WINDOW_PLANET, FILE_TEST_PLANET
    reader = rasterio_reader.RasterioReader([file], window_focus=window)
    reader.set_indexes([2,3], relative=True)

    data = reader.read()
    assert data.shape == (1, 2, window.height, window.width), f"Expected {(1, 2, window.height, window.width)} found {data.shape}"

    data = reader.read(indexes=2)
    assert data.shape == (1, window.height, window.width), f"Expected {(1, window.height, window.width)} found {data.shape}"

    with rasterio.open(file) as src:
        data_expected = src.read(window=window, indexes=3)

    assert np.allclose(data, data_expected), "Content of the array is different"

    # Same but with stack=False
    reader = rasterio_reader.RasterioReader([file], window_focus=window, stack=False)
    reader = reader.isel({"band": [1, 2]})
    data = reader.values

    assert data.shape == (2, window.height, window.width), f"Expected {(2, window.height, window.width)} found {data.shape}"

    data = reader.read(indexes=2)
    assert data.shape == (window.height, window.width), f"Expected {(window.height, window.width)} found {data.shape}"

    assert np.allclose(data, data_expected), "Content of the array is different"


def test_read_out_shape():
    window, file = WINDOW_PLANET, FILE_TEST_PLANET
    reader = rasterio_reader.RasterioReader([file], window_focus=window)
    reader.set_indexes([2, 3], relative=True)
    data = reader.read(indexes=2, out_shape=(64, 32))
    assert data.shape == (
    1, 64, 32), f"Expected {(1, 64, 32)} found {data.shape}"


def test_read_boundless_false():
    window, file = rasterio.windows.Window(col_off=-10, row_off=-10, width=128, height=64), FILE_TEST_PLANET
    reader = rasterio_reader.RasterioReader(file, window_focus=None)
    reader_subset = read.read_from_window(reader, window=window, boundless=False)
    xr_subset = read.read_from_window(reader, window=window, boundless=True).load(boundless=False)

    assert reader_subset.shape == (4, window.height+window.row_off, window.width+window.col_off), \
        f"Unexpected shape {reader_subset.shape} {(4, window.height+window.row_off, window.width+window.col_off)}"

    assert reader_subset.shape == xr_subset.shape, "Unexpected shapes"

    expected_window = rasterio.windows.Window(col_off=0, row_off=0, width=118, height=54)
    assert reader_subset.window_focus == expected_window, "Different windows"

    expected_bounds = rasterio.windows.bounds(expected_window, reader.transform)

    assert reader.transform == reader_subset.transform, "Expected same transform"

    assert xr_subset.transform == reader_subset.transform, "Expected same transform"
    assert xr_subset.bounds == reader_subset.bounds, "Expected same bounds"
    assert xr_subset.bounds == expected_bounds, "Expected same bounds"


def test_isel():
    window, file = WINDOW_PLANET, FILE_TEST_PLANET
    reader = rasterio_reader.RasterioReader([file], window_focus=window)
    reader.set_indexes([2, 3], relative=True)
    data = reader.isel({"time": 0}).values
    assert data.shape == (2, window.height, window.width), f"Expected {(2, window.height, window.width)} found {data.shape}"











