from georeader import rasterio_reader, geotensor, read
import rasterio.windows
import torch
import numpy as np
import itertools


FILE_TEST_PLANET = "gs://wotus_2020/PlanetCOG/SD_train_0/2018-06-15.tif" # (4, 25517, 7705)
WINDOW_OUT_PLANET = rasterio.windows.Window(col_off=10, row_off=10, width=138, height=148)

WINDOW_NORMAL = rasterio.windows.Window(col_off=10, row_off=5, width=30, height=20)
WINDOW_OUT_1 = rasterio.windows.Window(col_off=-10, row_off=5, width=30, height=20)
WINDOW_OUT_2 = rasterio.windows.Window(col_off=1, row_off=-5, width=30, height=20)
WINDOW_OUT_3 = rasterio.windows.Window(col_off=120, row_off=5, width=30, height=20)
WINDOW_OUT_4 = rasterio.windows.Window(col_off=1, row_off=130, width=30, height=20)

def test_read_window():
    window = WINDOW_OUT_PLANET
    rst_obj = rasterio_reader.RasterioReader(FILE_TEST_PLANET, window_focus=window)
    # C, H, W = "band", "y", "x"
    gtobj = rst_obj.load()

    assert rst_obj.shape == (4, window.height, window.width), f"Unexpected shape {gtobj.shape} {(1, window.height, window.width)}"
    assert gtobj.shape == (4, window.height, window.width), f"Unexpected shape {gtobj.shape} {(1, window.height, window.width)}"

    # Convert to torch.Tensor internal object
    gtobj.values = torch.tensor(gtobj.values)
    assert gtobj.shape == gtobj.shape, f"Unexpected shape {gtobj.shape} {rst_obj.shape}"

    assert rst_obj.width == gtobj.width, f"Unexpected width {rst_obj.width} {gtobj.width}"
    assert rst_obj.count == gtobj.count, f"Unexpected count {rst_obj.count} {gtobj.count}"
    assert rst_obj.height == gtobj.height, f"Unexpected height {rst_obj.height} {gtobj.height}"

    for subwindow, boundless in itertools.product([WINDOW_NORMAL, WINDOW_OUT_1,
                                                   WINDOW_OUT_2, WINDOW_OUT_3, WINDOW_OUT_4], [True, False]):

        rst_obj_isel = read.read_from_window(rst_obj, window=subwindow, boundless=boundless,
                                                     trigger_load=False)
        assert isinstance(rst_obj_isel, rasterio_reader.RasterioReader), f"Incorrect class {rst_obj_isel}"

        xarray_obj_isel = read.read_from_window(gtobj, window=subwindow, boundless=boundless)
        xarray_obj_isel_from_rst_obj_isel = rst_obj_isel.load(boundless=boundless)
        gtobj_isel = read.read_from_window(gtobj, window=subwindow, boundless=boundless)

        assert xarray_obj_isel.shape == gtobj_isel.shape, f"Different shapes {subwindow} {boundless}"

        assert rst_obj_isel.shape == xarray_obj_isel_from_rst_obj_isel.shape, f"Different shapes {subwindow} {boundless}"

        if boundless:
            assert xarray_obj_isel.shape == xarray_obj_isel_from_rst_obj_isel.shape, f"Different shapes {subwindow} {boundless}"
            assert rst_obj_isel.transform == gtobj_isel.transform, f"Different transforms {subwindow} {boundless}"
            assert rst_obj_isel.bounds == gtobj_isel.bounds, f"Different bounds {subwindow} {boundless}"

            assert gtobj_isel.bounds == xarray_obj_isel_from_rst_obj_isel.bounds, f"Different bounds {subwindow} {boundless}"
            assert gtobj_isel.transform == xarray_obj_isel_from_rst_obj_isel.transform, f"Different transforms {subwindow} {boundless}"

        assert gtobj_isel.transform == xarray_obj_isel.transform, f"Different transforms {subwindow} {boundless}"
        assert gtobj_isel.bounds == xarray_obj_isel.bounds, f"Different bounds {subwindow} {boundless}"

        assert np.allclose(xarray_obj_isel.values, np.array(gtobj_isel.values)), f"Content of the array is different {subwindow} {boundless}"

        # assert np.allclose(xarray_obj_isel.values,
        #                   xarray_obj_isel_from_rst_obj_isel.values), f"Content of the array is different {subwindow} {boundless}"
