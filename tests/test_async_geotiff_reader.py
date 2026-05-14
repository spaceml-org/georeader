"""
Tests for georeader.async_geotiff_reader.AsyncGeoTIFFReader.

The tests skip cleanly when the optional ``async`` extra (``async-geotiff``
plus an obstore backend) isn't installed, so they don't fail in lean
environments.

Covers:
- Metadata properties match RasterioReader for the same fixture.
- ``read_from_window`` produces a GeoTensor numerically equivalent to
  RasterioReader's read of the same window.
- ``read_from_bounds(target_crs=...)`` raises NotImplementedError (anti-goal
  documented in the design).
- Concurrent fan-out via ``asyncio.gather`` completes without errors.
- ``async with`` context manager works.
"""

import os
import tempfile

import numpy as np
import pytest
import rasterio
import rasterio.windows
from rasterio.transform import from_origin

# Skip the entire module if the optional async-geotiff stack isn't available.
async_geotiff = pytest.importorskip("async_geotiff")
obstore = pytest.importorskip("obstore")

from georeader.abstract_reader import AsyncGeoData  # noqa: E402
from georeader.async_geotiff_reader import AsyncGeoTIFFReader  # noqa: E402
from georeader.rasterio_reader import RasterioReader  # noqa: E402


@pytest.fixture(scope="module")
def cog_fixture():
    """A small tiled GeoTIFF + a LocalStore + the filename relative to that store."""
    tmpdir = tempfile.mkdtemp()
    fname = "demo.tif"
    path = os.path.join(tmpdir, fname)

    np.random.seed(0)
    data = np.random.randint(0, 5000, size=(3, 64, 64), dtype=np.int16)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=64, width=64, count=3, dtype=data.dtype,
        crs="EPSG:32631", transform=from_origin(500000.0, 4600000.0, 10.0, 10.0),
        tiled=True, blockxsize=32, blockysize=32, compress="deflate",
    ) as dst:
        dst.write(data)

    store = obstore.store.LocalStore(prefix=tmpdir)
    yield {"store": store, "fname": fname, "abs_path": path, "tmpdir": tmpdir}

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestAsyncGeoTIFFReader:
    """Smoke + parity tests for AsyncGeoTIFFReader."""

    @pytest.mark.asyncio
    async def test_open_populates_metadata(self, cog_fixture):
        """``await open()`` returns a reader whose metadata properties match the source file."""
        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        # Conforms to AsyncGeoData (structural — no inheritance check).
        assert isinstance(reader, AsyncGeoData)
        # Metadata matches the fixture.
        assert reader.shape == (3, 64, 64)
        assert reader.dtype == np.int16
        assert reader.dims == ["band", "y", "x"]
        # bounds is inherited from GeoDataBase default — matches the from_origin transform.
        assert reader.bounds == (500000.0, 4599360.0, 500640.0, 4600000.0)

    def test_metadata_before_open_raises(self, cog_fixture):
        """Accessing metadata before ``open()`` raises a clear RuntimeError."""
        reader = AsyncGeoTIFFReader(cog_fixture["fname"], store=cog_fixture["store"])

        with pytest.raises(RuntimeError, match="not opened"):
            _ = reader.crs

    @pytest.mark.asyncio
    async def test_read_parity_with_rasterio_reader(self, cog_fixture):
        """A windowed async read returns the same bytes as RasterioReader on the same window."""
        async_reader = await AsyncGeoTIFFReader.open(
            cog_fixture["fname"], store=cog_fixture["store"],
        )
        rio_reader = RasterioReader(cog_fixture["abs_path"])

        win = rasterio.windows.Window(col_off=8, row_off=4, width=32, height=24)
        async_gt = await async_reader.read_from_window(win)
        rio_gt = rio_reader.read_from_window(win).load()

        assert async_gt.values.shape == (3, 24, 32)
        assert np.array_equal(async_gt.values, rio_gt.values)

    @pytest.mark.asyncio
    async def test_load_returns_full_extent(self, cog_fixture):
        """``await load()`` reads the whole raster and matches a direct rasterio read."""
        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        gt = await reader.load()
        with rasterio.open(cog_fixture["abs_path"]) as src:
            expected = src.read()

        assert gt.values.shape == (3, 64, 64)
        assert np.array_equal(gt.values, expected)

    @pytest.mark.asyncio
    async def test_read_bounds_warp_not_implemented(self, cog_fixture):
        """``read_from_bounds(target_crs=...)`` raises ``NotImplementedError``.

        async-geotiff explicitly disclaims warp/resample; this reader follows
        suit and surfaces the limitation up front rather than silently
        falling back.
        """
        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        bounds = reader.bounds
        with pytest.raises(NotImplementedError, match="warp or resample"):
            await reader.read_from_bounds(bounds, target_crs="EPSG:4326")
        with pytest.raises(NotImplementedError, match="warp or resample"):
            await reader.read_from_bounds(bounds, target_resolution=(20.0, 20.0))

    @pytest.mark.asyncio
    async def test_concurrent_fan_out(self, cog_fixture):
        """``asyncio.gather`` across many windows from one reader completes successfully.

        Doesn't claim a speedup against this trivial local fixture — the
        point is to prove the reader survives concurrent ``await`` calls,
        which is the actual use case (tile servers fanning out 100s of
        reads).
        """
        import asyncio

        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        windows = [
            rasterio.windows.Window(col_off=c, row_off=r, width=16, height=16)
            for r in range(0, 64, 16) for c in range(0, 64, 16)
        ]
        results = await asyncio.gather(*[reader.read_from_window(w) for w in windows])

        assert len(results) == 16
        for gt in results:
            assert gt.values.shape == (3, 16, 16)

    @pytest.mark.asyncio
    async def test_async_context_manager(self, cog_fixture):
        """The ``async with`` context manager opens lazily and closes cleanly."""
        reader = AsyncGeoTIFFReader(cog_fixture["fname"], store=cog_fixture["store"])

        assert reader._geotiff is None
        async with reader:
            assert reader._geotiff is not None
            gt = await reader.load()
            assert gt.values.shape == (3, 64, 64)

    @pytest.mark.asyncio
    async def test_repr(self, cog_fixture):
        """``__repr__`` reflects open status."""
        reader = AsyncGeoTIFFReader(cog_fixture["fname"], store=cog_fixture["store"])
        assert "unopened" in repr(reader)

        await reader._open_geotiff()
        assert "opened" in repr(reader)
