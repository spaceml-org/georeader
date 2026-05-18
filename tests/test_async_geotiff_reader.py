"""
Tests for georeader.async_geotiff_reader.AsyncGeoTIFFReader.

The tests skip cleanly when the optional ``async`` extra (``async-geotiff``
plus an obstore backend) isn't installed, so they don't fail in lean
environments.

Covers:
- Metadata properties match RasterioReader for the same fixture.
- ``read_from_window`` returns a sync windowed view; ``await view.load()``
  produces a GeoTensor numerically equivalent to RasterioReader's read of
  the same window.
- Concurrent fan-out via ``asyncio.gather`` (over ``view.load()``) completes
  without errors.
- ``async with`` context manager works.
"""

import asyncio
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


@pytest.fixture(scope="module")
def cog_with_nodata_and_overviews():
    """A 256x256 tiled COG with explicit nodata=-9999 and a 2x/4x overview ladder.

    Used to exercise paths that the default ``cog_fixture`` doesn't:
    the ``nodata is not None`` branch of ``fill_value_default``,
    overview reads, and ``block_windows`` over multiple tile rows.
    """
    from rasterio.enums import Resampling
    tmpdir = tempfile.mkdtemp()
    fname = "withnodata.tif"
    path = os.path.join(tmpdir, fname)

    np.random.seed(1)
    data = np.random.randint(-1000, 5000, size=(3, 256, 256), dtype=np.int16)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=256, width=256, count=3, dtype=data.dtype,
        crs="EPSG:32631", transform=from_origin(500000.0, 4600000.0, 10.0, 10.0),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999,
    ) as dst:
        dst.write(data)
    with rasterio.open(path, "r+") as ds:
        ds.build_overviews([2, 4], Resampling.average)

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
        """Windowed view + load returns the same bytes as RasterioReader on the same window.

        Both readers follow the lazy-view + load pattern: ``read_from_window``
        is sync (returns a windowed view), ``load`` materialises.
        """
        async_reader = await AsyncGeoTIFFReader.open(
            cog_fixture["fname"], store=cog_fixture["store"],
        )
        rio_reader = RasterioReader(cog_fixture["abs_path"])

        win = rasterio.windows.Window(col_off=8, row_off=4, width=32, height=24)
        async_view = async_reader.read_from_window(win)
        # The view itself reports the windowed shape — no I/O yet.
        assert async_view.shape == (3, 24, 32)
        async_gt = await async_view.load()
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
    async def test_concurrent_fan_out(self, cog_fixture):
        """``asyncio.gather`` across many windowed-view loads completes successfully.

        The actual async I/O is in ``.load()``; ``read_from_window`` is
        sync window math. Fan-out pattern: build the views (sync),
        gather their loads (async).
        """
        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        windows = [
            rasterio.windows.Window(col_off=c, row_off=r, width=16, height=16)
            for r in range(0, 64, 16) for c in range(0, 64, 16)
        ]
        results = await asyncio.gather(
            *[reader.read_from_window(w).load() for w in windows]
        )

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
        """``__repr__`` is terse when unopened and shows metadata when opened."""
        reader = AsyncGeoTIFFReader(cog_fixture["fname"], store=cog_fixture["store"])
        assert "unopened" in repr(reader)

        await reader._open_geotiff()
        r = repr(reader)
        # Rich opened repr shows the same fields as RasterioReader.
        for field in ("path_or_url", "Shape", "Resolution", "Bounds", "CRS", "Transform"):
            assert field in r

    @pytest.mark.asyncio
    async def test_read_from_window_boundless(self, cog_fixture):
        """``boundless=True`` view pads to window shape on load; ``boundless=False`` clips at construction.

        Mirrors RasterioReader.read_from_window's boundless contract: edge
        windows return the full requested shape under boundless=True (with
        ``fill_value_default`` in the out-of-bounds region) and a clipped
        shape under boundless=False.
        """
        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        # 64x64 raster, request a 20x20 window starting at (60, 60) — only
        # the bottom-right 4x4 is inside the raster.
        edge_window = rasterio.windows.Window(col_off=60, row_off=60, width=20, height=20)

        boundless_view = reader.read_from_window(edge_window, boundless=True)
        clipped_view = reader.read_from_window(edge_window, boundless=False)

        # Views report the right shape before any I/O.
        assert boundless_view.shape == (3, 20, 20)
        assert clipped_view.shape == (3, 4, 4)

        boundless_gt = await boundless_view.load(boundless=True)
        clipped_gt = await clipped_view.load(boundless=False)

        assert boundless_gt.values.shape == (3, 20, 20)
        assert clipped_gt.values.shape == (3, 4, 4)
        # The clipped tensor should equal the inner 4x4 of the boundless tensor.
        assert np.array_equal(clipped_gt.values, boundless_gt.values[:, :4, :4])

    @pytest.mark.asyncio
    async def test_read_from_window_no_intersection_raises(self, cog_fixture):
        """``read_from_window(boundless=False)`` with a disjoint window raises ``WindowError`` synchronously.

        ``boundless=True`` does NOT raise at view-construction time
        (matches RasterioReader.set_window), but raises on load.
        """
        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        outside = rasterio.windows.Window(col_off=200, row_off=200, width=10, height=10)
        with pytest.raises(rasterio.windows.WindowError):
            reader.read_from_window(outside, boundless=False)

        # boundless=True view construction is permissive; the WindowError
        # surfaces on load via window_utils.get_slice_pad.
        view = reader.read_from_window(outside, boundless=True)
        with pytest.raises(rasterio.windows.WindowError):
            await view.load(boundless=True)

    # ----------------------------------------------- coverage gap #7
    def test_read_before_open_raises(self, cog_fixture):
        """``read_from_window`` and ``load`` raise ``RuntimeError`` before ``open()``.

        Complements ``test_metadata_before_open_raises`` — the same
        ``_require_open()`` guard should catch read attempts, not just
        property access.
        """
        reader = AsyncGeoTIFFReader(cog_fixture["fname"], store=cog_fixture["store"])

        with pytest.raises(RuntimeError, match="not opened"):
            reader.read_from_window(rasterio.windows.Window(0, 0, 4, 4))

        with pytest.raises(RuntimeError, match="not opened"):
            asyncio.run(reader.load())

    # ----------------------------------------------- coverage gap #2
    @pytest.mark.asyncio
    async def test_nested_read_from_window_view_of_view(self, cog_fixture):
        """``reader.read_from_window(w1).read_from_window(w2)`` translates correctly.

        ``w2`` is interpreted relative to the view's window, not the
        underlying raster — matches :meth:`RasterioReader.set_window`
        with ``relative=True``. Guards against off-by-one in the
        relative→absolute translation.
        """
        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        outer = rasterio.windows.Window(col_off=16, row_off=8, width=32, height=24)
        inner = rasterio.windows.Window(col_off=4, row_off=2, width=8, height=12)

        # Equivalent absolute window (outer offsets + inner offsets).
        absolute = rasterio.windows.Window(col_off=20, row_off=10, width=8, height=12)

        nested_gt = await reader.read_from_window(outer).read_from_window(inner).load()
        absolute_gt = await reader.read_from_window(absolute).load()

        assert nested_gt.values.shape == (3, 12, 8)
        assert nested_gt.transform == absolute_gt.transform
        assert np.array_equal(nested_gt.values, absolute_gt.values)

    # ----------------------------------------------- coverage gap #4
    @pytest.mark.asyncio
    async def test_fill_value_default_uses_explicit_nodata(
        self, cog_with_nodata_and_overviews,
    ):
        """When the COG has ``nodata=N``, ``fill_value_default`` returns ``N``.

        The default fixture has no nodata tag, so this branch is
        otherwise untested. Verifies edge-window padding under
        ``boundless=True`` uses the explicit nodata value (-9999).
        """
        reader = await AsyncGeoTIFFReader.open(
            cog_with_nodata_and_overviews["fname"],
            store=cog_with_nodata_and_overviews["store"],
        )
        assert reader.fill_value_default == -9999

        # 256x256 raster — request a 16x16 window straddling the right edge.
        edge_window = rasterio.windows.Window(col_off=250, row_off=8, width=16, height=16)
        gt = await reader.read_from_window(edge_window).load(boundless=True)

        assert gt.values.shape == (3, 16, 16)
        # The rightmost 10 cols (250..256 is 6 valid, then 10 padded) are -9999.
        assert (gt.values[:, :, 6:] == -9999).all()

    # ----------------------------------------------- coverage gap #5
    @pytest.mark.asyncio
    async def test_bounds_and_footprint_reflect_window_focus(self, cog_fixture):
        """Focused-view ``bounds`` and ``footprint`` derive from the focus, not the full raster."""
        reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])

        view = reader.read_from_window(rasterio.windows.Window(col_off=8, row_off=4, width=32, height=24))

        # Bounds should match the windowed transform/shape, NOT the full raster.
        assert view.bounds != reader.bounds
        # At 10m/pixel, origin (500000, 4600000) top-left, y descending:
        #   window (col=8, row=4, w=32, h=24)
        #   ⇒ top-left geo: (500000 + 80, 4600000 - 40) = (500080, 4599960)
        #   ⇒ bottom-right:  (500080 + 320, 4599960 - 240) = (500400, 4599720)
        assert view.bounds == (500080.0, 4599720.0, 500400.0, 4599960.0)

        pol = view.footprint()
        assert pol.bounds == view.bounds

    # ----------------------------------------------- overviews / reader_overview (gap #1)
    @pytest.mark.asyncio
    async def test_overviews_match_rasterio_reader(self, cog_with_nodata_and_overviews):
        """``overviews()`` returns the same decimation factors as RasterioReader."""
        reader = await AsyncGeoTIFFReader.open(
            cog_with_nodata_and_overviews["fname"],
            store=cog_with_nodata_and_overviews["store"],
        )
        rio = RasterioReader(cog_with_nodata_and_overviews["abs_path"])

        assert reader.overviews() == rio.overviews()
        # The fixture built [2, 4]; assert literal too as a sanity check.
        assert reader.overviews() == [2, 4]

    @pytest.mark.asyncio
    async def test_reader_overview_returns_correctly_sized_reader(
        self, cog_with_nodata_and_overviews,
    ):
        """``reader_overview(level)`` returns a reader pinned to that overview's shape."""
        reader = await AsyncGeoTIFFReader.open(
            cog_with_nodata_and_overviews["fname"],
            store=cog_with_nodata_and_overviews["store"],
        )

        # Index 0 = first overview = half resolution.
        ov0 = reader.reader_overview(0)
        assert ov0.shape == (3, 128, 128)
        gt0 = await ov0.load()
        assert gt0.values.shape == (3, 128, 128)

        # Negative indexing: -1 = coarsest overview.
        ovn1 = reader.reader_overview(-1)
        assert ovn1.shape == (3, 64, 64)

    @pytest.mark.asyncio
    async def test_reader_overview_warns_when_window_focus_set(
        self, cog_with_nodata_and_overviews,
    ):
        """``reader_overview`` warns and resets window_focus (parent's TODO behavior)."""
        import warnings
        reader = await AsyncGeoTIFFReader.open(
            cog_with_nodata_and_overviews["fname"],
            store=cog_with_nodata_and_overviews["store"],
        )
        focused = reader.read_from_window(rasterio.windows.Window(0, 0, 64, 64))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ov = focused.reader_overview(0)
            assert any("window_focus" in str(rec.message) for rec in w)
        # The overview reader is at full extent (focus reset).
        assert ov.shape == (3, 128, 128)

    # ----------------------------------------------- block_windows
    @pytest.mark.asyncio
    async def test_block_windows_match_rasterio_reader(self, cog_with_nodata_and_overviews):
        """``block_windows()`` returns the same (index, window) sequence as RasterioReader."""
        reader = await AsyncGeoTIFFReader.open(
            cog_with_nodata_and_overviews["fname"],
            store=cog_with_nodata_and_overviews["store"],
        )
        rio = RasterioReader(cog_with_nodata_and_overviews["abs_path"])

        async_blocks = reader.block_windows()
        rio_blocks = rio.block_windows()

        assert len(async_blocks) == len(rio_blocks)
        for (a_idx, a_win), (r_idx, r_win) in zip(async_blocks, rio_blocks):
            assert a_idx == r_idx
            assert (a_win.col_off, a_win.row_off, a_win.width, a_win.height) == (
                r_win.col_off, r_win.row_off, r_win.width, r_win.height,
            )

    @pytest.mark.asyncio
    async def test_block_windows_clipped_to_window_focus(self, cog_with_nodata_and_overviews):
        """When window_focus is set, ``block_windows`` returns only intersecting blocks, clipped."""
        reader = await AsyncGeoTIFFReader.open(
            cog_with_nodata_and_overviews["fname"],
            store=cog_with_nodata_and_overviews["store"],
        )
        # Focus on a region covering the upper-left 2×2 tiles (each tile is 64×64).
        focus = rasterio.windows.Window(col_off=32, row_off=32, width=80, height=80)
        view = reader.read_from_window(focus)

        blocks = view.block_windows()
        # Without focus there are 16 blocks; the focus intersects 4 in the upper-left.
        assert len(blocks) == 4
        # All returned blocks live inside the focus extent.
        for _, w in blocks:
            assert w.col_off >= focus.col_off
            assert w.row_off >= focus.row_off
            assert w.col_off + w.width <= focus.col_off + focus.width
            assert w.row_off + w.height <= focus.row_off + focus.height

    @pytest.mark.asyncio
    async def test_block_aligned_fan_out(self, cog_with_nodata_and_overviews):
        """Reading via ``block_windows`` + ``asyncio.gather`` produces full coverage with no overlap.

        This is the canonical tile-server pattern the reader was built
        for — fanning out reads tile-aligned with the COG's internal grid.
        """
        reader = await AsyncGeoTIFFReader.open(
            cog_with_nodata_and_overviews["fname"],
            store=cog_with_nodata_and_overviews["store"],
        )

        blocks = reader.block_windows()
        chips = await asyncio.gather(
            *[reader.read_from_window(w).load() for _, w in blocks]
        )
        assert len(chips) == 16  # 4×4 blocks of 64×64 in a 256×256 raster
        total_pixels = sum(c.values.shape[1] * c.values.shape[2] for c in chips)
        assert total_pixels == 256 * 256
