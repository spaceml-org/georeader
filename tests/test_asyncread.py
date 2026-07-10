"""Tests for the ``georeader.asyncread`` module.

Each test class mirrors a sync sibling in :mod:`georeader.read` and asserts
numerical parity between the async path (`AsyncGeoTIFFReader` →
`asyncread.*`) and the sync path (`RasterioReader` → `read.*`) on the same
on-disk fixture.

Gated on the optional ``async-geotiff`` extra via ``pytest.importorskip`` —
the whole module is skipped when the extra isn't installed.
"""
from __future__ import annotations

import os
import tempfile

import mercantile
import numpy as np
import pytest
import pytest_asyncio
import rasterio
import rasterio.warp
import rasterio.windows
from rasterio.transform import from_origin
from shapely.geometry import box

# Skip the entire module if the optional async-geotiff stack isn't available.
async_geotiff = pytest.importorskip("async_geotiff")
obstore = pytest.importorskip("obstore")

from georeader import asyncread, read  # noqa: E402
from georeader.async_geotiff_reader import AsyncGeoTIFFReader  # noqa: E402
from georeader.geotensor import GeoTensor  # noqa: E402
from georeader.rasterio_reader import RasterioReader  # noqa: E402


@pytest.fixture(scope="module")
def cog_fixture():
    """A 256×256 tiled COG plus a LocalStore + RasterioReader for sync parity.

    Resolution 10m at UTM 31N (EPSG:32631). 3 bands of int16 noise. Anchored
    so that several Web Mercator tiles at z=12 intersect it.
    """
    tmpdir = tempfile.mkdtemp()
    fname = "asyncread_demo.tif"
    path = os.path.join(tmpdir, fname)

    np.random.seed(7)
    data = np.random.randint(0, 5000, size=(3, 256, 256), dtype=np.int16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=256,
        width=256,
        count=3,
        dtype=data.dtype,
        crs="EPSG:32631",
        transform=from_origin(500000.0, 4600000.0, 10.0, 10.0),
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress="deflate",
    ) as dst:
        dst.write(data)

    store = obstore.store.LocalStore(prefix=tmpdir)
    yield {
        "store": store,
        "fname": fname,
        "abs_path": path,
        "tmpdir": tmpdir,
        # Geographic anchor: bounds (xmin, ymin, xmax, ymax)
        "bounds": (500000.0, 4597440.0, 502560.0, 4600000.0),
    }

    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest_asyncio.fixture
async def async_reader(cog_fixture):
    """A freshly-opened `AsyncGeoTIFFReader` for the module-scoped COG."""
    reader = await AsyncGeoTIFFReader.open(cog_fixture["fname"], store=cog_fixture["store"])
    try:
        yield reader
    finally:
        await reader.aclose()


@pytest.fixture
def sync_reader(cog_fixture):
    """A `RasterioReader` over the same on-disk COG, for sync-path parity."""
    return RasterioReader(cog_fixture["abs_path"])


# ──────────────────────────────────────────────────────────────────────────────
# read_from_window
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncReadFromWindow:
    """`asyncread.read_from_window` parity + return-shape contract."""

    @pytest.mark.asyncio
    async def test_basic_read_returns_geotensor_when_trigger_load(
        self, async_reader, sync_reader
    ):
        window = rasterio.windows.Window(col_off=10, row_off=10, width=100, height=80)
        async_gt = await asyncread.read_from_window(async_reader, window, trigger_load=True)
        sync_gt = read.read_from_window(sync_reader, window)
        assert isinstance(async_gt, GeoTensor)
        assert async_gt.shape == sync_gt.shape == (3, 80, 100)
        assert np.array_equal(np.asarray(async_gt.values), np.asarray(sync_gt.values))

    @pytest.mark.asyncio
    async def test_returns_view_when_not_triggered(self, async_reader):
        window = rasterio.windows.Window(col_off=0, row_off=0, width=32, height=32)
        view = await asyncread.read_from_window(async_reader, window)
        # Without trigger_load / return_only_data the async path returns the
        # unmaterialised view (an AsyncGeoData subclass), not a GeoTensor.
        assert not isinstance(view, GeoTensor)
        gt = await view.load()
        assert gt.shape == (3, 32, 32)

    @pytest.mark.asyncio
    async def test_return_only_data_yields_ndarray(self, async_reader, sync_reader):
        window = rasterio.windows.Window(col_off=20, row_off=30, width=40, height=40)
        async_arr = await asyncread.read_from_window(async_reader, window, return_only_data=True)
        sync_arr = read.read_from_window(sync_reader, window, return_only_data=True)
        assert isinstance(async_arr, np.ndarray) and not isinstance(async_arr, GeoTensor)
        assert np.array_equal(async_arr, sync_arr)

    @pytest.mark.asyncio
    async def test_boundless_no_intersection_returns_padded(self, async_reader, sync_reader):
        # Window far outside the raster extent in pixel coords.
        window = rasterio.windows.Window(col_off=10_000, row_off=10_000, width=64, height=64)
        async_gt = await asyncread.read_from_window(async_reader, window, boundless=True)
        sync_gt = read.read_from_window(sync_reader, window, boundless=True)
        assert async_gt.shape == sync_gt.shape == (3, 64, 64)
        # Both paths produce the synthetic fill — exact equality on the values.
        assert np.array_equal(np.asarray(async_gt.values), np.asarray(sync_gt.values))

    @pytest.mark.asyncio
    async def test_non_boundless_no_intersection_returns_none(self, async_reader):
        window = rasterio.windows.Window(col_off=10_000, row_off=10_000, width=64, height=64)
        out = await asyncread.read_from_window(async_reader, window, boundless=False)
        assert out is None


# ──────────────────────────────────────────────────────────────────────────────
# read_from_bounds
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncReadFromBounds:
    """`asyncread.read_from_bounds` parity vs sync."""

    @pytest.mark.asyncio
    async def test_basic_bounds(self, async_reader, sync_reader, cog_fixture):
        xmin, ymin, xmax, ymax = cog_fixture["bounds"]
        # Inset sub-extent inside the raster.
        sub = (xmin + 200, ymin + 200, xmax - 200, ymax - 200)
        async_gt = await asyncread.read_from_bounds(async_reader, sub, trigger_load=True)
        sync_gt = read.read_from_bounds(sync_reader, sub)
        assert async_gt.shape == sync_gt.shape
        assert np.array_equal(np.asarray(async_gt.values), np.asarray(sync_gt.values))

    @pytest.mark.asyncio
    async def test_pad_add_grows_window(self, async_reader, cog_fixture):
        xmin, ymin, xmax, ymax = cog_fixture["bounds"]
        sub = (xmin + 200, ymin + 200, xmax - 200, ymax - 200)
        no_pad = await asyncread.read_from_bounds(
            async_reader, sub, pad_add=(0, 0), trigger_load=True
        )
        with_pad = await asyncread.read_from_bounds(
            async_reader, sub, pad_add=(5, 5), trigger_load=True
        )
        assert with_pad.width > no_pad.width
        assert with_pad.height > no_pad.height

    @pytest.mark.asyncio
    async def test_bounds_with_different_crs(self, async_reader, sync_reader, cog_fixture):
        # Convert the inset sub-extent to WGS84 for the call, but expect the
        # same pixel result as the sync sibling on identical input.
        xmin, ymin, xmax, ymax = cog_fixture["bounds"]
        sub_utm = (xmin + 300, ymin + 300, xmax - 300, ymax - 300)
        sub_wgs = rasterio.warp.transform_bounds(
            "EPSG:32631", "EPSG:4326", *sub_utm, densify_pts=21
        )
        async_gt = await asyncread.read_from_bounds(
            async_reader, sub_wgs, crs_bounds="EPSG:4326", trigger_load=True
        )
        sync_gt = read.read_from_bounds(sync_reader, sub_wgs, crs_bounds="EPSG:4326")
        assert async_gt.shape == sync_gt.shape


# ──────────────────────────────────────────────────────────────────────────────
# read_from_polygon
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncReadFromPolygon:
    @pytest.mark.asyncio
    async def test_basic_polygon(self, async_reader, sync_reader, cog_fixture):
        xmin, ymin, xmax, ymax = cog_fixture["bounds"]
        poly = box(xmin + 300, ymin + 300, xmax - 300, ymax - 300)
        async_gt = await asyncread.read_from_polygon(async_reader, poly, trigger_load=True)
        sync_gt = read.read_from_polygon(sync_reader, poly)
        assert async_gt.shape == sync_gt.shape
        assert np.array_equal(np.asarray(async_gt.values), np.asarray(sync_gt.values))

    @pytest.mark.asyncio
    async def test_polygon_with_crs(self, async_reader, sync_reader, cog_fixture):
        xmin, ymin, xmax, ymax = cog_fixture["bounds"]
        sub_utm = (xmin + 400, ymin + 400, xmax - 400, ymax - 400)
        sub_wgs = rasterio.warp.transform_bounds("EPSG:32631", "EPSG:4326", *sub_utm)
        poly_wgs = box(*sub_wgs)
        async_gt = await asyncread.read_from_polygon(
            async_reader, poly_wgs, crs_polygon="EPSG:4326", trigger_load=True
        )
        sync_gt = read.read_from_polygon(sync_reader, poly_wgs, crs_polygon="EPSG:4326")
        assert async_gt.shape == sync_gt.shape


# ──────────────────────────────────────────────────────────────────────────────
# read_from_center_coords
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncReadFromCenterCoords:
    @pytest.mark.asyncio
    async def test_basic_center(self, async_reader, sync_reader, cog_fixture):
        xmin, ymin, xmax, ymax = cog_fixture["bounds"]
        center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
        shape = (40, 60)
        async_gt = await asyncread.read_from_center_coords(
            async_reader, center, shape, trigger_load=True
        )
        sync_gt = read.read_from_center_coords(sync_reader, center, shape)
        assert async_gt.shape == sync_gt.shape == (3, 40, 60)
        assert np.array_equal(np.asarray(async_gt.values), np.asarray(sync_gt.values))


# ──────────────────────────────────────────────────────────────────────────────
# read_reproject
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncReadReproject:
    @pytest.mark.asyncio
    async def test_reproject_to_wgs84(self, async_reader, sync_reader):
        """End-to-end reproject parity to a coarser dst grid."""
        # Use read_to_crs to pick a sensible default dst transform/window.
        sync_gt = read.read_to_crs(sync_reader, dst_crs="EPSG:4326")
        async_gt = await asyncread.read_reproject(
            async_reader,
            dst_crs="EPSG:4326",
            dst_transform=sync_gt.transform,
            window_out=rasterio.windows.Window(0, 0, sync_gt.width, sync_gt.height),
        )
        assert async_gt.shape == sync_gt.shape
        # Allow 1 DN tolerance — both paths invoke rasterio.warp.reproject
        # against the same input window, so the difference should be zero in
        # practice, but allow rounding noise in the boolean→float→threshold
        # branches just in case.
        assert np.allclose(
            np.asarray(async_gt.values), np.asarray(sync_gt.values), atol=1.0
        )

    @pytest.mark.asyncio
    async def test_reproject_fast_path_same_crs(self, async_reader, sync_reader):
        """When dst grid aligns with src, the fast path window-reads, no warp."""
        sync_gt = read.read_reproject(
            sync_reader,
            dst_crs=sync_reader.crs,
            dst_transform=sync_reader.transform,
            window_out=rasterio.windows.Window(0, 0, 64, 64),
        )
        async_gt = await asyncread.read_reproject(
            async_reader,
            dst_crs=async_reader.crs,
            dst_transform=async_reader.transform,
            window_out=rasterio.windows.Window(0, 0, 64, 64),
        )
        assert async_gt.shape == sync_gt.shape == (3, 64, 64)
        assert np.array_equal(np.asarray(async_gt.values), np.asarray(sync_gt.values))

    @pytest.mark.asyncio
    async def test_reproject_nonintersecting_returns_fill(self, async_reader, sync_reader):
        """Disjoint dst extent → returns nodata-filled GeoTensor (no I/O)."""
        # Pick a UTM zone on the other side of the planet — no overlap.
        far_bounds = (200000.0, 1000000.0, 210000.0, 1010000.0)
        sync_gt = read.read_reproject(
            sync_reader,
            dst_crs="EPSG:32601",  # UTM 1N
            bounds=far_bounds,
            resolution_dst_crs=100.0,
        )
        async_gt = await asyncread.read_reproject(
            async_reader,
            dst_crs="EPSG:32601",
            bounds=far_bounds,
            resolution_dst_crs=100.0,
        )
        assert async_gt.shape == sync_gt.shape
        assert np.array_equal(np.asarray(async_gt.values), np.asarray(sync_gt.values))


# ──────────────────────────────────────────────────────────────────────────────
# read_reproject_like
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncReadReprojectLike:
    @pytest.mark.asyncio
    async def test_match_geotensor_template(self, async_reader, sync_reader):
        # Build a small WGS84 template by reading sync_reader to that CRS.
        template = read.read_to_crs(sync_reader, dst_crs="EPSG:4326")
        sync_gt = read.read_reproject_like(sync_reader, template)
        async_gt = await asyncread.read_reproject_like(async_reader, template)
        assert async_gt.shape == sync_gt.shape
        assert async_gt.crs == template.crs
        assert np.allclose(
            np.asarray(async_gt.values), np.asarray(sync_gt.values), atol=1.0
        )

    @pytest.mark.asyncio
    async def test_return_only_data(self, async_reader, sync_reader):
        template = read.read_to_crs(sync_reader, dst_crs="EPSG:4326")
        out = await asyncread.read_reproject_like(async_reader, template, return_only_data=True)
        assert isinstance(out, np.ndarray) and not isinstance(out, GeoTensor)


# ──────────────────────────────────────────────────────────────────────────────
# read_to_crs
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncReadToCrs:
    @pytest.mark.asyncio
    async def test_basic_crs_conversion(self, async_reader, sync_reader):
        async_gt = await asyncread.read_to_crs(async_reader, dst_crs="EPSG:4326")
        sync_gt = read.read_to_crs(sync_reader, dst_crs="EPSG:4326")
        assert async_gt.shape == sync_gt.shape
        assert async_gt.crs == "EPSG:4326"
        assert np.allclose(
            np.asarray(async_gt.values), np.asarray(sync_gt.values), atol=1.0
        )

    @pytest.mark.asyncio
    async def test_same_crs_returns_input(self, async_reader):
        out = await asyncread.read_to_crs(async_reader, dst_crs="EPSG:32631")
        # Same-CRS short-circuit: returns the reader itself, unloaded.
        assert out is async_reader

    @pytest.mark.asyncio
    async def test_to_web_mercator(self, async_reader, sync_reader):
        async_gt = await asyncread.read_to_crs(async_reader, dst_crs="EPSG:3857")
        sync_gt = read.read_to_crs(sync_reader, dst_crs="EPSG:3857")
        assert async_gt.shape == sync_gt.shape
        assert async_gt.crs == "EPSG:3857"


# ──────────────────────────────────────────────────────────────────────────────
# resize
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncResize:
    @pytest.mark.asyncio
    async def test_resize_to_half_no_aa(self, async_reader, sync_reader):
        # anti_aliasing=False to avoid the lazy-input materialisation caveat.
        async_gt = await asyncread.resize(
            async_reader, resolution_dst=20.0, anti_aliasing=False
        )
        sync_gt = read.resize(sync_reader, resolution_dst=20.0, anti_aliasing=False)
        assert async_gt.shape == sync_gt.shape == (3, 128, 128)
        assert np.allclose(
            np.asarray(async_gt.values), np.asarray(sync_gt.values), atol=1.0
        )

    @pytest.mark.asyncio
    async def test_resize_to_double(self, async_reader, sync_reader):
        async_gt = await asyncread.resize(
            async_reader, resolution_dst=5.0, anti_aliasing=False
        )
        sync_gt = read.resize(sync_reader, resolution_dst=5.0, anti_aliasing=False)
        assert async_gt.shape == sync_gt.shape == (3, 512, 512)

    @pytest.mark.asyncio
    async def test_resize_to_half_with_anti_aliasing(self, async_reader, sync_reader):
        """Downsample with ``anti_aliasing=True`` (the default).

        Regression test: the sync ``apply_anti_aliasing`` helper calls
        ``data_in.load()`` on lazy inputs, which for an async reader
        returned an un-awaited coroutine and crashed with AttributeError.
        The async ``resize`` now awaits the load before filtering.
        """
        pytest.importorskip("scipy")
        async_gt = await asyncread.resize(async_reader, resolution_dst=20.0)
        sync_gt = read.resize(sync_reader, resolution_dst=20.0)
        assert async_gt.shape == sync_gt.shape == (3, 128, 128)
        assert np.allclose(
            np.asarray(async_gt.values), np.asarray(sync_gt.values), atol=1.0
        )

    @pytest.mark.asyncio
    async def test_resize_upsample_with_anti_aliasing_noop(self, async_reader, sync_reader):
        """Upsampling with AA on: the filter is a no-op and the reader stays lazy."""
        async_gt = await asyncread.resize(async_reader, resolution_dst=5.0)
        sync_gt = read.resize(sync_reader, resolution_dst=5.0)
        assert async_gt.shape == sync_gt.shape == (3, 512, 512)


# ──────────────────────────────────────────────────────────────────────────────
# read_from_tile
# ──────────────────────────────────────────────────────────────────────────────
class TestAsyncReadFromTile:
    @pytest.mark.asyncio
    async def test_basic_tile(self, async_reader, sync_reader):
        # z=0 always intersects (covers the whole world).
        async_gt = await asyncread.read_from_tile(async_reader, x=0, y=0, z=0)
        sync_gt = read.read_from_tile(sync_reader, x=0, y=0, z=0)
        assert async_gt is not None and sync_gt is not None
        assert async_gt.shape == sync_gt.shape

    @pytest.mark.asyncio
    async def test_non_intersecting_tile_returns_none(self, async_reader):
        # A tile in the far south Pacific, well outside our UTM 31N fixture.
        out = await asyncread.read_from_tile(async_reader, x=0, y=2**12 - 1, z=12)
        assert out is None

    @pytest.mark.asyncio
    async def test_assert_if_not_intersects(self, async_reader):
        with pytest.raises(AssertionError, match="does not intersect"):
            await asyncread.read_from_tile(
                async_reader, x=0, y=2**12 - 1, z=12, assert_if_not_intersects=True
            )

    @pytest.mark.asyncio
    async def test_tile_with_out_shape(self, async_reader, sync_reader):
        async_gt = await asyncread.read_from_tile(
            async_reader, x=0, y=0, z=0, out_shape=(128, 128)
        )
        sync_gt = read.read_from_tile(sync_reader, x=0, y=0, z=0, out_shape=(128, 128))
        assert async_gt.shape == sync_gt.shape == (3, 128, 128)

    @pytest.mark.asyncio
    async def test_tile_native_resolution_out_shape_none(
        self, async_reader, sync_reader, cog_fixture
    ):
        """``out_shape=None`` — grid over the TILE's extent, parity with sync.

        Regression test: this branch used to crash with AttributeError
        ('Affine' object has no attribute 'width') from unpacking
        ``calculate_transform_window`` in the wrong order, and — even
        unpacked correctly — computed the grid from the whole raster's
        bounds instead of the tile's.
        """
        # A z=13 tile covering the fixture's geographic centre (UTM 31N).
        xmin, ymin, xmax, ymax = cog_fixture["bounds"]
        lon, lat = rasterio.warp.transform(
            "EPSG:32631", "EPSG:4326", [(xmin + xmax) / 2], [(ymin + ymax) / 2]
        )
        t = mercantile.tile(lon[0], lat[0], 13)

        async_gt = await asyncread.read_from_tile(
            async_reader, x=t.x, y=t.y, z=t.z, out_shape=None
        )
        sync_gt = read.read_from_tile(sync_reader, x=t.x, y=t.y, z=t.z, out_shape=None)

        assert async_gt is not None and sync_gt is not None
        assert async_gt.shape == sync_gt.shape
        assert async_gt.transform == sync_gt.transform
        assert np.allclose(
            np.asarray(async_gt.values), np.asarray(sync_gt.values), atol=1.0
        )
        # The grid covers the TILE's Web Mercator extent (within a pixel),
        # not the whole raster's.
        tb = mercantile.xy_bounds(t.x, t.y, t.z)
        res_x = abs(async_gt.transform.a)
        assert abs(async_gt.bounds[0] - tb.left) <= res_x
        assert async_gt.bounds[2] >= tb.right - res_x
