"""
Tests for the georeader.abstract_reader module.

These tests verify the abstract base classes and protocols including:
- GeoDataBase protocol
- FakeGeoData dataclass
- GeoData abstract class
- AsyncGeoData abstract class
- same_extent comparison function
"""

import numpy as np
import pytest
from rasterio.transform import from_origin

from georeader.abstract_reader import AsyncGeoData, FakeGeoData, same_extent
from georeader.geotensor import GeoTensor


class TestFakeGeoData:
    """Tests for FakeGeoData dataclass."""

    def test_basic_creation(self):
        """Test creating a FakeGeoData instance."""
        transform = from_origin(0, 100, 10, 10)

        fake = FakeGeoData(transform=transform, crs="EPSG:32631", shape=(3, 100, 100))

        assert fake.transform == transform
        assert fake.crs == "EPSG:32631"
        assert fake.shape == (3, 100, 100)

    def test_minimal_creation(self):
        """Test creating FakeGeoData with minimal required fields."""
        transform = from_origin(0, 100, 10, 10)

        fake = FakeGeoData(transform=transform, crs="EPSG:4326", shape=(10, 10))

        assert fake.transform is not None
        assert fake.crs is not None

    def test_width_height_properties(self):
        """Test width and height computed properties."""
        transform = from_origin(0, 100, 10, 10)

        fake = FakeGeoData(transform=transform, crs="EPSG:32631", shape=(3, 100, 50))

        assert fake.height == 100
        assert fake.width == 50


class TestGeoDataProtocol:
    """Tests for GeoDataBase protocol compliance."""

    def test_geotensor_is_geodata(self):
        """Test that GeoTensor satisfies GeoDataBase protocol."""
        data = np.zeros((3, 100, 100))
        transform = from_origin(0, 100, 10, 10)

        gt = GeoTensor(data, transform=transform, crs="EPSG:32631")

        # GeoTensor should have all required attributes
        assert hasattr(gt, "transform")
        assert hasattr(gt, "crs")
        assert hasattr(gt, "shape")
        assert hasattr(gt, "bounds")

    def test_fake_geodata_is_geodata(self):
        """Test that FakeGeoData satisfies GeoDataBase protocol."""
        transform = from_origin(0, 100, 10, 10)

        fake = FakeGeoData(transform=transform, crs="EPSG:32631", shape=(3, 100, 100))

        # Should have all required attributes
        assert hasattr(fake, "transform")
        assert hasattr(fake, "crs")
        assert hasattr(fake, "shape")


class _FakeAsyncReader(AsyncGeoData):
    """Minimal concrete ``AsyncGeoData`` used to verify inherited defaults.

    Implements only the abstract surface (``transform``, ``crs``, ``shape``,
    ``dtype``, ``fill_value_default``); the read methods are left raising
    so the test focuses on metadata + derived-property defaults.
    """

    def __init__(self, transform, crs, shape, dtype=np.float32, fill_value_default=0):
        self._transform = transform
        self._crs = crs
        self._shape = shape
        self._dtype = dtype
        self._fill_value_default = fill_value_default

    @property
    def transform(self):
        return self._transform

    @property
    def crs(self):
        return self._crs

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def fill_value_default(self):
        return self._fill_value_default


class TestAsyncGeoData:
    """Tests for the AsyncGeoData abstract class."""

    def test_subclass_satisfies_surface(self):
        """A subclass exposing the required attributes satisfies AsyncGeoData."""
        transform = from_origin(0, 100, 10, 10)
        reader = _FakeAsyncReader(transform=transform, crs="EPSG:32631", shape=(3, 100, 100))

        # Inherited from GeoDataBase
        assert reader.width == 100
        assert reader.height == 100
        # Defaults inherited from AsyncGeoData (origin (0, 100), 10x10 pixels, 100x100 grid)
        assert reader.res == (10.0, 10.0)
        assert reader.bounds == (0.0, -900.0, 1000.0, 100.0)

    def test_footprint_native_crs(self):
        """Footprint returns the bounding polygon in the reader's native CRS."""
        transform = from_origin(0, 100, 10, 10)
        reader = _FakeAsyncReader(transform=transform, crs="EPSG:32631", shape=(3, 100, 100))

        pol = reader.footprint()
        # Same coverage as bounds — corners match
        assert pol.bounds == reader.bounds

    def test_default_load_raises_not_implemented(self):
        """Default async ``load`` raises NotImplementedError on a bare subclass."""
        import asyncio

        transform = from_origin(0, 100, 10, 10)
        reader = _FakeAsyncReader(transform=transform, crs="EPSG:32631", shape=(3, 100, 100))

        with pytest.raises(NotImplementedError):
            asyncio.run(reader.load())

    def test_default_read_from_window_raises_not_implemented(self):
        """Default sync ``read_from_window`` raises NotImplementedError on a bare subclass.

        ``read_from_window`` is sync (returns a windowed view), mirroring
        :meth:`RasterioReader.read_from_window`.
        """
        transform = from_origin(0, 100, 10, 10)
        reader = _FakeAsyncReader(transform=transform, crs="EPSG:32631", shape=(3, 100, 100))

        with pytest.raises(NotImplementedError):
            reader.read_from_window(window=None, boundless=True)


class TestSameExtent:
    """Tests for same_extent comparison function."""

    def test_same_extent_identical(self):
        """Test same_extent with identical GeoTensor objects."""
        data = np.zeros((3, 100, 100))
        transform = from_origin(0, 100, 10, 10)

        gt1 = GeoTensor(data, transform=transform, crs="EPSG:32631")
        gt2 = GeoTensor(data, transform=transform, crs="EPSG:32631")

        assert same_extent(gt1, gt2) is True

    def test_same_extent_different_transform(self):
        """Test same_extent with different transforms."""
        data = np.zeros((3, 100, 100))
        transform1 = from_origin(0, 100, 10, 10)
        transform2 = from_origin(100, 100, 10, 10)  # Different origin

        gt1 = GeoTensor(data, transform=transform1, crs="EPSG:32631")
        gt2 = GeoTensor(data, transform=transform2, crs="EPSG:32631")

        assert same_extent(gt1, gt2) is False

    def test_same_extent_different_crs(self):
        """Test same_extent with different CRS."""
        data = np.zeros((3, 100, 100))
        transform = from_origin(0, 100, 10, 10)

        gt1 = GeoTensor(data, transform=transform, crs="EPSG:32631")
        gt2 = GeoTensor(data, transform=transform, crs="EPSG:4326")  # Different CRS

        assert same_extent(gt1, gt2) is False

    def test_same_extent_different_shape(self):
        """Test same_extent with different spatial shapes."""
        data1 = np.zeros((3, 100, 100))
        data2 = np.zeros((3, 50, 50))
        transform = from_origin(0, 100, 10, 10)

        gt1 = GeoTensor(data1, transform=transform, crs="EPSG:32631")
        gt2 = GeoTensor(data2, transform=transform, crs="EPSG:32631")  # Different spatial shape

        assert same_extent(gt1, gt2) is False

    def test_same_extent_with_precision(self):
        """Test same_extent with precision tolerance."""
        data = np.zeros((3, 100, 100))
        transform1 = from_origin(0, 100, 10, 10)
        # Very slightly different transform
        transform2 = from_origin(0.0001, 100, 10, 10)

        gt1 = GeoTensor(data, transform=transform1, crs="EPSG:32631")
        gt2 = GeoTensor(data, transform=transform2, crs="EPSG:32631")

        # With low precision, should be considered same
        assert same_extent(gt1, gt2, precision=1e-2) is True
        # With high precision, should be different
        assert same_extent(gt1, gt2, precision=1e-6) is False

    def test_same_extent_geotensor(self):
        """Test same_extent with actual GeoTensor objects."""
        data = np.zeros((3, 100, 100))
        transform = from_origin(0, 100, 10, 10)

        gt1 = GeoTensor(data, transform=transform, crs="EPSG:32631")
        gt2 = GeoTensor(data, transform=transform, crs="EPSG:32631")

        assert same_extent(gt1, gt2) is True

    def test_same_extent_mixed_types(self):
        """Test same_extent with GeoTensor and FakeGeoData."""
        transform = from_origin(0, 100, 10, 10)

        data = np.zeros((3, 100, 100))
        gt = GeoTensor(data, transform=transform, crs="EPSG:32631")

        fake = FakeGeoData(transform=transform, crs="EPSG:32631", shape=(3, 100, 100))

        assert same_extent(gt, fake) is True
