"""Rasterize Carbon Mapper *sources* (point clusters) onto a target grid.

Carbon Mapper sources are point geometries (DBSCAN-clustered plume
locations). For training labels, QA overlays, and source-prior features
it is useful to project them onto the same grid as an L2B scene as a
binary mask. This module provides:

- :func:`rasterize_sources` — one-shot function: list of points →
  :class:`~georeader.geotensor.GeoTensor` mask.
- :class:`CMSourceRaster` — lazy wrapper that mirrors
  :class:`~georeader.readers.carbonmapper.rasters.CMImageRaster` shape
  (``read_polygon`` / ``read_window`` / ``read_window_to_crs``) so
  callers can compose the source mask with the L2B rasters.

Both delegate the actual burn-in to
:func:`georeader.rasterize.rasterize_geopandas_like` /
:func:`~georeader.rasterize.rasterize_from_geopandas` — no custom
rasterio.features call lives in this module.

The Carbon Mapper API does not publish a sources raster — these helpers
build it client-side from :func:`list_sources` (or any iterable of
:class:`~georeader.readers.carbonmapper.source.CMSource`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union, cast

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

from georeader import read
from georeader.abstract_reader import GeoData
from georeader.geotensor import GeoTensor
from georeader.rasterize import rasterize_from_geopandas, rasterize_geopandas_like

from georeader.readers.carbonmapper.api_queries import CMTileItem
from georeader.readers.carbonmapper.source import CMSource

BBox = tuple[float, float, float, float]   # (W, S, E, N) in WGS-84

#: Anything we know how to extract a (lon, lat) pair from.
SourceLike = Union[CMSource, Point, tuple[float, float]]


def _to_point(src: SourceLike) -> Point:
    if isinstance(src, CMSource):
        return src.point
    if isinstance(src, Point):
        return src
    lon, lat = src
    return Point(float(lon), float(lat))


def _sources_gdf(sources: Iterable[SourceLike]) -> gpd.GeoDataFrame:
    """Build an EPSG:4326 GeoDataFrame with a ``value=1`` column."""
    pts = [_to_point(s) for s in sources]
    return gpd.GeoDataFrame(
        {"value": np.ones(len(pts), dtype=np.uint8)},
        geometry=pts,
        crs="EPSG:4326",
    )


def _apply_buffer(
    gdf: gpd.GeoDataFrame, target_crs: rasterio.crs.CRS, buffer_m: float
) -> gpd.GeoDataFrame:
    """Reproject to ``target_crs`` and buffer each point by ``buffer_m`` metres.

    Requires ``target_crs`` to be projected so ``buffer_m`` is in the
    same units as the geometries.
    """
    if not target_crs.is_projected:
        raise ValueError(
            "buffer_m > 0 requires a projected CRS (metres). Got geographic "
            f"CRS {target_crs}. Reproject your grid first or use buffer_m=0."
        )
    gdf = gdf.to_crs(target_crs)
    gdf = gdf.assign(geometry=gdf.geometry.buffer(buffer_m))
    return gdf


def rasterize_sources(
    sources: Iterable[SourceLike],
    *,
    transform: rasterio.Affine,
    shape: tuple[int, int],
    crs: Union[str, rasterio.crs.CRS],
    buffer_m: float = 0.0,
) -> GeoTensor:
    """Rasterize source points onto a target grid as a binary mask.

    Each source contributes a value of ``1`` at its pixel; if
    ``buffer_m > 0`` a disk of that radius (in metres) is stamped
    instead. Sources falling outside the grid are silently dropped.

    Delegates to
    :func:`georeader.rasterize.rasterize_from_geopandas`.

    Parameters
    ----------
    sources:
        Iterable of :class:`CMSource`, Shapely :class:`Point`, or
        ``(lon, lat)`` tuples — all interpreted as WGS-84 lon/lat.
    transform:
        Affine transform of the target grid.
    shape:
        ``(height, width)`` of the target grid.
    crs:
        CRS of the target grid. Must be projected when
        ``buffer_m > 0``.
    buffer_m:
        Buffer radius in metres applied around each source point.
        ``0`` (default) → ``all_touched`` single-pixel stamp per source.

    Returns
    -------
    GeoTensor
        2D mask of ``shape`` with values in ``{0, 1}``.

    Raises
    ------
    ValueError
        If ``buffer_m > 0`` and ``crs`` is geographic, or if ``shape``
        is not 2D.
    """
    if len(shape) != 2:
        raise ValueError(f"Expected (H, W) shape, got {shape}")
    crs_obj = rasterio.crs.CRS.from_user_input(crs)

    gdf = _sources_gdf(sources)
    if len(gdf) == 0:
        return GeoTensor(
            np.zeros(shape, dtype=np.uint8),
            transform=transform, crs=crs_obj, fill_value_default=0,
        )

    if buffer_m > 0:
        gdf = _apply_buffer(gdf, crs_obj, buffer_m)
        all_touched = False
    else:
        gdf = gdf.to_crs(crs_obj)
        all_touched = True  # stamp the pixel containing each point

    height, width = shape
    window_out = rasterio.windows.Window(0, 0, width=width, height=height)
    return cast(
        GeoTensor,
        rasterize_from_geopandas(
            gdf,
            column="value",
            transform=transform,
            window_out=window_out,
            crs_out=crs_obj,
            fill=0,
            all_touched=all_touched,
        ),
    )


def rasterize_sources_like(
    sources: Iterable[SourceLike],
    data_like: GeoData,
    *,
    buffer_m: float = 0.0,
) -> GeoTensor:
    """Rasterize sources onto an existing :class:`GeoData` grid.

    Thin wrapper around
    :func:`georeader.rasterize.rasterize_geopandas_like`.
    """
    crs_obj = rasterio.crs.CRS.from_user_input(data_like.crs)
    gdf = _sources_gdf(sources)
    if len(gdf) == 0:
        return GeoTensor(
            np.zeros(data_like.shape[-2:], dtype=np.uint8),
            transform=data_like.transform,
            crs=crs_obj,
            fill_value_default=0,
        )

    if buffer_m > 0:
        gdf = _apply_buffer(gdf, crs_obj, buffer_m)
        all_touched = False
    else:
        gdf = gdf.to_crs(crs_obj)
        all_touched = True

    return cast(
        GeoTensor,
        rasterize_geopandas_like(
            gdf, data_like=data_like, column="value",
            fill=0, all_touched=all_touched,
        ),
    )


# ─────────────────────────────────────────────────────────────────────
#  Lazy class wrapper
# ─────────────────────────────────────────────────────────────────────


@dataclass(repr=False)
class CMSourceRaster:
    """Lazy binary-mask raster of Carbon Mapper sources on a target grid.

    Mirrors the read-helper surface of
    :class:`~georeader.readers.carbonmapper.rasters.CMImageRaster` so
    callers can compose source masks with L2B reads.

    Attributes
    ----------
    sources:
        Source points to rasterize.
    transform, shape, crs:
        Target grid spec. Use :meth:`from_cmtileitem` or
        :meth:`from_geodata` to inherit the spec from an existing
        raster.
    buffer_m:
        Per-point disk radius in metres. ``0`` → single pixel.
    """

    sources: Sequence[SourceLike]
    transform: rasterio.Affine
    shape: tuple[int, int]
    crs: rasterio.crs.CRS
    buffer_m: float = 0.0

    # ---- Constructors ----

    @classmethod
    def from_geodata(
        cls,
        sources: Sequence[SourceLike],
        template: GeoData,
        *,
        buffer_m: float = 0.0,
    ) -> "CMSourceRaster":
        """Build a source raster aligned to an existing :class:`GeoData`."""
        return cls(
            sources=sources,
            transform=template.transform,
            shape=(template.shape[-2], template.shape[-1]),
            crs=rasterio.crs.CRS.from_user_input(template.crs),
            buffer_m=buffer_m,
        )

    @classmethod
    def from_cmtileitem(
        cls,
        sources: Sequence[SourceLike],
        tile: CMTileItem,
        *,
        buffer_m: float = 0.0,
    ) -> "CMSourceRaster":
        """Build a source raster aligned to an L2B :class:`CMTileItem`.

        Resolves the tile's ``cmf`` GeoTIFF header to inherit
        ``(transform, shape, crs)``. Issues one HEAD/GET-range read.
        """
        cmf_url = tile.assets.get("cmf") or tile.assets.get("ch4-mfa")
        if cmf_url is None:
            raise ValueError(
                f"CMTileItem {tile.scene_id!r} has no 'cmf' asset to align to."
            )
        with rasterio.open(cmf_url) as ds:
            return cls(
                sources=sources,
                transform=ds.transform,
                shape=(ds.height, ds.width),
                crs=ds.crs,
                buffer_m=buffer_m,
            )

    # ---- Eager render ----

    def load(self) -> GeoTensor:
        """Rasterize all sources onto the full grid."""
        return rasterize_sources(
            self.sources,
            transform=self.transform,
            shape=self.shape,
            crs=self.crs,
            buffer_m=self.buffer_m,
        )

    # ---- Read helpers (mirror CMImageRaster) ----

    def read_polygon(
        self,
        polygon: BaseGeometry,
        *,
        crs_polygon: str = "EPSG:4326",
    ) -> GeoTensor:
        """Read a polygon clip of the source mask."""
        full = self.load()
        # `read_from_polygon` returns ``GeoData | NDArray``; with the
        # default ``return_only_data=False`` the GeoData arm is the one
        # we always hit.
        return cast(
            GeoTensor,
            read.read_from_polygon(
                cast(GeoData, full),
                polygon=polygon,
                crs_polygon=crs_polygon,
            ),
        )

    def read_window(self, bounds_4326: BBox) -> GeoTensor:
        """Read a WGS-84 bbox window of the source mask."""
        return self.read_polygon(box(*bounds_4326))

    def read_window_to_crs(
        self,
        bounds_4326: BBox,
        crs_dst: str,
    ) -> GeoTensor:
        """Read a window then reproject the mask to ``crs_dst``."""
        crop = self.read_window(bounds_4326)
        return cast(GeoTensor, read.read_to_crs(crop, crs_dst))

    # ---- Repr ----

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_sources={len(self.sources)}, "
            f"shape={self.shape}, buffer_m={self.buffer_m}, crs={self.crs})"
        )
