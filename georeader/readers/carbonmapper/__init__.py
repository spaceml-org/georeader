"""Carbon Mapper reader subpackage.

Reader for the `Carbon Mapper <https://carbonmapper.org>`_ STAC + plume
catalogue API. Mirrors the layout of the other mission readers in
:mod:`georeader.readers` (``emit``, ``prisma``, ``S2_SAFE_reader``)
but is split into multiple files because the surface area is larger:

- :mod:`~georeader.readers.carbonmapper.config` —
  :class:`CarbonMapperConfig` token persistence (env vars, file, or
  in-memory). Pure file-based; no Azure SDK dependency.
- :mod:`~georeader.readers.carbonmapper.download` — raw HTTP / JSON
  primitives (``obtain_token``, ``stac_get_items``,
  ``download_asset``, etc.). Pulls in :mod:`requests`.
- :mod:`~georeader.readers.carbonmapper.api_queries` — typed wrappers
  on top of ``download``: :class:`CMTileItem`, :func:`get_plume`,
  :func:`list_tiles`, exception hierarchy, etc. Returns
  :class:`CMRawPlume` / :class:`CMSource` / :class:`CMTileItem`
  instances — never raw dicts.
- :mod:`~georeader.readers.carbonmapper.plume` —
  :class:`CMRawPlume` Pydantic model accepting both CSV bulk-export
  and annotated-JSON payloads.
- :mod:`~georeader.readers.carbonmapper.source` — :class:`CMSource`
  dataclass for ``/catalog/sources.geojson`` features.
- :mod:`~georeader.readers.carbonmapper.rasters` —
  :class:`CMImageRaster` (L2B scene) and :class:`CMPlumeRaster` (L3A
  per-plume mask). Lazy ``RasterioReader``-backed band accessors that
  delegate to ``georeader.read``.

Optional install: ``pip install 'georeader-spaceml[carbonmapper]'``
(adds ``pydantic`` and ``requests`` — both Carbon-Mapper-only).

References
----------
- Product Guide: https://carbonmapper.org/articles/product-guide
- API Docs:      https://api.carbonmapper.org/api/v1/docs
- STAC Catalog:  https://api.carbonmapper.org/api/v1/stac
"""

from georeader.readers.carbonmapper.api_queries import (
    CMAPIError,
    CMPlumeNotFound,
    CMSceneNotPublished,
    CMSourceNotFound,
    CMTileItem,
    DEFAULT_L2B_COLLECTION,
    get_plume,
    get_plume_context,
    get_source,
    get_source_for_plume,
    get_tile,
    get_tile_for_plume,
    list_plumes,
    list_plumes_for_source,
    list_plumes_for_tile,
    list_sources,
    list_tiles,
    list_tiles_for_source,
)
from georeader.readers.carbonmapper.config import CarbonMapperConfig
from georeader.readers.carbonmapper.download import (
    download_asset,
    download_plume_assets,
    export_plumes_to_geojson,
    get_plume_by_id,
    get_plumes_annotated,
    get_plumes_csv,
    get_scenes,
    get_sources,
    obtain_token,
    paginate_plumes,
    refresh_token,
    stac_get_catalog,
    stac_get_collection,
    stac_get_items,
    stac_list_collections,
    stac_search,
)
from georeader.readers.carbonmapper.plume import (
    CARBONMAPPER_INSTRUMENTS,
    CARBONMAPPER_PLUME_PARAMS,
    CM_INSTRUMENT_TO_SATELLITE,
    CMRawPlume,
    CarbonMapperPlumeRaw,
    decompose_wind,
)
from georeader.readers.carbonmapper.rasters import (
    CM_L2B_BANDS,
    DEFAULT_L2B_RGB_COLLECTION,
    CMImageRaster,
    CMPlumeRaster,
)
from georeader.readers.carbonmapper.source import CMSource

__all__ = [
    "CARBONMAPPER_INSTRUMENTS",
    "CARBONMAPPER_PLUME_PARAMS",
    "CM_INSTRUMENT_TO_SATELLITE",
    "CM_L2B_BANDS",
    "CMAPIError",
    "CMImageRaster",
    "CMPlumeNotFound",
    "CMPlumeRaster",
    "CMRawPlume",
    "CMSceneNotPublished",
    "CMSource",
    "CMSourceNotFound",
    "CMTileItem",
    "CarbonMapperConfig",
    "CarbonMapperPlumeRaw",
    "DEFAULT_L2B_COLLECTION",
    "DEFAULT_L2B_RGB_COLLECTION",
    "decompose_wind",
    "download_asset",
    "download_plume_assets",
    "export_plumes_to_geojson",
    "get_plume",
    "get_plume_by_id",
    "get_plume_context",
    "get_plumes_annotated",
    "get_plumes_csv",
    "get_scenes",
    "get_source",
    "get_source_for_plume",
    "get_sources",
    "get_tile",
    "get_tile_for_plume",
    "list_plumes",
    "list_plumes_for_source",
    "list_plumes_for_tile",
    "list_sources",
    "list_tiles",
    "list_tiles_for_source",
    "obtain_token",
    "paginate_plumes",
    "refresh_token",
    "stac_get_catalog",
    "stac_get_collection",
    "stac_get_items",
    "stac_list_collections",
    "stac_search",
]
