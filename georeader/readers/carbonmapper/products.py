"""Explicit Carbon Mapper product registry.

Every product Carbon Mapper publishes is modelled here as a first-class,
frozen descriptor object — its asset key (the filename tail in the
asset-proxy URL), the collection *family* it lives in, and how to open
it. Consumers select the products they want explicitly instead of the
reader assuming a bundle::

    from georeader.readers.carbonmapper import products as P

    img = CMPlumeImage.from_plume_id(
        plume_id,
        token=token,
        products=(P.PLUME_TIF, P.PLUME_OUTLINE, P.RGB_TIF),
    )

Collection identity is factored into :class:`CMCollectionSpec` — the
``(gas, cmf_type, version)`` triple that composes every collection id
(``l3a-vis-ch4-mfa-v3d`` etc.). The spec is **resolved from the plume
record itself** (the collection segment of its ``plume_tif`` URL, or the
``gas`` / ``cmf_type`` / ``emission_version`` fields), never guessed
from a hardcoded version list. The 2026-07 audit
(``docs/carbonmapper/api_audit_2026-07.md``) verified:

- the asset-proxy URL pattern
  ``{base}/{coll}/{Y}/{M}/{D}/{item}/{item}_{coll}_{key}`` for both L3A
  (item = plume_id) and L2B (item = scene name) products;
- **same-version pairing** — a v3d L3A plume's L2B parent serves at
  ``l2b-ch4-mfa-v3d`` (not v3c, as previously assumed);
- the full asset sets per family (tables below), including the PNG
  quicklooks the reader previously ignored;
- ``/stac/collections`` stops at ``-v3a``: current-era collections
  exist only in the asset-proxy namespace.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import requests

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from georeader.rasterio_reader import RasterioReader

#: Asset-proxy base. Bearer-aware mirror of the signed CDN (no
#: signed-query expiry). Shared by every product's URL derivation.
CM_API_ASSET_BASE = "https://api.carbonmapper.org/api/v1/catalog/asset"


class CMProductNotSelected(KeyError):
    """A product was requested from a bundle that didn't select it.

    Raised instead of silently returning ``None`` so callers notice
    when they access a product they did not ask for at construction.
    """

    def __init__(self, product: "CMProduct", selected: tuple["CMProduct", ...]):
        super().__init__(
            f"Product {product.key!r} ({product.family.value}) was not "
            f"selected on this bundle. Selected products: "
            f"{sorted(p.key for p in selected)}. Pass it in `products=` "
            "at construction to use it."
        )
        self.product = product


class CMProductFamily(str, Enum):
    """Which collection family a product's assets live in.

    The family decides how the collection id is composed from a
    :class:`CMCollectionSpec` and which item id keys the URL (plume_id
    for L3A products, scene name for L2B products).
    """

    L3A_VIS = "l3a-vis"    #: per-plume visualisation products
    L3A_IME = "l3a-ime"    #: per-plume IME (emission-integral) products
    L2B = "l2b"            #: per-scene retrieval products (gas-typed)
    L2B_RGB = "l2b-rgb"    #: per-scene true-colour sibling (not gas-typed)


#: Collection-id patterns per family. ``l2b-rgb`` has no gas/cmf_type
#: segment; every other family carries the full triple.
_FAMILY_PATTERNS: dict[CMProductFamily, str] = {
    CMProductFamily.L3A_VIS: "l3a-vis-{gas}-{cmf_type}-{version}",
    CMProductFamily.L3A_IME: "l3a-ime-{gas}-{cmf_type}-{version}",
    CMProductFamily.L2B: "l2b-{gas}-{cmf_type}-{version}",
    CMProductFamily.L2B_RGB: "l2b-rgb-{version}",
}

#: Parses any collection id of the four families above.
_COLLECTION_ID_RE = re.compile(
    r"^(?P<family>l3a-vis|l3a-ime|l2b-rgb|l2b)"
    r"(?:-(?P<gas>ch4|co2))?"
    r"(?:-(?P<cmf_type>mf[a-z]*))?"
    r"-(?P<version>v[0-9][0-9a-z]*)$"
)


@dataclass(frozen=True)
class CMCollectionSpec:
    """The ``(gas, cmf_type, version)`` triple naming a processing run.

    Composes every collection id the reader touches, so that resolving
    the version **once** (from the plume record) pins all four product
    families consistently — the 2026-07 audit verified pairing is
    same-version across L3A vis / L3A ime / L2B / L2B-RGB.

    >>> spec = CMCollectionSpec(version="v3d")
    >>> spec.collection_id(CMProductFamily.L3A_VIS)
    'l3a-vis-ch4-mfa-v3d'
    >>> spec.collection_id(CMProductFamily.L2B)
    'l2b-ch4-mfa-v3d'
    >>> spec.collection_id(CMProductFamily.L2B_RGB)
    'l2b-rgb-v3d'
    """

    version: str
    gas: str = "ch4"
    cmf_type: str = "mfa"

    def collection_id(self, family: CMProductFamily) -> str:
        """Collection id for one product family under this spec."""
        return _FAMILY_PATTERNS[family].format(
            gas=self.gas, cmf_type=self.cmf_type, version=self.version,
        )

    @classmethod
    def from_collection_id(cls, collection_id: str) -> "CMCollectionSpec":
        """Parse a collection id (any family) into its spec.

        >>> CMCollectionSpec.from_collection_id("l3a-vis-ch4-mfa-v3d")
        CMCollectionSpec(version='v3d', gas='ch4', cmf_type='mfa')
        >>> CMCollectionSpec.from_collection_id("l3a-vis-ch4-mf-v1")
        CMCollectionSpec(version='v1', gas='ch4', cmf_type='mf')
        >>> CMCollectionSpec.from_collection_id("l2b-rgb-v3a")
        CMCollectionSpec(version='v3a', gas='ch4', cmf_type='mfa')

        Raises:
            ValueError: If ``collection_id`` doesn't match any known
                family pattern.
        """
        m = _COLLECTION_ID_RE.match(collection_id)
        if m is None:
            raise ValueError(
                f"Unrecognised Carbon Mapper collection id "
                f"{collection_id!r} — expected one of the "
                f"{[f.value for f in CMProductFamily]} families."
            )
        return cls(
            version=m.group("version"),
            gas=m.group("gas") or "ch4",
            cmf_type=m.group("cmf_type") or "mfa",
        )

    @classmethod
    def from_plume_record(cls, record: Mapping[str, Any]) -> "CMCollectionSpec":
        """Resolve the spec from a ``/catalog/plume/{id}`` record.

        Preferred source: the collection segment embedded in the
        record's own ``plume_tif`` URL (authoritative — it names the
        run the assets were actually published under). Fallback: the
        ``gas`` + ``cmf_type`` + ``emission_version`` fields, which the
        audit verified compose the same id.

        Raises:
            ValueError: If neither source is present on the record.
        """
        plume_tif = record.get("plume_tif")
        if plume_tif:
            parsed = _parse_asset_url(str(plume_tif))
            if parsed is not None:
                try:
                    return cls.from_collection_id(parsed.collection_id)
                except ValueError:
                    # Legacy families (`l3a-ch4-mf-v1` — pre vis/ime
                    # split) don't parse; fall through to the fields.
                    pass
        gas = record.get("gas")
        cmf_type = record.get("cmf_type") or record.get("emission_cmf_type")
        version = record.get("emission_version")
        if gas and cmf_type and version:
            return cls(
                version=str(version),
                gas=str(gas).lower(),
                cmf_type=str(cmf_type),
            )
        raise ValueError(
            f"Cannot resolve a CMCollectionSpec for plume "
            f"{record.get('plume_id')!r}: no parseable 'plume_tif' URL "
            "and no gas/cmf_type/emission_version fields."
        )


@dataclass(frozen=True)
class _ParsedAssetURL:
    """Decomposition of one asset-proxy / CDN asset URL."""

    collection_id: str
    yyyy: str
    mm: str
    dd: str
    item_id: str
    key: str


#: Asset URLs follow ``.../{coll}/{Y}/{M}/{D}/{item}/{item}_{coll}_{key}``
#: on both the signed CDN (``catalog.carbonmapper.org``) and the
#: Bearer-aware asset proxy. Query strings (CDN signatures) are ignored.
_ASSET_URL_RE = re.compile(
    r"/(?P<coll>[a-z0-9-]+)/(?P<yyyy>\d{4})/(?P<mm>\d{2})/(?P<dd>\d{2})/"
    r"(?P<item>[^/]+)/(?P=item)_(?P=coll)_(?P<key>[^/?]+)(?:\?.*)?$"
)


def _parse_asset_url(url: str) -> _ParsedAssetURL | None:
    """Parse an asset URL into its parts; ``None`` if it doesn't match."""
    m = _ASSET_URL_RE.search(url)
    if m is None:
        return None
    return _ParsedAssetURL(
        collection_id=m.group("coll"),
        yyyy=m.group("yyyy"),
        mm=m.group("mm"),
        dd=m.group("dd"),
        item_id=m.group("item"),
        key=m.group("key"),
    )


def parse_item_date(item_id: str) -> tuple[str, str, str]:
    """Extract ``(YYYY, MM, DD)`` from a scene name or plume_id.

    Both follow ``<3-char-instrument><YYYYMMDD>t<HHMMSS>...`` — the
    date sits at positions ``[3:11]``. Note the *record's* ``scene_id``
    field is a UUID and will NOT parse; use the scene name derived from
    ``plume_id.rsplit("-", 1)[0]``.

    >>> parse_item_date("tan20260623t124240c80s4001")
    ('2026', '06', '23')

    Raises:
        ValueError: If positions ``[3:11]`` aren't an 8-digit date.
    """
    if len(item_id) < 11 or not item_id[3:11].isdigit():
        raise ValueError(
            f"item id {item_id!r} does not carry an 8-digit date at "
            "positions [3:11] — is this a UUID scene_id? Use the scene "
            "name (plume_id.rsplit('-', 1)[0]) instead."
        )
    d = item_id[3:11]
    return d[:4], d[4:6], d[6:8]


# ─────────────────────────────────────────────────────────────────────
#  Product descriptor classes
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CMProduct:
    """One Carbon Mapper product: an asset key within a family.

    Identity is ``(family, key)`` — e.g. ``rgb.tif`` exists both as an
    L3A per-plume crop and as the L2B whole-scene sibling; they are two
    distinct products.

    Attributes:
        key: Asset filename tail (``"plume-concentrations.tif"``,
            ``"cmf.tif"``, ...). Carries its extension.
        family: The :class:`CMProductFamily` whose collections serve it.
        description: One-line human description (audit-grounded).
    """

    key: str
    family: CMProductFamily
    description: str = ""

    @property
    def name(self) -> str:
        """Python-identifier form of :attr:`key` (no extension,
        ``-`` → ``_``): ``"ime-cmf-mask.tif"`` → ``"ime_cmf_mask"``."""
        return self.band.replace("-", "_")

    @property
    def band(self) -> str:
        """:attr:`key` without its extension — the historical band name
        used as ``CMImageRaster.asset_paths`` key (``"cmf-unortho"``,
        ``"artifact-mask"``, ...)."""
        return self.key.rsplit(".", 1)[0]

    def asset_url(
        self,
        spec: CMCollectionSpec | None,
        item_id: str,
        *,
        date: tuple[str, str, str] | None = None,
        collection_id: str | None = None,
        base: str = CM_API_ASSET_BASE,
    ) -> str:
        """Compose this product's asset-proxy URL.

        Args:
            spec: Collection spec resolving the family to a collection
                id. May be ``None`` when ``collection_id`` is given.
            item_id: ``plume_id`` for L3A products, scene name for L2B.
            date: ``(YYYY, MM, DD)`` override; parsed from ``item_id``
                when omitted.
            collection_id: Explicit collection id override — use when
                the exact id is known from an existing asset URL
                (avoids recomposition drift, and is the only option
                for legacy families like ``l3a-ch4-mf-v1`` that predate
                the vis/ime split).
            base: URL base — swap for the signed-CDN host or a local
                mirror if needed.

        Raises:
            ValueError: If neither ``spec`` nor ``collection_id`` is
                given, or the date can't be parsed from ``item_id``.
        """
        if collection_id is not None:
            coll = collection_id
        elif spec is not None:
            coll = spec.collection_id(self.family)
        else:
            raise ValueError(
                f"asset_url for {self.key!r} needs a spec or an "
                "explicit collection_id."
            )
        yyyy, mm, dd = date if date is not None else parse_item_date(item_id)
        return f"{base}/{coll}/{yyyy}/{mm}/{dd}/{item_id}/{item_id}_{coll}_{self.key}"

    def open(
        self,
        path_or_url: str,
        *,
        token: str | None = None,
        overview_level: int | None = None,
        http_timeout: float = 30.0,
    ) -> Any:
        """Open the product from a URL or local path. Kind-specific."""
        raise NotImplementedError


class CMRasterProduct(CMProduct):
    """A georeferenced raster product (GeoTIFF) → ``RasterioReader``."""

    def open(
        self,
        path_or_url: str,
        *,
        token: str | None = None,
        overview_level: int | None = None,
        http_timeout: float = 30.0,
    ) -> "RasterioReader":
        del token, http_timeout  # raster auth flows via GDAL_HTTP_HEADERS
        from georeader.rasterio_reader import RasterioReader

        return RasterioReader(str(path_or_url), overview_level=overview_level)


class CMVectorProduct(CMProduct):
    """A GeoJSON vector product → shapely geometry (EPSG:4326)."""

    def open(
        self,
        path_or_url: str,
        *,
        token: str | None = None,
        overview_level: int | None = None,
        http_timeout: float = 30.0,
    ) -> "BaseGeometry | None":
        del overview_level
        data = _fetch_json_or_file(path_or_url, token=token, http_timeout=http_timeout)
        from georeader.readers.carbonmapper.image import (
            _parse_geojson_to_geometry,
        )

        return _parse_geojson_to_geometry(data)


class CMTextProduct(CMProduct):
    """A plain-text sidecar (``uas.txt``) → ``str``."""

    def open(
        self,
        path_or_url: str,
        *,
        token: str | None = None,
        overview_level: int | None = None,
        http_timeout: float = 30.0,
    ) -> str:
        del overview_level
        sp = str(path_or_url)
        if sp.startswith(("http://", "https://")):
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = requests.get(sp, headers=headers, timeout=http_timeout)
            r.raise_for_status()
            return r.text
        return Path(sp).read_text()


class CMQuicklookProduct(CMProduct):
    """A non-georeferenced PNG quicklook → raw ``bytes``."""

    def open(
        self,
        path_or_url: str,
        *,
        token: str | None = None,
        overview_level: int | None = None,
        http_timeout: float = 30.0,
    ) -> bytes:
        del overview_level
        sp = str(path_or_url)
        if sp.startswith(("http://", "https://")):
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            r = requests.get(sp, headers=headers, timeout=http_timeout)
            r.raise_for_status()
            return r.content
        return Path(sp).read_bytes()


def _fetch_json_or_file(
    path_or_url: str, *, token: str | None, http_timeout: float
) -> Any:
    sp = str(path_or_url)
    if sp.startswith(("http://", "https://")):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(sp, headers=headers, timeout=http_timeout)
        r.raise_for_status()
        return r.json()
    with open(sp, "r") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────
#  The registry — one instance per product Carbon Mapper publishes
#  (verified against the live API, 2026-07 audit §3–§4)
# ─────────────────────────────────────────────────────────────────────

# L3A vis — per-plume crops keyed by plume_id
PLUME_TIF = CMRasterProduct(
    "plume.tif", CMProductFamily.L3A_VIS,
    "RGBA visualisation with plume overlay; band-4 alpha = plume mask",
)
PLUME_CONCENTRATIONS = CMRasterProduct(
    "plume-concentrations.tif", CMProductFamily.L3A_VIS,
    "CH4 column density crop (ppm·m), thumbnail-grade (~41×48 px)",
)
PLUME_OUTLINE = CMVectorProduct(
    "plume-outline.geojson", CMProductFamily.L3A_VIS,
    "Plume polygon (EPSG:4326) — the broader full-mask outline",
)
RGB_TIF = CMRasterProduct(
    "rgb.tif", CMProductFamily.L3A_VIS,
    "3-band uint8 true-colour crop of the plume window",
)
PLUME_PNG = CMQuicklookProduct(
    "plume.png", CMProductFamily.L3A_VIS,
    "Non-georeferenced quicklook of plume.tif",
)
RGB_PNG = CMQuicklookProduct(
    "rgb.png", CMProductFamily.L3A_VIS,
    "Non-georeferenced quicklook of rgb.tif",
)
PLUME_RGB_PNG = CMQuicklookProduct(
    "plume-rgb.png", CMProductFamily.L3A_VIS,
    "Non-georeferenced quicklook — plume overlaid on RGB",
)

# L3A ime — the emission-integral products, keyed by plume_id
IME_CONCENTRATIONS = CMRasterProduct(
    "ime-cmf-concentrations.tif", CMProductFamily.L3A_IME,
    "IME-clipped CH4 column density (~11×11 px) — the emission_auto "
    "integrand (the record's con_tif field points here)",
)
IME_MASK = CMRasterProduct(
    "ime-cmf-mask.tif", CMProductFamily.L3A_IME,
    "Binary mask of pixels contributing to emission_auto",
)
IME_OUTLINE = CMVectorProduct(
    "ime-cmf-outline.geojson", CMProductFamily.L3A_IME,
    "IME-significance polygon (tighter than plume-outline)",
)
IME_CONCENTRATIONS_PNG = CMQuicklookProduct(
    "ime-cmf-concentrations.png", CMProductFamily.L3A_IME,
    "Non-georeferenced quicklook of ime-cmf-concentrations.tif",
)
IME_MASK_PNG = CMQuicklookProduct(
    "ime-cmf-mask.png", CMProductFamily.L3A_IME,
    "Non-georeferenced quicklook of ime-cmf-mask.tif",
)

# L2B — whole-scene retrieval products keyed by scene name
CMF = CMRasterProduct(
    "cmf.tif", CMProductFamily.L2B,
    "CH4 matched-filter retrieval, orthorectified (ppm·m)",
)
CMF_UNORTHO = CMRasterProduct(
    "cmf-unortho.tif", CMProductFamily.L2B,
    "CH4 retrieval in raw sensor frame (pre-orthorectification)",
)
UNCERTAINTY = CMRasterProduct(
    "uncertainty.tif", CMProductFamily.L2B,
    "Per-pixel retrieval uncertainty aligned with cmf",
)
UNCERTAINTY_UNORTHO = CMRasterProduct(
    "uncertainty-unortho.tif", CMProductFamily.L2B,
    "Per-pixel uncertainty in raw sensor frame",
)
ARTIFACT_MASK = CMRasterProduct(
    "artifact-mask.tif", CMProductFamily.L2B,
    "Geometric-anomaly flag layer (NOT a cloud mask)",
)
UAS = CMTextProduct(
    "uas.txt", CMProductFamily.L2B,
    "Sensor-metadata text sidecar",
)

# L2B rgb — whole-scene true-colour sibling (separate collection)
SCENE_RGB = CMRasterProduct(
    "rgb.tif", CMProductFamily.L2B_RGB,
    "3-band uint8 true-colour raster of the whole scene",
)


#: Per-plume (L3A) products, GeoTIFF/GeoJSON only — matches the legacy
#: ``CM_PLUME_IMAGE_ASSETS`` bundle exactly.
DEFAULT_PLUME_PRODUCTS: tuple[CMProduct, ...] = (
    PLUME_TIF, PLUME_CONCENTRATIONS, PLUME_OUTLINE, RGB_TIF,
    IME_CONCENTRATIONS, IME_MASK, IME_OUTLINE,
)

#: Every per-plume product, including the PNG quicklooks.
ALL_PLUME_PRODUCTS: tuple[CMProduct, ...] = DEFAULT_PLUME_PRODUCTS + (
    PLUME_PNG, RGB_PNG, PLUME_RGB_PNG,
    IME_CONCENTRATIONS_PNG, IME_MASK_PNG,
)

#: Per-scene (L2B) products of the gas-typed collection — matches the
#: legacy ``CM_L2B_BANDS`` + ``uas`` set.
DEFAULT_SCENE_PRODUCTS: tuple[CMProduct, ...] = (
    CMF, CMF_UNORTHO, UNCERTAINTY, UNCERTAINTY_UNORTHO, ARTIFACT_MASK, UAS,
)

#: Every product in the registry.
ALL_PRODUCTS: tuple[CMProduct, ...] = ALL_PLUME_PRODUCTS + DEFAULT_SCENE_PRODUCTS + (
    SCENE_RGB,
)


def product_for_key(
    key: str, *, family: CMProductFamily | None = None
) -> CMProduct:
    """Look up a registry product by asset key (and family for the
    ambiguous ``rgb.tif``).

    >>> product_for_key("plume-outline.geojson").name
    'plume_outline'
    >>> product_for_key("rgb.tif", family=CMProductFamily.L2B_RGB) is SCENE_RGB
    True

    Raises:
        KeyError: If no product matches (or several do and ``family``
            wasn't given).
    """
    matches = [
        p for p in ALL_PRODUCTS
        if p.key == key and (family is None or p.family == family)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No Carbon Mapper product with key {key!r}")
    raise KeyError(
        f"Ambiguous product key {key!r} — matches families "
        f"{[p.family.value for p in matches]}; pass family=."
    )
