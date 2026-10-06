"""Offline tests for georeader.readers.tileserver.read_from_tileserver.

``requests.get`` is replaced with a fake that returns real ``requests.Response`` objects, so the tests
need no network: a 256 x 256 JPEG for tiles that exist and an HTML error page for tiles that do not.
"""

from io import BytesIO

import mercantile
import pytest
import requests
from PIL import Image
from shapely.geometry import box

from georeader.geotensor import GeoTensor
from georeader.readers import tileserver

TILE_SERVER = "https://tiles.example.com/{z}/{x}/{y}"
ZOOM = 15
# A zoom-15 tile and a polygon strictly inside it (so exactly one tile is read).
TILE = mercantile.Tile(x=20763, y=13619, z=ZOOM)


def _inside(tile: mercantile.Tile, shrink: float = 0.25):
    b = mercantile.bounds(tile)
    dx, dy = (b.east - b.west) * shrink, (b.north - b.south) * shrink
    return box(b.west + dx, b.south + dy, b.east - dx, b.north - dy)


def _response(url: str, status: int) -> requests.Response:
    rsp = requests.Response()
    rsp.status_code, rsp.url = status, url
    rsp.reason = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}[status]
    if status == 200:
        buf = BytesIO()
        Image.new("RGB", (256, 256), (120, 110, 90)).save(buf, format="JPEG")
        rsp.headers["content-type"] = "image/jpeg"
        rsp._content = buf.getvalue()
    else:
        rsp.headers["content-type"] = "text/html; charset=UTF-8"
        rsp._content = b"<!DOCTYPE html><title>Error</title>"
    return rsp


@pytest.fixture
def fake_server(monkeypatch):
    """Serve every tile except those listed in ``missing`` (mapped to an HTTP status)."""
    missing: dict[tuple[int, int, int], int] = {}

    def fake_get(url, *args, **kwargs):
        for (z, x, y), status in missing.items():
            if url == TILE_SERVER.format(z=z, x=x, y=y):
                return _response(url, status)
        return _response(url, 200)

    monkeypatch.setattr(tileserver.requests, "get", fake_get)
    return missing


def test_reads_a_served_tile(fake_server):
    out = tileserver.read_from_tileserver(TILE_SERVER, _inside(TILE), zoom=ZOOM)
    assert isinstance(out, GeoTensor)
    assert out.shape[0] == 3


@pytest.mark.parametrize("status", [404, 500])
def test_http_error_names_status_and_tile(fake_server, status):
    fake_server[(TILE.z, TILE.x, TILE.y)] = status
    with pytest.raises(requests.HTTPError) as excinfo:
        tileserver.read_from_tileserver(TILE_SERVER, _inside(TILE), zoom=ZOOM)
    message = str(excinfo.value)
    assert str(status) in message
    assert TILE_SERVER.format(z=TILE.z, x=TILE.x, y=TILE.y) in message


def test_one_missing_tile_in_a_mosaic_raises(fake_server):
    """A polygon spanning a 2 x 2 block of tiles raises if any one of them is missing."""
    fake_server[(TILE.z, TILE.x, TILE.y)] = 404
    b0 = mercantile.bounds(TILE)
    b1 = mercantile.bounds(mercantile.Tile(TILE.x + 1, TILE.y + 1, ZOOM))
    polygon = box((b0.west + b0.east) / 2, (b1.south + b1.north) / 2,
                  (b1.west + b1.east) / 2, (b0.south + b0.north) / 2)
    with pytest.raises(requests.HTTPError, match="404"):
        tileserver.read_from_tileserver(TILE_SERVER, polygon, zoom=ZOOM)


def test_mosaic_of_served_tiles(fake_server):
    b0 = mercantile.bounds(TILE)
    b1 = mercantile.bounds(mercantile.Tile(TILE.x + 1, TILE.y + 1, ZOOM))
    polygon = box((b0.west + b0.east) / 2, (b1.south + b1.north) / 2,
                  (b1.west + b1.east) / 2, (b0.south + b0.north) / 2)
    out = tileserver.read_from_tileserver(TILE_SERVER, polygon, zoom=ZOOM)
    assert isinstance(out, GeoTensor)
