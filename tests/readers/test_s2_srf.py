"""Tests for S2_SAFE_reader.read_srf (issue #72).

The SRF file used to be downloaded from a hardcoded SentiWiki URL that
broke when SentiWiki reorganized its attachments. It is now bundled as
package data, so read_srf must work with no network access and an empty
``~/.georeader`` cache.
"""

import os
import socket

import pytest

from georeader.readers import S2_SAFE_reader

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]


@pytest.fixture
def offline_fresh_environment(monkeypatch, tmp_path):
    """Empty home dir (no ~/.georeader cache) and no network access."""
    monkeypatch.setenv("HOME", str(tmp_path))

    def guard(*args, **kwargs):
        raise RuntimeError("network access attempted during offline test")

    monkeypatch.setattr(socket.socket, "connect", guard)


def test_srf_file_default_is_bundled():
    assert not S2_SAFE_reader.SRF_FILE_DEFAULT.startswith("http")
    assert os.path.exists(S2_SAFE_reader.SRF_FILE_DEFAULT)


@pytest.mark.parametrize("satellite", ["S2A", "S2B", "S2C"])
def test_read_srf_offline_fresh_environment(satellite, offline_fresh_environment):
    srf = S2_SAFE_reader.read_srf(satellite, cache=False)
    assert list(srf.columns) == S2_BANDS
    assert srf.index.name == "SR_WL"
    assert len(srf) > 0
    # values are responses in [0, 1] with the all-zero rows dropped
    assert (srf.values >= 0).all() and (srf.values <= 1).all()
    assert (srf.values > 1e-6).any(axis=1).all()


def test_read_srf_rejects_unknown_satellite():
    with pytest.raises(AssertionError):
        S2_SAFE_reader.read_srf("S3A", cache=False)
