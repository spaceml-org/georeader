"""Tests for georeader.readers.carbonmapper.config module."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from georeader.readers.carbonmapper.config import (
    CONFIG_SEARCH_PATHS,
    DEFAULT_SAVE_PATH,
    CarbonMapperConfig,
    _ENV_EMAIL,
    _ENV_PASSWORD,
    _ENV_TOKEN,
)

# --- Construction ---


class TestCarbonMapperConfigInit:
    """Tests for CarbonMapperConfig constructor."""

    def test_defaults_all_none(self):
        cfg = CarbonMapperConfig()
        assert cfg.token is None
        assert cfg.email is None
        assert cfg.password is None
        assert cfg.extra == {}

    def test_explicit_values(self):
        cfg = CarbonMapperConfig(token="tok", email="a@b.com", password="pw")
        assert cfg.token == "tok"
        assert cfg.email == "a@b.com"
        assert cfg.password == "pw"

    def test_extra_kwargs_stored(self):
        cfg = CarbonMapperConfig(token="t", custom_key="val")
        assert cfg.extra == {"custom_key": "val"}


# --- from_env ---


class TestFromEnv:
    """Tests for CarbonMapperConfig.from_env."""

    def test_reads_all_env_vars(self):
        env = {
            _ENV_TOKEN: "my-token",
            _ENV_EMAIL: "user@example.com",
            _ENV_PASSWORD: "secret",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = CarbonMapperConfig.from_env()
        assert cfg.token == "my-token"
        assert cfg.email == "user@example.com"
        assert cfg.password == "secret"

    def test_missing_env_vars_are_none(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = CarbonMapperConfig.from_env()
        assert cfg.token is None
        assert cfg.email is None
        assert cfg.password is None


# --- from_file ---


class TestFromFile:
    """Tests for CarbonMapperConfig.from_file."""

    def test_loads_json_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "token": "file-token",
            "email": "file@example.com",
            "password": "file-pw",
        }))
        cfg = CarbonMapperConfig.from_file(config_file)
        assert cfg.token == "file-token"
        assert cfg.email == "file@example.com"
        assert cfg.password == "file-pw"

    def test_extra_fields_preserved(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "token": "t",
            "base_url": "https://custom.api",
        }))
        cfg = CarbonMapperConfig.from_file(config_file)
        assert cfg.extra["base_url"] == "https://custom.api"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            CarbonMapperConfig.from_file("/nonexistent/path/config.json")

    def test_invalid_json_raises(self, tmp_path):
        config_file = tmp_path / "bad.json"
        config_file.write_text("not json!")
        with pytest.raises(json.JSONDecodeError):
            CarbonMapperConfig.from_file(config_file)


# --- load ---


class TestLoad:
    """Tests for CarbonMapperConfig.load resolution order."""

    def test_explicit_path_takes_priority(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"token": "explicit"}))

        with patch.dict(os.environ, {}, clear=True):
            cfg = CarbonMapperConfig.load(path=config_file)
        assert cfg.token == "explicit"

    def test_env_overlays_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"token": "from-file", "email": "file@e.com"}))

        with patch.dict(os.environ, {_ENV_TOKEN: "from-env"}, clear=False):
            cfg = CarbonMapperConfig.load(path=config_file)
        assert cfg.token == "from-env"
        assert cfg.email == "file@e.com"

    def test_no_config_returns_empty(self):
        # ``create_placeholder=False`` keeps the filesystem untouched —
        # the placeholder behaviour is exercised separately below.
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(Path, "exists", return_value=False):
                cfg = CarbonMapperConfig.load(create_placeholder=False)
        assert cfg.token is None
        assert cfg.email is None

    def test_nonexistent_explicit_path_warns_and_continues(self, tmp_path):
        fake_path = tmp_path / "missing.json"
        with patch.dict(os.environ, {}, clear=True):
            # Explicit path is given — placeholder creation is skipped
            # by design (see config.py:load step 4 guard).
            cfg = CarbonMapperConfig.load(path=fake_path)
        assert cfg.token is None


# --- canonical path + placeholder behaviour ---


class TestCanonicalPath:
    """`~/.georeader/auth_carbonmapper.json` is the canonical path —
    first in :data:`CONFIG_SEARCH_PATHS`, matches the sibling-reader
    convention (emit, S2_SAFE_reader, scihubcopernicus_query)."""

    def test_default_save_path_is_in_dot_georeader(self):
        assert DEFAULT_SAVE_PATH == (
            Path.home() / ".georeader" / "auth_carbonmapper.json"
        )

    def test_dot_georeader_is_first_in_search_paths(self):
        assert CONFIG_SEARCH_PATHS[0] == DEFAULT_SAVE_PATH

    def test_legacy_paths_kept_as_fallbacks(self):
        # Back-compat — users with existing configs at the old paths
        # still resolve without action.
        rest = CONFIG_SEARCH_PATHS[1:]
        assert Path("config") / "carbonmapper_token.json" in rest
        assert (
            Path.home() / ".config" / "carbonmapper" / "config.json"
        ) in rest
        assert Path.home() / ".carbonmapper.json" in rest
        assert Path(".carbonmapper.json") in rest

    def test_dot_georeader_wins_over_legacy(self, tmp_path, monkeypatch):
        """When both the canonical path and a legacy path exist, the
        canonical one resolves."""
        # Redirect Path.home() so we don't touch the real user dir.
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        # Reimport-style: pick up the new home via the module's
        # configured constants. Easiest is to compute the canonical path
        # directly against the fake home.
        canonical = tmp_path / ".georeader" / "auth_carbonmapper.json"
        legacy = tmp_path / ".config" / "carbonmapper" / "config.json"

        canonical.parent.mkdir(parents=True, exist_ok=True)
        legacy.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(json.dumps({"token": "canonical-wins"}))
        legacy.write_text(json.dumps({"token": "legacy-loser"}))

        with patch.dict(os.environ, {}, clear=True):
            # Rebuild the search-paths list against the patched home —
            # the module's CONFIG_SEARCH_PATHS was bound at import time
            # and still references the real home, so iterate manually.
            search_paths = [
                tmp_path / ".georeader" / "auth_carbonmapper.json",
                Path("config") / "carbonmapper_token.json",
                tmp_path / ".config" / "carbonmapper" / "config.json",
                tmp_path / ".carbonmapper.json",
                Path(".carbonmapper.json"),
            ]
            with patch(
                "georeader.readers.carbonmapper.config.CONFIG_SEARCH_PATHS",
                search_paths,
            ):
                cfg = CarbonMapperConfig.load(create_placeholder=False)
        assert cfg.token == "canonical-wins"


class TestPlaceholderCreation:
    """`load()` auto-creates a stub ``~/.georeader/auth_carbonmapper.json``
    on first run when no credentials are found — matches emit / S2."""

    def test_load_creates_placeholder_when_no_config(self, tmp_path, monkeypatch):
        canonical = tmp_path / ".georeader" / "auth_carbonmapper.json"
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.DEFAULT_SAVE_PATH",
            canonical,
        )
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.CONFIG_SEARCH_PATHS",
            [canonical],
        )
        assert not canonical.exists()
        with patch.dict(os.environ, {}, clear=True):
            cfg = CarbonMapperConfig.load()
        assert canonical.exists(), "placeholder file should have been created"
        body = json.loads(canonical.read_text())
        assert body == {
            "email": "SET-EMAIL",
            "password": "SET-PASSWORD",
            "token": None,
        }
        # On the first run the returned cfg is empty — the placeholder
        # is just a UX target on disk; `cfg` itself doesn't pretend to
        # have credentials.
        assert cfg.email is None
        assert cfg.password is None
        assert cfg.token is None
        assert not cfg.has_credentials()

    def test_second_load_with_placeholder_file_filters_stubs(self, tmp_path, monkeypatch):
        """Once the placeholder is on disk, the next ``load()`` call
        opens it via ``from_file``. The stub values must be filtered to
        ``None`` — otherwise ``has_credentials()`` would falsely return
        ``True`` because both fields are non-empty strings."""
        canonical = tmp_path / ".georeader" / "auth_carbonmapper.json"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(json.dumps({
            "email": "SET-EMAIL", "password": "SET-PASSWORD", "token": None,
        }))
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.DEFAULT_SAVE_PATH",
            canonical,
        )
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.CONFIG_SEARCH_PATHS",
            [canonical],
        )
        with patch.dict(os.environ, {}, clear=True):
            cfg = CarbonMapperConfig.load(create_placeholder=False)
        assert cfg.email is None
        assert cfg.password is None
        assert not cfg.has_credentials()

    def test_load_skips_placeholder_when_disabled(self, tmp_path, monkeypatch):
        canonical = tmp_path / ".georeader" / "auth_carbonmapper.json"
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.DEFAULT_SAVE_PATH",
            canonical,
        )
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.CONFIG_SEARCH_PATHS",
            [canonical],
        )
        with patch.dict(os.environ, {}, clear=True):
            CarbonMapperConfig.load(create_placeholder=False)
        assert not canonical.exists(), "placeholder must not be written"

    def test_load_skips_placeholder_when_env_token_set(self, tmp_path, monkeypatch):
        """Env-var creds are sufficient — no placeholder needed."""
        canonical = tmp_path / ".georeader" / "auth_carbonmapper.json"
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.DEFAULT_SAVE_PATH",
            canonical,
        )
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.CONFIG_SEARCH_PATHS",
            [canonical],
        )
        with patch.dict(os.environ, {_ENV_TOKEN: "jwt-from-env"}, clear=True):
            cfg = CarbonMapperConfig.load()
        assert cfg.token == "jwt-from-env"
        assert not canonical.exists(), (
            "no placeholder when env credentials suffice"
        )

    def test_load_skips_placeholder_when_explicit_path_given(self, tmp_path, monkeypatch):
        """Explicit ``path=`` argument means the caller knows where their
        config lives; we shouldn't write a stub elsewhere."""
        canonical = tmp_path / ".georeader" / "auth_carbonmapper.json"
        explicit_missing = tmp_path / "some-other-place.json"
        monkeypatch.setattr(
            "georeader.readers.carbonmapper.config.DEFAULT_SAVE_PATH",
            canonical,
        )
        with patch.dict(os.environ, {}, clear=True):
            CarbonMapperConfig.load(path=explicit_missing)
        assert not canonical.exists(), (
            "explicit path means no opinionated placeholder"
        )


# --- save ---


class TestSave:
    """Tests for CarbonMapperConfig.save."""

    def test_save_creates_file(self, tmp_path):
        dest = tmp_path / "out" / "config.json"
        cfg = CarbonMapperConfig(token="save-test", email="s@e.com")
        result = cfg.save(path=dest)

        assert result == dest
        assert dest.exists()
        data = json.loads(dest.read_text())
        assert data["token"] == "save-test"
        assert data["email"] == "s@e.com"

    def test_save_omits_none_fields(self, tmp_path):
        dest = tmp_path / "config.json"
        cfg = CarbonMapperConfig(token="tok")
        cfg.save(path=dest)

        data = json.loads(dest.read_text())
        assert "token" in data
        assert "email" not in data
        assert "password" not in data

    def test_save_includes_extra(self, tmp_path):
        dest = tmp_path / "config.json"
        cfg = CarbonMapperConfig(token="tok", api_version="v2")
        cfg.save(path=dest)

        data = json.loads(dest.read_text())
        assert data["api_version"] == "v2"

    def test_save_sets_restrictive_permissions(self, tmp_path):
        dest = tmp_path / "config.json"
        cfg = CarbonMapperConfig(token="tok")
        cfg.save(path=dest)
        mode = dest.stat().st_mode & 0o777
        assert mode == 0o600


# --- reset ---


class TestReset:
    """Tests for CarbonMapperConfig.reset."""

    def test_reset_deletes_file(self, tmp_path):
        dest = tmp_path / "config.json"
        dest.write_text("{}")
        assert dest.exists()
        CarbonMapperConfig.reset(path=dest)
        assert not dest.exists()

    def test_reset_nonexistent_is_noop(self, tmp_path):
        dest = tmp_path / "missing.json"
        CarbonMapperConfig.reset(path=dest)


# --- get_token / has_credentials ---


class TestTokenAndCredentials:
    """Tests for get_token and has_credentials."""

    def test_get_token_returns_token(self):
        cfg = CarbonMapperConfig(token="bearer-123")
        assert cfg.get_token() == "bearer-123"

    def test_get_token_none_when_no_token(self):
        cfg = CarbonMapperConfig(email="a@b.com", password="pw")
        assert cfg.get_token() is None

    def test_has_credentials_with_token(self):
        cfg = CarbonMapperConfig(token="tok")
        assert cfg.has_credentials() is True

    def test_has_credentials_with_email_and_password(self):
        cfg = CarbonMapperConfig(email="a@b.com", password="pw")
        assert cfg.has_credentials() is True

    def test_has_credentials_false_when_empty(self):
        cfg = CarbonMapperConfig()
        assert cfg.has_credentials() is False

    def test_has_credentials_false_with_only_email(self):
        cfg = CarbonMapperConfig(email="a@b.com")
        assert cfg.has_credentials() is False


# --- repr ---


class TestRepr:
    """Tests for string representation."""

    def test_repr_contains_email(self):
        cfg = CarbonMapperConfig(email="a@b.com", token="secret")
        r = repr(cfg)
        assert "a@b.com" in r
        assert "has_token=True" in r
        assert "secret" not in r

    def test_repr_no_password_leak(self):
        cfg = CarbonMapperConfig(password="supersecret")
        r = repr(cfg)
        assert "supersecret" not in r
        assert "has_password=True" in r


# --- refresh_access_token ---


class TestRefreshAccessToken:
    """Tests for CarbonMapperConfig.refresh_access_token."""

    def test_raises_without_email(self):
        cfg = CarbonMapperConfig(password="pw")
        with pytest.raises(ValueError, match="email and password are required"):
            cfg.refresh_access_token()

    def test_raises_without_password(self):
        cfg = CarbonMapperConfig(email="a@b.com")
        with pytest.raises(ValueError, match="email and password are required"):
            cfg.refresh_access_token()

    def test_raises_when_both_missing(self):
        cfg = CarbonMapperConfig()
        with pytest.raises(ValueError, match="email and password are required"):
            cfg.refresh_access_token()

    @patch("georeader.readers.carbonmapper.download.obtain_token")
    def test_success_updates_token_and_refresh(self, mock_obtain):
        mock_obtain.return_value = {
            "access": "new-access-token",
            "refresh": "new-refresh-token",
        }
        cfg = CarbonMapperConfig(email="a@b.com", password="pw")
        result = cfg.refresh_access_token()

        mock_obtain.assert_called_once_with("a@b.com", "pw")
        assert result == "new-access-token"
        assert cfg.token == "new-access-token"
        assert cfg.extra["refresh"] == "new-refresh-token"

    @patch("georeader.readers.carbonmapper.download.obtain_token")
    def test_success_without_refresh_key(self, mock_obtain):
        mock_obtain.return_value = {"access": "tok-only"}
        cfg = CarbonMapperConfig(email="a@b.com", password="pw")
        result = cfg.refresh_access_token()

        assert result == "tok-only"
        assert cfg.token == "tok-only"
        assert cfg.extra.get("refresh") is None
