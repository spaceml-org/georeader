"""
config.py
=========

Lightweight credentials and configuration handler for the Carbon Mapper
Data Platform API.

Credentials can be supplied in three ways (checked in priority order):

1. **Environment variables** — set ``CARBONMAPPER_TOKEN`` (access token),
   ``CARBONMAPPER_EMAIL`` and ``CARBONMAPPER_PASSWORD`` (login credentials).
2. **Config file** — a JSON file at one of the well-known paths listed in
   :data:`CONFIG_SEARCH_PATHS`, or a custom path passed to
   :meth:`CarbonMapperConfig.load`. The canonical location matches the
   sibling readers (``emit.py`` / ``S2_SAFE_reader.py``):
   ``~/.georeader/auth_carbonmapper.json``.
3. **Explicit arguments** — pass ``token=`` directly to API functions in
   ``download.py``.

If no config file exists when :meth:`CarbonMapperConfig.load` is called
without an explicit path and no env-var credentials are set, a
placeholder ``~/.georeader/auth_carbonmapper.json`` is auto-created
with stub values so users have a clear edit target.

Quick start
-----------
>>> from georeader.readers.carbonmapper.config import CarbonMapperConfig
>>> cfg = CarbonMapperConfig.load()
>>> token = cfg.get_token()  # resolves from env var or file
>>> # — or — store credentials in the default config file:
>>> cfg.email = "user@example.com"
>>> cfg.password = "s3cret"
>>> cfg.save()                # writes to ~/.georeader/auth_carbonmapper.json

References
----------
- API docs      : https://api.carbonmapper.org/api/v1/docs
- Registration  : https://data.carbonmapper.org
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config search locations (in priority order)
# ---------------------------------------------------------------------------

#: Canonical credentials path — matches the sibling-reader convention
#: (``~/.georeader/auth_emit.json`` for EMIT,
#: ``~/.georeader/auth_S2.json`` for Sentinel-2 / SciHub, etc.).
#: Used by :meth:`CarbonMapperConfig.save` and :meth:`CarbonMapperConfig.reset`,
#: and is the first entry in :data:`CONFIG_SEARCH_PATHS`.
DEFAULT_SAVE_PATH: Path = Path.home() / ".georeader" / "auth_carbonmapper.json"

#: Search order, first-match-wins. The canonical path is checked first;
#: legacy paths are kept so users who already configured them continue
#: to work without action.
CONFIG_SEARCH_PATHS: list[Path] = [
    DEFAULT_SAVE_PATH,                                          # canonical
    Path("config") / "carbonmapper_token.json",                 # legacy — project-local
    Path.home() / ".config" / "carbonmapper" / "config.json",   # legacy
    Path.home() / ".carbonmapper.json",                         # legacy
    Path(".carbonmapper.json"),                                 # legacy — cwd
]

# Environment variable names
_ENV_TOKEN = "CARBONMAPPER_TOKEN"
_ENV_EMAIL = "CARBONMAPPER_EMAIL"
_ENV_PASSWORD = "CARBONMAPPER_PASSWORD"

# Placeholder values written into the stub config on first run. Users
# replace these with real credentials. Matches the ``SET-USER`` /
# ``SET-PASSWORD`` style of ``emit.py``.
_PLACEHOLDER_EMAIL = "SET-EMAIL"
_PLACEHOLDER_PASSWORD = "SET-PASSWORD"


def _create_placeholder_config() -> Path:
    """Write a stub config file to :data:`DEFAULT_SAVE_PATH` if missing.

    Matches the behaviour of :mod:`georeader.readers.emit` and
    :mod:`georeader.readers.scihubcopernicus_query`, which auto-create
    a placeholder JSON on first run so users have a clear edit target.

    Returns the destination path. No-op if the file already exists.
    """
    dest = DEFAULT_SAVE_PATH.expanduser().resolve()
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    placeholder = {
        "email": _PLACEHOLDER_EMAIL,
        "password": _PLACEHOLDER_PASSWORD,
        "token": None,
    }
    dest.write_text(json.dumps(placeholder, indent=2))
    try:
        os.chmod(dest, 0o600)
    except (PermissionError, OSError) as exc:
        logger.warning(
            "Created placeholder config at %s but couldn't set 0o600 perms: %s",
            dest,
            exc,
        )
    logger.warning(
        "Carbon Mapper credentials not configured. Created placeholder "
        "at %s — edit it with your Carbon Mapper email + password "
        "(or set CARBONMAPPER_TOKEN / CARBONMAPPER_EMAIL / "
        "CARBONMAPPER_PASSWORD env vars).",
        dest,
    )
    return dest


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------


class CarbonMapperConfig:
    """Simple credentials and configuration container for the Carbon Mapper API.

    Attributes
    ----------
    token:
        A pre-obtained JWT bearer token.  If set, it takes precedence over
        *email* / *password* when :meth:`get_token` is called.
    email:
        Registered Carbon Mapper account e-mail address.
    password:
        Account password.  Stored only in memory or in the config file on
        disk — never sent anywhere except the token endpoint.
    extra:
        Any additional key/value pairs loaded from or saved to the config
        file (for forward compatibility).

    Examples
    --------
    Load from environment or disk and retrieve a usable token:

    >>> cfg = CarbonMapperConfig.load()
    >>> token = cfg.get_token()  # may return None if no credentials found
    >>> if token:
    ...     data = get_plumes_annotated(plume_gas="CH4", token=token)

    Persist credentials to the default config file:

    >>> cfg = CarbonMapperConfig(email="user@example.com", password="s3cret")
    >>> cfg.save()

    Reset (delete) the stored config file:

    >>> CarbonMapperConfig.reset()
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        email: str | None = None,
        password: str | None = None,
        **extra: Any,
    ) -> None:
        self.token = token
        self.email = email
        self.password = password
        self.extra: dict[str, Any] = extra

    # ------------------------------------------------------------------ #
    # Class-level factory / persistence methods                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "CarbonMapperConfig":
        """Build a :class:`CarbonMapperConfig` purely from environment variables.

        Reads :envvar:`CARBONMAPPER_TOKEN`, :envvar:`CARBONMAPPER_EMAIL`,
        and :envvar:`CARBONMAPPER_PASSWORD`.  Fields that are absent from
        the environment are left as ``None``.

        Returns
        -------
        CarbonMapperConfig
            A new config object populated from the environment.

        Examples
        --------
        >>> import os
        >>> os.environ["CARBONMAPPER_TOKEN"] = "eyJ..."
        >>> cfg = CarbonMapperConfig.from_env()
        >>> cfg.token
        'eyJ...'
        """
        return cls(
            token=os.environ.get(_ENV_TOKEN),
            email=os.environ.get(_ENV_EMAIL),
            password=os.environ.get(_ENV_PASSWORD),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "CarbonMapperConfig":
        """Load a :class:`CarbonMapperConfig` from a specific JSON file.

        Parameters
        ----------
        path:
            Path to a JSON config file containing any combination of the
            keys ``"token"``, ``"email"``, ``"password"``, plus any extra
            fields.

        Returns
        -------
        CarbonMapperConfig
            Config populated from the file.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        json.JSONDecodeError
            If the file cannot be parsed as JSON.

        Examples
        --------
        >>> cfg = CarbonMapperConfig.from_file("~/.georeader/auth_carbonmapper.json")
        """
        path = Path(path).expanduser().resolve()
        with path.open() as fh:
            data: dict[str, Any] = json.load(fh)
        token = data.pop("token", None)
        email = data.pop("email", None) or data.pop("username", None)
        password = data.pop("password", None)
        # Filter stub values — if the user hasn't yet edited a freshly
        # auto-created placeholder, treat the fields as un-set rather
        # than letting ``"SET-EMAIL"`` flow into has_credentials() as
        # if it were a real value.
        if email == _PLACEHOLDER_EMAIL:
            email = None
        if password == _PLACEHOLDER_PASSWORD:
            password = None
        return cls(token=token, email=email, password=password, **data)

    @classmethod
    def load(
        cls,
        path: Path | str | None = None,
        *,
        create_placeholder: bool = True,
    ) -> "CarbonMapperConfig":
        """Load config using the standard resolution order.

        Resolution order
        ~~~~~~~~~~~~~~~~
        1. If *path* is given, load that file.
        2. Otherwise search :data:`CONFIG_SEARCH_PATHS` for the first file
           that exists.
        3. Overlay environment variables — env values overwrite file values.
        4. If still nothing is configured (no file found, no env vars set)
           AND ``create_placeholder`` is True, write a stub config to
           :data:`DEFAULT_SAVE_PATH` with ``SET-EMAIL`` / ``SET-PASSWORD``
           placeholders so users have a clear edit target. Matches the
           ``emit.py`` / ``S2_SAFE_reader.py`` behaviour.

        Parameters
        ----------
        path:
            Optional explicit path to a config file.  Skips the search
            when provided.
        create_placeholder:
            When ``True`` (default), auto-create a stub config file at
            :data:`DEFAULT_SAVE_PATH` if no credentials could be
            resolved. Set to ``False`` in tests / non-interactive
            contexts to keep the filesystem untouched.

        Returns
        -------
        CarbonMapperConfig
            The resolved config.  Fields without a value (from file *and*
            env) are ``None``.

        Examples
        --------
        >>> cfg = CarbonMapperConfig.load()
        >>> print(cfg.email)   # None if not configured

        >>> cfg = CarbonMapperConfig.load("~/my_project/.carbonmapper.json")
        """
        cfg: CarbonMapperConfig | None = None
        loaded_from_file = False

        # 1. Explicit path
        if path is not None:
            resolved = Path(path).expanduser().resolve()
            if resolved.exists():
                try:
                    cfg = cls.from_file(resolved)
                    loaded_from_file = True
                    logger.debug("Loaded Carbon Mapper config from %s", resolved)
                except Exception as exc:
                    logger.warning("Failed to load config from %s: %s", resolved, exc)
            else:
                logger.warning("Config path %s does not exist; ignoring.", resolved)

        # 2. Search well-known paths
        if cfg is None:
            for candidate in CONFIG_SEARCH_PATHS:
                resolved_candidate = candidate.expanduser().resolve()
                if resolved_candidate.exists():
                    try:
                        cfg = cls.from_file(resolved_candidate)
                        loaded_from_file = True
                        logger.debug("Loaded Carbon Mapper config from %s", resolved_candidate)
                        break
                    except Exception as exc:
                        logger.warning(
                            "Failed to load config from %s: %s",
                            resolved_candidate,
                            exc,
                        )

        if cfg is None:
            cfg = cls()

        # 3. Overlay environment variables (env takes priority over file)
        env_token = os.environ.get(_ENV_TOKEN)
        env_email = os.environ.get(_ENV_EMAIL)
        env_password = os.environ.get(_ENV_PASSWORD)
        if env_token:
            cfg.token = env_token
        if env_email:
            cfg.email = env_email
        if env_password:
            cfg.password = env_password

        # 4. Placeholder — only when caller didn't pass an explicit path,
        #    no config file was found, and env vars didn't supply creds.
        if (
            create_placeholder
            and path is None
            and not loaded_from_file
            and not cfg.has_credentials()
        ):
            _create_placeholder_config()

        return cfg

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: Path | str | None = None) -> Path:
        """Persist the config to a JSON file.

        Parameters
        ----------
        path:
            Destination file path.  Defaults to
            :data:`DEFAULT_SAVE_PATH`
            (``~/.georeader/auth_carbonmapper.json``), matching the
            sibling-reader convention (emit, S2). User-level location
            outside the working tree so credentials are never
            accidentally committed.

        Returns
        -------
        Path
            The resolved path of the file that was written.

        Examples
        --------
        >>> cfg = CarbonMapperConfig(email="user@example.com", password="s3cret")
        >>> saved_path = cfg.save()
        >>> print(saved_path)
        /home/user/.georeader/auth_carbonmapper.json
        """
        if path is not None:
            dest = Path(path).expanduser().resolve()
        else:
            dest = DEFAULT_SAVE_PATH.expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {**self.extra}
        if self.token is not None:
            data["token"] = self.token
        if self.email is not None:
            data["email"] = self.email
        if self.password is not None:
            data["password"] = self.password
        dest.write_text(json.dumps(data, indent=2))
        try:
            os.chmod(dest, 0o600)
        except PermissionError:
            logger.warning(
                "Carbon Mapper config saved to %s but restrictive permissions "
                "(0o600) could not be set due to insufficient permissions.",
                dest,
            )
        except OSError as exc:
            logger.warning(
                "Carbon Mapper config saved to %s but setting restrictive "
                "permissions (0o600) failed: %s",
                dest,
                exc,
            )
        logger.info("Carbon Mapper config saved to %s", dest)
        return dest

    @classmethod
    def reset(cls, path: Path | str | None = None) -> None:
        """Delete the stored config file, if it exists.

        Parameters
        ----------
        path:
            Path to the config file to remove.  Defaults to
            :data:`DEFAULT_SAVE_PATH`
            (``~/.georeader/auth_carbonmapper.json``).

        Examples
        --------
        >>> CarbonMapperConfig.reset()  # removes ~/.georeader/auth_carbonmapper.json
        """
        dest = (
            Path(path).expanduser().resolve()
            if path is not None
            else DEFAULT_SAVE_PATH.expanduser().resolve()
        )
        if dest.exists():
            dest.unlink()
            logger.info("Carbon Mapper config removed: %s", dest)
        else:
            logger.debug("No config file to remove at %s", dest)

    # ------------------------------------------------------------------ #
    # Token resolution                                                     #
    # ------------------------------------------------------------------ #

    def get_token(self) -> str | None:
        """Return the best available bearer token.

        If :attr:`token` is set, it is returned directly.  Otherwise
        ``None`` is returned — callers that need a fresh token should call
        :meth:`refresh_access_token` or
        :func:`~georeader.readers.carbonmapper.download.obtain_token`
        with :attr:`email` and :attr:`password`.

        Returns
        -------
        str or None
            A JWT bearer token string, or ``None`` if none is configured.

        Examples
        --------
        >>> cfg = CarbonMapperConfig.load()
        >>> token = cfg.get_token()
        >>> if token is None:
        ...     token = cfg.refresh_access_token()
        """
        return self.token

    def refresh_access_token(self) -> str:
        """Obtain a fresh JWT access token using stored email/password.

        Calls :func:`~georeader.readers.carbonmapper.download.obtain_token` with the
        stored :attr:`email` and :attr:`password`, updates :attr:`token`
        in-place, and returns the new access token.

        Returns
        -------
        str
            The new JWT access token.

        Raises
        ------
        ValueError
            If *email* or *password* is not set.
        requests.HTTPError
            If the Carbon Mapper API rejects the credentials.

        Examples
        --------
        >>> cfg = CarbonMapperConfig.load()  # ~/.georeader/auth_carbonmapper.json
        >>> token = cfg.refresh_access_token()
        """
        if not self.email or not self.password:
            raise ValueError(
                "Cannot refresh token: email and password are required. "
                "Provide them via config file, environment variables, or "
                "constructor arguments."
            )
        from georeader.readers.carbonmapper.download import obtain_token

        tokens = obtain_token(self.email, self.password)
        self.token = tokens["access"]
        self.extra["refresh"] = tokens.get("refresh")
        logger.info("Carbon Mapper access token refreshed for %s", self.email)
        return self.token

    def has_credentials(self) -> bool:
        """Return ``True`` if any usable credentials are present.

        A config is considered to have credentials when at least one of the
        following is set: :attr:`token`, or both :attr:`email` *and*
        :attr:`password`.

        Examples
        --------
        >>> cfg = CarbonMapperConfig(email="u@example.com", password="pw")
        >>> cfg.has_credentials()
        True
        >>> CarbonMapperConfig().has_credentials()
        False
        """
        return bool(self.token) or bool(self.email and self.password)

    # ------------------------------------------------------------------ #
    # String representations                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"CarbonMapperConfig("
            f"email={self.email!r}, "
            f"has_token={self.token is not None}, "
            f"has_password={self.password is not None}"
            f")"
        )
