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
   :meth:`CarbonMapperConfig.load`.
3. **Explicit arguments** — pass ``token=`` directly to API functions in
   ``download.py``.

Quick start
-----------
>>> from georeader.readers.carbonmapper.config import CarbonMapperConfig
>>> cfg = CarbonMapperConfig.load()
>>> token = cfg.get_token()  # resolves from env var or file
>>> # — or — store credentials in the default config file:
>>> cfg.email = "user@example.com"
>>> cfg.password = "s3cret"
>>> cfg.save()

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

CONFIG_SEARCH_PATHS: list[Path] = [
    Path("config") / "carbonmapper_token.json",  # project-local convention
    Path.home() / ".config" / "carbonmapper" / "config.json",
    Path.home() / ".carbonmapper.json",
    Path(".carbonmapper.json"),  # current working directory
]

#: Default path used by :meth:`CarbonMapperConfig.save` and
#: :meth:`CarbonMapperConfig.reset` — a user-level config file that is
#: outside the working tree to avoid accidentally committing credentials.
DEFAULT_SAVE_PATH: Path = Path.home() / ".config" / "carbonmapper" / "config.json"

# Environment variable names
_ENV_TOKEN = "CARBONMAPPER_TOKEN"
_ENV_EMAIL = "CARBONMAPPER_EMAIL"
_ENV_PASSWORD = "CARBONMAPPER_PASSWORD"


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
        >>> cfg = CarbonMapperConfig.from_file("~/.config/carbonmapper/config.json")
        """
        path = Path(path).expanduser().resolve()
        with path.open() as fh:
            data: dict[str, Any] = json.load(fh)
        token = data.pop("token", None)
        email = data.pop("email", None) or data.pop("username", None)
        password = data.pop("password", None)
        return cls(token=token, email=email, password=password, **data)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "CarbonMapperConfig":
        """Load config using the standard resolution order.

        Resolution order
        ~~~~~~~~~~~~~~~~
        1. If *path* is given, load that file.
        2. Otherwise search :data:`CONFIG_SEARCH_PATHS` for the first file
           that exists.
        3. Overlay environment variables — env values overwrite file values.

        Parameters
        ----------
        path:
            Optional explicit path to a config file.  Skips the search
            when provided.

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

        # 1. Explicit path
        if path is not None:
            resolved = Path(path).expanduser().resolve()
            if resolved.exists():
                try:
                    cfg = cls.from_file(resolved)
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
            (``~/.config/carbonmapper/config.json``), a user-level
            location outside the working tree so credentials are never
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
        /home/user/.config/carbonmapper/config.json
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
            (``~/.config/carbonmapper/config.json``).

        Examples
        --------
        >>> CarbonMapperConfig.reset()  # removes ~/.config/carbonmapper/config.json
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
        >>> cfg = CarbonMapperConfig.load("config/carbonmapper_token.json")
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
