"""
Pytest configuration for the notebook integration tests under ``docs/``.

The notebooks in this folder double as integration tests: ``make test-notebooks``
executes them with ``pytest --nbmake``. Most of them need a large raster, a
cloud credential, or network access that is not always available, so each
notebook is *skipped automatically* unless everything it needs is present. This
keeps the suite green on a machine with no data and lets CI run exactly the
subset of notebooks whose inputs (data files in ``examples/`` and/or secrets in
the environment) have been provided.

How requirements are expressed
------------------------------
For every gated notebook we list one or more :class:`Requirement` groups. A
notebook runs only if **all** of its groups are satisfied, and a single group is
satisfied if **any** of these is true:

* one of its ``files`` exists under ``examples/`` (repo root), or
* one of its ``env`` environment variables is set (the notebook then downloads /
  authenticates by itself), or
* one of its ``paths`` credential files exists (e.g. ``~/.georeader/...``).

Notebooks that are **not** listed here always run -- these are the ones that
stream public data and only need network access (Sentinel-2 SAFE from the
Google public bucket, a public Hugging Face GeoTIFF, ...).

Providing the inputs (see ``examples/README.md`` for details)
-------------------------------------------------------------
* ``examples/`` data files.
* PRISMA / EnMAP (Azure):  ``SAS_TOKEN``, ``AZURE_STORAGE_ACCOUNT``, ``CONTAINER_NAME``
* EMIT (NASA Earthdata):        ``EARTHDATA_TOKEN`` or ``~/.georeader/auth_emit.json``
* Carbon Mapper:                ``CARBONMAPPER_TOKEN`` or ``~/.georeader/auth_carbonmapper.json``
* Google Earth Engine:          ``EARTHENGINE_SERVICE_ACCOUNT_KEY`` (service-account JSON key: a file path or the raw JSON)

In GitHub Actions these can be wired as repository secrets and exported as the
matching environment variables before ``make test-notebooks`` runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Resolve examples/ relative to this file (docs/conftest.py -> repo root -> examples/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# Load credentials/config from a repo-root .env file if present (and python-dotenv
# is installed) so they are available both for the gating below and for the
# notebook kernels (which inherit this process's environment). See .env.sample.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


def _file_available(path: Path) -> bool:
    """True if ``path`` exists as real data (not an un-smudged Git LFS pointer).

    Some example rasters are stored with Git LFS. On a clone without git-lfs the
    file is present but is a tiny text pointer, which would make a notebook fail
    rather than skip. Treat such pointers as "not available" so the notebook is
    skipped cleanly.
    """
    if not path.exists():
        return False
    try:
        if path.stat().st_size < 1024:
            with open(path, "rb") as fh:
                if fh.read(40).startswith(b"version https://git-lfs"):
                    return False
    except OSError:
        return False
    return True

_ENMAP_TILE = "ENMAP01-____L1B-DT0000074101_20240511T080843Z_001_V010402_20240514T093550Z"


@dataclass
class Requirement:
    """A single requirement group (satisfied if *any* member is available)."""

    files: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def satisfied(self) -> bool:
        if any(_file_available(_EXAMPLES_DIR / f) for f in self.files):
            return True
        if any(os.environ.get(e) for e in self.env):
            return True
        if any(Path(p).expanduser().exists() for p in self.paths):
            return True
        return False

    def describe(self) -> str:
        bits = []
        if self.files:
            bits.append(f"a data file in {_EXAMPLES_DIR} ({', '.join(self.files)})")
        if self.env:
            bits.append(f"one of env vars [{', '.join(self.env)}]")
        if self.paths:
            bits.append(f"one of credential files [{', '.join(self.paths)}]")
        return " OR ".join(bits)


# Keyed by notebook basename (basenames are unique across docs/).
NOTEBOOK_REQUIREMENTS: dict[str, list[Requirement]] = {
    # --- PRISMA / EnMAP: local file or Azure download -----------------------
    "prisma_with_cloudsen12.ipynb": [
        Requirement(
            files=["PRISMA/PRS_L1_STD_OFFL_20241109073054_20241109073059_0001.he5"],
            env=["SAS_TOKEN","AZURE_STORAGE_ACCOUNT","CONTAINER_NAME"],
        ),
    ],
    "enmap_with_cloudsen12.ipynb": [
        Requirement(
            files=[f"EnMAP/{_ENMAP_TILE}/{_ENMAP_TILE}-METADATA.XML"],
            env=["SAS_TOKEN","AZURE_STORAGE_ACCOUNT","CONTAINER_NAME"],
        ),
    ],
    # --- EMIT: local file, NASA Earthdata token or auth file ----------------
    "emit_explore.ipynb": [
        Requirement(
            files=["EMIT/EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc"],
            env=["EARTHDATA_TOKEN"],
            paths=["~/.georeader/auth_emit.json"],
        ),
    ],
    # --- PRISMA (Azure) AND EMIT (NASA): both groups must be satisfied ------
    "simultaneous_prisma_emit.ipynb": [
        Requirement(
            files=["PRISMA/PRS_L1_STD_OFFL_20230929102749_20230929102753_0001.he5"],
            env=["SAS_TOKEN","AZURE_STORAGE_ACCOUNT","CONTAINER_NAME"],
        ),
        Requirement(
            files=["EMIT/EMIT_L1B_RAD_001_20230929T122534_2327208_039.nc"],
            env=["EARTHDATA_TOKEN"],
            paths=["~/.georeader/auth_emit.json"],
        ),
    ],
    # --- Local-only data files ----------------------------------------------
    "reading_overlapping_sentinel2_aviris.ipynb": [
        Requirement(files=["ang20190928t185111-4_r6871_c424_rgb.tif"]),
    ],
    # Both the Sentinel-2 crop and the Proba-V product must be supplied locally.
    # The VITO data pool that used to serve the Proba-V file is no longer
    # reachable, so it can no longer be downloaded at run time. Two groups -> the
    # notebook runs only if *both* files are present in examples/.
    "read_overlapping_probav_and_sentinel2.ipynb": [
        Requirement(files=["S2L1C.tif"]),
        Requirement(files=["PROBAV_S1_TOA_X07Y05_20190209_100M_V101.HDF5"]),
    ],
    # --- Carbon Mapper API token --------------------------------------------
    "api_explore.ipynb": [
        Requirement(
            env=["CARBONMAPPER_TOKEN"],
            paths=["~/.georeader/auth_carbonmapper.json"],
        ),
    ],
    "products_explore.ipynb": [
        Requirement(
            env=["CARBONMAPPER_TOKEN"],
            paths=["~/.georeader/auth_carbonmapper.json"],
        ),
    ],
    # --- Google Earth Engine: a service-account key (a path to the JSON key
    # file or the raw JSON string). The notebooks initialize EE via
    # georeader.readers.ee_image.initialize(); a Cloud project is not required
    # for service-account auth.
    "convert_to_radiance.ipynb": [
        Requirement(env=["EARTHENGINE_SERVICE_ACCOUNT_KEY"]),
    ],
    "run_in_gee_image.ipynb": [
        Requirement(env=["EARTHENGINE_SERVICE_ACCOUNT_KEY"]),
    ],
    "s2_mosaic_from_gee.ipynb": [
        Requirement(env=["EARTHENGINE_SERVICE_ACCOUNT_KEY"]),
    ],
}


# Notebooks that are *always* skipped, regardless of available data or
# credentials, because they cannot run for reasons unrelated to missing local
# inputs (e.g. they depend on a decommissioned service). Keyed by basename.
ALWAYS_SKIP: dict[str, str] = {
    # Queries the Copernicus Open Access Hub (scihub.copernicus.eu), which was
    # decommissioned. Needs migrating to the Copernicus Data Space Ecosystem
    # before it can run again.
    "query_mosaic_s2_images.ipynb": "reads from the decommissioned scihub.copernicus.eu",
}


def pytest_collection_modifyitems(config, items):
    """Skip notebook tests whose required data files / credentials are missing."""
    for item in items:
        if item.fspath.ext != ".ipynb":
            continue

        name = Path(item.fspath).name

        always_skip_reason = ALWAYS_SKIP.get(name)
        if always_skip_reason:
            item.add_marker(pytest.mark.skip(reason=always_skip_reason))
            continue

        requirements = NOTEBOOK_REQUIREMENTS.get(name)
        if not requirements:
            continue

        for req in requirements:
            if not req.satisfied():
                item.add_marker(
                    pytest.mark.skip(reason=f"missing notebook input: needs {req.describe()}")
                )
                break
