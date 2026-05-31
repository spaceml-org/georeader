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
* PRISMA / EnMAP (IMEO Azure):  ``SAS_TOKEN``, ``AZURE_STORAGE_ACCOUNT``, ``CONTAINER_NAME``
* EMIT (NASA Earthdata):        ``EARTHDATA_TOKEN`` or ``~/.georeader/auth_emit.json``
* Carbon Mapper:                ``CARBONMAPPER_TOKEN`` (or ``CARBONMAPPER_EMAIL`` + ``CARBONMAPPER_PASSWORD``) or ``~/.georeader/auth_carbonmapper.json``
* Google Earth Engine:          ``EARTHENGINE_TOKEN`` or ``~/.config/earthengine/credentials``

In GitHub Actions these can be wired as repository secrets and exported as the
matching environment variables before ``make test-notebooks`` runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Resolve examples/ relative to this file (docs/conftest.py -> repo root -> examples/)
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

_ENMAP_TILE = "ENMAP01-____L1B-DT0000074101_20240511T080843Z_001_V010402_20240514T093550Z"


@dataclass
class Requirement:
    """A single requirement group (satisfied if *any* member is available)."""

    files: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def satisfied(self) -> bool:
        if any((_EXAMPLES_DIR / f).exists() for f in self.files):
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
    # --- PRISMA / EnMAP: local file or IMEO Azure download ------------------
    "prisma_with_cloudsen12.ipynb": [
        Requirement(
            files=["PRISMA/PRS_L1_STD_OFFL_20241109073054_20241109073059_0001.he5"],
            env=["SAS_TOKEN"],
        ),
    ],
    "enmap_with_cloudsen12.ipynb": [
        Requirement(
            files=[f"EnMAP/{_ENMAP_TILE}/{_ENMAP_TILE}-METADATA.XML"],
            env=["SAS_TOKEN"],
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
            env=["SAS_TOKEN"],
        ),
        Requirement(
            files=["EMIT/EMIT_L1B_RAD_001_20230929T122534_2327208_039.nc"],
            env=["EARTHDATA_TOKEN"],
            paths=["~/.georeader/auth_emit.json"],
        ),
    ],
    # --- Local-only data files ----------------------------------------------
    "query_mosaic_s2_images.ipynb": [Requirement(files=["liria.geojson"])],
    "reading_overlapping_sentinel2_aviris.ipynb": [
        Requirement(files=["ang20190928t185111-4_r6871_c424_rgb.tif"]),
    ],
    # Proba-V is downloaded from the public VITO data pool at run time; only
    # the Sentinel-2 crop needs to be supplied locally.
    "read_overlapping_probav_and_sentinel2.ipynb": [Requirement(files=["S2L1C.tif"])],
    # --- Carbon Mapper API token --------------------------------------------
    "api_explore.ipynb": [
        Requirement(
            env=["CARBONMAPPER_TOKEN", "CARBONMAPPER_EMAIL"],
            paths=["~/.georeader/auth_carbonmapper.json"],
        ),
    ],
    "products_explore.ipynb": [
        Requirement(
            env=["CARBONMAPPER_TOKEN", "CARBONMAPPER_EMAIL"],
            paths=["~/.georeader/auth_carbonmapper.json"],
        ),
    ],
    # --- Google Earth Engine credentials ------------------------------------
    "convert_to_radiance.ipynb": [
        Requirement(env=["EARTHENGINE_TOKEN"], paths=["~/.config/earthengine/credentials"]),
    ],
    "run_in_gee_image.ipynb": [
        Requirement(env=["EARTHENGINE_TOKEN"], paths=["~/.config/earthengine/credentials"]),
    ],
    "s2_mosaic_from_gee.ipynb": [
        Requirement(env=["EARTHENGINE_TOKEN"], paths=["~/.config/earthengine/credentials"]),
    ],
}


def pytest_collection_modifyitems(config, items):
    """Skip notebook tests whose required data files / credentials are missing."""
    for item in items:
        if item.fspath.ext != ".ipynb":
            continue

        requirements = NOTEBOOK_REQUIREMENTS.get(Path(item.fspath).name)
        if not requirements:
            continue

        for req in requirements:
            if not req.satisfied():
                item.add_marker(
                    pytest.mark.skip(reason=f"missing notebook input: needs {req.describe()}")
                )
                break
