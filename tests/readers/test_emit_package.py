"""Tests for the layout of the georeader.readers.emit package.

``georeader/readers/emit.py`` was split into ``utils``, ``download``, ``mask`` and ``image``.
The package must keep the old module namespace, import without cycles in any order, and
keep forwarding ``emit.TOKEN`` / ``emit.AUTH_METHOD`` assignments to the download code.
"""

from __future__ import annotations
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from georeader.readers import emit
from georeader.readers.emit import download


_PACKAGE_DIR = Path(emit.__file__).parent

# Names the former emit.py module exposed that callers use.
FORMER_MODULE_NAMES = [
    "AUTH_METHOD", "DAAC_URL", "EMITImage", "EMITProductID", "EMIT_PRODUCT_RE", "HAS_XARRAY",
    "L1B_COMPANIONS", "L1B_VERSIONS_WITH_ORBIT_SCENE", "MASK_BUFFER_FLAGS", "MASK_INVALID_FLAGS",
    "MASK_INVALID_FLAGS_IF_PRESENT", "MASK_SPECTF_FLAGS", "TOKEN", "WAVELENGTHS_RGB",
    "_bounds_indexes_raw", "_companion_version", "_l1b_radiance_id", "_normalise_mask_label",
    "download_product", "get_auth", "get_ch4enhancement_link", "get_headers", "get_l2amask_link",
    "get_obs_link", "get_radiance_link", "mask_band_index", "mask_flag_indexes",
    "parse_product_name", "product_name_from_params", "split_product_name", "valid_mask",
]

# Submodule -> sibling submodules it may import. Must stay acyclic.
ALLOWED_DEPENDENCIES = {
    "utils": set(),
    "download": {"utils"},
    "mask": {"utils"},
    "image": {"utils", "download", "mask"},
}


@pytest.mark.parametrize("name", FORMER_MODULE_NAMES)
def test_former_module_names_available(name):
    assert hasattr(emit, name)


def test_all_is_importable():
    for name in emit.__all__:
        assert hasattr(emit, name), name


def _sibling_imports(module: str) -> set[str]:
    tree = ast.parse((_PACKAGE_DIR / f"{module}.py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "georeader.readers.emit":
                found |= {alias.name for alias in node.names}
            elif node.module.startswith("georeader.readers.emit."):
                found.add(node.module.rsplit(".", 1)[-1])
            assert node.level == 0, f"{module}.py uses a relative import"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("georeader.readers.emit."):
                    found.add(alias.name.rsplit(".", 1)[-1])
    return found


@pytest.mark.parametrize("module", sorted(ALLOWED_DEPENDENCIES))
def test_submodule_dependencies_are_acyclic(module):
    assert _sibling_imports(module) <= ALLOWED_DEPENDENCIES[module]


def test_package_has_no_unexpected_submodules():
    modules = {p.stem for p in _PACKAGE_DIR.glob("*.py")} - {"__init__"}
    assert modules == set(ALLOWED_DEPENDENCIES)


@pytest.mark.parametrize("module", sorted(ALLOWED_DEPENDENCIES))
def test_submodule_imports_first_in_fresh_interpreter(module):
    code = (f"import georeader.readers.emit.{module}; "
            "from georeader.readers.emit import EMITImage, get_l2amask_link, valid_mask")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=_PACKAGE_DIR.parents[2])


def test_emitimage_uses_the_shared_link_builders():
    assert emit.image.get_l2amask_link is emit.get_l2amask_link is download.get_l2amask_link


# ── Download settings forwarded to emit.download ──────────────────────────


def test_token_assignment_on_package_reaches_download(monkeypatch):
    monkeypatch.setattr(emit, "AUTH_METHOD", "token")
    monkeypatch.setattr(emit, "TOKEN", "abc")
    assert (download.AUTH_METHOD, download.TOKEN) == ("token", "abc")
    assert (emit.AUTH_METHOD, emit.TOKEN) == ("token", "abc")

    calls = []
    monkeypatch.setattr(download, "download_product_base",
                        lambda link, **kwargs: calls.append((link, kwargs)) or kwargs["filename"])
    emit.download_product("https://example/x.nc", "x.nc", display_progress_bar=False)
    assert calls[0][1]["headers"] == {"Authorization": "Bearer abc"}

    monkeypatch.undo()
    assert (download.AUTH_METHOD, download.TOKEN) == ("auth", None)


def test_from_import_of_token():
    from georeader.readers.emit import TOKEN  # noqa: F401


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        emit.not_a_name
