"""Unit tests for make/OEM-group + component normalisation."""

from __future__ import annotations

from pathlib import Path

import pytest

from nhtsa_curator import taxonomy


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Tests may use different paths; clear the cache each time."""
    taxonomy._load_oem_data.cache_clear()


def _taxonomy_path() -> str:
    """Locate the project's oem_groups.yml regardless of CWD."""
    here = Path(__file__).resolve().parent.parent
    return str(here / "src" / "nhtsa_curator" / "ref" / "oem_groups.yml")


def test_make_to_oem_group_known_make() -> None:
    canonical, group = taxonomy.make_to_oem_group("CHEVROLET", _taxonomy_path())
    assert canonical == "Chevrolet"
    assert group == "General Motors"


def test_make_alias_resolution() -> None:
    canonical, group = taxonomy.make_to_oem_group("CHEVY", _taxonomy_path())
    assert canonical == "Chevrolet"
    assert group == "General Motors"


def test_make_unknown_passes_through() -> None:
    """Unknown makes are title-cased (NHTSA writes everything UPPER) and
    yield ``None`` for oem_group so downstream queries can spot gaps."""
    canonical, group = taxonomy.make_to_oem_group("ZYX-MOTORS", _taxonomy_path())
    assert canonical == "Zyx-Motors"
    assert group is None


def test_make_none_returns_double_none() -> None:
    assert taxonomy.make_to_oem_group(None, _taxonomy_path()) == (None, None)
    assert taxonomy.make_to_oem_group("", _taxonomy_path()) == (None, None)


def test_tesla_group_self() -> None:
    canonical, group = taxonomy.make_to_oem_group("TESLA", _taxonomy_path())
    assert canonical == "Tesla"
    assert group == "Tesla"


def test_component_group_strips_hierarchy() -> None:
    assert taxonomy.component_group(
        "ENGINE AND ENGINE COOLING:ENGINE:CYLINDER BLOCK"
    ) == "Engine And Engine Cooling"
    assert taxonomy.component_group("BATTERY") == "Battery"
    assert taxonomy.component_group(None) is None
    assert taxonomy.component_group("") is None


def test_component_leaf_extracts_tail() -> None:
    assert taxonomy.component_leaf(
        "ENGINE AND ENGINE COOLING:ENGINE:CYLINDER BLOCK"
    ) == "Cylinder Block"
    assert taxonomy.component_leaf("BATTERY") == "Battery"
    assert taxonomy.component_leaf(None) is None
