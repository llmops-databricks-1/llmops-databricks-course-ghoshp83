"""Make / OEM-group / component normalisation helpers.

Two responsibilities:

1. **Make normalisation** — fold submitter spelling variants into a
   canonical make name, then look up the parent corporate group from
   :file:`src/nhtsa_curator/ref/oem_groups.yml`. Used by silver writers.

2. **Component grouping** — NHTSA component names are hierarchical
   strings like ``"ENGINE AND ENGINE COOLING:ENGINE:CYLINDER BLOCK"``.
   For dim_component we keep the leaf, and pull the top-level segment
   out as ``component_group`` for dashboard grouping.

The data file is loaded once and cached in module globals. Reload-on-
demand is fine for batch jobs; if we ever need hot-reload for a
long-running process we'll add an explicit invalidation hook.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


def _resolve_taxonomy_path(path: str | Path) -> Path:
    """Search up to 3 parent dirs for the taxonomy YAML, mirroring config.py."""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    cur = Path.cwd()
    for _ in range(3):
        cand = cur / path
        if cand.exists():
            return cand
        cur = cur.parent
    return Path(path)


@lru_cache(maxsize=4)
def _load_oem_data(path: str = "src/nhtsa_curator/ref/oem_groups.yml") -> dict:
    """Load and cache the OEM YAML; keyed on path so tests can override."""
    resolved = _resolve_taxonomy_path(path)
    with open(resolved) as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=4)
def _oem_lookup(
    path: str = "src/nhtsa_curator/ref/oem_groups.yml",
) -> dict[str, str]:
    """Flat ``{canonical_make_lower: oem_group}`` map for O(1) lookup.

    The silver complaints writer calls the OEM resolver ~2.5M times per
    run. The old implementation scanned every ``oem_groups`` entry × its
    ``makes`` list on each call — ~500 string compares per row, ~1.25B
    per run. Flattening once at load time cuts that to a single dict
    get per row.
    """
    data = _load_oem_data(path)
    lookup: dict[str, str] = {}
    for entry in data.get("oem_groups") or []:
        group = entry["oem_group"]
        for m in entry.get("makes", []):
            lookup[m.lower()] = group
    return lookup


def make_to_oem_group(
    make: str | None,
    taxonomy_path: str = "src/nhtsa_curator/ref/oem_groups.yml",
) -> tuple[str | None, str | None]:
    """Return ``(canonical_make, oem_group)`` for a raw make string.

    Either field may be ``None`` if the input is empty or the make is
    not in the taxonomy. Unknown makes are passed through unchanged for
    ``canonical_make`` (better than discarding useful raw data) but get
    ``None`` for ``oem_group`` so downstream queries can identify gaps.
    """
    if not make:
        return None, None
    data = _load_oem_data(taxonomy_path)
    aliases: dict = data.get("make_aliases") or {}

    upper = make.strip().upper()
    canonical = aliases.get(upper, make.strip())

    # Title-case fallback for make names not in aliases — NHTSA uses
    # all-caps everywhere; we want "Toyota" not "TOYOTA" downstream.
    if canonical == make.strip() and canonical.isupper():
        canonical = canonical.title()

    group = _oem_lookup(taxonomy_path).get(canonical.lower())
    return canonical, group


def component_group(component_name: str | None) -> str | None:
    """Extract the top-level grouping from a NHTSA component string.

    Examples:
        ``"ENGINE AND ENGINE COOLING:ENGINE:CYLINDER BLOCK"`` →
        ``"Engine And Engine Cooling"``.
        ``"BATTERY"`` → ``"Battery"``.
    """
    if not component_name:
        return None
    head = component_name.split(":", 1)[0].strip()
    return head.title() if head.isupper() else head


def component_leaf(component_name: str | None) -> str | None:
    """Return the most-specific segment of a NHTSA component path."""
    if not component_name:
        return None
    leaf = component_name.split(":")[-1].strip()
    return leaf.title() if leaf.isupper() else leaf
