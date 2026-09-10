"""Config loading with a single level of inheritance.

Every run is defined by a config file plus a seed, and no hyperparameters on the
command line (CLAUDE.md, Conventions). A config may name a parent with
``_base_``; keys merge recursively, child wins.
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any, Mapping

import yaml


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | pathlib.Path) -> dict[str, Any]:
    """Load a YAML config, resolving ``_base_`` relative to the config's own dir."""
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    base_name = cfg.pop("_base_", None)
    if base_name is None:
        cfg.setdefault("name", path.stem)
        return cfg

    base = load_config(path.parent / base_name)
    merged = _deep_merge(base, cfg)
    merged["name"] = cfg.get("name", path.stem)
    return merged
