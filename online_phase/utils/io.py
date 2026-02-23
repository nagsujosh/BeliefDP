"""I/O utilities: YAML/JSON loading, seed setting, path resolution."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=_json_default)


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve a path relative to the project root if not absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    if project_root is None:
        # Default: BeliefDP project root (two levels up from this file)
        project_root = Path(__file__).resolve().parent.parent.parent
    return Path(project_root) / path


def merge_configs(*configs: dict) -> dict:
    """Merge multiple config dicts (later overrides earlier)."""
    merged = {}
    for cfg in configs:
        if cfg is not None:
            merged.update(cfg)
    return merged
