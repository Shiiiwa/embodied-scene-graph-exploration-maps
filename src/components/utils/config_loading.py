"""Helpers for loading and normalising experiment configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.components.utils.utility_functions import read_config


def normalize_config_schema(conf: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise old and new config schemas into a shared structure.

    The returned dictionary always contains the keys ``seed``, ``agent``,
    ``navigation``, ``env`` and ``exploration``.  Older config files that use
    ``*_config`` keys are mapped to the new layout.  Missing exploration blocks
    fall back to a minimal disabled configuration so downstream code does not
    have to guard against ``None`` values.
    """

    # Work on a shallow copy so we never mutate the caller's dictionary.
    data = dict(conf)

    seed = (
        data.get("seed")
        or data.get("agent", {}).get("seed")
        or data.get("navigation", {}).get("seed")
        or data.get("env", {}).get("seed")
        or data.get("agent_config", {}).get("seed")
        or data.get("navigation_config", {}).get("seed")
        or data.get("env_config", {}).get("seed")
    )
    norm: dict[str, Any] = {"seed": seed if seed is not None else 0}

    if any(key in data for key in ("agent", "navigation", "env")):
        norm["agent"] = dict(data.get("agent", {}))
        norm["navigation"] = dict(data.get("navigation", {}))
        norm["env"] = dict(data.get("env", {}))
        exploration_block = data.get("exploration", data.get("exploration_config"))
    else:
        norm["agent"] = dict(data.get("agent_config", {}))
        norm["navigation"] = dict(data.get("navigation_config", {}))
        norm["env"] = dict(data.get("env_config", {}))
        exploration_block = data.get("exploration_config")

    if exploration_block is None:
        exploration_block = {
            "active": False,
            "map_dim": 64,
            "map_version": "metric_semantic_v1",
        }
    else:
        exploration_block = dict(exploration_block)

    norm["exploration"] = exploration_block

    norm["agent"].setdefault("seed", norm["seed"])
    norm["navigation"].setdefault("seed", norm["seed"])
    if isinstance(norm["env"], dict):
        norm["env"].setdefault("seed", norm["seed"])

    return norm


def load_normalized_config(config_path: Path) -> dict[str, Any]:
    """Read ``config_path`` and return it in normalised form."""

    config_path = Path(config_path)
    print(f"Reading config from {config_path}")
    raw = read_config(config_path)
    return normalize_config_schema(raw)


def derive_experiment_tag(
    config_path: Path | None,
    config: Mapping[str, Any],
    fallback: str | None = None,
) -> str:
    """Derive a human-friendly experiment tag.

    Preference order:
      1. If ``config_path`` lives inside ``configs/slurm/<*>/<tag>/...``, return
         that ``<tag>`` component (e.g. ``a2c_lstm_none``).
      2. If exploration is disabled, fall back to ``"none"``.
      3. Otherwise use ``exploration.map_version`` when available.
      4. Finally fall back to the provided ``fallback`` or ``"default"``.
    """

    if config_path is not None:
        try:
            parts = Path(config_path).resolve().parts
        except FileNotFoundError:
            parts = Path(config_path).parts

        if "slurm" in parts:
            idx = parts.index("slurm")
            if idx + 2 < len(parts):
                return parts[idx + 2]

    exploration_cfg = config.get("exploration", {}) if config else {}
    if isinstance(exploration_cfg, Mapping):
        if not exploration_cfg.get("active", True):
            return fallback or "none"
        map_version = exploration_cfg.get("map_version")
        if isinstance(map_version, str) and map_version:
            return map_version

    return fallback or "default"
