import os
import json
import math
from pathlib import Path

from src.components.utils.paths import SLURM_BASE_DIR, SLURM_GENERATED_DIR
from src.components.utils.utility_functions import generate_seeds


def create_sl_configs(
    config_base_dir: Path = SLURM_BASE_DIR,
    out_root: Path = SLURM_GENERATED_DIR,
    num_seeds: int = 20,
    episodes_override: int = 1000,
    group_size: int = 1,
):
    """
    For each agent configs, generate N seeds, grouped into conf_0, conf_1, ...
    and write combined configs files into slurm/generated/<AgentName>/conf_<n>/config_<seed>.json.
    """

    out_root.mkdir(parents=True, exist_ok=True)

    seeds = generate_seeds(num_seeds)

    agent_dirs = [
        d for d in sorted(os.listdir(config_base_dir))
        if (config_base_dir / d).is_dir()
    ]
    if not agent_dirs:
        print(f"[WARN] No agent folders found at: {config_base_dir}")

    for agent_name in sorted(agent_dirs):
        agent_path = config_base_dir / agent_name

        try:
            with open(agent_path / "agent.json") as f:
                agent_template = json.load(f)
            with open(agent_path / "env.json") as f:
                env_template = json.load(f)
            with open(agent_path / "navigation.json") as f:
                nav_template = json.load(f)
            try:
                with open(agent_path / "exploration.json") as f:
                    _exp_raw = json.load(f)
                    exploration_template = _exp_raw.get("exploration", _exp_raw)
            except FileNotFoundError:
                exploration_template = {
                    "active": False,
                    "map_dim": 64,
                    "map_version": "metric_semantic_v1",
                }
        except Exception as e:
            print(f"[ERROR] Failed to load configs for {agent_name}: {e}")
            continue

        num_groups = math.ceil(num_seeds / group_size)
        for group_idx in range(num_groups):
            group_seeds = seeds[group_idx * group_size : (group_idx + 1) * group_size]
            for seed in group_seeds:
                agent_cfg = json.loads(json.dumps(agent_template))
                env_cfg   = json.loads(json.dumps(env_template))
                nav_cfg   = json.loads(json.dumps(nav_template))
                expl_cfg  = json.loads(json.dumps(exploration_template))

                agent_cfg["seed"]     = seed
                agent_cfg["episodes"] = episodes_override
                env_cfg["seed"]       = seed
                nav_cfg["seed"]       = seed

                full_cfg = {
                    "seed": seed,
                    "agent_config": agent_cfg,
                    "env_config": env_cfg,
                    "navigation_config": nav_cfg,
                    "exploration_config": expl_cfg,
                }

                out_dir = out_root / agent_name / f"conf_{group_idx}"
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"config_{seed}.json").write_text(json.dumps(full_cfg, indent=2))

        print(f"[INFO] Generated {num_seeds} configs for '{agent_name}' in {num_groups} groups.")


if __name__ == "__main__":
    create_sl_configs(num_seeds=10, episodes_override=2000)