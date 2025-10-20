import datetime
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Optional

import torch

from src.reinforcement.a2c_agent import A2CAgent
from src.reinforcement.rl_train_runner import RLTrainRunner
from src.components.enviroments.precomputed_thor_env import PrecomputedThorEnv
from src.components.utils.paths import CONFIG_DIR, describe, MODEL_WEIGHTS
from src.components.utils.utility_functions import set_seeds, read_config
from src.components.utils.config_loading import normalize_config_schema, derive_experiment_tag


def set_working_directory():
    """Ensure cwd is project root for consistent relative imports."""
    desired_directory = Path(__file__).resolve().parents[2]  # project-root
    current_directory = Path.cwd()
    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(str(desired_directory))
        print(f"[INFO] Changed working directory from '{current_directory}' to '{desired_directory}'")
    else:
        print(f"[INFO] Current working directory: {current_directory}")


def resolve_encoder_path_by_tag(weights_tag: str) -> Path:
    """
    Resolve the newest/best FeatureEncoder checkpoint from a top-level tag folder
    under MODEL_WEIGHTS. Example tags: 'none', 'metric_semantic_v1', 'neural_slam'.
    """
    base_dir = MODEL_WEIGHTS / weights_tag
    if not base_dir.exists():
        raise FileNotFoundError(f"[ERROR] Base weights directory not found: {base_dir}")

    best_dirs = sorted(
        (d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("best_")),
        key=lambda d: int(re.match(r"best_(\d+)_acc_", d.name).group(1) or -1),
        reverse=True,
    )
    if not best_dirs:
        raise FileNotFoundError(f"[ERROR] No 'best_*' folders found in {base_dir}")

    print(f"[INFO] Using encoder from: {best_dirs[0]}")
    return best_dirs[0]

def find_il_submodule_paths(best_dir: Path):
    """Locate imitation learning encoder and policy checkpoints inside ``best_dir``."""
    enc = next(iter(best_dir.glob("feature_encoder.pth")), None)
    if enc is None:
        encs = sorted(best_dir.glob("feature_encoder_*.pth"))
        enc = encs[0] if encs else None

    pol = next(iter(best_dir.glob("navigation_policy.pth")), None)
    if pol is None:
        pols = sorted(best_dir.glob("navigation_policy_*.pth"))
        pol = pols[0] if pols else None

    return enc, pol
def main(
    seed: int,
    config: dict[str, Any],
    experiment_tag: Optional[str] = None,
    save_model: bool = True,
    weights_tag_override: Optional[str] = None,
    skip_il_warm_start: bool = False,
):
    """Train an RL agent for a single seed using the provided, normalised configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(seed)

    agent_config = config["agent"]
    navigation_config = config["navigation"]
    exploration_config = config["exploration"]
    env_config = config["env"]
    navigation_config["seed"] = seed

    map_version = exploration_config["map_version"]
    exploration_active = bool(exploration_config.get("active", True))
    derived_tag = derive_experiment_tag(None, config)
    weights_tag = experiment_tag or derived_tag

    if weights_tag_override is not None:
        selected_weights_tag = weights_tag_override
        print(f"[INFO] Weights tag override provided: '{selected_weights_tag}'")
    else:
        selected_weights_tag = weights_tag
        print(f"[INFO] Weights tag derived from config: '{selected_weights_tag}'")

    if skip_il_warm_start:
        print("[INFO] Skipping IL warm start – starting with randomly initialised weights")
        encoder_path = None
        policy_path = None
        neural_slam_path = None
    else:
        best_dir = resolve_encoder_path_by_tag(selected_weights_tag)
        encoder_path, policy_path = find_il_submodule_paths(best_dir)

        print(f"[INFO] Encoder weights: {encoder_path}")
        print(f"[INFO] Policy weights: {policy_path}")

        if exploration_config.get("map_version") == "neural_slam":
            neural_slam_path = best_dir / "neural_slam_networks"
            if neural_slam_path.exists():
                print(f"[INFO] Neural SLAM warm-start directory: {neural_slam_path}")
            else:
                neural_slam_path = None
                print("[INFO] Expected Neural SLAM warm-start directory but none was found; continuing without it")
        else:
            neural_slam_path = None



    env = PrecomputedThorEnv(
        rho=env_config["rho"],
        max_actions=agent_config["num_steps"],
        map_version=map_version,
    )

    if agent_config["name"] == "a2c":
        agent = A2CAgent(env, navigation_config, agent_config, exploration_config)
    else:
        raise RuntimeError("Invalid agent name")

    agent.load_weights(
        encoder_path=str(encoder_path) if encoder_path else None,
        policy_path=str(policy_path) if policy_path else None,
        neural_slam_path=str(neural_slam_path) if neural_slam_path else None,
        device=device
    )
    # print("[INFO] Successfully loaded weights")

    runner = RLTrainRunner(env, agent, device)
    runner.run()
    env.close()
    print("[INFO] Training completed.")

    if save_model:
        save_folder = MODEL_WEIGHTS / weights_tag
        save_folder.mkdir(parents=True, exist_ok=True)
        run_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_ckpt_path = save_folder / f"{run_start}_{agent_config['name']}_agent.pth"
        agent.save_model(str(final_ckpt_path))
        print(f"[INFO] Saved final agent weights to {final_ckpt_path}")


if __name__ == "__main__":
    set_working_directory()

    parser = ArgumentParser()
    parser.add_argument("--conf_path", type=str, help="Path to config file or folder")
    parser.add_argument("--num_seeds", type=int, default=3, help="How many seeds to run")
    parser.add_argument("--save_model", action="store_true", help="Save model weights")
    parser.add_argument("--precomputed", action="store_true", help="Use precomputed environment")
    parser.add_argument(
        "--weights_tag",
        type=str,
        help="Optional override for the MODEL_WEIGHTS/<tag> folder used for warm starts.",
    )
    parser.add_argument(
        "--skip_il_warm_start",
        action="store_true",
        help="Initialise the agent without loading imitation learning or Neural SLAM checkpoints.",
    )
    parser.add_argument(
        "--skip_configs",
        type=int,
        default=0,
        help="Number of config files to skip from the start before running.",
    )
    args = parser.parse_args()

    if args.conf_path:
        base = Path(args.conf_path)
    else:
        # fallback: point to generated slurm configs
        base = CONFIG_DIR / "slurm" / "generated"

    if base.is_dir():
        conf_files = sorted(base.rglob("config_*.json"))
    else:
        conf_files = [base]

    if not conf_files:
        raise FileNotFoundError(f"No config files found under: {base}")


    if args.skip_configs:
        if args.skip_configs >= len(conf_files):
            raise ValueError(
                f"Requested to skip {args.skip_configs} configs but only {len(conf_files)} available."
            )
        print(f"[INFO] Skipping the first {args.skip_configs} config(s)")
        conf_files = conf_files[args.skip_configs :]

    for conf_file in conf_files:
        raw = read_config(conf_file)
        norm = normalize_config_schema(raw)
        experiment_tag = derive_experiment_tag(conf_file, norm)
        print("\n----------------------\n")
        print(f"[INFO] Running with seed: {norm['seed']}")
        print(f"[INFO] Exploration map active: {norm['exploration']['active']}")
        set_seeds(norm["seed"])
        main(
            norm["seed"],
            norm,
            experiment_tag=experiment_tag,
            save_model=args.save_model,
            weights_tag_override=args.weights_tag,
            skip_il_warm_start=args.skip_il_warm_start,
        )
        torch.cuda.empty_cache()
