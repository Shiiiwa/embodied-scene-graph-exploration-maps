import json
import os
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from typing import List

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.storages import RDBStorage

from src.components.utils.paths import CONFIG_DIR, PROJECT_ROOT, OPTIM_DIR
from src.components.enviroments.precomputed_thor_env import PrecomputedThorEnv
from src.components.utils.utility_functions import set_seeds, read_config
from src.optim.runners.trial_runner import RLTrialRunner


def save_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Persist progress after each *completed* trial."""
    # Make sure output dirs exist
    out_dir = Path("optim")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1) Save current best params (robust even if best_trial doesn't change)
    best = study.best_trial
    output_model_name = "transformer" if args.use_transformer else "lstm"
    best_path = out_dir / f"best_params_{agent}_{output_model_name}.json"
    with open(best_path, "w") as f:
        json.dump(best.params, f, indent=2)

    # 2) Optional: dump full trials table for auditing/analysis
    #    Note: `trials_dataframe` is cheap enough at this cadence.
    df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs", "system_attrs"))
    df.to_csv(out_dir / f"trials_{agent}_{output_model_name}.csv", index=False)


def _generate_rho_grid(start: float, stop: float, step: float) -> List[float]:
    """Generate an inclusive float range."""

    if step <= 0:
        raise ValueError("rho step size must be positive")

    values: List[float] = []
    max_steps = int(round((stop - start) / step))
    for i in range(max_steps + 1):
        value = start + i * step
        if value > stop + 1e-9:  # numeric stability guard
            break
        values.append(round(value, 10))

    if values and values[-1] < stop:
        values.append(round(stop, 10))

    return values


def get_rho_search_space(agent_name: str, use_transformer: bool):
    if agent_name != "a2c":
        raise ValueError(f"Unknown agent/model combination: {agent_name}, transformer={use_transformer}")

    if args.rho_values:
        return list(dict.fromkeys(args.rho_values))

    if args.rho_step is not None and args.rho_step > 0:
        return _generate_rho_grid(args.rho_min, args.rho_max, args.rho_step)

    # Continuous search space – use optuna.suggest_float later on
    return None


def objective(trial, agent, use_transformer):
    set_seeds(42)

    rho_space = get_rho_search_space(agent, use_transformer)
    if rho_space is None:
        rho = trial.suggest_float(
            "rho",
            args.rho_min,
            args.rho_max,
            log=args.rho_log_uniform,
            step=args.rho_float_step,
        )
    else:
        rho = trial.suggest_categorical("rho", rho_space)

    full_config = read_config(CONFIG_DIR / "config.json")

    agent_config = full_config["agent"]
    exploration_config = full_config["exploration"]
    navigation_config = full_config["navigation"]
    env_config = {
        "rho": full_config["navigation"]["rho"],
        "map_version": full_config["exploration"]["map_version"],
        "active": full_config["exploration"]["active"]
    }

    agent_config["name"] = agent
    navigation_config["use_transformer"] = use_transformer
    env_config["rho"] = rho

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = PrecomputedThorEnv(render=False, rho=rho, max_actions=agent_config["num_steps"], map_version=exploration_config["map_version"])

    runner = RLTrialRunner(
        trial,
        env,
        navigation_config,
        agent_config,
        exploration_config,
        device,
        params={"rho": rho},
        num_agents=args.num_agents,
        max_episodes=args.max_episodes,
        n_jobs=args.n_jobs,
    )

    try:
        return runner.run()
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception as e:
                print(f"[WARNING] Failed to stop AI2-THOR controller: {e}")



def set_working_directory():
    desired_directory = Path(__file__).resolve().parents[3]
    current_directory = os.getcwd()

    if current_directory != desired_directory:
        os.chdir(desired_directory)
        sys.path.append(desired_directory)
        print(f"Current working director changed from '{current_directory}', to '{desired_directory}'")
        return

    print("Current working directory:", os.getcwd())


if __name__ == "__main__":
    # set_working_directory()

    parser = ArgumentParser()

    parser.add_argument(
        "--use_transformer", action="store_true", help="Use transformer model for the agent. If not set, LSTM will be used."
    )
    parser.add_argument("--max_episodes", type=int, default=600, help="Maximum number of episodes to run for each trial.")
    parser.add_argument("--num_agents", type=int, default=1, help="Number of agents to run in parallel during optimization.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel processes to run in parallel.")
    parser.add_argument(
        "--rho-values",
        type=float,
        nargs="+",
        help="Explicit rho values to evaluate. Overrides --rho-min/--rho-max/--rho-step.",
    )
    parser.add_argument(
        "--rho-min",
        type=float,
        default=0.008,
        help="Lower bound for rho search when using continuous/grid sampling.",
    )
    parser.add_argument(
        "--rho-max",
        type=float,
        default=0.025,
        help="Upper bound for rho search when using continuous/grid sampling.",
    )
    parser.add_argument(
        "--rho-step",
        type=float,
        default=0.0005,
        help=(
            "Step size for generating a discrete rho grid. Set to 0 to switch to a continuous "
            "search with --rho-float-step/--rho-log-uniform."
        ),
    )
    parser.add_argument(
        "--rho-float-step",
        type=float,
        default=None,
        help="Optional Optuna float step when sampling rho continuously (requires --rho-step 0).",
    )
    parser.add_argument(
        "--rho-log-uniform",
        action="store_true",
        help="Sample rho on a log-uniform scale when using the continuous search space.",
    )
    args = parser.parse_args()

    if args.rho_values and args.rho_step == 0:
        print("[INFO] --rho-values provided; ignoring --rho-step 0 toggle for continuous search.")

    if args.rho_min >= args.rho_max:
        parser.error("--rho-min must be smaller than --rho-max")

    if args.rho_step is not None and args.rho_step < 0:
        parser.error("--rho-step must be non-negative")

    if args.rho_step == 0 and args.rho_float_step is None and not args.rho_log_uniform:
        print("[INFO] Using continuous rho search without discretisation.")

    if args.rho_step == 0 and args.rho_float_step is not None and args.rho_log_uniform:
        parser.error("--rho-float-step cannot be combined with --rho-log-uniform")

    if args.rho_log_uniform and (args.rho_min <= 0 or args.rho_max <= 0):
        parser.error("Log-uniform rho search requires positive bounds")

    if args.rho_values and any(value <= 0 for value in args.rho_values) and args.rho_log_uniform:
        parser.error("Log-uniform sampling cannot be used with non-positive rho values")

    if args.rho_values:
        rho_preview = list(dict.fromkeys(args.rho_values))
        print(f"[INFO] Optimising rho over explicit grid: {rho_preview}")
    elif args.rho_step and args.rho_step > 0:
        rho_preview = _generate_rho_grid(args.rho_min, args.rho_max, args.rho_step)
        print(
            f"[INFO] Optimising rho over discrete grid of {len(rho_preview)} values from "
            f"{rho_preview[0]:.4f} to {rho_preview[-1]:.4f} (step={args.rho_step})."
        )
    else:
        mode = "log-uniform" if args.rho_log_uniform else "uniform"
        step_msg = f", step={args.rho_float_step}" if args.rho_float_step else ""
        print(
            f"[INFO] Optimising rho over {mode} continuous range "
            f"[{args.rho_min}, {args.rho_max}]{step_msg}."
        )

    agent = "a2c"
    config = read_config(CONFIG_DIR / "config.json")
    use_transformer = args.use_transformer

    print(f"Agent: {agent}")
    print(f"Using model: {'transformer' if use_transformer else 'lstm'}")
    if config['exploration']['active']:
        print(f"Using map: {config['exploration']['map_version']}")
    else:
        print(f"Not using any exploration map")

    db_path = OPTIM_DIR / f"optuna_a2c_{'transformer' if use_transformer else 'lstm'}.db"
    storage = RDBStorage(
        url=f"sqlite:///{db_path}",
        heartbeat_interval=60,
        grace_period=120,
    )

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=1, multivariate=True)  # first 3 trials are random for exploration
    pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=6, interval_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="param_opt_a2c",
        storage=storage,
        load_if_exists=True,  # critical for resume
    )

    # --- run optimization with callback to persist after each trial ---
    study.optimize(
        partial(objective, agent=agent, use_transformer=use_transformer),
        n_trials=60,
        n_jobs=args.n_jobs,
        callbacks=[save_progress_callback],  # persist after every finished trial
    )

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best rho: {study.best_trial.params['rho']}")

    output_model_name = "lstm" if not use_transformer else "transformer"
    with open(f"optim/best_params_{agent}_{output_model_name}.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=2)