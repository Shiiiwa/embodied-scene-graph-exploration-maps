# Embodied Scene Graph Navigation

This repository provides a practical workflow for experimenting with **scene graph–based navigation** in embodied environments such as **AI2-THOR**.  The focus is on how different exploration maps and training regimes affect an agent’s navigation performance.  The guide below targets practitioners who want to reproduce experiments without diving into implementation details.

> **Data & outputs** – Every script writes its results under the project-level `data/` directory.  Sub-folders are created automatically (e.g., `data/gt_graphs`, `data/transition_tables`, `data/il_dataset`, `data/model_weights`, `data/runs`).

---

## 📑 Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Quickstart Workflow](#quickstart-workflow)
  1. [Create ground-truth scene graphs](#1-create-ground-truth-scene-graphs)
  2. [Generate transition tables](#2-generate-transition-tables)
  3. [Generate Slurm-ready configs](#3-generate-slurm-ready-configs)
  4. [Build imitation learning datasets](#4-build-imitation-learning-datasets)
  5. [Run imitation learning (optional)](#5-run-imitation-learning-optional)
  6. [Run reinforcement learning](#6-run-reinforcement-learning)
  7. [Aggregate experiment runs](#7-aggregate-experiment-runs)
- [Monitoring & utilities](#monitoring--utilities)
- [Project structure](#project-structure)
- [Citation](#citation)
- [License](#license)

---

## Overview
The workflow combines **imitation learning (IL)** for supervised warm-starting with **reinforcement learning (RL)** for reward-driven fine-tuning.  By following the steps below you will:

1. Precompute environment assets (scene graphs and transition tables).
2. Prepare experiment configurations.
3. Produce datasets, checkpoints, and TensorBoard logs for IL and RL.
4. Summarise results across seeds.

All commands assume you run them from the project root (`/workspace/scene_graph_generation`).

---

## Setup
### Requirements
- Python 3.10+
- PyTorch (GPU)
- AI2-THOR and additional packages listed in `requirements.txt`

### Installation
```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If you intend to render scenes locally, ensure you have the system dependencies required by AI2-THOR (X11, GLX, etc.).

---

## Quickstart Workflow
Follow the steps in order.  Each stage expects the outputs of the previous one to already exist under `data/` or `configs/`.

### 1. Create ground-truth scene graphs
Precompute semantic graphs for every reachable viewpoint in the target scenes.

```bash
PYTHONPATH=. python3 src/scripts/generate_gt_graphs.py \
  --scenes 1 2 3          # optional subset (accepts numbers or FloorPlan names)
```

- Output directory: `data/gt_graphs/`
- One JSON file per scene (e.g., `FloorPlan1.json`).
- Omit `--scenes` to process all 30 default environments.

### 2. Generate transition tables
Transition tables map `(x, z, rotation)` tuples to AI2-THOR events.  They allow offline replay without repeatedly querying the simulator.

```bash
PYTHONPATH=. python3 src/scripts/generate_transition_tables.py \
  --scenes 1 2 3          # optional subset (accepts numbers or FloorPlan names)
```

- Output directory: `data/transition_tables/`
- One pickle file per scene containing the table and grid size information.
- Omit `--scenes` to reuse the default curated list of environments.
- Ensure the same scene coverage as in step 1 so IL/RL have consistent assets.

### 3. Generate Slurm-ready configs
Create per-seed RL configuration bundles from the templates in `configs/slurm/base/`.

```bash
PYTHONPATH=. python3 src/scripts/generate_slurm_configs.py
```

- Output directory: `configs/slurm/generated/<agent_name>/conf_<idx>/config_<seed>.json`
- Every file contains `agent`, `env`, `navigation`, and `exploration` blocks ready for RL training.
- Set `"exploration": {"intrinsic": {"active": true}}` to enable potential-based reward shaping that augments the
  environment reward with exploration map coverage potentials during RL fine-tuning.
- Tweak the defaults (`num_seeds`, `episodes_override`, `group_size`) by editing `create_sl_configs` in the script, or by importing the function in a short Python snippet and overriding the arguments.

### 4. Build imitation learning datasets
Use the precomputed assets to assemble IL trajectories.

```bash
PYTHONPATH=. python3 src/imitation/scripts/generate_il_dataset.py \
  --scenes 1 2 3          # optional subset (defaults to all 30 scenes) \
  --visualize-path        # optional flag for debugging
```

- Output directory: `data/il_dataset/<experiment_tag>/<FloorPlanX>/`
- Each pickle file encodes demonstrations for a particular start position and rotation.
- Requires `data/transition_tables/` and `data/gt_graphs/` from steps 1–2 plus a valid configuration (defaults to `configs/config.json`, override via `--config-path`).

### 5. Run imitation learning (optional)
Train supervised navigation policies.  Skip this step only if you plan to initialise RL from scratch.

```bash
PYTHONPATH=. python3 src/imitation/scripts/train_il.py \
  --config_path configs/slurm/generated/<agent_name>/conf_0/config_<seed>.json \
  --dataset_tag <dataset-subfolder> \
  --epochs 70 --batch_size 8
```

- Checkpoints: `data/model_weights/<experiment_tag>/`
  - Within each run you will find `best_*` directories containing encoder/policy weights (and optional Neural SLAM modules).
- Progress is reported in the console; copy the printed summaries into your experiment notes if you need persistent logs.

### 6. Run reinforcement learning
Fine-tune agents using the generated Slurm configs.  Provide IL checkpoints (from step 5) unless you intentionally skip warm-starting.

```bash
PYTHONPATH=. python3 src/reinforcement/train_rl.py \
  --conf_path configs/slurm/generated/<agent_name>/conf_0/config_<seed>.json \
  --weights_tag <map_version-from-IL> \
  --skip_il_warm_start  # optional flag to start from scratch
```

- TensorBoard logs & checkpoints: `data/runs/<AgentName>_<map_tag>_<timestamp>_<rho>/`
- RL model snapshots (if enabled by the config) are stored alongside the logs in the same folder.
- The script automatically expands the config to multiple seeds when run through job arrays; invoke it once per config file.

### 7. Aggregate experiment runs
Summarise TensorBoard metrics across seeds and produce publication-ready plots.

```bash
PYTHONPATH=. python3 src/scripts/aggregate_runs.py
```

- Reads from: `data/runs/`
- Produces plots interactively (default) using the aggregated scalar statistics.
- Modify the script to save figures (e.g., `plt.savefig(...)`) if you need persistent artefacts.

---

## Monitoring & utilities
- **TensorBoard** – launch a live dashboard covering all runs:
  ```bash
  ./src/scripts/start_tensorboard.sh  # logs under data/runs
  ```
- **Environment helpers** – `src/scripts/get_all_object_types.py` and `get_number_of_viewpoints_per_scene.py` provide quick statistics for debugging asset coverage.

---

## Project structure
```
configs/                # Base and generated configuration files
src/                    # Python package with scripts, components, IL, and RL modules
data/                   # Created on demand; holds datasets, weights, logs, and results
tests/                  # Automated checks
```
---

## License
This project is licensed under the MIT License.  See the `LICENSE` file for details.
