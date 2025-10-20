from pathlib import Path

# Resolve project root automatically (2 levels up from this file)
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[3]  # adjust if you place paths.py deeper

# === Top-level directories ===
CONFIG_DIR   = PROJECT_ROOT / "configs"
DATA_DIR     = PROJECT_ROOT / "data"
SRC_DIR      = PROJECT_ROOT / "src"
TESTS_DIR    = PROJECT_ROOT / "tests"

# === Configs ===
MAIN_CONFIG      = CONFIG_DIR / "main.json"
SLURM_CONFIG_DIR = CONFIG_DIR / "slurm"
SLURM_BASE_DIR = CONFIG_DIR / "slurm" / "base"
SLURM_GENERATED_DIR = CONFIG_DIR / "slurm" / "generated"

# === Data subdirs ===
IL_DATASET_DIR   = DATA_DIR / "il_dataset"
PROCESSED_DATA   = DATA_DIR / "processed"
RUNS_DIR         = DATA_DIR / "runs"
MODEL_WEIGHTS    = DATA_DIR / "model_weights"
TRANSITION_TABLES = DATA_DIR / "transition_tables"
GT_GRAPHS_DIR    = DATA_DIR / "gt_graphs"

# === Source subdirs ===
APP_DIR          = SRC_DIR / "app"
IL_DIR           = SRC_DIR / "imitation"
RL_DIR           = SRC_DIR / "reinforcement"
OPTIM_DIR        = SRC_DIR / "optim"
SCRIPTS_DIR      = SRC_DIR / "scripts"

def ensure_dirs() -> None:
    """
    Ensure that frequently used directories exist.
    """
    for p in [
        RUNS_DIR,
        MODEL_WEIGHTS,
        IL_DATASET_DIR,
        PROCESSED_DATA,
    ]:
        p.mkdir(parents=True, exist_ok=True)

def describe() -> str:
    """
    Return a human-readable description of important paths.
    """
    return "\n".join([
        f"PROJECT_ROOT={PROJECT_ROOT}",
        f"CONFIG_DIR={CONFIG_DIR}",
        f"MAIN_CONFIG={MAIN_CONFIG}",
        f"DATA_DIR={DATA_DIR}",
        f"RUNS_DIR={RUNS_DIR}",
        f"MODEL_WEIGHTS={MODEL_WEIGHTS}",
        f"IL_DATASET_DIR={IL_DATASET_DIR}",
        f"PROCESSED_DATA={PROCESSED_DATA}",
        f"TRANSITION_TABLES={TRANSITION_TABLES}",
        f"GT_GRAPHS_DIR={GT_GRAPHS_DIR}",
        f"SRC_DIR={SRC_DIR}",
        f"APP_DIR={APP_DIR}",
        f"IL_DIR={IL_DIR}",
        f"RL_DIR={RL_DIR}",
        f"OPTIM_DIR={OPTIM_DIR}",
        f"SCRIPTS_DIR={SCRIPTS_DIR}",
        f"SLURM_CONFIG_DIR={SLURM_CONFIG_DIR}",
    ])
