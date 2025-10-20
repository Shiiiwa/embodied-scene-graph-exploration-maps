import argparse
import csv
import os
import json
from collections import defaultdict
from pathlib import Path
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator

from src.components.utils.paths import RUNS_DIR, DATA_DIR

# ---------------- Styling (Paper ready) ----------------
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.35,
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "legend.frameon": False,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

})

# ---------------- Utils ----------------
def moving_average(x, w, align="center", pad_mode="reflect"):
    x = np.asarray(x, dtype=float)
    if w is None or w <= 1 or len(x) == 0:
        return x.copy()

    kernel = np.ones(w, dtype=float) / w

    if align == "center":
        pad_left = (w - 1) // 2
        pad_right = w - 1 - pad_left
        x_pad = np.pad(x, (pad_left, pad_right), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    elif align == "left":
        x_pad = np.pad(x, (w - 1, 0), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    elif align == "right":
        x_pad = np.pad(x, (0, w - 1), mode=pad_mode)
        y = np.convolve(x_pad, kernel, mode="valid")
    else:
        raise ValueError("align must be 'center', 'left', or 'right'.")

    return y[: len(x)]


def get_config_from_event(event_path: Path):
    """Load the configuration dict embedded in a TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(str(event_path))
    ea.Reload()
    tags = ea.Tags()
    if "full_config/text_summary" not in tags.get("tensors", []):
        return None
    try:
        raw = ea.Tensors("full_config/text_summary")[0].tensor_proto.string_val[0]
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None

def _resolve_scalar_tag(ea, requested_tag: str):
    """Resolve ``requested_tag`` to an actual TensorBoard scalar tag.

    ``metric_semantic_v1`` runs log certain metrics (e.g. ``cov_auc`` and
    ``revisit_rate``) with a numerical suffix denoting the evaluation
    threshold.  Historically the plotting script passed the prefix ending with
    ``@`` which used to match the full tag.  Newer runs therefore expose tags
    such as ``efficiency/cov_auc@0.10`` that are missed by an exact lookup.

    This helper performs a prefix lookup so the rest of the code can keep using
    the shorter identifier.  When multiple suffixes exist we pick the largest
    numeric value under the assumption that higher thresholds are the more
    conservative metric usually inspected in reports.
    """

    scalar_tags = ea.Tags().get("scalars", [])
    if requested_tag in scalar_tags:
        return requested_tag

    # Allow callers to specify a prefix (e.g. ``foo@``) and automatically map
    # it to the logged scalar tag (e.g. ``foo@0.10``).
    if requested_tag.endswith("@"):
        matches = [t for t in scalar_tags if t.startswith(requested_tag)]
        if matches:
            def parse_suffix(tag):
                suffix = tag[len(requested_tag) :]
                try:
                    return float(suffix)
                except ValueError:
                    return float("nan")

            numeric_matches = [(t, parse_suffix(t)) for t in matches]
            numeric_matches = [m for m in numeric_matches if np.isfinite(m[1])]
            if numeric_matches:
                # Prefer the largest threshold, otherwise fall back to the
                # lexicographically smallest option to keep behaviour stable.
                numeric_matches.sort(key=lambda item: item[1], reverse=True)
                return numeric_matches[0][0]
            matches.sort()
            return matches[0]

    return None


def extract_scalar_from_event(event_path: Path, tag: str):
    """Extract scalar step/value arrays for a single TensorBoard tag."""
    ea = event_accumulator.EventAccumulator(str(event_path))
    ea.Reload()

    resolved_tag = _resolve_scalar_tag(ea, tag)
    if resolved_tag is None:
        return np.array([], dtype=int), np.array([], dtype=float)

    events = ea.Scalars(resolved_tag)
    steps = np.array([e.step for e in events], dtype=int)
    values = np.array([e.value for e in events], dtype=float)
    return steps, values

def config_without_seed(cfg):
    """Drop seed values to compare configurations independent of randomness."""
    cfg = deepcopy(cfg)
    for k in ["agent_config", "navigation_config"]:
        if k in cfg and "seed" in cfg[k]:
            del cfg[k]["seed"]
    return cfg

def group_agent_dirs(base_dir: Path):
    """Group run directories by agent name prefix (before the timestamp suffix)."""
    agent_groups = defaultdict(list)
    for d in os.listdir(base_dir):
        path = base_dir / d
        if not path.is_dir():
            continue
        # Everything before the first ``_20..`` segment corresponds to the agent label
        parts = d.split("_20")[0]
        agent_groups[parts].append(path)
    return agent_groups

def find_valid_event_files(event_files):
    """Keep only event files that share the same configuration (ignoring seeds)."""
    configs = []
    for f in event_files:
        cfg = get_config_from_event(f)
        if cfg is not None:
            configs.append((f, config_without_seed(cfg)))
    # Older training runs did not log the ``full_config`` summary. In that case we
    # should simply use all discovered event files instead of discarding them.
    if not configs:
        return list(event_files)
    ref_cfg = configs[0][1]
    return [f for f, c in configs if c == ref_cfg]

def collect_all_valid_event_files(agent_dir: Path):
    """Recursively gather all valid event files under ``agent_dir``."""
    event_files = []
    for root, _, files in os.walk(agent_dir):
        for fname in files:
            if fname.startswith("events.out.tfevents."):
                event_files.append(Path(root) / fname)
    return find_valid_event_files(event_files) if event_files else []

def aggregate_seeds(event_files, tag, min_required=10):
    """Align runs to the shortest common length and compute mean/SEM curves."""
    all_values = []
    all_steps = []
    for ef in event_files:
        steps, values = extract_scalar_from_event(ef, tag)
        if steps.size > 0 and steps.size >= min_required:
            all_steps.append(steps)
            all_values.append(values)
    if not all_steps:
        return None, None, None, 0

    min_len = min(len(s) for s in all_steps)
    steps_common = all_steps[0][:min_len]
    arr = np.stack([v[:min_len] for v in all_values], axis=0).astype(float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1)
    assert all(np.array_equal(s[:min_len], all_steps[0][:min_len]) for s in all_steps)
    n_used = arr.shape[0]
    return steps_common, mean, std, n_used


    # sem = arr.std(axis=0, ddof=0) / np.sqrt(n_used)
    # return steps_common, mean, sem, n_used

# ---------------- Plot ----------------
def plot_metric(
    base_dir,
    tags,
    ylabel,
    title,
    max_episodes=4000,
    ylim=None,
    smooth=20,
    label_overrides=None,
    save_path=None,
    as_percent=False,
    exclude_predicate=None,
):
    """Plot mean/SEM curves across seeds for one of multiple scalar tags.

    ``tags`` may be provided either as a single string or as an iterable of
    strings.  The first tag that yields valid data for a given agent group is
    used, allowing us to maintain compatibility with historical runs that used
    slightly different naming conventions.
    """
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)

    any_plotted = False
    y_upper_candidates = []

    label_overrides = label_overrides or {}
    if isinstance(tags, str):
        tags = [tags]

    agent_groups = group_agent_dirs(base_dir)
    for agent_name, run_dirs in agent_groups.items():
        base_label = agent_name.replace("A2C_Agent_", "")
        display_label = label_overrides.get(base_label, base_label)
        if exclude_predicate and exclude_predicate(agent_name, base_label, display_label):
            continue

        all_event_files = []
        for rd in run_dirs:
            all_event_files.extend(collect_all_valid_event_files(rd))
        if not all_event_files:
            continue

        steps = mean = std = None
        n_used = 0
        for tag in tags:
            steps, mean, std, n_used = aggregate_seeds(all_event_files, tag)
            if steps is not None and n_used > 0:
                break
        if steps is None or n_used == 0:
            continue

        #  = moving_average(mean, smooth)
        # lower = moving_average(mean - std, smooth)
        # upper = moving_average(mean + std, smooth)

        se = std / max(1, np.sqrt(n_used))

        # Mittelwert und SE separat glätten (Logik aus dem Vergleichsskript)
        mean_s = moving_average(mean, smooth) if (smooth and smooth > 1) else mean
        se_s = moving_average(se, smooth) if (smooth and smooth > 1) else se

        # Band = mean ± SE
        lower = mean_s - se_s
        upper = mean_s + se_s

        # Truncate sequences to the shared valid length for numerical stability
        L = min(len(steps), len(mean_s), len(lower), len(upper))
        steps = steps[:L]
        mean_s = mean_s[:L]
        lower = lower[:L]
        upper = upper[:L]

        # Keep track of the upper envelope to derive axis limits later
        y_upper_candidates.append(upper.max())

        # ax.plot(steps, mean_s, linewidth=2, label=f"{display_label} (n={n_used})")
        if "metric_semantic_v3" in base_label.lower():
            color = "#9467bd"  # z. B. Lila aus dem tab10-Set
        else:
            color = None  # Matplotlib nutzt automatisch den Standard-Zyklus

        ax.plot(steps, mean_s, linewidth=2, label=display_label, color=color)
        ax.fill_between(steps, lower, upper, alpha=0.20, color=color)

        any_plotted = True

    if not any_plotted:
        print("No curves plotted.")
        return

    ax.set_xlabel("Episoden")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if as_percent:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))

    # Apply gentle padding on the axes
    ax.set_xlim(0, max_episodes)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymax = max(y_upper_candidates) if y_upper_candidates else None
        if ymax is not None and np.isfinite(ymax):
            ax.set_ylim(bottom=None, top=ymax * 1.05)  # 5% Headroom
    ax.margins(y=0.02)

    # Place legend below the plot to avoid overlapping lines
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()

def parse_label_overrides(raw_pairs):
    overrides = {}
    for item in raw_pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid label override '{item}'. Expected format 'original=new'."
            )
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise argparse.ArgumentTypeError(
                f"Invalid label override '{item}'. Keys and values must be non-empty."
            )
        overrides[key] = value
    return overrides

def _ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path

def export_final_values(base_dir: Path, tag, last_n: int = 200, output_file: str | None = None):
    """
    Aggregiert pro Agent den Durchschnitt der letzten `last_n` Episoden der
    (über Seeds gemittelten) Kurve und gibt eine Liste von Dicts zurück.
    Optional wird eine CSV geschrieben.

    Rückgabe: List[Dict[str, Any]] mit Schlüsseln:
      - "Agent": Anzeigename des Agenten (aus Ordnernamen abgeleitet)
      - f"Mean_{last_n}": float
      - f"Std_{last_n}":  float (Std der letzten last_n Mittelwerte, nicht über Seeds)
      - "Num_Seeds": int  (Anzahl Seeds, die in die Mittelkurve eingingen)

    ``tag`` kann entweder ein einzelner TensorBoard-Tag oder eine Sequenz von
    Tags sein. Im letzteren Fall wird – analog zur Plotfunktion – der erste
    verfügbare Tag pro Agent verwendet.
    """
    if isinstance(tag, (list, tuple, set)):
        tags = list(tag)
    else:
        tags = [tag]

    rows = []
    agent_groups = group_agent_dirs(base_dir)

    for agent_name, run_dirs in sorted(agent_groups.items()):
        # Alle Eventfiles unter allen Run-Dirs einsammeln (wie im Plot)
        all_event_files = []
        for rd in run_dirs:
            all_event_files.extend(collect_all_valid_event_files(rd))
        if not all_event_files:
            print(f"[WARN] Keine gültigen Eventfiles für '{agent_name}', übersprungen.")
            continue

        selected_tag = None
        steps = mean = std = n_used = None
        for tag_candidate in tags:
            steps, mean, std, n_used = aggregate_seeds(all_event_files, tag_candidate)
            if steps is not None and mean is not None and len(mean) >= max(1, last_n):
                selected_tag = tag_candidate
                break

        if selected_tag is None:
            length = 0 if mean is None else len(mean)
            print(
                f"[WARN] Nicht genug Daten für '{agent_name}' (len={length}), übersprungen."
            )
            continue

        mean_last = float(np.mean(mean[-last_n:]))
        std_last  = float(np.std(mean[-last_n:]))

        display_label = agent_name.replace("A2C_Agent_", "")

        rows.append({
            "Agent": display_label,
            f"Mean_{last_n}": mean_last,
            f"Std_{last_n}": std_last,
            "Num_Seeds": int(n_used),
        })

    if output_file:
        _ensure_dir(os.path.dirname(output_file))
        fieldnames = ["Agent", f"Mean_{last_n}", f"Std_{last_n}", "Num_Seeds"]
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in sorted(rows, key=lambda r: r[f"Mean_{last_n}"], reverse=True):
                writer.writerow(row)
        print(f"[INFO] Ergebnisse gespeichert unter: {output_file}")


    return sorted(rows, key=lambda r: r[f"Mean_{last_n}"], reverse=True)


def export_all_tables(base_dir: Path, last_n: int = 200, out_dir: str = "tables"):
    """
    Convenience: schreibt drei CSVs (Scores, Steps, Steps_for_score_1)
    in `out_dir` und gibt ein Dict {filename: rows} zurück.
    """
    _ensure_dir(out_dir)
    tables = {
        "scores.csv": "last_episode/Mean_Score",
        "steps.csv": "last_episode/Mean_Steps",
        "steps_for_score1.csv": "Rollout/Steps_for_score_1",
    }
    out = {}
    for fname, tag in tables.items():
        output_path = os.path.join(out_dir, fname)
        rows = export_final_values(base_dir, tag=tag, last_n=last_n, output_file=output_path)
        out[fname] = rows
    return out


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Aggregate and plot training runs.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing run subdirectories (default: RUNS_DIR).",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export aggregated CSV tables alongside the plots.",
    )
    parser.add_argument(
        "--label-alias",
        metavar="ORIGINAL=NEW",
        nargs="*",
        default=None,
        help=(
            "Optional mappings to rename legend entries. Provide multiple pairs as "
            "separate arguments, e.g. --label-alias old_name=New Label"
        ),
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.label_alias:
        try:
            label_overrides = parse_label_overrides(args.label_alias)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    else:
        label_overrides = {}

    metrics = [
        {
            "name": "mean_score",
            "tags": ["Rollout/Mean_Score", "last_episode/Mean_Score"],
            "ylabel": "Score",
            "title": "Mittlerer Score pro Episode",
            "max_episodes": 2000,
            "ylim": None,
            "smooth": 100,
            "as_percent": False,
        },
        {
            "name": "steps_for_score_1",
            "tags": "Rollout/Steps_for_score_1",
            "ylabel": "Schritte",
            "title": "Schritte bis Score 1",
            "max_episodes": 2000,
            "ylim": None,
            "smooth": 100,
            "as_percent": False,
        },
        {
            "name": "mean_steps",
            "tags": ["Rollout/Mean_Steps", "last_episode/Mean_Steps"],
            "ylabel": "Schritte",
            "title": "Mittlere Schrittzahl pro Episode",
            "max_episodes": 1000,
            "ylim": None,
            "smooth": 100,
            "as_percent": False,
        },
        {
            "name": "efficiency_cov_auc",
            "tags": "efficiency/cov_auc@",
            "ylabel": "AUC",
            "title": "Effizienz: AUC der Abdeckung",
            "max_episodes": 1000,
            "ylim": None,
            "smooth": 100,
            "as_percent": False,
        },
        {
            "name": "Mean_Coverage_per_Step",
            "tags": "Rollout/Mean_Coverage_per_Step",
            "ylabel": "Abdeckung",
            "title": "Relative Abdeckung pro Schritt",
            "max_episodes": 1000,
            "ylim": None,
            "smooth": 100,
            "as_percent": True,
        },
        {
            "name": "map_influence_share",
            "tags": "Rollout/Mean_Map_Influence_Share",
            "ylabel": "Einfluss",
            "title": "Relativer Einfluss der Explorations Map",
            "max_episodes": 2000,
            "ylim": None,
            "smooth": 100,
            "as_percent": True,
            "exclude_predicate": lambda agent_name, base_label, display_label: (
                "neural" in base_label.lower() and "slam" in base_label.lower()
            ),
        },
        {
            "name": "scenegraph_influence_share",
            "tags": "Rollout/Mean_SceneGraph_Influence_Share",
            "ylabel": "Einfluss",
            "title": "Relativer Einfluss des Scene Graphs",
            "max_episodes": 2000,
            "ylim": None,
            "smooth": 100,
            "as_percent": True,
        },
        {
            "name": "rgb_influence_share",
            "tags": "Rollout/Mean_RGB_Influence_Share",
            "ylabel": "Einfluss",
            "title": "Relativer Einfluss des RGB-Bildes",
            "max_episodes": 2000,
            "ylim": None,
            "smooth": 100,
            "as_percent": True,
        }

    ]

    for metric in metrics:
        save_dir = DATA_DIR / "plots" / metric["name"]
        pdf_path = save_dir / f"{metric['name']}.pdf"
        plot_metric(
            args.base_dir,
            tags=metric["tags"],
            ylabel=metric["ylabel"],
            title=metric["title"],
            max_episodes=metric["max_episodes"],
            ylim=metric["ylim"],
            smooth=metric["smooth"],
            label_overrides=label_overrides,
            save_path=pdf_path,
            exclude_predicate=metric.get("exclude_predicate"),
        )

        if args.export_csv:
            csv_path = save_dir / f"{metric['name']}.csv"
            export_final_values(
                args.base_dir,
                tag=metric["tags"],
                last_n=200,
                output_file=str(csv_path),
            )


# ---------------- Main ----------------
if __name__ == "__main__":
    main()
