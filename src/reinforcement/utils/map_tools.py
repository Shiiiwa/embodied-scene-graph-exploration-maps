"""Helper utilities shared by reinforcement learning runners."""

from __future__ import annotations

from typing import Dict, Iterable, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def ensure_map_in_state(agent, obs):
    """Ensure the observation state contains the exploration map expected by the agent."""
    if not getattr(agent, "use_map", False):
        return obs.state

    state = list(obs.state) if isinstance(obs.state, (list, tuple)) else [obs.state]
    if len(state) < 4:
        state += [None] * (4 - len(state))

    map_value = state[3]
    exploration_mode = getattr(agent, "exploration_mode", None)
    if exploration_mode == "raster_v2":
        policy_map = obs.info.get("policy_map", None)
        map_value = policy_map if policy_map is not None else obs.info.get("exploration_map", map_value)
    else:
        if not isinstance(map_value, dict):
            map_value = obs.info.get("exploration_map", None)

    if (
        map_value is not None
        and not isinstance(map_value, dict)
        and hasattr(map_value, "to_dict")
    ):
        try:
            map_value = map_value.to_dict()
        except Exception:
            map_value = None

    state[3] = map_value
    return state


def count_visited_cells(emap: dict) -> int:
    """Count how many cells in the exploration map are marked as visited."""
    if not isinstance(emap, dict):
        return 0

    if "memory" in emap:
        mem = np.array(emap["memory"], dtype=np.float32)
        return int((mem[0] >= 0.5).sum())

    entries = emap.get("visited") or emap.get("cells") or emap.get("map")
    if isinstance(entries, list) and entries:
        if isinstance(entries[0], dict):
            return len(entries)
        if isinstance(entries[0], list):
            count = 0
            for row in entries:
                for cell in row:
                    if isinstance(cell, dict) and (cell.get("visited", 0) or cell.get("vis", 0)):
                        count += 1
            return count
    return 0


def safe_mean(values: Iterable[Optional[float]]) -> float:
    """Return the mean of numeric values while ignoring ``None`` entries."""
    filtered = [float(x) for x in values if x is not None]
    return float(np.mean(filtered)) if filtered else 0.0


def pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute a numerically safe Pearson correlation coefficient."""
    if not xs or not ys:
        return 0.0

    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    x = np.asarray(xs[:n], dtype=np.float32)
    y = np.asarray(ys[:n], dtype=np.float32)

    x_center = x - x.mean()
    y_center = y - y.mean()

    denom = float(np.linalg.norm(x_center) * np.linalg.norm(y_center))
    if denom <= 0:
        return 0.0
    return float(np.dot(x_center, y_center) / denom)


def map_shape(emap: dict) -> Tuple[int, int]:
    """Infer the height/width of the exploration map container."""
    if not isinstance(emap, dict):
        return (0, 0)
    shape = emap.get("map_shape") or emap.get("shape") or emap.get("size")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    if "memory" in emap:
        mem = emap["memory"]
        return len(mem[0]), len(mem[0][0])
    return (0, 0)


def coverage_fraction(emap: dict) -> float:
    """Compute visited cell ratio for the exploration map."""
    height, width = map_shape(emap)
    area = max(1, height * width)
    return float(count_visited_cells(emap)) / float(area)


def pose_belief_error(emap: dict, true_index) -> Optional[float]:
    """Estimate the localisation error between the belief map and the ground-truth index."""
    if not isinstance(emap, dict):
        return None

    pose_belief = emap.get("pose_belief")
    if pose_belief is None:
        return None

    try:
        pb = np.asarray(pose_belief, dtype=np.float32)
    except Exception:
        return None

    if pb.ndim != 2 or pb.size == 0:
        return None

    if not isinstance(true_index, (list, tuple)) or len(true_index) < 2:
        return None

    gi, gj = true_index[0], true_index[1]
    if not isinstance(gi, (int, np.integer)) or not isinstance(gj, (int, np.integer)):
        return None

    pred_flat = int(pb.argmax())
    pred_i, pred_j = np.unravel_index(pred_flat, pb.shape)

    di = float(pred_i - int(gi))
    dj = float(pred_j - int(gj))
    return float(np.hypot(di, dj))


def compute_intrinsic_potential(intrinsic_cfg: Optional[MutableMapping], emap: dict) -> float:
    """Potential-based intrinsic reward using visited cells or unit coverage fraction."""
    if not intrinsic_cfg or not intrinsic_cfg.get("active", False):
        return 0.0
    mode = intrinsic_cfg.get("mode", "cells")
    if mode == "cells":
        return float(intrinsic_cfg.get("w_cell", 0.0)) * float(count_visited_cells(emap))

    coverage = coverage_fraction(emap)
    weight = float(intrinsic_cfg.get("w_cell", intrinsic_cfg.get("w_cov", 0.02)))
    return weight * coverage


def discretise_position(env, pos):
    """Convert an (x, z) world coordinate to a discrete grid cell identifier."""
    if pos is None:
        return None
    if isinstance(pos, dict):
        pos = (pos.get("x"), pos.get("z"))
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    grid = float(getattr(env, "grid_size", 0.25) or 0.25)
    if grid <= 0:
        grid = 0.25
    try:
        x_val = float(pos[0])
        z_val = float(pos[1])
    except (TypeError, ValueError):
        return None
    i = int(round(z_val / grid))
    j = int(round(x_val / grid))
    return (i, j)


def extract_cell(env, obs):
    """Extract a discrete cell identifier from an observation."""
    if obs is None or not hasattr(obs, "info"):
        return None
    map_index = obs.info.get("map_index")
    if isinstance(map_index, (list, tuple)) and len(map_index) >= 2:
        try:
            return (int(map_index[0]), int(map_index[1]))
        except (TypeError, ValueError):
            pass
    if hasattr(map_index, "__array__"):
        try:
            arr = map_index.__array__()
            if len(arr) >= 2:
                return (int(arr[0]), int(arr[1]))
        except Exception:
            pass
    agent_pos = obs.info.get("agent_pos")
    return discretise_position(env, agent_pos)


def get_exploration_map(obs):
    """Retrieve the exploration map dictionary from an observation if available."""
    if obs is None:
        return None
    info = getattr(obs, "info", {}) or {}

    policy_map = info.get("policy_map")
    if isinstance(policy_map, dict):
        return policy_map

    exploration_map = info.get("exploration_map")
    if isinstance(exploration_map, dict):
        return exploration_map

    map_dict = policy_map or exploration_map
    if isinstance(map_dict, dict):
        return map_dict
    state = getattr(obs, "state", None)
    if isinstance(state, (list, tuple)) and len(state) >= 4:
        candidate = state[3]
        if isinstance(candidate, dict):
            return candidate
    return None


def estimate_total_cells(env, scene_number: int, map_dict: Optional[dict], cache: MutableMapping[int, int]) -> int:
    """Estimate the number of reachable cells for the current scene."""
    cached = cache.get(scene_number)
    if isinstance(cached, (int, float)) and cached > 0:
        return int(cached)

    total = 0
    if isinstance(map_dict, dict):
        shape = map_dict.get("map_shape") or map_dict.get("shape") or map_dict.get("size")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            try:
                total = int(shape[0]) * int(shape[1])
            except (TypeError, ValueError):
                total = 0
        elif "memory" in map_dict:
            mem = map_dict["memory"]
            try:
                total = int(len(mem[0]) * len(mem[0][0]))
            except Exception:
                total = 0

    if total <= 0:
        occ = getattr(env, "occupancy_map", None)
        if isinstance(occ, np.ndarray) and occ.ndim >= 2:
            total = int(occ.shape[0] * occ.shape[1])

    if total <= 0 and hasattr(env, "mapping"):
        mapping = getattr(env, "mapping", None)
        if isinstance(mapping, dict) and mapping:
            grid = float(getattr(env, "grid_size", 0.25) or 0.25)
            if grid <= 0:
                grid = 0.25
            positions = set()
            for key, event in mapping.items():
                if event is None or not isinstance(key, tuple) or len(key) < 2:
                    continue
                try:
                    x_val, z_val = float(key[0]), float(key[1])
                except (TypeError, ValueError):
                    continue
                positions.add((int(round(z_val / grid)), int(round(x_val / grid))))
            total = len(positions)

    if total <= 0:
        total = 1

    cache[scene_number] = int(total)
    return int(total)


def pad_history(history: Sequence[float], horizon: int, default_value: float) -> Sequence[float]:
    if horizon <= 0:
        return []
    if not history:
        return [default_value] * horizon
    if len(history) >= horizon:
        return list(history[:horizon])
    pad_value = history[-1] if history else default_value
    return list(history) + [pad_value] * (horizon - len(history))


def normalized_coverage_auc(coverage_history: Sequence[float], horizon: int) -> float:
    padded = pad_history(coverage_history, horizon, 0.0)
    if not padded:
        return 0.0
    return float(sum(padded)) / float(horizon)


def revisit_rate_at_k(revisit_history: Sequence[float], horizon: int) -> float:
    padded = pad_history(revisit_history, horizon, 1.0)
    if not padded:
        return 0.0
    return float(sum(padded)) / float(horizon)


def js_divergence(p, q, eps: float = 1e-8) -> Optional[float]:
    """Compute the Jensen–Shannon divergence between two probability vectors."""
    if p is None or q is None:
        return None

    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)

    if p_arr.size == 0 or q_arr.size == 0 or p_arr.shape != q_arr.shape:
        return None

    p_arr = np.clip(p_arr, eps, 1.0)
    q_arr = np.clip(q_arr, eps, 1.0)

    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)

    kl_pm = np.sum(p_arr * (np.log(p_arr) - np.log(m)))
    kl_qm = np.sum(q_arr * (np.log(q_arr) - np.log(m)))
    js_value = 0.5 * (kl_pm + kl_qm)

    return float(js_value / np.log(2.0))


def shannon_entropy(p) -> Optional[float]:
    if p is None:
        return None
    p_arr = np.asarray(p, dtype=np.float64)
    if p_arr.size == 0:
        return None
    p_arr = np.clip(p_arr, 1e-12, 1.0)
    p_arr = p_arr / p_arr.sum()
    return float(-(p_arr * np.log(p_arr)).sum())


def frontier_follow(prev_map_dict: dict, prev_idx, next_idx) -> int:
    """Return 1 if the agent moves from a frontier cell into an unknown four-neighbour cell."""
    if not isinstance(prev_map_dict, dict) or prev_idx is None or next_idx is None:
        return 0
    dense = prev_map_dict.get("dense", {})
    frontier = np.asarray(dense.get("frontier", []), dtype=np.uint8)
    visited = np.asarray(dense.get("visited_mask", []), dtype=np.uint8)
    if frontier.ndim != 2 or visited.ndim != 2:
        return 0
    pi, pj = prev_idx
    ni, nj = next_idx
    height, width = frontier.shape
    if not (0 <= pi < height and 0 <= pj < width and 0 <= ni < height and 0 <= nj < width):
        return 0
    if (pi == ni and pj == nj):
        return 0
    if frontier[pi, pj] == 1 and visited[ni, nj] == 0 and (abs(pi - ni) + abs(pj - nj) == 1):
        return 1
    return 0


def local_uncertainty(prev_map_dict: dict, idx, radius: int = 2) -> Optional[float]:
    """Average lack of confidence in a square window centred on the agent."""
    if not isinstance(prev_map_dict, dict) or idx is None:
        return None
    dense = prev_map_dict.get("dense", {})
    conf = np.asarray(dense.get("confidence", []), dtype=np.float32)
    if conf.ndim != 2 or conf.size == 0:
        return None
    i, j = idx
    height, width = conf.shape
    if not (0 <= i < height and 0 <= j < width):
        return None
    i0, i1 = max(0, i - radius), min(height, i + radius + 1)
    j0, j1 = max(0, j - radius), min(width, j + radius + 1)
    window = conf[i0:i1, j0:j1]
    if window.size == 0:
        return None
    return float(np.mean(1.0 - window))


def time_to_thresholds(coverage_hist: Sequence[float], thresholds: Sequence[float] = (0.25, 0.5, 0.75)) -> Dict[float, Optional[int]]:
    """Return the first step index where coverage crosses each threshold."""
    out: Dict[float, Optional[int]] = {}
    if not coverage_hist:
        for threshold in thresholds:
            out[threshold] = None
        return out

    for threshold in thresholds:
        step = next((k for k, c in enumerate(coverage_hist, start=1) if c >= threshold), None)
        out[threshold] = step
    return out

