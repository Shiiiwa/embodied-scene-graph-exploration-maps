import json
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.components.enviroments.precomputed_thor_env import PrecomputedThorEnv
from src.components.utils.observation import Observation
from src.components.utils.paths import RUNS_DIR
from src.reinforcement.utils import (
    compute_intrinsic_potential,
    count_visited_cells,
    coverage_fraction,
    discretise_position,
    ensure_map_in_state,
    estimate_total_cells,
    extract_cell,
    frontier_follow,
    get_exploration_map,
    js_divergence,
    local_uncertainty,
    map_shape,
    normalized_coverage_auc,
    pad_history,
    pearson_corr,
    pose_belief_error,
    revisit_rate_at_k,
    safe_mean,
    shannon_entropy,
    time_to_thresholds,
)


class RLTrainRunner:
    """Handle reinforcement learning training loops, logging, and bookkeeping."""

    def __init__(self, env, agent, device=None):
        """Set up logging, thresholds, and intrinsic reward configuration for training."""
        self.env = env
        self.agent = agent
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent.to(self.device)

        self.agent_config = agent.agent_config
        self.navigation_config = agent.navigation_config

        self.total_episodes = self.agent_config.get("episodes", 500)
        self.scene_numbers = agent.scene_numbers
        self.log_buffer_size = 40
        self.coverage_horizon = max(1, int(self.agent_config.get("num_steps", 1)))
        self._total_cell_cache = {}
        self._blank_map_state_cache = {}
        self._blank_map_buffer_cache = {}

        exploration_mode = getattr(self.agent, "exploration_mode", "")
        default_stride = 4 if exploration_mode == "neural" else 1
        self.map_influence_stride = max(
            1, int(self.agent_config.get("map_influence_stride", default_stride))
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name = agent.get_agent_info().get("Agent Name", "Agent").replace(" ", "_")
        print(f"use transformer: ", self.navigation_config.get("use_transformer", False))
        if self.navigation_config.get("use_transformer", False):
            agent_name += "_Transformer"
        else:
            agent_name += "_LSTM"

        rho_str = getattr(self.env, "rho", "na")
        exp_cfg = getattr(self.agent, "exploration_config", None)
        if isinstance(exp_cfg, dict) and exp_cfg.get("active", False):
            map_tag = str(exp_cfg.get("map_version", "unknown"))
        else:
            map_tag = "none"

        log_dir = (RUNS_DIR / f"{agent_name}_{map_tag}_{timestamp}_{rho_str}").as_posix()
        self.writer = SummaryWriter(log_dir)
        print(f"[INFO] TensorBoard logs will be saved to: {log_dir}")

        full_config = {"agent_config": self.agent_config, "navigation_config": self.navigation_config}
        self.writer.add_text("full_config", json.dumps(full_config, indent=2), 0)

        self.ep_info_buffer = deque(maxlen=self.log_buffer_size)

        exp_cfg = getattr(self.agent, "exploration_config", {}) or {}
        icfg = exp_cfg.get("intrinsic", {}) if isinstance(exp_cfg, dict) else {}

        default_active = (exploration_mode == "neural")

        self.intrinsic_cfg = {
            "active": bool(icfg.get("active", default_active)),
            "mode": icfg.get("mode", "cells"),
            "w_cell": float(icfg.get("w_cell", 0.02)),
            "w_cov": float(icfg.get("w_cov", 0.0)),
            "w_vw": float(icfg.get("w_vw", 0.0)),
            "gamma": float(icfg.get("gamma", 1.0)),
        }

        print("=====================")
        for k, v in self.intrinsic_cfg.items():
            print(f"[REWARD]: Intrinsic {k} = {v}")
        print("=====================")

        self.discount_gamma = float(self.agent_config.get("gamma", 0.99))


    def _recreate_env_with_same_cfg(self):
        """Recreate the environment using the current configuration to recover from errors."""
        mv = None
        if hasattr(self.agent, "exploration_config"):
            mv = self.agent.exploration_config.get("map_version", None)

        self.env = PrecomputedThorEnv(
            render=getattr(self.env, "render", False),
            rho=getattr(self.env, "rho", 0.0),
            max_actions=self.agent_config["num_steps"],
            grid_size=getattr(self.env, "grid_size", 0.25),
            map_version=mv
        )

    def robust_reset(self, scene_number, max_retries=3):
        """Reset the environment with retries to handle transient simulator failures."""
        for attempt in range(max_retries):
            try:
                obs = self.env.reset(scene_number=scene_number, random_start=True)
                st = self._ensure_map_in_state(obs)
                if st is not obs.state:
                    obs = Observation(state=tuple(st), reward=obs.reward, terminated=obs.terminated,
                                      truncated=obs.truncated, info=obs.info)
                return obs
            except TimeoutError:
                print(f"[WARNING] TimeoutError at reset, trying restart... ({attempt + 1}/{max_retries})")
                try:
                    self.env.close()
                except Exception:
                    pass

                self._recreate_env_with_same_cfg()
        print("[ERROR] Multiple Timeouts at reset - skipping episode.")
        return None

    def robust_step(self, action, max_retries=3):
        """Step the environment with retries, recreating it if AI2-THOR times out."""
        for attempt in range(max_retries):
            try:
                obs = self.env.step(action)
                st = self._ensure_map_in_state(obs)
                if st is not obs.state:
                    obs = Observation(state=tuple(st), reward=obs.reward, terminated=obs.terminated,
                                      truncated=obs.truncated, info=obs.info)
                return obs
            except TimeoutError:
                print(f"[WARNING] TimeoutError at step, trying restart... ({attempt + 1}/{max_retries})")
                try:
                    self.env.close()
                except Exception:
                    pass

                self._recreate_env_with_same_cfg()
                return None
        print("[ERROR] Multiple Timeouts at step - skipping episode.")
        return None

    def _ensure_map_in_state(self, obs):
        """Wrapper around :func:`ensure_map_in_state` for backward compatibility."""
        return ensure_map_in_state(self.agent, obs)

    def _count_visited_cells(self, emap: dict) -> int:
        return count_visited_cells(emap)

    def _safe_mean(self, xs):
        return safe_mean(xs)

    def _pearson_corr(self, xs, ys):
        return pearson_corr(xs, ys)

    def _map_shape(self, emap):
        return map_shape(emap)

    def _coverage_frac(self, emap: dict) -> float:
        return coverage_fraction(emap)

    def _pose_belief_error(self, emap: dict, true_index) -> Optional[float]:
        return pose_belief_error(emap, true_index)

    def _phi(self, emap: dict, mi) -> float:
        return compute_intrinsic_potential(self.intrinsic_cfg, emap)

    def _discretise_position(self, pos):
        return discretise_position(self.env, pos)

    def _extract_cell(self, obs):
        return extract_cell(self.env, obs)

    def _get_exploration_map(self, obs):
        return get_exploration_map(obs)

    def _estimate_total_cells(self, scene_number, map_dict=None):
        return estimate_total_cells(self.env, scene_number, map_dict, self._total_cell_cache)

    def _pad_history(self, history, horizon, default_value):
        return pad_history(history, horizon, default_value)

    def _normalized_coverage_auc(self, coverage_history):
        return normalized_coverage_auc(coverage_history, self.coverage_horizon)

    def _revisit_rate_at_k(self, revisit_history):
        return revisit_rate_at_k(revisit_history, self.coverage_horizon)

    def _js_divergence(self, p, q, eps=1e-8):
        return js_divergence(p, q, eps)

    def _peek_policy_probs(self, obs):
        try:
            with torch.no_grad():
                _, _, _, probs = self.agent.peek_policy(obs)  # expects numpy or list
            p = np.asarray(probs, dtype=np.float64)
            p = np.clip(p, 1e-8, 1.0)
            p = p / p.sum()
            return p
        except Exception:
            return None

    def _shannon_entropy(self, p):
        return shannon_entropy(p)

    def _map_influence_and_entropy(self, obs):
        """Return map influence score and policy entropy for the given observation."""
        if obs is None or not getattr(self.agent, "use_map", False):
            return 0.0, None

        # Build an observation without map information for comparison.
        try:
            st = list(obs.state) if isinstance(obs.state, (list, tuple)) else [obs.state]
            if len(st) < 4:
                st += [None] * (4 - len(st))
            st[3] = None  # Remove map channel
            info_off = dict(obs.info) if hasattr(obs, "info") and isinstance(obs.info, dict) else {}
            info_off.pop("exploration_map", None)
            info_off.pop("policy_map", None)
            obs_off = Observation(state=st, info=info_off)
        except Exception:
            return 0.0, None

        p_full = self._peek_policy_probs(obs)
        p_off = self._peek_policy_probs(obs_off)
        js = self._js_divergence(p_full, p_off) if (p_full is not None and p_off is not None) else 0.0
        ent = self._shannon_entropy(p_full)
        if js is None or not np.isfinite(js):
            js = 0.0
        return float(js), (None if ent is None or not np.isfinite(ent) else float(ent))

    def _frontier_follow(self, prev_map: dict, prev_idx, next_idx) -> int:
        return frontier_follow(prev_map, prev_idx, next_idx)

    def _local_uncertainty(self, prev_map: dict, idx, radius: int = 2) -> Optional[float]:
        return local_uncertainty(prev_map, idx, radius)

    def _time_to_thresholds(self, coverage_hist, thresholds=(0.25, 0.5, 0.75)):
        return time_to_thresholds(coverage_hist, thresholds)

    def _get_blank_map_state(self, snapshot):
        if not snapshot or not isinstance(snapshot, dict):
            return None

        map_shape = snapshot.get("map_shape")
        cache_key = None
        if isinstance(map_shape, (list, tuple)) and len(map_shape) >= 2:
            cache_key = (int(map_shape[0]), int(map_shape[1]))
        else:
            memory = snapshot.get("memory")
            if isinstance(memory, torch.Tensor) and memory.dim() >= 4:
                cache_key = (int(memory.size(-2)), int(memory.size(-1)))

        if cache_key is None:
            return None

        blank_state = self._blank_map_state_cache.get(cache_key)
        if blank_state is None:
            memory = snapshot.get("memory")
            if not isinstance(memory, torch.Tensor):
                return None

            blank_memory = torch.zeros_like(memory)
            mask = snapshot.get("mask")
            blank_mask = torch.zeros_like(mask) if isinstance(mask, torch.Tensor) else None

            blank_state = {
                "map_shape": cache_key,
                "memory": blank_memory,
                "mask": blank_mask,
            }
            self._blank_map_state_cache[cache_key] = blank_state

        return blank_state

    def _acquire_blank_map_buffers(self, memory, mask):
        blank_mem = None
        blank_mask = None

        if isinstance(memory, torch.Tensor):
            key = ("mem", tuple(memory.shape), memory.dtype, memory.device)
            blank_mem = self._blank_map_buffer_cache.get(key)
            if blank_mem is None or blank_mem.shape != memory.shape or blank_mem.device != memory.device:
                blank_mem = torch.zeros_like(memory)
                self._blank_map_buffer_cache[key] = blank_mem
            else:
                blank_mem.zero_()

        if isinstance(mask, torch.Tensor):
            key = ("mask", tuple(mask.shape), mask.dtype, mask.device)
            blank_mask = self._blank_map_buffer_cache.get(key)
            if blank_mask is None or blank_mask.shape != mask.shape or blank_mask.device != mask.device:
                blank_mask = torch.zeros_like(mask)
                self._blank_map_buffer_cache[key] = blank_mask
            else:
                blank_mask.zero_()

        return blank_mem, blank_mask

    def _fast_snapshot_map_state(self, map_encoder):
        if map_encoder is None:
            return None

        memory = getattr(map_encoder, "memory", None)
        if not isinstance(memory, torch.Tensor):
            return None

        mask = getattr(map_encoder, "memory_mask", None)
        with torch.no_grad():
            mem_clone = memory.detach().clone()
            mask_clone = mask.detach().clone() if isinstance(mask, torch.Tensor) else None
        mem_shape = tuple(getattr(map_encoder, "_mem_shape", ()) or ())
        return mem_clone, mask_clone, mem_shape

    def _restore_fast_snapshot(self, map_encoder, snapshot):
        if map_encoder is None or not snapshot:
            return False

        mem_clone, mask_clone, mem_shape = snapshot
        restored = False

        with torch.no_grad():
            if isinstance(mem_clone, torch.Tensor):
                memory = getattr(map_encoder, "memory", None)
                if isinstance(memory, torch.Tensor) and memory.shape == mem_clone.shape:
                    memory.copy_(mem_clone)
                else:
                    map_encoder.memory = mem_clone.clone()
                restored = True
            else:
                map_encoder.memory = None

            if isinstance(mask_clone, torch.Tensor):
                memory_mask = getattr(map_encoder, "memory_mask", None)
                if isinstance(memory_mask, torch.Tensor) and memory_mask.shape == mask_clone.shape:
                    memory_mask.copy_(mask_clone)
                else:
                    map_encoder.memory_mask = mask_clone.clone()
            else:
                map_encoder.memory_mask = None

            map_encoder._mem_shape = tuple(mem_shape) if mem_shape else None

        return restored

    def _swap_in_blank_map(self, map_encoder):
        if map_encoder is None:
            return None

        memory = getattr(map_encoder, "memory", None)
        mask = getattr(map_encoder, "memory_mask", None)

        blank_mem, blank_mask = self._acquire_blank_map_buffers(memory, mask)
        if blank_mem is None and blank_mask is None:
            return None

        prev_shape = getattr(map_encoder, "_mem_shape", None)

        with torch.no_grad():
            if isinstance(blank_mem, torch.Tensor):
                map_encoder.memory = blank_mem
                if blank_mem.dim() >= 4:
                    map_encoder._mem_shape = (int(blank_mem.size(-2)), int(blank_mem.size(-1)))
            else:
                map_encoder.memory = None
                map_encoder._mem_shape = prev_shape

            if isinstance(blank_mask, torch.Tensor):
                map_encoder.memory_mask = blank_mask
            else:
                map_encoder.memory_mask = None

        return (memory, mask, prev_shape, blank_mem, blank_mask)

    def _restore_blank_swap(self, map_encoder, swap_ctx):
        if map_encoder is None or not swap_ctx:
            return

        prev_mem, prev_mask, prev_shape, blank_mem, blank_mask = swap_ctx

        with torch.no_grad():
            map_encoder.memory = prev_mem
            map_encoder.memory_mask = prev_mask
            map_encoder._mem_shape = prev_shape

            if isinstance(blank_mem, torch.Tensor):
                blank_mem.zero_()
            if isinstance(blank_mask, torch.Tensor):
                blank_mask.zero_()

    def _snapshot_policy_state(self):
        """Capture the agent's recurrent policy state without cloning tensors."""
        agent = self.agent
        state = {
            "use_transformer": bool(getattr(agent, "use_transformer", False)),
            "last_action": getattr(agent, "last_action", None),
        }

        if state["use_transformer"]:
            state["obs_buffer"] = list(getattr(agent, "obs_buffer", []))
            state["action_buffer"] = list(getattr(agent, "action_buffer", []))
        else:
            state["lssg_hidden"] = getattr(agent, "lssg_hidden", None)
            state["gssg_hidden"] = getattr(agent, "gssg_hidden", None)
            state["policy_hidden"] = getattr(agent, "policy_hidden", None)

        return state

    @staticmethod
    def _copy_policy_state(state):
        if not state:
            return {}

        copied = dict(state)
        if copied.get("use_transformer", False):
            copied["obs_buffer"] = list(state.get("obs_buffer", []))
            copied["action_buffer"] = list(state.get("action_buffer", []))
        return copied

    def _apply_policy_state(self, state):
        """Restore a previously captured policy state on the agent."""
        if not state:
            return

        agent = self.agent
        if "last_action" in state:
            agent.last_action = state["last_action"]

        if state.get("use_transformer", False):
            if hasattr(agent, "obs_buffer"):
                agent.obs_buffer = list(state.get("obs_buffer", []))
            if hasattr(agent, "action_buffer"):
                agent.action_buffer = list(state.get("action_buffer", []))
        else:
            for attr in ("lssg_hidden", "gssg_hidden", "policy_hidden"):
                if attr in state and hasattr(agent, attr):
                    setattr(agent, attr, state[attr])

    def _build_blank_policy_state(self):
        """Generate an empty policy state compatible with the agent type."""
        if getattr(self.agent, "use_transformer", False):
            return {
                "use_transformer": True,
                "last_action": -1,
                "obs_buffer": [],
                "action_buffer": [],
            }

        return {
            "use_transformer": False,
            "last_action": -1,
            "lssg_hidden": None,
            "gssg_hidden": None,
            "policy_hidden": None,
        }

    def _compute_map_influence_score(self, obs, policy_snapshot=None, p_full=None):
        """Compute JS divergence between policy with and without map features."""
        if obs is None or not getattr(self.agent, "use_map", False):
            return 0.0

        try:
            st = list(obs.state) if isinstance(obs.state, (list, tuple)) else [obs.state]
            if len(st) < 4:
                st += [None] * (4 - len(st))
            st[3] = None
            info_off = dict(getattr(obs, "info", {}) or {})
            info_off.pop("exploration_map", None)
            info_off.pop("policy_map", None)
            obs_off = Observation(state=tuple(st), info=info_off)
        except Exception as e:
            print(f"[MIS] Failed to build obs_off: {e}")
            return 0.0

        encoder = getattr(self.agent, "encoder", None)
        neural_map_active = (
            getattr(self.agent, "is_neural_slam", False)
            and getattr(encoder, "exploration_mode", None) == "neural"
        )

        if policy_snapshot is None:
            policy_snapshot = self._snapshot_policy_state()
        blank_policy_state = self._build_blank_policy_state()
        policy_snapshot = self._snapshot_policy_state()
        off_policy_state = self._copy_policy_state(policy_snapshot)

        map_snapshot = None
        cached_hidden = None
        snapshot_fn = None
        restore_fn = None

        if neural_map_active and encoder is not None:
            snapshot_fn = getattr(self.agent, "_snapshot_map_state", None)
            restore_fn = getattr(self.agent, "_restore_map_state", None)

        has_map_snapshot = callable(snapshot_fn) and callable(restore_fn)
        p_full_local = p_full
        p_off = None

        if has_map_snapshot:
            try:
                map_snapshot = snapshot_fn()
                cached_hidden = getattr(encoder, "_map_hidden_state", None)

                if map_snapshot is not None:
                    restore_fn(map_snapshot)
                    if cached_hidden is not None:
                        encoder._map_hidden_state = cached_hidden

                self._apply_policy_state(policy_snapshot)
                if p_full_local is None:
                    _, _, _, p_full_local = self.agent.peek_policy(obs)

                if map_snapshot is not None:
                    restore_fn(map_snapshot)
                    if cached_hidden is not None:
                        encoder._map_hidden_state = cached_hidden

                    blank_snapshot = self._get_blank_map_state(map_snapshot)
                    if blank_snapshot is not None:
                        restore_fn(blank_snapshot)
                    encoder._map_hidden_state = None

                # Evaluate policy without map using the copied policy state
                self._apply_policy_state(off_policy_state)
                _, _, _, p_off = self.agent.peek_policy(obs_off)
            finally:
                if map_snapshot is not None:
                    restore_fn(map_snapshot)
                    if cached_hidden is not None:
                        encoder._map_hidden_state = cached_hidden

                self._apply_policy_state(policy_snapshot)
        else:
            try:
                self._apply_policy_state(policy_snapshot)
                if p_full_local is None:
                    _, _, _, p_full_local = self.agent.peek_policy(obs)

                # Evaluate policy without map using the copied policy state
                self._apply_policy_state(off_policy_state)
                _, _, _, p_off = self.agent.peek_policy(obs_off)
            finally:
                self._apply_policy_state(policy_snapshot)

        js = self._js_divergence(p_full_local, p_off)
        return float(js) if (js is not None and np.isfinite(js)) else 0.0

    def _compute_scene_graph_influence_score(self, obs, policy_snapshot=None, p_full=None):
        if obs is None or not getattr(self.agent, "use_scene_graph", False):
            return 0.0

        try:
            st = list(obs.state) if isinstance(obs.state, (list, tuple)) else [obs.state]
            if len(st) < 3:
                st += [None] * (3 - len(st))
            st[1] = None
            st[2] = None
            info_off = dict(getattr(obs, "info", {}) or {})
            obs_off = Observation(
                state=tuple(st),
                info=info_off,
                terminated=getattr(obs, "terminated", False),
                truncated=getattr(obs, "truncated", False),
                reward=getattr(obs, "reward", 0.0),
            )
        except Exception as e:
            print(f"[SG MIS] Failed to build obs_off: {e}")
            return 0.0

        if policy_snapshot is None:
            policy_snapshot = self._snapshot_policy_state()

        p_full_local = p_full
        p_off = None

        try:
            self._apply_policy_state(policy_snapshot)
            if p_full_local is None:
                _, _, _, p_full_local = self.agent.peek_policy(obs)

            sg_off_state = self._copy_policy_state(policy_snapshot)
            if not sg_off_state.get("use_transformer", False):
                sg_off_state["lssg_hidden"] = None
                sg_off_state["gssg_hidden"] = None

            self._apply_policy_state(sg_off_state)
            _, _, _, p_off = self.agent.peek_policy(obs_off)
        finally:
            self._apply_policy_state(policy_snapshot)

        if p_full_local is None or p_off is None:
            return 0.0
        js = self._js_divergence(p_full_local, p_off)
        return float(js) if (js is not None and np.isfinite(js)) else 0.0

    def _compute_rgb_influence_score(self, obs, policy_snapshot=None, p_full=None):
        if obs is None or not getattr(self.agent, "use_rgb", False):
            return 0.0

        try:
            st = list(obs.state) if isinstance(obs.state, (list, tuple)) else [obs.state]
            if len(st) < 1:
                st += [None]
            st[0] = None
            info_off = dict(getattr(obs, "info", {}) or {})
            obs_off = Observation(
                state=tuple(st),
                info=info_off,
                terminated=getattr(obs, "terminated", False),
                truncated=getattr(obs, "truncated", False),
                reward=getattr(obs, "reward", 0.0),
            )
        except Exception as e:
            print(f"[RGB MIS] Failed to build obs_off: {e}")
            return 0.0

        if policy_snapshot is None:
            policy_snapshot = self._snapshot_policy_state()

        encoder = getattr(self.agent, "encoder", None)
        p_full_local = p_full
        p_off = None
        restore_use_rgb = None

        try:
            self._apply_policy_state(policy_snapshot)
            if p_full_local is None:
                _, _, _, p_full_local = self.agent.peek_policy(obs)

            rgb_off_state = self._copy_policy_state(policy_snapshot)
            self._apply_policy_state(rgb_off_state)

            if encoder is not None and hasattr(encoder, "use_rgb"):
                restore_use_rgb = encoder.use_rgb
                encoder.use_rgb = False

            _, _, _, p_off = self.agent.peek_policy(obs_off)
        finally:
            if encoder is not None and restore_use_rgb is not None:
                encoder.use_rgb = restore_use_rgb
            self._apply_policy_state(policy_snapshot)

        if p_full_local is None or p_off is None:
            return 0.0
        js = self._js_divergence(p_full_local, p_off)
        return float(js) if (js is not None and np.isfinite(js)) else 0.0

    def _compute_last_action_influence_score(self, obs, policy_snapshot=None, p_full=None):
        if obs is None:
            return 0.0

        if policy_snapshot is None:
            policy_snapshot = self._snapshot_policy_state()

        p_full_local = p_full
        p_off = None

        try:
            self._apply_policy_state(policy_snapshot)
            if p_full_local is None:
                _, _, _, p_full_local = self.agent.peek_policy(obs)

            actionless_state = self._copy_policy_state(policy_snapshot)
            actionless_state["last_action"] = -1
            self._apply_policy_state(actionless_state)
            _, _, _, p_off = self.agent.peek_policy(obs)
        finally:
            self._apply_policy_state(policy_snapshot)

        if p_full_local is None or p_off is None:
            return 0.0
        js = self._js_divergence(p_full_local, p_off)
        return float(js) if (js is not None and np.isfinite(js)) else 0.0

    def _compute_modality_influence_divergences(self, obs):
        """Compute raw JS divergences for active modalities using a single snapshot."""
        divergences = {}
        if obs is None:
            return divergences

        policy_snapshot = self._snapshot_policy_state()
        p_full = None

        try:
            self._apply_policy_state(policy_snapshot)
            _, _, _, p_full = self.agent.peek_policy(obs)
        except Exception:
            p_full = None
        finally:
            self._apply_policy_state(policy_snapshot)

        if p_full is None:
            return divergences

        p_full_array = np.asarray(p_full, dtype=np.float64)
        if p_full_array.size == 0 or not np.all(np.isfinite(p_full_array)):
            return divergences

        if getattr(self.agent, "use_map", False):
            mis = self._compute_map_influence_score(obs, policy_snapshot=policy_snapshot, p_full=p_full)
            if mis is not None and np.isfinite(mis):
                divergences["map"] = float(mis)

        if getattr(self.agent, "use_scene_graph", False):
            sg_mis = self._compute_scene_graph_influence_score(obs, policy_snapshot=policy_snapshot, p_full=p_full)
            if sg_mis is not None and np.isfinite(sg_mis):
                divergences["scene_graph"] = float(sg_mis)

        if getattr(self.agent, "use_rgb", False):
            rgb_mis = self._compute_rgb_influence_score(obs, policy_snapshot=policy_snapshot, p_full=p_full)
            if rgb_mis is not None and np.isfinite(rgb_mis):
                divergences["rgb"] = float(rgb_mis)

        la_mis = self._compute_last_action_influence_score(obs, policy_snapshot=policy_snapshot, p_full=p_full)
        if la_mis is not None and np.isfinite(la_mis):
            divergences["last_action"] = float(la_mis)

        self._apply_policy_state(policy_snapshot)
        return divergences

    def _to_action_idx(self, a):
        """Convert various action containers to a plain integer index."""
        import numpy as _np, torch as _torch
        if isinstance(a, (list, tuple, _np.ndarray)):
            if len(a) == 0:
                raise ValueError("Empty action container")
            a = a[0]
        if isinstance(a, _torch.Tensor):
            a = a.item()
        return int(a)

    def run(self):
        """Execute the multi-scene training loop with logging and diagnostics."""

        # Helper utilities scoped to this method only.
        def shannon_entropy(p):
            if p is None:
                return None
            p = np.asarray(p, dtype=np.float64)
            if p.size == 0:
                return None
            p = np.clip(p, 1e-12, 1.0)
            p = p / p.sum()
            return float(-(p * np.log(p)).sum())

        def finite_mean(xs):
            xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
            return float(np.mean(xs)) if xs else 0.0

        def time_to_thresholds(coverage_hist, thresholds=(0.25, 0.5, 0.75)):
            out = {t: None for t in thresholds}
            if not coverage_hist:
                return out
            for t in thresholds:
                step = next((k for k, c in enumerate(coverage_hist, start=1) if c >= t), None)
                out[t] = step  # None indicates the threshold was not reached
            return out

        def frontier_follow(prev_map_dict, prev_idx, next_idx):
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
            H, W = frontier.shape
            if not (0 <= pi < H and 0 <= pj < W and 0 <= ni < H and 0 <= nj < W):
                return 0
            if (pi == ni and pj == nj):
                return 0
            if frontier[pi, pj] == 1 and visited[ni, nj] == 0 and (abs(pi - ni) + abs(pj - nj) == 1):
                return 1
            return 0

        def local_uncertainty(prev_map_dict, idx, radius=2):
            """Average lack of confidence in a square window centred on the agent."""
            if not isinstance(prev_map_dict, dict) or idx is None:
                return None
            dense = prev_map_dict.get("dense", {})
            conf = np.asarray(dense.get("confidence", []), dtype=np.float32)
            if conf.ndim != 2 or conf.size == 0:
                return None
            i, j = idx
            H, W = conf.shape
            if not (0 <= i < H and 0 <= j < W):
                return None
            i0, i1 = max(0, i - radius), min(H, i + radius + 1)
            j0, j1 = max(0, j - radius), min(W, j + radius + 1)
            win = conf[i0:i1, j0:j1]
            if win.size == 0:
                return None
            return float(np.mean(1.0 - win))

        # Training loop
        pbar = tqdm(total=self.total_episodes, desc="Training Episodes", ncols=160, leave=False)
        episode_count = 0
        max_score = 0.0

        while episode_count < self.total_episodes:
            episode_number = episode_count + 1
            neural_slam_active = getattr(self.agent, "is_neural_slam", False)
            compute_map_influence_this_episode = True
            if neural_slam_active:
                # Skip some evaluations to reduce Neural SLAM overhead
                compute_map_influence_this_episode = ((episode_number - 1) % 3 == 0)

            # Aggregated metrics across all scenes in the episode
            episode_scores, episode_steps, episode_rewards = [], [], []
            episode_intrinsic_rewards, episode_extrinsic_rewards = [], []
            episode_losses, episode_entropies, episode_policy_losses, episode_value_losses = [], [], [], []
            episode_mapper_losses = []

            # Exploration and map metrics collected per episode
            episode_coverage_per_step, episode_revisit_rates, episode_cov_auc = [], [], []
            episode_map_influence_scores = []
            episode_scene_graph_influence_scores = []
            episode_rgb_influence_scores = []
            episode_last_action_influence_scores = []
            episode_value_deltas = []
            episode_frontier_follow_rates = []
            episode_ttc25, episode_ttc50, episode_ttc75 = [], [], []
            episode_corr_entropy_uncertainty = []
            episode_num_frontiers_means = []

            for scene_number in self.scene_numbers:
                obs = self.robust_reset(scene_number)
                if obs is None:
                    continue
                self.agent.reset()

                episode_reward = 0.0
                episode_intrinsic_reward = 0.0
                episode_extrinsic_reward = 0.0
                episode_steps_scene = 0
                unfinished_episode = False

                # Scene-level efficiency statistics
                scene_total_cells = self._estimate_total_cells(scene_number, obs.info.get("exploration_map"))
                scene_visited_cells = set()
                first_cell = self._extract_cell(obs)
                if first_cell is not None:
                    scene_visited_cells.add(first_cell)
                if scene_total_cells < len(scene_visited_cells):
                    scene_total_cells = len(scene_visited_cells) or 1
                    self._total_cell_cache[scene_number] = scene_total_cells

                scene_new_cells_total = 0.0
                scene_revisit_steps = 0
                scene_metric_steps = 0
                scene_coverage_history, scene_revisit_history = [], []

                # Frontier and uncertainty statistics
                scene_frontier_follow = 0
                scene_move_steps = 0
                num_frontiers_hist = []
                policy_entropy_hist = []
                local_uncertainty_hist = []
                scene_last_map_influence = None

                while not (obs.terminated or obs.truncated):
                    def _with_policy_map(o):
                        st = list(o.state) if isinstance(o.state, (list, tuple)) else [o.state]
                        if len(st) < 4:  # Ensure space for the map channel
                            st += [None] * (4 - len(st))
                        m = o.info.get("policy_map")
                        if m is None:
                            m = o.info.get("exploration_map")  # Fallback
                        st[3] = m

                        # Rebuild the observation if mutation is not supported
                        try:
                            o.state = tuple(st)
                            return o
                        except Exception:
                            return Observation(
                                state=tuple(st),
                                info=o.info,
                                terminated=getattr(o, "terminated", False),
                                truncated=getattr(o, "truncated", False),
                                reward=getattr(o, "reward", 0.0),
                            )

                    obs_fixed = _with_policy_map(obs)

                    self.agent.prepare_neural_map_update(obs_fixed, env=self.env)
                    neural_training = None
                    if getattr(self.agent, "exploration_mode", "") == "neural":
                        neural_training = obs_fixed.info.get("neural_slam_training")

                    # Compute modality influence metrics using the map-aware observation
                    if (
                        compute_map_influence_this_episode
                        and (
                            self.map_influence_stride == 1
                            or scene_metric_steps % self.map_influence_stride == 0
                        )
                    ):
                        raw_divergences = self._compute_modality_influence_divergences(obs_fixed)
                        enabled_divergences = {}
                        if getattr(self.agent, "use_map", False) and "map" in raw_divergences:
                            val = raw_divergences["map"]
                            if np.isfinite(val) and val >= 0.0:
                                enabled_divergences["map"] = float(val)
                        if getattr(self.agent, "use_scene_graph", False) and "scene_graph" in raw_divergences:
                            val = raw_divergences["scene_graph"]
                            if np.isfinite(val) and val >= 0.0:
                                enabled_divergences["scene_graph"] = float(val)
                        if getattr(self.agent, "use_rgb", False) and "rgb" in raw_divergences:
                            val = raw_divergences["rgb"]
                            if np.isfinite(val) and val >= 0.0:
                                enabled_divergences["rgb"] = float(val)
                        if "last_action" in raw_divergences:
                            val = raw_divergences["last_action"]
                            if np.isfinite(val) and val >= 0.0:
                                enabled_divergences["last_action"] = float(val)

                        normalized_divergences = {}
                        if enabled_divergences:
                            total_divergence = sum(enabled_divergences.values())
                            if np.isfinite(total_divergence) and total_divergence > 0.0:
                                normalized_divergences = {
                                    k: float(v / total_divergence) for k, v in enabled_divergences.items()
                                }
                            else:
                                share = 1.0 / float(len(enabled_divergences))
                                normalized_divergences = {k: share for k in enabled_divergences}

                        if normalized_divergences:
                            if "map" in normalized_divergences:
                                scene_last_map_influence = float(normalized_divergences["map"])
                                episode_map_influence_scores.append(scene_last_map_influence)
                            if "scene_graph" in normalized_divergences:
                                episode_scene_graph_influence_scores.append(
                                    float(normalized_divergences["scene_graph"])
                                )
                            if "rgb" in normalized_divergences:
                                episode_rgb_influence_scores.append(
                                    float(normalized_divergences["rgb"])
                                )
                            if "last_action" in normalized_divergences:
                                episode_last_action_influence_scores.append(
                                    float(normalized_divergences["last_action"])
                                )

                    # Diagnostic metric: influence of the map on the critic value
                    try:
                        v_full = self.agent.peek_value(obs_fixed)

                        st = list(obs_fixed.state) if isinstance(obs_fixed.state, (list, tuple)) else [obs_fixed.state]
                        if len(st) < 4:
                            st += [None] * (4 - len(st))
                        st[3] = None

                        info_off = dict(getattr(obs_fixed, "info", {}) or {})
                        info_off["exploration_map"] = None
                        info_off["policy_map"] = None
                        obs_off = Observation(
                            state=tuple(st),
                            info=info_off,
                            terminated=getattr(obs_fixed, "terminated", False),
                            truncated=getattr(obs_fixed, "truncated", False),
                            reward=getattr(obs_fixed, "reward", 0.0),
                        )

                        v_off = self.agent.peek_value(obs_off)
                        if np.isfinite(v_full) and np.isfinite(v_off):
                            episode_value_deltas.append(float(v_full - v_off))
                    except Exception:
                        pass

                    try:
                        with torch.no_grad():
                            _, _, _, probs_full = self.agent.peek_policy(obs_fixed)
                        ent = shannon_entropy(probs_full)
                        if ent is not None and np.isfinite(ent):
                            policy_entropy_hist.append(float(ent))
                    except Exception:
                        pass

                    # Use the same map source for frontier and uncertainty statistics
                    prev_map_dict = self._get_exploration_map(obs_fixed)
                    prev_idx = self._extract_cell(obs_fixed)

                    # Execute the policy step using the map-aware observation
                    action, lssg_h, gssg_h, policy_h, last_a, value = self.agent.get_action(obs_fixed)

                    mapper_logits = head_params = None
                    if getattr(self.agent.encoder, "exploration_mode", None) == "neural":
                        mapper_logits, head_params = self.agent.compute_neural_mapper_outputs(obs_fixed)
                        if hasattr(self.env, "set_mapper_outputs") and (
                            mapper_logits is not None or head_params is not None
                        ):
                            self.env.set_mapper_outputs(mapper_logits, head_params)

                    next_obs = self.robust_step(action)
                    if next_obs is None:
                        unfinished_episode = True
                        break

                    extrinsic_reward = next_obs.reward
                    reward = extrinsic_reward
                    done = next_obs.terminated or next_obs.truncated

                    intrinsic_bonus = 0.0
                    if self.intrinsic_cfg.get("active", False):
                        phi_prev = self._phi(prev_map_dict, prev_idx) if isinstance(prev_map_dict, dict) else 0.0
                        next_map_dict = self._get_exploration_map(next_obs)
                        next_idx = self._extract_cell(next_obs)
                        phi_next = self._phi(next_map_dict, next_idx) if isinstance(next_map_dict, dict) else 0.0
                        intrinsic_gamma = float(self.intrinsic_cfg.get("gamma", 1.0))
                        intrinsic_bonus = (intrinsic_gamma * phi_next) - phi_prev
                        if np.isfinite(intrinsic_bonus):
                            reward += intrinsic_bonus
                        else:
                            intrinsic_bonus = 0.0

                    # Store the observation in the rollout buffer with a consistent map entry
                    rgb, lssg, gssg = obs_fixed.state[:3]
                    occ = obs_fixed.state[3]  # map from policy_map/exploration_map (see above)
                    state_for_buffer = (rgb, lssg, gssg, occ)

                    hiddens = (lssg_h, gssg_h, policy_h)
                    map_index = prev_idx
                    agent_pos = next_obs.info.get("agent_pos", None)
                    self.agent.rollout_buffers.add(
                        state_for_buffer,
                        action,
                        reward,
                        done,
                        hiddens,
                        last_a,
                        agent_pos,
                        map_index=map_index,
                        neural_slam=neural_training,
                    )

                    if getattr(self.agent, "exploration_mode", "") == "neural":
                        self.agent.enqueue_neural_slam_sample(neural_training)
                        self.agent.maybe_online_neural_slam_step()

                    # Update episode statistics
                    episode_reward += reward
                    episode_extrinsic_reward += extrinsic_reward
                    episode_intrinsic_reward += intrinsic_bonus
                    episode_steps_scene += 1

                    # Update efficiency metrics
                    prev_unique = len(scene_visited_cells)
                    prev_cov_fraction = float(prev_unique) / float(scene_total_cells) if scene_total_cells > 0 else 0.0
                    scene_metric_steps += 1

                    cell_next = self._extract_cell(next_obs)
                    if cell_next is not None:
                        if cell_next in scene_visited_cells:
                            scene_revisit_steps += 1
                            scene_revisit_history.append(1.0)
                        else:
                            scene_visited_cells.add(cell_next)
                            scene_new_cells_total += 1.0
                            scene_revisit_history.append(0.0)
                            if len(scene_visited_cells) > scene_total_cells:
                                scene_total_cells = len(scene_visited_cells)
                                self._total_cell_cache[scene_number] = scene_total_cells
                        coverage_fraction = (
                            float(len(scene_visited_cells)) / float(scene_total_cells)
                            if scene_total_cells > 0
                            else 0.0
                        )
                    else:
                        scene_revisit_steps += 1
                        scene_revisit_history.append(1.0)
                        coverage_fraction = prev_cov_fraction

                    scene_coverage_history.append(coverage_fraction)

                    # Frontier-follow statistics and local uncertainty (read-only)
                    if prev_map_dict is not None and prev_idx is not None and cell_next is not None:
                        ff = frontier_follow(prev_map_dict, prev_idx, cell_next)
                        scene_frontier_follow += ff
                        if cell_next != prev_idx:
                            scene_move_steps += 1

                        lu = local_uncertainty(prev_map_dict, prev_idx, radius=2)
                        if lu is not None and np.isfinite(lu):
                            local_uncertainty_hist.append(float(lu))

                        dense = prev_map_dict.get("dense", {}) if isinstance(prev_map_dict, dict) else {}
                        fmat = np.asarray(dense.get("frontier", []), dtype=np.uint8)
                        if fmat.ndim == 2:
                            num_frontiers_hist.append(float(fmat.sum()))

                    obs = next_obs

                if unfinished_episode:
                    self.agent.reset()
                    continue

                # Aggregate scene-level metrics into episode statistics
                denom = scene_metric_steps if scene_metric_steps > 0 else episode_steps_scene
                if denom > 0:
                    episode_coverage_per_step.append(scene_new_cells_total / float(denom))
                    episode_revisit_rates.append(scene_revisit_steps / float(denom))

                if scene_coverage_history:
                    episode_cov_auc.append(self._normalized_coverage_auc(scene_coverage_history))
                    ttc = time_to_thresholds(scene_coverage_history, thresholds=(0.25, 0.5, 0.75))
                    episode_ttc25.append(float(ttc[0.25]) if ttc[0.25] is not None else float("inf"))
                    episode_ttc50.append(float(ttc[0.5]) if ttc[0.5] is not None else float("inf"))
                    episode_ttc75.append(float(ttc[0.75]) if ttc[0.75] is not None else float("inf"))

                if scene_move_steps > 0:
                    episode_frontier_follow_rates.append(scene_frontier_follow / float(scene_move_steps))

                if policy_entropy_hist and local_uncertainty_hist:
                    corr_eu = self._pearson_corr(policy_entropy_hist, local_uncertainty_hist)
                    episode_corr_entropy_uncertainty.append(float(corr_eu))

                if num_frontiers_hist:
                    episode_num_frontiers_means.append(float(np.mean(num_frontiers_hist)))

                # Perform the A2C update
                result = self.agent.update()
                loss = result.get("loss", None)
                policy_loss = result.get("policy_loss", None)
                value_loss = result.get("value_loss", None)
                entropy = result.get("entropy", None)
                mapper_loss = result.get("mapper_loss", None)

                if episode_count % 10 == 0 and hasattr(self.agent, 'encoder') and hasattr(self.agent.encoder,
                                                                                          'map_encoder'):
                    # Gradient norms for different modules
                    rgb_grad_norm = 0.0
                    map_grad_norm = 0.0
                    policy_grad_norm = 0.0

                    for name, param in self.agent.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if 'rgb_encoder' in name:
                                rgb_grad_norm += grad_norm
                            elif 'map_encoder' in name:
                                map_grad_norm += grad_norm
                            elif 'policy' in name:
                                policy_grad_norm += grad_norm

                    self.writer.add_scalar("diagnostics/grad_norm_rgb", rgb_grad_norm, episode_count)
                    self.writer.add_scalar("diagnostics/grad_norm_map", map_grad_norm, episode_count)
                    self.writer.add_scalar("diagnostics/grad_norm_policy", policy_grad_norm, episode_count)

                    # Gradient ratio indicating relative learning of map vs. RGB encoders
                    if rgb_grad_norm > 0:  # Only compute the ratio if RGB gradients are available
                        self.writer.add_scalar(
                            "diagnostics/map_grad_to_rgb_ratio",
                            map_grad_norm / (rgb_grad_norm + 1e-8),
                            episode_count,
                        )

                final_score = obs.info.get("score", 0.0)
                episode_scores.append(final_score)
                episode_steps.append(episode_steps_scene)
                episode_rewards.append(episode_reward)
                episode_intrinsic_rewards.append(episode_intrinsic_reward)
                episode_extrinsic_rewards.append(episode_extrinsic_reward)
                episode_losses.append(loss)
                episode_entropies.append(entropy)
                if policy_loss is not None:
                    episode_policy_losses.append(policy_loss)
                if value_loss is not None:
                    episode_value_losses.append(value_loss)
                if mapper_loss is not None:
                    episode_mapper_losses.append(mapper_loss)

            # Episode-wide means across scenes
            mean_score = self._safe_mean(episode_scores)
            mean_steps = self._safe_mean(episode_steps)
            mean_reward = self._safe_mean(episode_rewards)
            mean_intrinsic_reward = self._safe_mean(episode_intrinsic_rewards)
            mean_extrinsic_reward = self._safe_mean(episode_extrinsic_rewards)
            mean_loss = self._safe_mean(episode_losses)
            mean_entropy = self._safe_mean(episode_entropies)

            if episode_policy_losses:
                self.writer.add_scalar("Loss/policy_loss", self._safe_mean(episode_policy_losses), episode_count)
            if episode_value_losses:
                self.writer.add_scalar("Loss/value_loss", self._safe_mean(episode_value_losses), episode_count)
            if episode_mapper_losses:
                self.writer.add_scalar("Loss/mapper_loss", self._safe_mean(episode_mapper_losses), episode_count)
            self.writer.add_scalar("policy/entropy", mean_entropy, episode_count)
            self.writer.add_scalar("Loss/total", mean_loss, episode_count)
            if episode_extrinsic_rewards:
                self.writer.add_scalar("reward/extrinsic_mean", mean_extrinsic_reward, episode_count)
            if self.intrinsic_cfg.get("active", False) and episode_intrinsic_rewards:
                self.writer.add_scalar("reward/intrinsic_mean", mean_intrinsic_reward, episode_count)

            # Exploration and map logging
            if episode_coverage_per_step:
                self.writer.add_scalar("exploration/coverage_per_step", self._safe_mean(episode_coverage_per_step),
                                       episode_count)
            if episode_revisit_rates:
                self.writer.add_scalar("exploration/revisit_rate", self._safe_mean(episode_revisit_rates),
                                       episode_count)
            if episode_cov_auc:
                self.writer.add_scalar(f"efficiency/cov_auc@{self.coverage_horizon}",
                                       self._safe_mean(episode_cov_auc), episode_count)

            if episode_map_influence_scores:
                mean_map_share = self._safe_mean(episode_map_influence_scores)
                self.writer.add_scalar("map/mis_js_normalized", mean_map_share, episode_count)

            if episode_scene_graph_influence_scores:
                self.writer.add_scalar(
                    "scene_graph/mis_js_normalized",
                    self._safe_mean(episode_scene_graph_influence_scores),
                    episode_count,
                )

            if episode_rgb_influence_scores:
                self.writer.add_scalar(
                    "rgb/mis_js_normalized",
                    self._safe_mean(episode_rgb_influence_scores),
                    episode_count,
                )

            if episode_last_action_influence_scores:
                self.writer.add_scalar(
                    "action/mis_js_normalized",
                    self._safe_mean(episode_last_action_influence_scores),
                    episode_count,
                )

            if episode_value_deltas:
                self.writer.add_scalar(
                    "map/delta_value",
                    self._safe_mean(episode_value_deltas),
                    episode_count,
                )

            if episode_frontier_follow_rates:
                self.writer.add_scalar("map/frontier_follow_rate",
                                       self._safe_mean(episode_frontier_follow_rates), episode_count)

            # Time to coverage thresholds (lower is better; ignore infinite values)
            def finite_mean_or_inf(xs):
                xs = [x for x in xs if x is not None and np.isfinite(x)]
                return float(np.mean(xs)) if xs else float('inf')

            t25 = finite_mean_or_inf(episode_ttc25)
            t50 = finite_mean_or_inf(episode_ttc50)
            t75 = finite_mean_or_inf(episode_ttc75)
            if np.isfinite(t25):
                self.writer.add_scalar("efficiency/time_to_25pct_cov", t25, episode_count)
            if np.isfinite(t50):
                self.writer.add_scalar("efficiency/time_to_50pct_cov", t50, episode_count)
            if np.isfinite(t75):
                self.writer.add_scalar("efficiency/time_to_75pct_cov", t75, episode_count)

            if episode_corr_entropy_uncertainty:
                self.writer.add_scalar("map/corr_entropy_local_uncertainty",
                                       self._safe_mean(episode_corr_entropy_uncertainty), episode_count)

            if episode_num_frontiers_means:
                self.writer.add_scalar("map/num_frontiers_mean",
                                       self._safe_mean(episode_num_frontiers_means), episode_count)

            # Maintain a rolling window for reporting
            episode_record = {"reward": mean_reward, "steps": mean_steps, "score": mean_score}
            if episode_coverage_per_step:
                episode_record["coverage_per_step"] = self._safe_mean(episode_coverage_per_step)
            if episode_revisit_rates:
                episode_record["revisit_rate"] = self._safe_mean(episode_revisit_rates)
            if episode_cov_auc:
                episode_record["cov_auc"] = self._safe_mean(episode_cov_auc)
            if episode_map_influence_scores:
                episode_record["map_influence_share"] = self._safe_mean(episode_map_influence_scores)
            if episode_scene_graph_influence_scores:
                episode_record["scene_graph_influence_share"] = self._safe_mean(
                    episode_scene_graph_influence_scores
                )
            if episode_rgb_influence_scores:
                episode_record["rgb_influence_share"] = self._safe_mean(episode_rgb_influence_scores)
            if episode_last_action_influence_scores:
                episode_record["last_action_influence_share"] = self._safe_mean(
                    episode_last_action_influence_scores
                )
            if episode_value_deltas:
                episode_record["delta_value"] = self._safe_mean(episode_value_deltas)
            self.ep_info_buffer.append(episode_record)

            recent = list(self.ep_info_buffer)
            recent_scores = [ep["score"] for ep in recent]
            recent_steps = [ep["steps"] for ep in recent]
            recent_rewards = [ep["reward"] for ep in recent]
            recent_cov_step = [ep.get("coverage_per_step") for ep in recent if ep.get("coverage_per_step") is not None]
            recent_revisit = [ep.get("revisit_rate") for ep in recent if ep.get("revisit_rate") is not None]
            recent_cov_auc = [ep.get("cov_auc") for ep in recent if ep.get("cov_auc") is not None]
            recent_map_infl = [ep.get("map_influence_share") for ep in recent if
                               ep.get("map_influence_share") is not None]
            recent_sg_infl = [ep.get("scene_graph_influence_share") for ep in recent if
                              ep.get("scene_graph_influence_share") is not None]
            recent_rgb_infl = [ep.get("rgb_influence_share") for ep in recent if
                               ep.get("rgb_influence_share") is not None]
            recent_la_infl = [ep.get("last_action_influence_share") for ep in recent if
                              ep.get("last_action_influence_share") is not None]

            mean_score_total = float(np.mean(recent_scores)) if recent_scores else 0.0
            mean_steps_total = float(np.mean(recent_steps)) if recent_steps else 0.0
            mean_reward_total = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            mean_cov_step_total = float(np.mean(recent_cov_step)) if recent_cov_step else 0.0
            mean_revisit_total = float(np.mean(recent_revisit)) if recent_revisit else 0.0
            mean_cov_auc_total = float(np.mean(recent_cov_auc)) if recent_cov_auc else 0.0
            mean_map_infl_total = float(np.mean(recent_map_infl)) if recent_map_infl else 0.0
            mean_sg_infl_total = float(np.mean(recent_sg_infl)) if recent_sg_infl else 0.0
            mean_rgb_infl_total = float(np.mean(recent_rgb_infl)) if recent_rgb_infl else 0.0
            mean_la_infl_total = float(np.mean(recent_la_infl)) if recent_la_infl else 0.0

            max_score = max(max_score, mean_score_total)

            self.writer.add_scalar("Rollout/Mean_Reward", mean_reward_total, episode_count)
            self.writer.add_scalar("Rollout/Mean_Steps", mean_steps_total, episode_count)
            self.writer.add_scalar("Rollout/Mean_Score", mean_score_total, episode_count)
            if mean_score_total > 0:
                self.writer.add_scalar("Rollout/Steps_for_score_1", mean_steps_total / mean_score_total, episode_count)
            if recent_cov_step:
                self.writer.add_scalar("Rollout/Mean_Coverage_per_Step", mean_cov_step_total, episode_count)
            if recent_revisit:
                self.writer.add_scalar("Rollout/Mean_Revisit_Rate", mean_revisit_total, episode_count)
            if recent_cov_auc:
                self.writer.add_scalar(f"Rollout/Mean_Cov_AUC@{self.coverage_horizon}", mean_cov_auc_total,
                                       episode_count)
            if recent_map_infl:
                self.writer.add_scalar("Rollout/Mean_Map_Influence_Share", mean_map_infl_total, episode_count)
            if recent_sg_infl:
                self.writer.add_scalar("Rollout/Mean_SceneGraph_Influence_Share", mean_sg_infl_total, episode_count)
            if recent_rgb_infl:
                self.writer.add_scalar("Rollout/Mean_RGB_Influence_Share", mean_rgb_infl_total, episode_count)
            if recent_la_infl:
                self.writer.add_scalar("Rollout/Mean_LastAction_Influence_Share", mean_la_infl_total, episode_count)

            pbar.set_postfix({
                "Loss": f"{mean_loss:.3f}",
                "Mean Score": f"{mean_score:.2f}",
                "Mean Steps": f"{mean_steps:4.1f}",
            })
            if episode_count % 5 == 0:
                print(
                    f"\nEp {episode_count:4d} | "
                    f"MA Score: {mean_score_total:5.2f} | Max Score: {max_score:5.2f} | "
                    f"MA Steps: {mean_steps_total:5.1f} | MA Reward: {mean_reward_total:6.2f}"
                )

            episode_count += 1
            pbar.update(1)

        pbar.close()
        self.writer.close()
        self.env.close()
        print("\n[INFO] Training finished.")

