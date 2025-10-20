import inspect
import math
import os
import random
from collections import deque
from pathlib import Path
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical

from src.components.agents.utils import (
    prepare_neural_map_update as util_prepare_neural_map_update,
)
from src.components.encoders.feature_encoder import FeatureEncoder
from src.components.models.navigation_policy import NavigationPolicy
from src.components.utils.observation import Observation
from src.components.utils.paths import DATA_DIR
from src.components.utils.rollout_buffer import RolloutBuffer
from src.imitation.models.neural_slam_il import NeuralSlamController, NeuralSlamMemoryCore, NeuralSlamPoseEstimator

from contextlib import contextmanager


class AbstractAgent(nn.Module):

    def _get_network_component(self, networks, name):
        """Retrieve a component from `networks` that may be a dict, ModuleDict, or object."""
        if networks is None:
            return None

        if isinstance(networks, dict):
            return networks.get(name)

        if isinstance(networks, nn.ModuleDict):
            return networks[name] if name in networks else None

        return getattr(networks, name, None)

    @contextmanager
    def _suppress_map_features(self):
        """Temporarily zero out map features while leaving all other components unchanged."""
        if not (self.encoder.use_map and self.encoder.map_encoder is not None):
            yield
            return

        map_enc = self.encoder.map_encoder
        map_dim = self.encoder.map_dim
        device = self.device

        orig_forward = map_enc.forward

        def zero_forward(map_list):
            N = len(map_list)
            return torch.zeros(N, map_dim, device=device)

        map_enc.forward = zero_forward
        try:
            yield
        finally:
            map_enc.forward = orig_forward

    @torch.no_grad()
    def peek_policy(self, obs, topk=5):
        """Inspect the current policy distribution without mutating the agent state."""
        was_training = self.training
        _la = self.last_action
        _lh = self.lssg_hidden
        _gh = self.gssg_hidden
        _ph = self.policy_hidden

        try:
            self.eval()
            logits, value = self.forward(obs)
            logits = logits[:, -1] if self.use_transformer else logits
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * (probs.clamp_min(1e-8)).log()).sum(-1).item()
            k = min(topk, probs.size(-1))
            top_p, top_i = torch.topk(probs, k=k, dim=-1)
            return (top_i.squeeze(0).tolist(), top_p.squeeze(0).tolist(), float(ent), probs.squeeze(0).cpu().tolist())
        finally:
            self.last_action = _la
            self.lssg_hidden = _lh
            self.gssg_hidden = _gh
            self.policy_hidden = _ph
            if was_training:
                self.train()

    @torch.no_grad()
    def peek_value(self, obs):
        """Return the critic value estimate for an observation without altering hidden state."""
        was_training = self.training
        _la = self.last_action
        _lh = self.lssg_hidden
        _gh = self.gssg_hidden
        _ph = self.policy_hidden

        try:
            self.eval()
            _, value = self.forward(obs)
            if value is None:
                return 0.0
            if self.use_transformer:
                value_tensor = value.reshape(-1)[-1]
            else:
                value_tensor = value.reshape(-1)[-1]
            return float(value_tensor.item())
        finally:
            self.last_action = _la
            self.lssg_hidden = _lh
            self.gssg_hidden = _gh
            self.policy_hidden = _ph
            if was_training:
                self.train()

    @torch.no_grad()
    def peek_policy_map_off(self, obs, topk=5):
        """Inspect the policy with map inputs disabled while keeping internal state intact."""
        was_training = self.training
        _la = self.last_action
        _lh = self.lssg_hidden
        _gh = self.gssg_hidden
        _ph = self.policy_hidden

        if hasattr(obs, "state"):
            st = list(obs.state) if isinstance(obs.state, (list, tuple)) else [obs.state]
            if len(st) < 4:
                st += [None] * (4 - len(st))
            st[3] = None  # Force map features to zeros inside the encoder
            info = dict(getattr(obs, "info", {}) or {})
            info["exploration_map"] = None
            info["policy_map"] = None
            obs0 = Observation(
                state=tuple(st),
                info=info,
                terminated=getattr(obs, "terminated", False),
                truncated=getattr(obs, "truncated", False),
                reward=getattr(obs, "reward", 0.0),
            )
        else:
            # Fallback: leave observation unchanged if state layout is unknown
            obs0 = obs

        try:
            self.eval()
            logits, value = self.forward(obs0)
            logits = logits[:, -1] if self.use_transformer else logits
            probs = F.softmax(logits, dim=-1)
            ent = -(probs * (probs.clamp_min(1e-8)).log()).sum(-1).item()
            k = min(topk, probs.size(-1))
            top_p, top_i = torch.topk(probs, k=k, dim=-1)
            return (top_i.squeeze(0).tolist(), top_p.squeeze(0).tolist(), float(ent), probs.squeeze(0).cpu().tolist())
        finally:
            self.last_action = _la
            self.lssg_hidden = _lh
            self.gssg_hidden = _gh
            self.policy_hidden = _ph
            if was_training:
                self.train()

    def _snapshot_map_state(self):
        if not (getattr(self.encoder, "use_map", False) and hasattr(self.encoder, "map_encoder")):
            return None

        map_enc = self.encoder.map_encoder
        if not hasattr(map_enc, "get_state"):
            return None

        try:
            state_dict = map_enc.get_state()
        except Exception:
            return None

        if not state_dict:
            return None
        return state_dict

    def _restore_map_state(self, snapshot):
        if not snapshot or not (getattr(self.encoder, "use_map", False) and hasattr(self.encoder, "map_encoder")):
            return

        map_enc = self.encoder.map_encoder
        device = self.device

        with torch.no_grad():
            memory = snapshot.get("memory")
            if memory is not None:
                map_enc.memory = memory.to(device)
            mask = snapshot.get("mask")
            if mask is not None:
                map_enc.memory_mask = mask.to(device)
            map_shape = snapshot.get("map_shape")
            if map_shape is not None:
                map_enc._mem_shape = tuple(map_shape)

    def _neural_map_enabled(self) -> bool:
        return (
            getattr(self, "exploration_mode", None) == "neural"
            and isinstance(self.neural_slam_networks, nn.ModuleDict)
            and len(self.neural_slam_networks) > 0
        )

    def _get_neural_component(self, name: str):
        if not isinstance(self.neural_slam_networks, nn.ModuleDict):
            return None
        if name in self.neural_slam_networks:
            return self.neural_slam_networks[name]
        return None

    def _reset_neural_map_state(self):
        self._neural_state = {"read_vector": None, "map": None}

        if not self._neural_map_enabled():
            return

        device = self.device
        batch_size = 1

        controller = self._get_neural_component("controller")
        memory_core = self._get_neural_component("memory_core")

        if controller is not None and hasattr(controller, "reset_state"):
            controller.reset_state(batch_size, device)
        if memory_core is not None and hasattr(memory_core, "reset"):
            memory_core.reset(batch_size, device)

    def prepare_neural_map_update(self, obs, env=None):
        return util_prepare_neural_map_update(self, obs, env)

    def compute_map_contribution(self, obs, action_idx=None):
        """Return gradient-based attribution metrics of log-policy w.r.t. map embeddings."""
        if not getattr(self, "use_map", False):
            return {"log_grad_norm": 0.0, "grad_x_input": 0.0}

        was_training = self.training
        _la = self.last_action
        _lh = self.lssg_hidden
        _gh = self.gssg_hidden
        _ph = self.policy_hidden
        map_snapshot = self._snapshot_map_state()

        try:
            self.eval()

            batch_dict = self.encoder.obs_to_dict(obs) if not isinstance(obs, dict) else obs
            device = self.device
            last_action_val = self.last_action if self.last_action is not None else -1
            last_action_tensor = torch.tensor([[last_action_val]], dtype=torch.long, device=device)

            with torch.no_grad():
                _, _, _, map_feats = self.encoder.forward_seq(
                    batch_dict, last_action_tensor, lssg_hidden=self.lssg_hidden, gssg_hidden=self.gssg_hidden, return_map_features=True
                )

            if map_feats is None:
                return {"log_grad_norm": 0.0, "grad_x_input": 0.0}

            map_override = map_feats.detach().clone().requires_grad_(True)

            self.zero_grad(set_to_none=True)

            with torch.backends.cudnn.flags(enabled=False):
                state_seq, _, _ = self.encoder.forward_seq(
                    batch_dict, last_action_tensor, lssg_hidden=self.lssg_hidden, gssg_hidden=self.gssg_hidden, map_override=map_override
                )

                logits, _, _ = self.policy(state_seq, hidden=self.policy_hidden)

                if logits.dim() == 3:
                    logits_step = logits[:, -1, :]
                else:
                    logits_step = logits

                probs = F.softmax(logits_step, dim=-1)
                dist = Categorical(probs=probs)

                if action_idx is None:
                    action_tensor = torch.argmax(probs, dim=-1)
                else:
                    action_tensor = torch.full((probs.size(0),), int(action_idx), dtype=torch.long, device=device)

                log_prob = dist.log_prob(action_tensor)
                log_prob.sum().backward()

            grad = map_override.grad

            if grad is None:
                return {"log_grad_norm": 0.0, "grad_x_input": 0.0}

            grad_abs = grad.abs()
            log_grad_norm = torch.log(grad_abs + 1e-8).mean().item()
            grad_x_input = (grad * map_override.detach()).mean().item()

            return {"log_grad_norm": log_grad_norm, "grad_x_input": grad_x_input}
        finally:
            self.zero_grad(set_to_none=True)
            self.last_action = _la
            self.lssg_hidden = _lh
            self.gssg_hidden = _gh
            self.policy_hidden = _ph
            self._restore_map_state(map_snapshot)
            if was_training:
                self.train()

    def compute_counterfactual_map_kl(self, obs, include_shuffle=True):
        """Measure KL divergence between original policy and counterfactual map manipulations."""
        if not getattr(self, "use_map", False):
            return {"kl_zero": 0.0, "kl_shuffle": 0.0}

        was_training = self.training
        _la = self.last_action
        _lh = self.lssg_hidden
        _gh = self.gssg_hidden
        _ph = self.policy_hidden
        map_snapshot = self._snapshot_map_state()

        try:
            self.eval()

            batch_dict = self.encoder.obs_to_dict(obs) if not isinstance(obs, dict) else obs
            device = self.device
            last_action_val = self.last_action if self.last_action is not None else -1
            last_action_tensor = torch.tensor([[last_action_val]], dtype=torch.long, device=device)

            state_seq, _, _, map_feats = self.encoder.forward_seq(
                batch_dict, last_action_tensor, lssg_hidden=self.lssg_hidden, gssg_hidden=self.gssg_hidden, return_map_features=True
            )

            if map_feats is None:
                return {"kl_zero": 0.0, "kl_shuffle": 0.0}

            logits_ref, _, _ = self.policy(state_seq, hidden=self.policy_hidden)
            if logits_ref.dim() == 3:
                logits_ref = logits_ref[:, -1, :]

            probs_ref = F.softmax(logits_ref, dim=-1)

            def _kl(p, q):
                p = p.clamp_min(1e-8)
                q = q.clamp_min(1e-8)
                return torch.sum(p * (p.log() - q.log()), dim=-1)

            results = {"kl_zero": 0.0, "kl_shuffle": 0.0}

            map_zero = torch.zeros_like(map_feats)
            state_zero, _, _ = self.encoder.forward_seq(
                batch_dict, last_action_tensor, lssg_hidden=self.lssg_hidden, gssg_hidden=self.gssg_hidden, map_override=map_zero
            )
            logits_zero, _, _ = self.policy(state_zero, hidden=self.policy_hidden)
            if logits_zero.dim() == 3:
                logits_zero = logits_zero[:, -1, :]
            probs_zero = F.softmax(logits_zero, dim=-1)
            results["kl_zero"] = _kl(probs_ref, probs_zero).mean().item()

            if include_shuffle:
                flat = map_feats.reshape(-1, map_feats.size(-1))
                if flat.size(0) > 1:
                    perm = torch.randperm(flat.size(0), device=flat.device)
                    shuffled = flat[perm].view_as(map_feats)
                else:
                    shuffled = map_feats.flip(-1)
                state_shuffle, _, _ = self.encoder.forward_seq(
                    batch_dict, last_action_tensor, lssg_hidden=self.lssg_hidden, gssg_hidden=self.gssg_hidden, map_override=shuffled
                )
                logits_shuffle, _, _ = self.policy(state_shuffle, hidden=self.policy_hidden)
                if logits_shuffle.dim() == 3:
                    logits_shuffle = logits_shuffle[:, -1, :]
                probs_shuffle = F.softmax(logits_shuffle, dim=-1)
                results["kl_shuffle"] = _kl(probs_ref, probs_shuffle).mean().item()

            return results
        finally:
            self.last_action = _la
            self.lssg_hidden = _lh
            self.gssg_hidden = _gh
            self.policy_hidden = _ph
            self._restore_map_state(map_snapshot)
            if was_training:
                self.train()

    @torch.no_grad()
    def forward_for_eval(self, obs, map_gain: float = 1.0, drop_map: bool = False):
        """Single-step eval forward, erlaubt Map-Manipulationen (Gain/Drop)."""
        self.eval()
        # in das Batch-Format der Encoder bringen
        batch_dict = self.encoder.obs_to_dict(obs) if not isinstance(obs, dict) else obs
        # last_action -> [B,T] (wie in forward)
        device = self.device
        last_action = torch.tensor([[self.last_action if self.last_action is not None else -1]], dtype=torch.long, device=device)

        seq, _, _ = self.encoder.forward_seq(batch_dict, last_action)
        logits, _, _ = self.policy(seq)  # [B, T, A]
        logits = logits.reshape(-1, logits.size(-1))  # [1, A]
        return logits  # ohne Softmax

    def __init__(self, env, navigation_config, agent_config, exploration_config, device=None, mapping_path=None):
        super().__init__()
        self.env = env

        self.navigation_config = navigation_config
        self.agent_config = agent_config
        self.exploration_config = exploration_config

        self.alpha = agent_config.get("alpha", 1e-4)
        self.gamma = agent_config.get("gamma", 0.99)
        self.entropy_coef = agent_config.get("entropy_coef", 0.1)
        self.weight_decay = float(agent_config.get("weight_decay", 0.0))

        self.use_transformer = navigation_config["use_transformer"]
        self.use_map = exploration_config["active"]
        self.is_neural_slam = exploration_config.get("map_version") == "neural_slam"
        self.use_scene_graph = navigation_config.get("use_scene_graph", True)
        self.use_rgb = navigation_config.get("use_rgb", True)
        print(f"use_scene_graph: {self.use_scene_graph}")
        print(f"use_rgb: {self.use_rgb}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_actions = len(env.get_actions())

        self._rgb_norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._rgb_norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        self.all_scene_numbers = list(range(1, 6)) + list(range(7, 8)) + list(range(9, 31))
        self.num_scenes = 5

        self.scene_numbers = self.all_scene_numbers[: self.num_scenes]
        if mapping_path is None:
            mapping_path = DATA_DIR / "scene_graph_mappings" / "default"

        self.map_dim = int(exploration_config.get("map_dim", navigation_config.get("map_dim", 64)))

        self.exploration_mode = self._map_version_to_mode(exploration_config["map_version"])
        self.encoder = FeatureEncoder(
            self.num_actions,
            rgb_dim=navigation_config["rgb_dim"],
            action_dim=navigation_config["action_dim"],
            sg_dim=navigation_config["sg_dim"],
            mapping_path=mapping_path,
            use_transformer=self.use_transformer,
            use_map=self.use_map,
            map_dim=self.map_dim,
            exploration_mode=self.exploration_mode,
            use_scene_graph=self.use_scene_graph,
            use_rgb=self.use_rgb,
        ).to(self.device)

        # Policy input dim
        base_dim = int(navigation_config["rgb_dim"] + navigation_config["action_dim"] + 2 * navigation_config["sg_dim"])
        self.input_dim = base_dim + (self.map_dim if self.use_map else 0)

        self.policy = NavigationPolicy(
            input_dim=self.input_dim,
            hidden_dim=navigation_config["policy_hidden"],
            output_dim=self.num_actions,
            use_transformer=navigation_config["use_transformer"],
            use_map=exploration_config["active"],
            value_head=True if agent_config["name"] in ["a2c", "a2c_v2"] else False,
            device=self.device,
        ).to(self.device)

        self.neural_slam_networks = None
        self.neural_slam_config = exploration_config.get("neural_slam", {}).get("training", {}) if self.is_neural_slam else {}
        if self.is_neural_slam:
            networks = self._create_neural_slam_networks()
            if networks:
                self.neural_slam_networks = nn.ModuleDict(networks)
                self.neural_slam_networks.to(self.device)

        self._neural_state = {}
        self._reset_neural_map_state()
        # Track whether the Neural SLAM sub-modules need to be reset before the
        # next invocation.  This is toggled whenever a new episode begins or an
        # error occurs while producing mapper outputs.
        self._neural_mapper_needs_reset = True

        self._neural_online_buffer = (
            deque(maxlen=int(self.neural_slam_config.get("online_replay_size", 128))) if self._neural_map_enabled() else None
        )
        self._neural_online_interval = int(self.neural_slam_config.get("online_update_interval", 8))
        self._neural_online_batch_size = int(self.neural_slam_config.get("online_batch_size", 4))
        self._neural_online_tick = 0
        self.neural_slam_online_optimizer = None
        if self._neural_map_enabled() and self.neural_slam_networks:
            params = list(self.neural_slam_networks.parameters())
            if params:
                lr = float(self.neural_slam_config.get("online_lr", self.alpha))
                weight_decay = float(self.neural_slam_config.get("weight_decay", 0.0))
                self.neural_slam_online_optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # Adam optimizer for ALL parameters
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        # Internal buffers
        self.rollout_buffers = RolloutBuffer(agent_config["num_steps"])
        self.last_action = -1
        self.lssg_hidden = None
        self.gssg_hidden = None
        self.policy_hidden = None
        self.obs_buffer = []
        self.action_buffer = []

        self.to(self.device)

    def forward(self, obs):
        if self.use_transformer:
            self.obs_buffer.append(obs)
            if len(self.action_buffer) == 0:
                self.action_buffer.append(-1)
            else:
                self.action_buffer.append(self.last_action)
            state_vector, _, _ = self.encoder(self.obs_buffer, self.action_buffer)
            logits, value, _ = self.policy(state_vector)
        else:
            state_vector, self.lssg_hidden, self.gssg_hidden = self.encoder(
                obs, self.last_action, lssg_hidden=self.lssg_hidden, gssg_hidden=self.gssg_hidden
            )
            logits, value, self.policy_hidden = self.policy(state_vector, hidden=self.policy_hidden)

        return logits, value.squeeze(-1) if value is not None else None

    @torch.no_grad()
    def compute_neural_mapper_outputs(self, obs):
        """Run Neural SLAM networks and return mapper logits and head params.

        Returns ``(mapper_logits, head_params)`` when the agent operates in
        neural exploration mode.  For all other modes ``(None, None)`` is
        returned so callers can remain agnostic to the active map variant.
        """

        if getattr(self.encoder, "exploration_mode", None) != "neural":
            return None, None

        networks = getattr(self, "neural_slam_networks", None)
        if not networks:
            return None, None

        if obs is None or not hasattr(obs, "info"):
            return None, None

        event = obs.info.get("event") if isinstance(obs.info, dict) else None
        if event is None:
            return None, None

        frame = getattr(event, "frame", None)
        if frame is None and isinstance(event, dict):  # pragma: no cover - safety
            frame = event.get("frame")
        if frame is None:
            return None, None

        try:
            rgb = torch.as_tensor(frame, dtype=torch.float32, device=self.device)
        except Exception:
            return None, None

        if rgb.dim() == 3:
            rgb = rgb.permute(2, 0, 1).unsqueeze(0)
        elif rgb.dim() == 4:
            if rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 3, 1, 2)
            if rgb.size(0) > 1:
                rgb = rgb[:1]
        else:
            return None, None

        rgb_min = float(rgb.min()) if rgb.numel() > 0 else 0.0
        rgb_max = float(rgb.max()) if rgb.numel() > 0 else 0.0
        if rgb_min >= 0.0 and rgb_max > 1.0:
            rgb = rgb / 255.0

        metadata = getattr(event, "metadata", None)
        pose_tensor = None
        if isinstance(metadata, dict):
            agent = metadata.get("agent", {})
            position = agent.get("position", {})
            rotation = agent.get("rotation", {})
            pose_tensor = torch.tensor(
                [[float(position.get("x", 0.0)), float(position.get("z", 0.0)), float(rotation.get("y", 0.0))]],
                dtype=torch.float32,
                device=self.device,
            )

        controller = self._get_network_component(networks, "controller")
        memory_core = self._get_network_component(networks, "memory_core")
        if controller is None or memory_core is None:
            return None, None

        batch = rgb.size(0)
        if self._neural_mapper_needs_reset:
            if hasattr(controller, "reset_state"):
                controller.reset_state(batch, self.device)
            if hasattr(memory_core, "reset"):
                memory_core.reset(batch, self.device)
            self._neural_mapper_needs_reset = False

        try:
            prev_read = None
            if hasattr(memory_core, "get_last_read"):
                prev_read = memory_core.get_last_read(batch, self.device)

            controller_out = controller(rgb, pose_tensor, prev_read)
            mapper_logits = controller_out.get("map_logits")
            head_params = controller_out.get("head_params") or {}

            if not isinstance(head_params, dict):
                head_params = dict(head_params)

            memory_core(head_params)

            if hasattr(controller, "detach_state"):
                controller.detach_state()

            mapper_out = mapper_logits.detach().cpu() if mapper_logits is not None else None
            head_out = {k: v.detach().cpu() for k, v in head_params.items()}
        except Exception:
            self._neural_mapper_needs_reset = True
            return None, None

        if mapper_out is None and not head_out:
            return None, None

        return mapper_out, head_out if head_out else None

    def enqueue_neural_slam_sample(self, sample):
        if not self._neural_map_enabled() or self._neural_online_buffer is None:
            return

        formatted = self._format_neural_tuple(sample)
        if formatted is None:
            return

        self._neural_online_buffer.append(formatted)

    def maybe_online_neural_slam_step(self):
        if not self._neural_map_enabled():
            return

        if self._neural_online_buffer is None or self.neural_slam_online_optimizer is None:
            return

        self._neural_online_tick += 1
        if self._neural_online_interval <= 0 or (self._neural_online_tick % self._neural_online_interval != 0):
            return

        if len(self._neural_online_buffer) < self._neural_online_batch_size:
            return

        try:
            batch_samples = random.sample(self._neural_online_buffer, min(self._neural_online_batch_size, len(self._neural_online_buffer)))
        except ValueError:
            return

        prepared = self._prepare_online_neural_batch(batch_samples)
        if prepared is None:
            return

        controller = self._get_neural_component("controller")
        pose_estimator = self._get_neural_component("pose_estimator")
        if controller is None:
            return

        batch_size = prepared["rgb_curr"].size(0)
        self.set_neural_slam_training_mode(True)
        controller.train()
        if hasattr(controller, "reset_state"):
            controller.reset_state(batch_size, self.device)

        self.neural_slam_online_optimizer.zero_grad(set_to_none=True)

        pose_input = torch.zeros(batch_size, 3, device=self.device)
        read_dim = getattr(controller, "read_dim", getattr(controller, "map_channels", 2))
        read_vector = torch.zeros(batch_size, read_dim, device=self.device)

        try:
            controller_out = controller(prepared["rgb_curr"], pose_input, read_vector)
            logits = controller_out.get("map_logits")
        except Exception:
            return

        if logits is None:
            return

        map_channels = getattr(controller, "map_channels", 1)
        map_resolution = getattr(controller, "map_resolution", None)
        if map_resolution is None:
            total = logits.size(-1)
            map_resolution = int(math.sqrt(max(total // max(map_channels, 1), 1)))
        occupancy_logits = logits.view(batch_size, map_channels, map_resolution, map_resolution)
        occupancy_logits = occupancy_logits[:, 0]

        target = prepared["fp_proj_gt"]
        if occupancy_logits.shape[-2:] != target.shape[-2:]:
            occupancy_logits = F.interpolate(
                occupancy_logits.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)

        mask = prepared.get("fp_explored_gt")
        if mask is not None and mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(mask.unsqueeze(1), size=target.shape[-2:], mode="nearest").squeeze(1)

        if mask is not None:
            loss_map = F.binary_cross_entropy_with_logits(occupancy_logits, target, reduction="none")
            denom = mask.sum().clamp_min(1.0)
            map_loss = (loss_map * mask).sum() / denom
        else:
            map_loss = F.binary_cross_entropy_with_logits(occupancy_logits, target, reduction="mean")

        total_loss = map_loss

        if pose_estimator is not None and "pose_delta" in prepared:
            pose_estimator.train()
            pose_target = prepared["pose_delta"]
            pose_channels = getattr(controller, "map_channels", 2)
            prev_map = torch.zeros(batch_size, pose_channels, target.shape[-2], target.shape[-1], device=self.device)
            curr_map = prev_map.clone()
            curr_map[:, 0] = target
            if mask is not None and pose_channels > 1:
                curr_map[:, 1] = mask
            pose_pred = pose_estimator(prev_map, curr_map)
            pose_loss = F.mse_loss(pose_pred, pose_target)
            total_loss = total_loss + self.neural_slam_config.get("online_pose_weight", 1.0) * pose_loss

        try:
            total_loss.backward()
            clip = float(self.neural_slam_config.get("online_grad_clip", 5.0))
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.neural_slam_networks.parameters(), clip)
            self.neural_slam_online_optimizer.step()
        except Exception:
            # Prevent stale gradients on errors
            self.neural_slam_online_optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _format_neural_tuple(sample):
        if sample is None:
            return None
        if isinstance(sample, dict):
            keys = ("rgb_prev", "rgb_curr", "pose_delta", "fp_proj_gt", "fp_explored_gt")
            if all(k in sample for k in keys):
                return tuple(sample[k] for k in keys)
            inner = sample.get("neural_slam_training")
            if isinstance(inner, (list, tuple)) and len(inner) == 5:
                return tuple(inner)
        if isinstance(sample, (list, tuple)) and len(sample) == 5:
            return tuple(sample)
        return None

    def _prepare_online_neural_batch(self, samples):
        rgb_prev_list, rgb_curr_list = [], []
        pose_list, proj_list, explored_list = [], [], []

        for sample in samples:
            formatted = self._format_neural_tuple(sample)
            if formatted is None:
                continue
            rgb_prev, rgb_curr, pose_delta, proj_gt, explored_gt = formatted
            try:
                rgb_prev_list.append(self._normalise_online_rgb(rgb_prev))
                rgb_curr_list.append(self._normalise_online_rgb(rgb_curr))

                pose_tensor = torch.as_tensor(pose_delta, dtype=torch.float32).view(-1)
                if pose_tensor.numel() < 3:
                    pose_tensor = F.pad(pose_tensor, (0, 3 - pose_tensor.numel()))
                pose_list.append(pose_tensor[:3])

                proj_tensor = torch.as_tensor(proj_gt, dtype=torch.float32)
                if proj_tensor.dim() == 3:
                    proj_tensor = proj_tensor[0]
                proj_list.append(proj_tensor.clamp(0.0, 1.0))

                if explored_gt is None:
                    explored_tensor = torch.ones_like(proj_tensor)
                else:
                    explored_tensor = torch.as_tensor(explored_gt, dtype=torch.float32)
                    if explored_tensor.dim() == 3:
                        explored_tensor = explored_tensor[0]
                    explored_tensor = explored_tensor.clamp(0.0, 1.0)
                explored_list.append(explored_tensor)
            except Exception:
                continue

        if not rgb_curr_list:
            return None

        device = self.device
        return {
            "rgb_prev": torch.stack(rgb_prev_list).to(device),
            "rgb_curr": torch.stack(rgb_curr_list).to(device),
            "pose_delta": torch.stack(pose_list).to(device),
            "fp_proj_gt": torch.stack(proj_list).to(device),
            "fp_explored_gt": torch.stack(explored_list).to(device),
        }

    def _normalise_online_rgb(self, value):
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)
        if tensor.dim() == 3 and tensor.size(0) != 3 and tensor.size(-1) == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.dim() != 3:
            raise ValueError("RGB tensor must have 3 dimensions")
        if tensor.numel() > 0 and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        tensor = F.interpolate(tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
        mean = self._rgb_norm_mean.to(tensor.device)
        std = self._rgb_norm_std.to(tensor.device)
        return (tensor - mean) / std

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            logits, value = self.forward(obs)
            if self.use_transformer:
                logits = logits[:, -1]
            else:
                logits = logits

            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            probs = F.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            valid = torch.isfinite(probs).all() and torch.all(probs >= 0) and torch.all(probs.sum(dim=-1) > 0)

            if not valid:
                warnings.warn("Encountered invalid action probabilities; falling back to uniform distribution.", RuntimeWarning)
                probs = torch.full_like(probs, 1.0 / probs.size(-1))

            dist = Categorical(probs)
            if deterministic:
                action = torch.argmax(probs).item()
            else:
                action = dist.sample().item()
        last_action = self.last_action
        self.last_action = action
        if self.use_transformer:
            return action, None, None, None, last_action, value.reshape(-1)[-1].item() if value is not None else None
        else:
            return action, self.lssg_hidden, self.gssg_hidden, self.policy_hidden, last_action, value.item() if value is not None else None

    def reset(self):
        self.last_action = -1
        self.rollout_buffers.clear()
        if self.use_transformer:
            self.obs_buffer.clear()
            self.action_buffer.clear()
        else:
            self.lssg_hidden = None
            self.gssg_hidden = None
            self.policy_hidden = None

        if hasattr(self.encoder, "reset_map_state"):
            self.encoder.reset_map_state()

        self._reset_neural_map_state()
        if getattr(self.encoder, "exploration_mode", None) == "neural":
            self._neural_mapper_needs_reset = True
            if self.neural_slam_networks:
                controller = self._get_neural_component("controller")
                memory_core = self._get_neural_component("memory_core")
                if controller is not None and hasattr(controller, "detach_state"):
                    controller.detach_state()
                if memory_core is not None and hasattr(memory_core, "reset"):
                    memory_core.reset(1, self.device)

    def _get_update_values(self):
        b = self.rollout_buffers.get(self.gamma)

        # Map-Feld konsistent machen
        if "map" not in b and "occ" in b:
            b["map"] = b["occ"]
            maps = b.get("occ", [])
            empty_frac = sum(1 for m in maps if (m is None or (hasattr(m, "__len__") and len(m) == 0))) / max(1, len(maps))
            if empty_frac > 0:
                print(f"[WARN] {empty_frac:.1%} der Maps im Batch sind leer/None")
        if "map" not in b:
            b["map"] = []

        batch = {
            "rgb": b["rgb"],
            "lssg": b["lssg"],
            "gssg": b["gssg"],
            "map": b["map"],
            "map_policy": b.get("map_policy", []),
            "actions": b["actions"],
            "returns": b["returns"],
            "last_actions": b["last_actions"],
            "agent_pos": b.get("agent_positions", []),  # <-- direkt als agent_pos
            "map_index": b.get("map_index", []),
            # WICHTIG: initiale Hidden-States für korrektes TBPTT
            "initial_lssg_hidden": b.get("initial_lssg_hidden", None),
            "initial_gssg_hidden": b.get("initial_gssg_hidden", None),
            "initial_policy_hidden": b.get("initial_policy_hidden", None),
        }

        # Tensors & dtypes
        if not isinstance(batch["actions"], torch.Tensor):
            batch["actions"] = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        if not isinstance(batch["last_actions"], torch.Tensor):
            batch["last_actions"] = torch.tensor(batch["last_actions"], dtype=torch.long, device=self.device)
        if not isinstance(batch["returns"], torch.Tensor):
            batch["returns"] = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device)

        # [T] -> [1, T]
        for k in ["actions", "returns", "last_actions"]:
            if batch[k].dim() == 1:
                batch[k] = batch[k].unsqueeze(0)

        # Listen zu [B=1, T] packen (Encoder erwartet [B,T,...])
        for k in ["rgb", "lssg", "gssg", "map", "agent_pos", "map_index"]:
            if isinstance(batch[k], list):
                batch[k] = [batch[k]]

        return batch

    def forward_update(self, batch):
        state_seq, _, _ = self.encoder.forward_seq(batch, batch["last_actions"])
        init_pol_h = batch.get("initial_policy_hidden")
        logits, value, _ = self.policy(state_seq, hidden=init_pol_h)

        if value is None:
            return logits
        else:
            value = value.squeeze(0)
            return logits, value

    def load_weights(self, encoder_path=None, policy_path=None, model_path=None, neural_slam_path=None, device="cpu"):
        print("===== STARTING LOADING WEIGHTS =====")

        if model_path is not None:
            payload = torch.load(model_path, map_location=device)
            state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
            self.load_state_dict(state_dict, strict=True)
            self.to(device)
            print("[INFO] Full agent checkpoint loaded")
            return

        # ENCODER (IL)
        if encoder_path is not None:
            enc_payload = torch.load(encoder_path, map_location=device)
            enc_sd = enc_payload["state_dict"] if isinstance(enc_payload, dict) and "state_dict" in enc_payload else enc_payload
            enc_meta = enc_payload.get("meta", {})

            # (deine Checks bleiben wie gehabt …)

            strict_flag = True

            if not getattr(self.encoder, "use_map", False):
                map_keys = [k for k in enc_sd.keys() if k.startswith("map_encoder.")]
                if map_keys:
                    enc_sd = {k: v for k, v in enc_sd.items() if not k.startswith("map_encoder.")}
                    print("[ENC][INFO] Dropping map encoder weights from checkpoint " "because the RL agent does not use a map.")

            # >>> Backward-Compat: env_proj ggf. nachrüsten
            needs_env_proj = hasattr(self.encoder, "map_encoder") and hasattr(self.encoder.map_encoder, "env_proj")
            has_env_proj_in_ckpt = any(k.endswith("map_encoder.env_proj.weight") for k in enc_sd.keys())

            if needs_env_proj and not has_env_proj_in_ckpt:
                # fehlendes Gewicht mit der aktuellen (random/kaiming) Init des Modells füllen
                with torch.no_grad():
                    w = self.encoder.map_encoder.env_proj.weight.data.clone()
                enc_sd["map_encoder.env_proj.weight"] = w
                print("[ENC][BC] Checkpoint ohne 'env_proj'; fülle Gewicht aus Modell-Init und lade strict.")

            # >>> Backward-Compat: readout_proj ggf. nachrüsten
            needs_readout_proj = (
                hasattr(self.encoder, "map_encoder")
                and hasattr(self.encoder.map_encoder, "readout_proj")
                and isinstance(getattr(self.encoder.map_encoder, "readout_proj", None), torch.nn.Linear)
            )
            readout_weight_key = "map_encoder.readout_proj.weight"
            readout_bias_key = "map_encoder.readout_proj.bias"

            if needs_readout_proj and readout_weight_key not in enc_sd:
                with torch.no_grad():
                    enc_sd[readout_weight_key] = self.encoder.map_encoder.readout_proj.weight.data.clone()
                print("[ENC][BC] Checkpoint ohne 'readout_proj.weight'; übernehme aktuelle Initialisierung.")

            if needs_readout_proj and readout_bias_key not in enc_sd:
                with torch.no_grad():
                    enc_sd[readout_bias_key] = self.encoder.map_encoder.readout_proj.bias.data.clone()
                print("[ENC][BC] Checkpoint ohne 'readout_proj.bias'; übernehme aktuelle Initialisierung.")

            # >>> Backward-Compat: persistente Speicher (neural map) ggf. nachrüsten
            needs_memory_buffers = (
                needs_env_proj and hasattr(self.encoder.map_encoder, "memory") and hasattr(self.encoder.map_encoder, "memory_mask")
            )

            if needs_memory_buffers:
                mem_key = "map_encoder.memory"
                mask_key = "map_encoder.memory_mask"

                def _resize_neural_buffer(current_tensor, loaded_tensor, key_name):
                    if not isinstance(current_tensor, torch.Tensor):
                        return loaded_tensor
                    if not isinstance(loaded_tensor, torch.Tensor):
                        return loaded_tensor
                    if current_tensor.shape == loaded_tensor.shape:
                        return loaded_tensor

                    if current_tensor.dim() != loaded_tensor.dim():
                        resized = current_tensor.clone()
                    else:
                        resized = current_tensor.clone()
                        overlap_slices = tuple(slice(0, min(c, l)) for c, l in zip(current_tensor.shape, loaded_tensor.shape))
                        resized[overlap_slices] = loaded_tensor[overlap_slices].to(resized.device)

                    print(
                        "[ENC][BC] Resized neural map buffer '{}' from {} to {}.".format(
                            key_name, tuple(loaded_tensor.shape), tuple(current_tensor.shape)
                        )
                    )
                    return resized

                if mem_key not in enc_sd:
                    with torch.no_grad():
                        enc_sd[mem_key] = self.encoder.map_encoder.memory.data.clone()
                    print("[ENC][BC] Checkpoint ohne 'memory'; übernehme aktuellen Buffer.")
                else:
                    enc_sd[mem_key] = _resize_neural_buffer(self.encoder.map_encoder.memory.data, enc_sd[mem_key], mem_key)

                if mask_key not in enc_sd:
                    with torch.no_grad():
                        enc_sd[mask_key] = self.encoder.map_encoder.memory_mask.data.clone()
                    print("[ENC][BC] Checkpoint ohne 'memory_mask'; übernehme aktuellen Buffer.")
                else:
                    enc_sd[mask_key] = _resize_neural_buffer(self.encoder.map_encoder.memory_mask.data, enc_sd[mask_key], mask_key)

            adapter_weight_key = "map_encoder._readout_input_adapter.weight"
            adapter_bias_key = "map_encoder._readout_input_adapter.bias"
            if (
                self.encoder.use_map
                and getattr(self.encoder, "map_encoder", None) is not None
                and getattr(self.encoder, "exploration_mode", "raster") == "neural"
                and (adapter_weight_key in enc_sd or adapter_bias_key in enc_sd)
            ):
                adapter_module = getattr(self.encoder.map_encoder, "_readout_input_adapter", None)

                if not isinstance(adapter_module, nn.Linear):
                    in_features = None
                    out_features = None

                    weight_tensor = enc_sd.get(adapter_weight_key)
                    bias_tensor = enc_sd.get(adapter_bias_key)

                    if isinstance(weight_tensor, torch.Tensor):
                        out_features, in_features = weight_tensor.shape[0], weight_tensor.shape[1]

                    if isinstance(bias_tensor, torch.Tensor):
                        out_features = bias_tensor.shape[0]

                    if in_features is None:
                        in_features = getattr(getattr(self.encoder.map_encoder, "readout_proj", None), "in_features", None)

                    if out_features is None:
                        out_features = getattr(getattr(self.encoder.map_encoder, "readout_proj", None), "in_features", None)

                    if in_features is not None and out_features is not None:
                        adapter_module = nn.Linear(in_features, out_features)
                        adapter_module.to(self.encoder.map_encoder.readout_proj.weight.device)
                        self.encoder.map_encoder._readout_input_adapter = adapter_module
                        print("[ENC][BC] Initialisiere 'readout_input_adapter' für Neural-Map-Checkpoint.")

            # >>> Backward-Compat: ältere Raster-Encoder (ohne Agent-Channel)
            conv_key = "map_encoder.encoder.0.weight"
            if (
                self.encoder.use_map
                and hasattr(self.encoder, "map_encoder")
                and hasattr(self.encoder.map_encoder, "encoder")
                and conv_key in enc_sd
            ):
                current_weight = self.encoder.map_encoder.encoder[0].weight.data
                loaded_weight = enc_sd[conv_key]

                if (
                    isinstance(current_weight, torch.Tensor)
                    and isinstance(loaded_weight, torch.Tensor)
                    and current_weight.dim() == 4
                    and loaded_weight.dim() == 4
                    and current_weight.shape[0] == loaded_weight.shape[0]
                    and current_weight.shape[2:] == loaded_weight.shape[2:]
                    and loaded_weight.shape[1] == current_weight.shape[1] - 1
                ):
                    # Der neue Agent-Channel (4. Kanal) existiert im aktuellen Modell,
                    # aber nicht im Checkpoint. Wir behalten die gewürfelte Init für
                    # den neuen Kanal und kopieren die restlichen Gewichte aus dem Checkpoint.
                    patched_weight = current_weight.clone()
                    patched_weight[:, : loaded_weight.shape[1], :, :] = loaded_weight
                    enc_sd[conv_key] = patched_weight
                    print("[ENC][BC] Checkpoint ohne Agent-Channel; ergänze vierten Kanal aus aktueller Init.")

            # >>> Backward-Compat: Map-Dimensionswechsel (z.B. 64 -> 128)
            fc_weight_key = "map_encoder.fc.weight"
            fc_bias_key = "map_encoder.fc.bias"
            if self.encoder.use_map and hasattr(self.encoder, "map_encoder") and hasattr(self.encoder.map_encoder, "fc"):
                current_weight = getattr(self.encoder.map_encoder.fc, "weight", None)
                current_bias = getattr(self.encoder.map_encoder.fc, "bias", None)
                loaded_weight = enc_sd.get(fc_weight_key)
                loaded_bias = enc_sd.get(fc_bias_key)

                def _resize_param(current_param, loaded_param):
                    if not isinstance(current_param, torch.Tensor) or not isinstance(loaded_param, torch.Tensor):
                        return loaded_param

                    if current_param.shape == loaded_param.shape:
                        return loaded_param

                    if current_param.dim() != loaded_param.dim():
                        return loaded_param

                    # We always keep the current parameter as base and copy/trim the overlapping region.
                    resized = current_param.clone()

                    # Build slice objects for overlapping dimensions
                    overlap_slices = tuple(slice(0, min(c, l)) for c, l in zip(current_param.shape, loaded_param.shape))

                    if any(c == 0 for c in current_param.shape):
                        return loaded_param

                    resized[overlap_slices] = loaded_param[overlap_slices]
                    return resized

                if loaded_weight is not None and isinstance(current_weight, torch.Tensor):
                    resized_weight = _resize_param(current_weight.data, loaded_weight)
                    if resized_weight.shape == current_weight.shape and resized_weight is not loaded_weight:
                        enc_sd[fc_weight_key] = resized_weight
                        print("[ENC][BC] Angepasste map_encoder.fc.weight auf neue Dimensionen.")

                if loaded_bias is not None and isinstance(current_bias, torch.Tensor):
                    resized_bias = _resize_param(current_bias.data, loaded_bias)
                    if resized_bias.shape == current_bias.shape and resized_bias is not loaded_bias:
                        enc_sd[fc_bias_key] = resized_bias
                        print("[ENC][BC] Angepasste map_encoder.fc.bias auf neue Dimensionen.")

            self.encoder.load_state_dict(enc_sd, strict=strict_flag)
            self.encoder.to(device)

            print(f"[INFO] Encoder weights loaded (strict={strict_flag})")
        else:
            print("[WARN] No feature encoder loaded")

        # POLICY (IL → RL)
        if policy_path is not None and os.path.exists(policy_path):
            pol_payload = torch.load(policy_path, map_location=device)
            pol_sd = pol_payload["state_dict"] if isinstance(pol_payload, dict) and "state_dict" in pol_payload else pol_payload
            pol_meta = pol_payload.get("meta", {})

            def _soft_warn(tag, il_val, rl_val):
                if il_val is not None and il_val != rl_val:
                    print(f"[POL][WARN] {tag} mismatch: IL={il_val} RL={rl_val}")

            if pol_meta:
                _soft_warn("use_transformer", pol_meta.get("use_transformer"), self.policy.use_transformer)
                # Wenn RL Map nutzt, aber IL-Policy ohne Map trainiert wurde → harter Abbruch
                if self.encoder.use_map and (pol_meta.get("use_map", True) is False):
                    raise RuntimeError("[POL] Policy was trained WITHOUT map, but RL uses map.")
                _soft_warn("input_dim", pol_meta.get("input_dim"), self.policy.input_dim)
                _soft_warn("hidden_dim", pol_meta.get("hidden_dim"), self.policy.hidden_dim)
                _soft_warn("output_dim", pol_meta.get("output_dim"), self.num_actions)
                _soft_warn("exploration_mode", pol_meta.get("exploration_mode"), getattr(self, "exploration_mode", None))

            # IL-Policy has no Value-Head
            filtered_sd = {k: v for k, v in pol_sd.items() if not k.startswith("value_head.")}

            # value heaed stays random
            missing, unexpected = self.policy.load_state_dict(filtered_sd, strict=False)
            if missing:
                print(f"[POL][INFO] Missing keys (expected in RL): {missing}")
            if unexpected:
                print(f"[POL][INFO] Unexpected keys (ignored): {unexpected}")

            self.policy.to(device)
            print("[INFO] Policy weights (IL) loaded (value-head randomly initialized)")
        else:
            print("[WARN] No IL policy weights provided – policy head starts random")

        if self.is_neural_slam and neural_slam_path is not None:
            self._load_neural_slam_weights(neural_slam_path, device=device)

        print("===== FINISHED LOADING WEIGHTS =====")

    def _create_neural_slam_networks(self):
        """Instantiate Neural SLAM controller, memory core and pose estimator."""
        try:
            controller = NeuralSlamController()
            map_channels = getattr(controller, "map_channels", 2)

            networks = {
                "controller": controller,
                "memory_core": NeuralSlamMemoryCore(map_channels=map_channels),
                "pose_estimator": NeuralSlamPoseEstimator(map_channels=map_channels),
            }

            return networks
        except Exception as exc:
            print(f"[WARN] Failed to create Neural SLAM networks: {exc}")
            return {}

    def _resolve_neural_weight_path(self, base_path: Path, candidates: list[str]) -> Optional[Path]:
        for name in candidates:
            candidate = base_path / name
            if candidate.exists():
                return candidate
        return None

    def _load_neural_slam_weights(self, neural_slam_path, device="cpu"):
        if self.neural_slam_networks is None:
            print("[WARN] Neural SLAM networks not initialised; skipping warm start")
            return

        base = Path(neural_slam_path)
        if not base.exists():
            print(f"[WARN] Neural SLAM weights directory not found: {base}")
            return

        mapping = {
            "controller": ["controller.pth", "mapper.pth"],
            "memory_core": ["memory_core.pth", "memory.pth"],
            "pose_estimator": ["pose_estimator.pth", "pose.pth"],
        }

        controller_config = None

        for net_name, candidates in mapping.items():
            if net_name not in self.neural_slam_networks:
                continue

            module = self.neural_slam_networks[net_name]

            weight_path = self._resolve_neural_weight_path(base, candidates)
            if weight_path is None:
                print(f"[WARN] No weights found for Neural SLAM component '{net_name}' in {base}")
                continue

            state_dict = torch.load(weight_path, map_location=device)
            if net_name == "pose_estimator" and isinstance(state_dict, dict):
                keys = [k for k in state_dict.keys() if isinstance(k, str)]
                if keys and all(not k.startswith("pose_head.") for k in keys):
                    state_dict = {f"pose_head.{k}": v for k, v in state_dict.items()}
            try:
                module.load_state_dict(state_dict)
            except RuntimeError as exc:
                rebuilt = False
                if isinstance(module, NeuralSlamController):
                    inferred = self._infer_controller_config(state_dict)
                    if inferred is not None:
                        read_dim = int(inferred.get("read_dim", inferred.get("map_channels", 0)))
                        map_channels = int(inferred.get("map_channels", read_dim))
                        if read_dim != map_channels:
                            adapted = self._adapt_legacy_controller_state_dict(state_dict, map_channels)
                            if adapted is not None:
                                state_dict = adapted
                                inferred["read_dim"] = map_channels
                                print(
                                    "[INFO] Adapted legacy Neural SLAM controller weights from "
                                    f"read_dim={read_dim} to read_dim={map_channels}"
                                )
                            else:
                                raise exc
                        controller_config = inferred
                        module = NeuralSlamController(**inferred)
                        module.load_state_dict(state_dict)
                        rebuilt = True
                        print("[INFO] Reconstructed Neural SLAM controller with inferred configuration: " f"{inferred}")
                elif isinstance(module, NeuralSlamMemoryCore):
                    inferred = self._infer_memory_core_config(state_dict, controller_config, module)
                    if inferred is not None:
                        module = NeuralSlamMemoryCore(**inferred)
                        module.load_state_dict(state_dict, strict=False)
                        rebuilt = True
                        print("[INFO] Reconstructed Neural SLAM memory core with inferred configuration: " f"{inferred}")
                elif isinstance(module, NeuralSlamPoseEstimator):
                    inferred_channels = self._infer_pose_estimator_channels(state_dict)
                    if inferred_channels is not None:
                        module = NeuralSlamPoseEstimator(map_channels=inferred_channels)
                        module.load_state_dict(state_dict)
                        rebuilt = True
                        print("[INFO] Reconstructed Neural SLAM pose estimator with inferred map_channels=" f"{inferred_channels}")

                if not rebuilt:
                    raise exc

                self.neural_slam_networks[net_name] = module

            module.to(self.device)
            print(f"[INFO] Loaded Neural SLAM {net_name} weights from {weight_path}")

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        # Save encoder separately in the specified directory (falls benötigt)
        self.encoder.save_model(path)

        # Gather relevant configs parameters for filename
        rgb_dim = self.encoder.rgb_encoder.output_dim
        action_dim = self.encoder.action_emb.embedding.embedding_dim
        sg_dim = self.encoder.lssg_encoder.lstm.hidden_size if not self.use_transformer else self.encoder.lssg_encoder.output_dim
        policy_hidden = self.policy_hidden
        agent_name = self.get_agent_info().get("Agent Name", "Agent").replace(" ", "_")

        # Create filename including configs parameters
        filename = f"{agent_name}_{rgb_dim}_{action_dim}_{sg_dim}_{policy_hidden}_{self.use_transformer}.pth"
        full_path = os.path.join(path, filename)
        # Save model state dict
        torch.save(self.state_dict(), full_path)
        print(f"Saved model to {full_path}")

    def get_agent_info(self):
        """
        Return basic information about the agent.
        """
        pass

    @staticmethod
    def _map_version_to_mode(map_version) -> str:
        if map_version == "metric_semantic_v2":
            return "raster_v2"
        if map_version == "metric_semantic_v1":
            return "raster"
        else:
            return "neural"

    def _infer_controller_config(self, state_dict):
        try:
            hidden_size = int(state_dict["pose_encoder.0.weight"].shape[0])
            read_dim = int(state_dict["read_proj.weight"].shape[1])
            map_channels = int(state_dict["add_net.weight"].shape[0])
            decoder_out = int(state_dict["decoder.2.weight"].shape[0])
            if map_channels <= 0 or decoder_out % map_channels != 0:
                return None
            map_resolution = int(round(math.sqrt(decoder_out / map_channels)))
            if map_resolution <= 0:
                return None
            shift_elems = int(state_dict["shift_net.weight"].shape[0])
            shift_kernel_size = int(round(math.sqrt(shift_elems)))
            if shift_kernel_size * shift_kernel_size != shift_elems:
                return None
            rgb_in_channels = int(state_dict["rgb_encoder.0.weight"].shape[1])
            use_depth = rgb_in_channels > 3
            return {
                "map_channels": map_channels,
                "map_resolution": map_resolution,
                "hidden_size": hidden_size,
                "read_dim": read_dim,
                "shift_kernel_size": shift_kernel_size,
                "use_depth": use_depth,
            }
        except KeyError:
            return None

    def _infer_memory_core_config(self, state_dict, controller_config, module):
        map_resolution = None
        uniform = state_dict.get("uniform_weights")
        if uniform is not None:
            map_resolution = int(uniform.shape[-1])

        map_channels = None
        read_dim = None
        shift_kernel_size = None

        if controller_config is not None:
            map_channels = controller_config.get("map_channels")
            read_dim = controller_config.get("read_dim")
            shift_kernel_size = controller_config.get("shift_kernel_size")

        if map_channels is None:
            map_channels = getattr(module, "map_channels", None)
        if read_dim is None:
            read_dim = getattr(module, "read_dim", None)
        if shift_kernel_size is None:
            shift_kernel_size = getattr(module, "shift_kernel_size", 3)
        if map_resolution is None:
            map_resolution = getattr(module, "map_resolution", None)

        if None in (map_channels, map_resolution, read_dim):
            return None

        return {
            "map_channels": int(map_channels),
            "map_resolution": int(map_resolution),
            "read_dim": int(read_dim),
            "shift_kernel_size": int(shift_kernel_size),
        }

    @staticmethod
    def _infer_pose_estimator_channels(state_dict):
        if not isinstance(state_dict, dict):
            return None

        weight = state_dict.get("pose_head.net.0.weight")
        if weight is None:
            weight = state_dict.get("net.0.weight")
        if weight is None:
            return None

        in_channels = int(weight.shape[1])
        if in_channels % 2 != 0:
            return None

        return in_channels // 2

    @staticmethod
    def _adapt_legacy_controller_state_dict(state_dict, target_read_dim):
        required = ("read_proj.weight", "key_net.weight", "key_net.bias")
        if any(key not in state_dict for key in required):
            return None

        read_proj_weight = state_dict["read_proj.weight"]
        key_weight = state_dict["key_net.weight"]
        key_bias = state_dict["key_net.bias"]

        current_read_dim = read_proj_weight.shape[1]
        if current_read_dim < target_read_dim:
            return None

        def _reduce_tensor(tensor, dim):
            current = tensor.shape[dim]
            if current == target_read_dim:
                return tensor.clone()
            if current % target_read_dim == 0:
                group = current // target_read_dim
                shape = list(tensor.shape)
                shape.insert(dim + 1, group)
                shape[dim] = target_read_dim
                return tensor.reshape(shape).mean(dim=dim + 1).clone()
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(0, target_read_dim)
            return tensor[tuple(slices)].clone()

        adapted = dict(state_dict)
        adapted["read_proj.weight"] = _reduce_tensor(read_proj_weight, 1)
        adapted["key_net.weight"] = _reduce_tensor(key_weight, 0)
        adapted["key_net.bias"] = _reduce_tensor(key_bias, 0)
        return adapted
