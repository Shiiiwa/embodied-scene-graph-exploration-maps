import os

import torch
from torch import nn

from src.components.agents.abstract_agent import AbstractAgent
from src.components.encoders.feature_encoder import FeatureEncoder
from src.components.models.navigation_policy import NavigationPolicy
from src.imitation.models.neural_slam_il import (
    NeuralSlamController,
    NeuralSlamMemoryCore,
    NeuralSlamModule,
    NeuralSlamPoseEstimator,
)


class ImitationAgent(nn.Module):
    """
    Imitation learning agent that predicts the next action given multimodal state inputs.
    Consists of a feature encoder and a navigation policy network.
    Now supports Neural SLAM training alongside regular IL training.
    """

    def __init__(self, config, num_actions, device=None, mapping_path=None, **kwargs):
        """
        Combines a multimodal FeatureEncoder (RGB, previous actions, local/global scene graphs)
        with a NavigationPolicy to predict the next action. Optionally includes Neural SLAM networks.
        """
        super().__init__()

        navigation_config = config["navigation"]
        exploration_config = config["exploration"]

        self.use_map = exploration_config["active"]
        self.navigation_config = navigation_config
        self.exploration_config = exploration_config
        self.num_actions = num_actions
        self.use_transformer = navigation_config["use_transformer"]
        self.use_rgb = navigation_config.get("use_rgb", True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache the configured map dimensionality so encoders and policies stay aligned.
        self.map_dim = int(exploration_config.get("map_dim", navigation_config.get("map_dim", 64)))

        # Determine if we're using Neural SLAM
        self.is_neural_slam = exploration_config.get("map_version") == "neural_slam"

        exploration_mode =  AbstractAgent._map_version_to_mode(exploration_config["map_version"])
        print(f"[INFO]: Exploration mode: {exploration_mode}")

        if mapping_path is None:
            mapping_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "scene_graph_mappings",
                                        "default")

        # Feature encoder for multimodal input (RGB images, scene graphs, occupancy maps, etc.)
        self.encoder = FeatureEncoder(
            self.num_actions,
            rgb_dim=navigation_config["rgb_dim"],
            action_dim=navigation_config["action_dim"],
            sg_dim=navigation_config["sg_dim"],
            map_dim=self.map_dim,
            mapping_path=mapping_path,
            use_transformer=self.use_transformer,
            use_map=self.use_map,
            exploration_mode=exploration_mode,
            use_scene_graph=navigation_config.get("use_scene_graph", True),
            use_rgb=self.use_rgb,
        ).to(self.device)

        # Define the input dimension for the policy network based on encoder output
        input_dim = (
            navigation_config["rgb_dim"]
            + navigation_config["action_dim"]
            + 2 * navigation_config["sg_dim"]  # Local and global scene graphs
        )
        if self.use_map:
            print(f"Adding map-dim {self.map_dim} to input-dim {input_dim}")
            input_dim += self.map_dim

        # Policy network that predicts action logits from encoded state representation
        self.policy = NavigationPolicy(
            input_dim=input_dim,
            hidden_dim=navigation_config["policy_hidden"],
            output_dim=self.num_actions,
            use_transformer=self.use_transformer,
            value_head=self.is_neural_slam,
        ).to(self.device)

        # Neural SLAM networks (only if using neural_slam map_version)
        self.neural_slam_networks = None
        self._neural_slam_wrappers = {}
        if self.is_neural_slam:
            print("[INFO]: Initializing Neural SLAM networks")
            networks = self._create_neural_slam_networks()
            self.neural_slam_networks = nn.ModuleDict(networks)
            # Move networks to same device as main agent
            for network in self.neural_slam_networks.values():
                if hasattr(network, "to"):
                    network.to(self.device)
            # Wrappers expose legacy controller/pose estimator handles.
            module = None
            if hasattr(self.neural_slam_networks, "__contains__") and "module" in self.neural_slam_networks:
                module = self.neural_slam_networks["module"]
            elif isinstance(networks, dict):
                module = networks.get("module")

            if module is not None:
                self._neural_slam_wrappers = {
                    "controller": NeuralSlamController(module=module),
                    "pose_estimator": NeuralSlamPoseEstimator(module=module),
                }
            else:
                self._neural_slam_wrappers = {}

    def forward(self, x_batch, last_actions, neural_slam_data=None):
        """
        Runs a forward pass through encoder and policy.

        Args:
            x_batch: Standard multimodal input batch
            last_actions: Previous actions
            neural_slam_data: Optional dict with Neural SLAM training data

        Returns:
            logits: Action predictions
            neural_slam_outputs: Optional dict with Neural SLAM network outputs (for training)
        """
        if not self.use_map and "map" in x_batch:
            # modify map and fill with None
            x_batch["map"] = [[None for _ in seq] for seq in x_batch["map"]]

        if isinstance(last_actions, torch.Tensor):
            last_actions = last_actions.detach()

        neural_slam_outputs = None
        neural_map_read = None
        if self.is_neural_slam and neural_slam_data is not None and self.neural_slam_networks is not None:
            batch_size = len(x_batch.get("rgb", [])) if isinstance(x_batch.get("rgb"), list) else None
            neural_slam_outputs = self._forward_neural_slam(neural_slam_data, batch_size=batch_size)
            if neural_slam_outputs is not None:
                neural_map_read = neural_slam_outputs.get("read_vector")

        # Forward pass through encoder
        state_seq, _, _ = self.encoder.forward_seq(
            x_batch,
            last_actions,
            neural_map_read=neural_map_read,
        )

        # Forward pass through policy network
        if self.use_transformer:
            gssg_mask = x_batch["gssg_mask"]
            if not isinstance(gssg_mask, torch.Tensor):
                gssg_mask = torch.tensor(gssg_mask, device=self.device)
            gssg_mask = gssg_mask.bool()
            pad_mask = ~gssg_mask
            logits, values, _ = self.policy(state_seq, pad_mask=pad_mask)
        else:
            logits, values, _ = self.policy(state_seq)

        return {
            "logits": logits,
            "value": values,
            "neural_slam": neural_slam_outputs,
        }

    def _forward_neural_slam(self, neural_slam_data, batch_size=None):
        """Run the Neural SLAM controller, memory, and pose estimator for one batch."""
        if not getattr(self, "is_neural_slam", False) or self.neural_slam_networks is None or not neural_slam_data:
            return None

        device = self.device
        controller = self._get_neural_slam_component("controller")
        memory_core = self._get_neural_slam_component("memory_core")
        poser = self._get_neural_slam_component("pose_estimator")
        out = {}

        rgb = neural_slam_data.get("rgb_current", None)
        prev_map = neural_slam_data.get("ego_map_prev", None)
        pose_curr = neural_slam_data.get("pose_curr_sensor", None)

        # Basic sanity checks and normalisation for RGB input
        if rgb is None or controller is None or memory_core is None:
            return None

        if isinstance(rgb, torch.Tensor):
            rgb = rgb.to(device)
        else:
            rgb = torch.as_tensor(rgb, device=device)

        rgb = rgb.float()
        if rgb.min() >= 0.0 and rgb.max() > 1.0:
            rgb = rgb / 255.0

        if isinstance(pose_curr, torch.Tensor):
            pose_curr = pose_curr.to(device).float()
        elif pose_curr is not None:
            pose_curr = torch.as_tensor(pose_curr, dtype=torch.float32, device=device)

        # Controller forward pass (produces map logits and addressing parameters)
        try:
            batch = rgb.size(0)
            if hasattr(controller, "reset_state"):
                controller.reset_state(batch, device)
            if hasattr(memory_core, "reset"):
                memory_core.reset(batch, device)

            if prev_map is not None:
                if not isinstance(prev_map, torch.Tensor):
                    prev_map = torch.as_tensor(prev_map)
                prev_map = prev_map.to(device).float()
                if prev_map.ndim == 4 and prev_map.shape[1] != 2 and prev_map.shape[-1] == 2:
                    prev_map = prev_map.permute(0, 3, 1, 2).contiguous()
                memory_core.load_external_map(prev_map)

            prev_read = memory_core.get_last_read(batch, device)
            controller_out = controller(rgb, pose_curr, prev_read)
            ego_map_pred = controller_out.get("map_logits")
            head_params = controller_out.get("head_params", {})
            exploration_mask = controller_out.get("exploration_mask_logits")

            if head_params is None:
                head_params = {}
            else:
                head_params = {k: v for k, v in head_params.items()}

            if ego_map_pred is not None:
                head_params["add_patch_logits"] = ego_map_pred
                erase_vec = head_params.get("erase")
                if erase_vec is not None and isinstance(erase_vec, torch.Tensor) and ego_map_pred.dim() >= 3:
                    erase_patch = erase_vec.view(erase_vec.size(0), erase_vec.size(1), 1, 1)
                    erase_patch = erase_patch.expand(-1, -1, ego_map_pred.size(-2), ego_map_pred.size(-1))
                    head_params["erase_patch_logits"] = erase_patch
                out["map_logits_raw"] = ego_map_pred
            if exploration_mask is not None:
                out["exploration_mask_logits"] = exploration_mask

            memory_out = memory_core(head_params)
            read_vec = memory_out.get("read_vector")
            updated_map = memory_out.get("updated_map")
            if updated_map is not None:
                out["ego_map_current"] = updated_map
            if "weights" in memory_out:
                out["address_weights"] = memory_out["weights"]
            if read_vec is not None:
                if batch_size is not None and isinstance(neural_slam_data.get("batch_indices"), torch.Tensor):
                    idx = neural_slam_data["batch_indices"].long().to(read_vec.device)
                    full = torch.zeros(batch_size, read_vec.size(-1), device=read_vec.device, dtype=read_vec.dtype)
                    full = full.index_copy(0, idx, read_vec)
                    out["read_vector"] = full
                else:
                    out["read_vector"] = read_vec
            if head_params:
                out["head_params"] = head_params
        except Exception as e:
            if getattr(self, "_log_slam_shapes", False):
                print(f"[SLAM][WARN] Controller forward failed: {e}")
            return out if out else None

        # Optional pose-estimator pass
        if poser is not None and prev_map is not None and "ego_map_current" in out:
            try:
                prev_for_pose = prev_map
                if not isinstance(prev_for_pose, torch.Tensor):
                    prev_for_pose = torch.as_tensor(prev_for_pose)
                prev_for_pose = prev_for_pose.to(device).float()
                if prev_for_pose.ndim == 4 and prev_for_pose.shape[1] != 2 and prev_for_pose.shape[-1] == 2:
                    prev_for_pose = prev_for_pose.permute(0, 3, 1, 2).contiguous()
                curr_map = out["ego_map_current"]
                curr_map = curr_map.to(device)
                bmin = min(prev_for_pose.size(0), curr_map.size(0))
                if bmin > 0:
                    pose_delta = poser(prev_for_pose[:bmin], curr_map[:bmin])
                    out["pose_correction"] = pose_delta
            except Exception as e:
                if getattr(self, "_log_slam_shapes", False):
                    print(f"[SLAM][WARN] Pose estimator forward failed: {e}")

        # Optional shape logging for debugging
        if getattr(self, "_log_slam_shapes", False):
            print(
                "[SLAM] shapes:",
                f"rgb={tuple(rgb.shape) if rgb is not None else None},",
                f"prev={tuple(prev_map.shape) if prev_map is not None else None},",
                f"ego_map_current={tuple(out['ego_map_current'].shape) if 'ego_map_current' in out else None},",
                f"pose_correction={tuple(out['pose_correction'].shape) if 'pose_correction' in out else None}"
            )

        return out if out else None

    def _get_neural_slam_component(self, name):
        """Safely fetch a Neural SLAM sub-module if it is available."""
        if not getattr(self, "is_neural_slam", False):
            return None

        if hasattr(self, "_neural_slam_wrappers") and name in self._neural_slam_wrappers:
            return self._neural_slam_wrappers[name]

        networks = getattr(self, "neural_slam_networks", None)
        if networks is None:
            return None

        if isinstance(networks, dict):
            return networks.get(name)

        if hasattr(networks, "__contains__") and name in networks:
            return networks[name]

        return None

    def compute_neural_slam_loss(self, neural_slam_outputs, neural_slam_targets):
        """Compute the auxiliary Neural SLAM losses for mapper, memory, and pose heads."""
        if not self.is_neural_slam or neural_slam_outputs is None:
            return {}

        losses = {}

        # Mapper / controller loss (binary cross entropy with logits)
        occupancy_pred = None
        if isinstance(neural_slam_outputs, dict):
            occupancy_pred = (
                neural_slam_outputs.get("occupancy_logits")
                or neural_slam_outputs.get("fp_proj_logits")
                or neural_slam_outputs.get("ego_map_current")
            )

        occupancy_target = (
            neural_slam_targets.get("occupancy_target")
            or neural_slam_targets.get("ego_map_gt")
        )

        if occupancy_pred is not None and occupancy_target is not None:
            pred = occupancy_pred
            tgt = occupancy_target.to(pred.device).float()
            if pred.min().item() >= 0.0 and pred.max().item() <= 1.0:
                eps = 1e-6
                p = pred.clamp(eps, 1.0 - eps)
                pred = torch.log(p / (1.0 - p))

            mask = neural_slam_targets.get("exploration_mask")
            if mask is not None:
                mask = mask.to(pred.device).float()
                if mask.dim() < pred.dim():
                    expand_dims = pred.dim() - mask.dim()
                    for _ in range(expand_dims):
                        mask = mask.unsqueeze(1)
                    mask = mask.expand_as(pred)
                loss_map = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, tgt, reduction="none"
                )
                denom = mask.sum().clamp_min(1.0)
                losses["mapper_loss"] = (loss_map * mask).sum() / denom
            else:
                losses["mapper_loss"] = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, tgt, reduction="mean"
                )

        head_params = neural_slam_outputs.get("head_params") if isinstance(neural_slam_outputs, dict) else None
        if head_params:
            reg_terms = []
            key = head_params.get("key")
            add = head_params.get("add")
            erase = head_params.get("erase")
            beta = head_params.get("beta")
            if key is not None:
                reg_terms.append(key.pow(2).mean())
            if add is not None:
                reg_terms.append(add.pow(2).mean())
            if erase is not None:
                reg_terms.append(torch.clamp(erase, min=0.0, max=1.0).pow(2).mean())
            if beta is not None:
                reg_terms.append(torch.nn.functional.softplus(beta).mean())
            if reg_terms:
                losses["controller_reg"] = 1e-3 * sum(reg_terms)

        # Pose head loss: mean squared error with yaw converted to radians
        pose_pred = None
        if isinstance(neural_slam_outputs, dict):
            pose_pred = neural_slam_outputs.get("pose_delta") or neural_slam_outputs.get("pose_correction")

        pose_target = neural_slam_targets.get("pose_delta_gt")
        if pose_pred is not None and pose_target is not None:
            pred = pose_pred
            tgt = pose_target.to(pred.device).float()

            mb = min(pred.size(0), tgt.size(0))
            if mb > 0:
                dx_dz_tgt = tgt[:mb, :2]
                dyaw_tgt_rad = tgt[:mb, 2] * (torch.pi / 180.0)

                dx_dz_pred = pred[:mb, :2]
                dyaw_pred_rad = pred[:mb, 2] * (torch.pi / 180.0)

                w_dx, w_dz, w_yaw = 1.0, 1.0, 0.5
                pose_loss = (w_dx * torch.nn.functional.mse_loss(dx_dz_pred[:, 0], dx_dz_tgt[:, 0], reduction="mean")
                             + w_dz * torch.nn.functional.mse_loss(dx_dz_pred[:, 1], dx_dz_tgt[:, 1], reduction="mean")
                             + w_yaw * torch.nn.functional.mse_loss(dyaw_pred_rad, dyaw_tgt_rad, reduction="mean"))
                losses["pose_loss"] = pose_loss

        return losses

    def save_model(self, path):
        """Persist the full agent (encoder, policy/value, memory modules)."""
        if path is None:
            raise ValueError("Path for saving model must not be None")

        suffix = os.path.splitext(path)[1]
        is_directory_like = os.path.isdir(path) or path.endswith(os.sep) or suffix == ""
        if is_directory_like:
            target_dir = path.rstrip(os.sep)
            if not target_dir:
                target_dir = "."
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, "imitation_agent_full.pth")
        else:
            target_dir = os.path.dirname(path) or "."
            os.makedirs(target_dir, exist_ok=True)
            file_path = path

        payload = {
            "state_dict": self.state_dict(),
            "meta": {
                "map_version": "neural_slam" if self.is_neural_slam else "standard",
                "num_actions": int(self.num_actions),
                "uses_transformer": bool(self.use_transformer),
                "map_dim": int(self.map_dim),
            },
        }
        torch.save(payload, file_path)
        print(f"[INFO]: Saved imitation agent checkpoint to {file_path}")

        target_dir = os.path.dirname(file_path) or "."

        # Preserve legacy artefacts so other training stages keep working without adjustments.
        self.encoder.save_model(target_dir)
        self.policy.save_model(target_dir)

        if self.is_neural_slam and self.neural_slam_networks:
            neural_slam_dir = os.path.join(target_dir, "neural_slam_networks")
            os.makedirs(neural_slam_dir, exist_ok=True)
            saved_states = {}
            for net_name, network in self.neural_slam_networks.items():
                if not hasattr(network, "state_dict"):
                    continue
                state_dict = network.state_dict()
                if not state_dict:
                    continue
                filename = f"{net_name}.pth"
                torch.save(state_dict, os.path.join(neural_slam_dir, filename))
                saved_states[net_name] = state_dict

            module_state = saved_states.get("module")
            if module_state:
                # Legacy compatibility: also export controller/pose heads separately.
                controller_state = {
                    k.split("controller.", 1)[1]: v
                    for k, v in module_state.items()
                    if k.startswith("controller.")
                }
                if controller_state:
                    torch.save(controller_state, os.path.join(neural_slam_dir, "controller.pth"))

                pose_state = {
                    k: v
                    for k, v in module_state.items()
                    if k.startswith("pose_head.")
                }
                if pose_state:
                    torch.save(pose_state, os.path.join(neural_slam_dir, "pose_estimator.pth"))
            print(f"[INFO]: Saved Neural SLAM component weights to {neural_slam_dir}")

    def load_weights(self, encoder_path, policy_path=None, neural_slam_path=None, device="cpu"):
        """
        Load model weights including optional Neural SLAM networks.

        Args:
            encoder_path: Path to feature encoder weights
            policy_path: Path to policy weights (optional)
            neural_slam_path: Path to Neural SLAM network weights (optional)
            device: Device to load weights on
        """
        if encoder_path and os.path.isfile(encoder_path):
            payload = torch.load(encoder_path, map_location=device)
            if isinstance(payload, dict) and "state_dict" in payload:
                state_dict = payload["state_dict"]
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[INFO] Missing keys while loading combined checkpoint: {missing}")
                if unexpected:
                    print(f"[INFO] Unexpected keys in combined checkpoint: {unexpected}")
                self.to(device)
                return

        self.encoder.load_weights(encoder_path, device=device)

        if policy_path and os.path.exists(policy_path):
            self.policy.load_weights(policy_path, device=device)

        if self.is_neural_slam and neural_slam_path and os.path.exists(neural_slam_path):
            self._load_neural_slam_weights(neural_slam_path, device)

        self.to(device)

    def _load_neural_slam_weights(self, neural_slam_path, device="cpu"):
        """Load Neural SLAM network weights."""
        if self.neural_slam_networks is None:
            print("[WARN]: Neural SLAM networks not initialized, cannot load weights")
            return

        for net_name, network in self.neural_slam_networks.items():
            if not hasattr(network, "load_state_dict"):
                continue

            candidate_files = [f"{net_name}.pth"]
            if net_name == "module":
                candidate_files = ["module.pth", "controller.pth", "pose_estimator.pth"]
            elif net_name == "memory_core":
                candidate_files = ["memory_core.pth"]

            state_dict = None
            chosen_path = None
            for filename in candidate_files:
                network_path = os.path.join(neural_slam_path, filename)
                if os.path.exists(network_path):
                    state_dict = torch.load(network_path, map_location=device)
                    chosen_path = network_path
                    break

            if state_dict is None and net_name == "module":
                controller_path = os.path.join(neural_slam_path, "controller.pth")
                pose_path = os.path.join(neural_slam_path, "pose_estimator.pth")
                if os.path.exists(controller_path):
                    base_state = torch.load(controller_path, map_location=device)
                    if isinstance(base_state, dict):
                        state_dict = {f"controller.{k}": v for k, v in base_state.items()}
                        chosen_path = controller_path
                        if os.path.exists(pose_path):
                            pose_state = torch.load(pose_path, map_location=device)
                            if isinstance(pose_state, dict):
                                state_dict.update({f"pose_head.{k}": v for k, v in pose_state.items()})

            if state_dict is None:
                print(
                    f"[WARN]: Neural SLAM {net_name} weights not found (looked for {', '.join(candidate_files)})"
                )
                continue

            if isinstance(state_dict, dict):
                network.load_state_dict(state_dict, strict=False)
            else:
                print(f"[WARN]: Unexpected state dict format for {net_name} at {chosen_path}")
                continue

            if hasattr(network, "to"):
                network.to(device)
            print(f"[INFO]: Loaded Neural SLAM {net_name} weights from {chosen_path}")

    def _create_neural_slam_networks(self):
        """Create Neural SLAM controller, memory and pose estimator modules."""
        map_resolution = int(getattr(self, "map_dim", 64))
        navigation_cfg = getattr(self, "navigation_config", {}) or {}
        exploration_cfg = getattr(self, "exploration_config", {}) or {}
        use_depth = bool(navigation_cfg.get("use_depth", exploration_cfg.get("use_depth", False)))

        map_channels = int(exploration_cfg.get("map_channels", 4))

        module = NeuralSlamModule(
            map_channels=map_channels,
            map_resolution=map_resolution,
            use_depth=use_depth,
            estimate_pose=True,
        ).to(self.device)
        memory_core = NeuralSlamMemoryCore(map_channels=map_channels, map_resolution=map_resolution).to(self.device)

        return {
            "module": module,
            "memory_core": memory_core,
        }


    def set_neural_slam_training_mode(self, training=True):
        """Set Neural SLAM networks to training or evaluation mode."""
        if self.is_neural_slam and self.neural_slam_networks is not None:
            for network in self.neural_slam_networks.values():
                if hasattr(network, 'train'):
                    network.train(training)
