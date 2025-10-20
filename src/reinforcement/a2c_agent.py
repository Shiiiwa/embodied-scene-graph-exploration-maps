import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.components.agents.abstract_agent import AbstractAgent

class A2CAgent(AbstractAgent):
    """Advantage Actor-Critic agent with shared encoder, policy, and value heads."""

    def __init__(self, env, navigation_config, agent_config, exploration_config, device=None):
        """Initialise the agent and cache the loss coefficients used during optimisation."""
        super().__init__(env, navigation_config, agent_config, exploration_config, device)
        self.value_coef = agent_config["value_coef"]
        self.entropy_coef = agent_config["entropy_coef"]

    def update(self, obs=None):
        batch = self._get_update_values()

        logits, values = self.forward_update(batch)
        values = values.view(-1)

        probs = F.softmax(logits, dim=-1)
        dist  = Categorical(probs=probs)

        actions = batch["actions"].to(self.device).view(-1)
        returns = batch["returns"].to(self.device).view(-1)

        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy().mean()

        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss  = F.mse_loss(values, returns)
        loss        = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        aux_losses = {}
        if getattr(self, "exploration_mode", "") == "neural":
            aux_losses = self._compute_neural_aux_losses(batch)
            if hasattr(self, "neural_slam_loss_weights"):
                mapper_weight = self.neural_slam_loss_weights.get("mapper", 0.0)
            else:
                mapper_weight = 0.0
            mapper_loss = aux_losses.get("mapper")
            if mapper_loss is not None and mapper_weight > 0:
                loss = loss + mapper_weight * mapper_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
        self.optimizer.step()

        self.reset()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            **{k + "_loss": float(v.item()) for k, v in aux_losses.items()},
        }

    def _compute_neural_aux_losses(self, batch):
        if not getattr(self, "is_neural_slam", False):
            return {}
        networks = getattr(self, "neural_slam_networks", None)
        if networks is None:
            return {}

        def _get_component(container, name):
            if container is None:
                return None
            if isinstance(container, dict):
                return container.get(name)
            if hasattr(container, "__getitem__"):
                try:
                    return container[name]
                except KeyError:
                    return None
            return None

        controller = _get_component(networks, "controller")
        memory_core = _get_component(networks, "memory_core")
        if controller is None or memory_core is None:
            return {}

        neural_data = batch.get("neural_slam")
        if not neural_data:
            return {}

        rgb_sequences = batch.get("rgb", [])
        rgb_seq = rgb_sequences[0] if isinstance(rgb_sequences, list) and rgb_sequences else []

        device = self.device
        controller.reset_state(batch_size=1, device=device)
        memory_core.reset(batch_size=1, device=device)

        map_losses = []

        def _rgb_to_tensor(entry):
            if isinstance(entry, torch.Tensor):
                tensor = entry.to(device=device, dtype=torch.float32)
            else:
                tensor = torch.as_tensor(entry, dtype=torch.float32, device=device)
            if tensor.dim() == 3:
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            return tensor

        def _pose_to_tensor(pose_dict):
            if not isinstance(pose_dict, dict):
                return None
            x = float(pose_dict.get("x", 0.0))
            z = float(pose_dict.get("z", 0.0))
            yaw = float(pose_dict.get("yaw", 0.0))
            return torch.tensor([[x, z, yaw]], dtype=torch.float32, device=device)

        def _map_gt_to_tensor(gt, target_hw):
            tensor = gt
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
            else:
                tensor = tensor.to(device=device, dtype=torch.float32)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.size(-2) != target_hw[0] or tensor.size(-1) != target_hw[1]:
                tensor = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
            return tensor

        for idx, training in enumerate(neural_data):
            if not training:
                continue
            if idx >= len(rgb_seq):
                break

            rgb_tensor = _rgb_to_tensor(rgb_seq[idx])
            pose_tensor = _pose_to_tensor(training.get("agent_world_pose"))
            read_vec = memory_core.get_last_read(batch_size=1, device=device)

            controller_out = controller(rgb_tensor, pose_tensor, read_vec)
            map_logits = controller_out.get("map_logits")
            head_params = controller_out.get("head_params", {}) or {}
            if map_logits is not None:
                head_params = dict(head_params)
                head_params["add_patch_logits"] = map_logits

            if head_params:
                memory_core(head_params)

            map_gt = training.get("ego_map_gt")
            if map_gt is None or map_logits is None:
                continue

            target_tensor = _map_gt_to_tensor(map_gt, map_logits.shape[-2:])
            map_losses.append(F.binary_cross_entropy_with_logits(map_logits, target_tensor))

        if not map_losses:
            return {}

        mapper_loss = torch.stack(map_losses).mean()
        return {"mapper": mapper_loss}

    def get_agent_info(self):
        """Return a dictionary describing the agent configuration."""
        return {
            "Agent Name": "A2C Agent",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
        }
