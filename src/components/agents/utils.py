"""Utility helpers for agent implementations."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import torch


def prepare_neural_map_update(agent, obs, env=None) -> Optional[Dict[str, Any]]:
    """Generate mapper logits and head parameters for neural exploration maps."""
    if not agent._neural_map_enabled():
        return None

    target_env = env if env is not None else getattr(agent, "env", None)
    if target_env is None or not hasattr(target_env, "set_mapper_outputs"):
        return None

    controller = agent._get_neural_component("controller")
    memory_core = agent._get_neural_component("memory_core")
    if controller is None or memory_core is None or obs is None:
        return None

    state = getattr(obs, "state", None)
    info = getattr(obs, "info", {}) or {}

    rgb = None
    if isinstance(state, (list, tuple)) and state:
        rgb = state[0]
    elif isinstance(state, dict):
        rgb = state.get("rgb")

    if rgb is None:
        event = info.get("event")
        if event is not None and hasattr(event, "frame"):
            rgb = event.frame

    if rgb is None:
        return None

    device = agent.device

    try:
        with torch.no_grad():
            if isinstance(rgb, torch.Tensor):
                rgb_tensor = rgb.to(device=device, dtype=torch.float32)
            else:
                rgb_tensor = torch.as_tensor(rgb, dtype=torch.float32, device=device)

            if rgb_tensor.dim() == 3:
                if rgb_tensor.shape[0] in (3, 4) and rgb_tensor.shape[-1] not in (3, 4):
                    rgb_tensor = rgb_tensor.unsqueeze(0)
                else:
                    rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0)
            elif rgb_tensor.dim() != 4:
                raise ValueError(f"[NeuralMap] Unexpected RGB tensor shape {tuple(rgb_tensor.shape)}")

            controller_impl = controller
            if hasattr(controller_impl, "module"):
                controller_impl = controller_impl.module
            if hasattr(controller_impl, "controller"):
                controller_impl = controller_impl.controller
            controller_use_depth = bool(getattr(controller_impl, "use_depth", False))

            if rgb_tensor.size(1) < 3:
                raise ValueError(
                    f"[NeuralMap] Expected RGB tensor with at least 3 channels, got shape {tuple(rgb_tensor.shape)}"
                )

            depth_tensor = None
            if rgb_tensor.size(1) >= 4:
                depth_tensor = rgb_tensor[:, 3:4, ...].contiguous()
            rgb_tensor = rgb_tensor[:, :3, ...].contiguous()

            if torch.isfinite(rgb_tensor).any() and rgb_tensor.max().item() > 1.0:
                rgb_tensor = rgb_tensor / 255.0

            if not controller_use_depth:
                depth_tensor = None
            elif depth_tensor is not None and depth_tensor.dtype != rgb_tensor.dtype:
                depth_tensor = depth_tensor.to(dtype=rgb_tensor.dtype)

            event = info.get("event")
            pose_tensor = None
            if event is not None:
                metadata = getattr(event, "metadata", {}) or {}
                agent_meta = metadata.get("agent") or {}
                position = agent_meta.get("position") or {}
                rotation = agent_meta.get("rotation") or {}
                pose_vals = [
                    float(position.get("x", 0.0)),
                    float(position.get("z", 0.0)),
                    float(rotation.get("y", 0.0)),
                ]
                pose_tensor = torch.tensor(pose_vals, dtype=torch.float32, device=device).view(1, 3)

            batch = rgb_tensor.size(0)
            if pose_tensor is None:
                pose_tensor = torch.zeros(batch, 3, device=device)
            elif pose_tensor.size(0) != batch:
                pose_tensor = pose_tensor.expand(batch, -1)

            if agent._neural_mapper_needs_reset:
                if hasattr(controller, "reset_state"):
                    controller.reset_state(batch, device)
                if hasattr(memory_core, "reset"):
                    memory_core.reset(batch, device)
                agent._neural_mapper_needs_reset = False
            elif hasattr(controller, "hidden"):
                hidden = getattr(controller, "hidden", None)
                if (hidden is None or hidden.size(0) != batch) and hasattr(controller, "reset_state"):
                    controller.reset_state(batch, device)

            read_vector = agent._neural_state.get("read_vector")
            if not isinstance(read_vector, torch.Tensor) or read_vector.size(0) != batch:
                read_vector = memory_core.get_last_read(batch, device)
            else:
                read_vector = read_vector.to(device)

            forward_params = inspect.signature(controller.forward).parameters
            param_names = [name for name in forward_params if name != "self"]
            controller_kwargs: Dict[str, Any] = {}
            controller_args = []

            def _assign(param_candidates, value):
                for cand in param_candidates:
                    if cand in forward_params:
                        controller_kwargs[cand] = value
                        return True
                return False

            if not _assign(("curr_rgb", "rgb", "observation"), rgb_tensor):
                if param_names:
                    controller_kwargs.setdefault(param_names[0], rgb_tensor)
                else:
                    controller_args.append(rgb_tensor)

            _assign(("prev_read", "read_vector"), read_vector)
            _assign(("pose", "agent_pose", "curr_pose"), pose_tensor)
            if depth_tensor is not None:
                _assign(("curr_depth", "depth", "depth_map"), depth_tensor)

            if controller_args and controller_kwargs:
                controller_out = controller(*controller_args, **controller_kwargs)
            elif controller_kwargs:
                controller_out = controller(**controller_kwargs)
            else:
                controller_out = controller(*controller_args)

            map_logits = controller_out.get("map_logits")
            if map_logits is None or not isinstance(map_logits, torch.Tensor):
                raise ValueError("[NeuralMap] Controller did not return 'map_logits' tensor")
            if map_logits.dim() != 4:
                raise ValueError(
                    f"[NeuralMap] Expected map_logits shape (B, C, H, W) but received {tuple(map_logits.shape)}"
                )
            if map_logits.size(0) != batch:
                raise ValueError(
                    f"[NeuralMap] Expected map_logits batch dimension {batch} but received {map_logits.size(0)}"
                )

            head_params = controller_out.get("head_params") or {}
            exploration_mask = controller_out.get("exploration_mask")
            pose_delta = controller_out.get("pose_delta")
            required_keys = ("key", "beta", "gate", "shift", "sharpen", "erase", "add")
            validated_params: Dict[str, torch.Tensor] = {}
            for key in required_keys:
                tensor = head_params.get(key)
                if tensor is None:
                    raise ValueError(f"[NeuralMap] Missing '{key}' in controller outputs")
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"[NeuralMap] Expected '{key}' to be a torch.Tensor, got {type(tensor).__name__}")
                if tensor.size(0) != batch:
                    raise ValueError(
                        f"[NeuralMap] Expected '{key}' batch dimension {batch} but received {tensor.size(0)}"
                    )
                validated_params[key] = tensor.to(device)

            memory_inputs: Dict[str, Any] = {k: v for k, v in validated_params.items()}
            memory_inputs["add_patch_logits"] = map_logits

            erase_vec = validated_params.get("erase")
            if erase_vec is not None and erase_vec.dim() == 2:
                erase_patch = erase_vec.view(batch, erase_vec.size(1), 1, 1)
                erase_patch = erase_patch.expand(-1, -1, map_logits.size(-2), map_logits.size(-1))
                memory_inputs["erase_patch_logits"] = erase_patch
            elif erase_vec is not None and erase_vec.dim() not in (3, 4):
                raise ValueError(
                    f"[NeuralMap] Unexpected 'erase' tensor shape {tuple(erase_vec.shape)} for neural mapper"
                )

            map_state = getattr(memory_core, "map_state", None)
            if isinstance(map_state, torch.Tensor) and map_state.numel() > 0:
                if map_state.device != device:
                    map_state = map_state.to(device)
                if torch.count_nonzero(map_state).item() == 0:
                    sanitized_map = map_state.clone()
                    sanitized_map.view(sanitized_map.size(0), -1)[:, 0] = 1e-6
                    memory_core.load_external_map(sanitized_map)

            memory_out = memory_core(memory_inputs)
            read_vec_new = memory_out.get("read_vector")
            if isinstance(read_vec_new, torch.Tensor):
                agent._neural_state["read_vector"] = read_vec_new.detach()
            updated_map = memory_out.get("updated_map")
            if isinstance(updated_map, torch.Tensor):
                agent._neural_state["map"] = updated_map.detach()

            if hasattr(controller, "detach_state"):
                controller.detach_state()

            mapper_cpu = map_logits.detach().cpu()
            mask_cpu = exploration_mask.detach().cpu() if isinstance(exploration_mask, torch.Tensor) else None
            pose_delta_cpu = pose_delta.detach().cpu() if isinstance(pose_delta, torch.Tensor) else None
            head_params_cpu: Dict[str, Any] = {}
            for key, value in memory_inputs.items():
                if isinstance(value, torch.Tensor):
                    head_params_cpu[key] = value.detach().cpu()
                else:
                    head_params_cpu[key] = value

            target_env.set_mapper_outputs(
                mapper_logits=mapper_cpu,
                head_params=head_params_cpu,
                exploration_mask=mask_cpu,
                pose_delta=pose_delta_cpu,
                prev_event=event,
            )

            result: Dict[str, Any] = {"mapper_logits": mapper_cpu, "head_params": head_params_cpu}
            if mask_cpu is not None:
                result["exploration_mask"] = mask_cpu
            if pose_delta_cpu is not None:
                result["pose_delta"] = pose_delta_cpu

            return result
    except Exception:
        agent._neural_mapper_needs_reset = True
        agent._reset_neural_map_state()
        raise

