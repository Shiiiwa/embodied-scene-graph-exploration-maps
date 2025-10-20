"""Differentiable Neural SLAM modules used during IL pre-training.

This module provides an AI2-THOR-oriented implementation of Neural SLAM by
bundling the controller, egocentric map decoder and pose head inside
:class:`NeuralSlamModule`.  Thin wrappers keep the legacy
``NeuralSlamController`` / ``NeuralSlamMemoryCore`` /
``NeuralSlamPoseEstimator`` interfaces stable so the rest of the code base
continues to work without changes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _circular_convolution(weights: torch.Tensor, shift_kernel: torch.Tensor) -> torch.Tensor:
    """Apply batched circular convolution used for differentiable shifting."""

    b, h, w = weights.shape
    kh, kw = shift_kernel.shape[-2:]
    pad_h = kh // 2
    pad_w = kw // 2
    result = torch.zeros_like(weights)

    for dh in range(kh):
        for dw in range(kw):
            coeff = shift_kernel[:, dh, dw]
            if torch.allclose(coeff, torch.zeros_like(coeff)):
                continue
            roll_h = dh - pad_h
            roll_w = dw - pad_w
            result = result + coeff.view(b, 1, 1) * torch.roll(weights, shifts=(roll_h, roll_w), dims=(1, 2))
    return result


@dataclass
class NeuralSlamState:
    map: torch.Tensor
    weights: torch.Tensor
    read_vector: torch.Tensor


class _PoseHead(nn.Module):
    """Pose head used by the AI2-THOR Neural SLAM stack."""

    def __init__(self, map_channels: int = 4) -> None:
        super().__init__()
        in_channels = map_channels * 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

    def forward(self, prev_map: torch.Tensor, updated_map: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([prev_map, updated_map], dim=1)
        return self.net(stacked)


class _ControllerImpl(nn.Module):
    """Controller with ResNet-18 encoder and transposed-convolution decoders."""

    def __init__(
        self,
        map_channels: int = 4,
        map_resolution: int = 64,
        hidden_size: int = 512,
        read_dim: Optional[int] = None,
        shift_kernel_size: int = 3,
        use_depth: bool = False,
    ) -> None:
        super().__init__()
        self.map_channels = map_channels
        self.map_resolution = map_resolution
        self.hidden_size = hidden_size
        self.use_depth = bool(use_depth)
        if read_dim is None:
            read_dim = map_channels
        if int(read_dim) != int(map_channels):
            raise ValueError(
                "NeuralSlam controller requires read_dim to match map_channels for content addressing"
            )
        self.read_dim = int(read_dim)
        self.shift_kernel_size = shift_kernel_size

        in_channels = 3 + (1 if use_depth else 0)
        backbone = models.resnet18(weights=None)
        if in_channels != 3:
            conv1 = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias is not None,
            )
            with torch.no_grad():
                backbone.conv1.weight[:, :3] = conv1.weight
                if in_channels > 3:
                    nn.init.kaiming_normal_(backbone.conv1.weight[:, 3:], mode="fan_out", nonlinearity="relu")
        backbone.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Conv2d(512, 128, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.pose_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.read_proj = nn.Linear(self.read_dim, hidden_size)
        controller_in_dim = 128 + hidden_size + hidden_size
        self.controller_cell = nn.GRUCell(controller_in_dim, hidden_size)

        base_res = max(1, map_resolution // 8)
        self.decoder_base_res = base_res
        decoder_channels = 128
        self.decoder_linear = nn.Linear(hidden_size, decoder_channels * base_res * base_res)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.map_head = nn.ConvTranspose2d(64, map_channels, kernel_size=4, stride=2, padding=1)
        self.exploration_head = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

        self.key_net = nn.Linear(hidden_size, self.read_dim)
        self.beta_net = nn.Linear(hidden_size, 1)
        self.gate_net = nn.Linear(hidden_size, 1)
        self.shift_net = nn.Linear(hidden_size, shift_kernel_size * shift_kernel_size)
        self.sharpen_net = nn.Linear(hidden_size, 1)
        self.erase_net = nn.Linear(hidden_size, map_channels)
        self.add_net = nn.Linear(hidden_size, map_channels)

        self.hidden: Optional[torch.Tensor] = None

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = state_dict.copy()

        decoder_weight = state_dict.get("decoder_linear.weight")
        decoder_bias = state_dict.get("decoder_linear.bias")

        expected_out = self.decoder_linear.out_features
        decoder_channels = expected_out // (self.decoder_base_res * self.decoder_base_res)

        def _resize_decoder_weight(weight: torch.Tensor) -> torch.Tensor:
            actual_out, hidden_dim = weight.shape
            if hidden_dim != self.decoder_linear.in_features:
                return weight
            base_area, remainder = divmod(actual_out, decoder_channels)
            if remainder != 0:
                return weight
            prev_base = int(round(math.sqrt(base_area)))
            if prev_base * prev_base != base_area or prev_base <= 0:
                return weight
            if prev_base == self.decoder_base_res:
                return weight

            reshaped = weight.view(decoder_channels, prev_base, prev_base, hidden_dim)
            reshaped = reshaped.permute(3, 0, 1, 2)
            resized = F.interpolate(
                reshaped,
                size=(self.decoder_base_res, self.decoder_base_res),
                mode="bilinear",
                align_corners=False,
            )
            resized = resized.permute(1, 2, 3, 0).reshape(expected_out, hidden_dim)
            warnings.warn(
                "Resized Neural SLAM decoder_linear weights from base resolution "
                f"{prev_base} to {self.decoder_base_res} to match map resolution.",
                RuntimeWarning,
            )
            return resized

        def _resize_decoder_bias(bias: torch.Tensor) -> torch.Tensor:
            actual_out = bias.shape[0]
            base_area, remainder = divmod(actual_out, decoder_channels)
            if remainder != 0:
                return bias
            prev_base = int(round(math.sqrt(base_area)))
            if prev_base * prev_base != base_area or prev_base <= 0:
                return bias
            if prev_base == self.decoder_base_res:
                return bias

            reshaped = bias.view(1, decoder_channels, prev_base, prev_base)
            resized = F.interpolate(
                reshaped,
                size=(self.decoder_base_res, self.decoder_base_res),
                mode="bilinear",
                align_corners=False,
            )
            resized = resized.view(expected_out)
            warnings.warn(
                "Resized Neural SLAM decoder_linear bias from base resolution "
                f"{prev_base} to {self.decoder_base_res} to match map resolution.",
                RuntimeWarning,
            )
            return resized

        if decoder_weight is not None and decoder_weight.shape[0] != expected_out:
            state_dict["decoder_linear.weight"] = _resize_decoder_weight(decoder_weight)
        if decoder_bias is not None and decoder_bias.shape[0] != expected_out:
            state_dict["decoder_linear.bias"] = _resize_decoder_bias(decoder_bias)

        return super().load_state_dict(state_dict, strict=strict)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.hidden = torch.zeros(batch_size, self.hidden_size, device=device)

    def detach_state(self) -> None:
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

    def forward(
        self,
        rgb: torch.Tensor,
        pose: Optional[torch.Tensor],
        read_vector: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch = rgb.size(0)
        device = rgb.device

        if depth is None:
            if self.use_depth:
                depth = torch.zeros(
                    batch,
                    1,
                    rgb.size(-2),
                    rgb.size(-1),
                    device=device,
                    dtype=rgb.dtype,
                )
        else:
            if depth.size(-1) != rgb.size(-1) or depth.size(-2) != rgb.size(-2):
                depth = F.interpolate(depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            if depth.dtype != rgb.dtype:
                depth = depth.to(dtype=rgb.dtype)

        if depth is not None:
            rgb = torch.cat([rgb, depth], dim=1)

        if read_vector is None:
            read_vector = torch.zeros(batch, self.read_dim, device=device)
        if self.hidden is None or self.hidden.size(0) != batch:
            self.reset_state(batch, device)

        feats = self.backbone(rgb)
        feats = self.projection(feats)
        pooled = self.avgpool(feats).view(batch, -1)

        pose_embed = (
            self.pose_encoder(pose) if pose is not None else torch.zeros(batch, self.hidden_size, device=device)
        )
        read_embed = self.read_proj(read_vector)

        controller_in = torch.cat([pooled, pose_embed, read_embed], dim=1)
        hidden = self.controller_cell(controller_in, self.hidden)
        self.hidden = hidden

        decoder_input = self.decoder_linear(hidden)
        decoder_input = decoder_input.view(batch, 128, self.decoder_base_res, self.decoder_base_res)
        decoder_feats = self.decoder(decoder_input)

        map_logits = self.map_head(decoder_feats)
        exploration_mask_logits = self.exploration_head(decoder_feats)

        if map_logits.shape[-1] != self.map_resolution or map_logits.shape[-2] != self.map_resolution:
            map_logits = F.interpolate(
                map_logits,
                size=(self.map_resolution, self.map_resolution),
                mode="bilinear",
                align_corners=False,
            )
        if exploration_mask_logits.shape[-1] != self.map_resolution or exploration_mask_logits.shape[-2] != self.map_resolution:
            exploration_mask_logits = F.interpolate(
                exploration_mask_logits,
                size=(self.map_resolution, self.map_resolution),
                mode="bilinear",
                align_corners=False,
            )

        head_params = {
            "key": self.key_net(hidden),
            "beta": self.beta_net(hidden),
            "gate": self.gate_net(hidden),
            "shift": self.shift_net(hidden),
            "sharpen": self.sharpen_net(hidden),
            "erase": self.erase_net(hidden),
            "add": self.add_net(hidden),
        }

        return {
            "map_logits": map_logits,
            "exploration_mask_logits": exploration_mask_logits,
            "head_params": head_params,
            "hidden": hidden,
        }


class NeuralSlamModule(nn.Module):
    """Bundled Neural SLAM controller, map decoder and optional pose head."""

    def __init__(
        self,
        map_channels: int = 4,
        map_resolution: int = 64,
        hidden_size: int = 512,
        read_dim: Optional[int] = None,
        shift_kernel_size: int = 3,
        use_depth: bool = False,
        estimate_pose: bool = True,
    ) -> None:
        super().__init__()
        self.controller = _ControllerImpl(
            map_channels=map_channels,
            map_resolution=map_resolution,
            hidden_size=hidden_size,
            read_dim=read_dim,
            shift_kernel_size=shift_kernel_size,
            use_depth=use_depth,
        )
        self.estimate_pose = estimate_pose
        self.pose_head: Optional[_PoseHead]
        if estimate_pose:
            self.pose_head = _PoseHead(map_channels=map_channels)
        else:
            self.pose_head = None

    def reset_controller_state(self, batch_size: int, device: torch.device) -> None:
        self.controller.reset_state(batch_size, device)

    def detach_controller_state(self) -> None:
        self.controller.detach_state()

    def controller_forward(
        self,
        rgb: torch.Tensor,
        pose: Optional[torch.Tensor],
        read_vector: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.controller(rgb, pose, read_vector, depth)

    def pose_forward(self, prev_map: torch.Tensor, updated_map: torch.Tensor) -> torch.Tensor:
        if self.pose_head is None:
            raise RuntimeError("Pose estimation disabled for this NeuralSlamModule")
        return self.pose_head(prev_map, updated_map)

    def forward(
        self,
        rgb: torch.Tensor,
        pose: Optional[torch.Tensor],
        read_vector: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.controller_forward(rgb, pose, read_vector, depth)


class NeuralSlamController(nn.Module):
    """Backwards-compatible wrapper around :class:`NeuralSlamModule`."""

    def __init__(
        self,
        module: Optional[NeuralSlamModule] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if module is None:
            module = NeuralSlamModule(estimate_pose=False, **kwargs)
        # Store without registering to keep parameters in the shared module only.
        self.__dict__["_module"] = module

    @property
    def module(self) -> NeuralSlamModule:
        return self.__dict__["_module"]

    @property
    def map_channels(self) -> int:
        return int(self.module.controller.map_channels)

    @property
    def read_dim(self) -> int:
        return int(self.module.controller.read_dim)

    @property
    def hidden_size(self) -> int:
        return int(self.module.controller.hidden_size)

    @property
    def shift_kernel_size(self) -> int:
        return int(self.module.controller.shift_kernel_size)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.module.reset_controller_state(batch_size, device)

    def detach_state(self) -> None:
        self.module.detach_controller_state()

    def forward(
        self,
        rgb: torch.Tensor,
        pose: Optional[torch.Tensor],
        read_vector: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # ``NeuralSlamController`` stores the underlying :class:`NeuralSlamModule`
        # outside of PyTorch's module registration system.  In most cases the
        # custom ``_apply`` implementation keeps the wrapped module on the same
        # device as this controller.  However, during RL fine-tuning the wrapper
        # is sometimes rebuilt or loaded after the enclosing ``ModuleDict`` has
        # already been moved to CUDA.  In that scenario the ResNet backbone can
        # stay on CPU which triggers a device mismatch once the first batch is
        # processed.  To guard against this subtle ordering issue we make a
        # best-effort attempt to keep the wrapped module synchronised with the
        # incoming tensor device.
        module = self.module
        try:
            first_param = next(module.controller.parameters())
        except StopIteration:
            first_param = None

        target_device = rgb.device
        if first_param is not None and first_param.device != target_device:
            module.to(target_device)

        return self.module.controller_forward(rgb, pose, read_vector, depth)

    def state_dict(self, *args, **kwargs):
        return self.module.controller.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        if isinstance(state_dict, dict) and not any(k.startswith("controller.") for k in state_dict.keys()):
            return self.module.controller.load_state_dict(state_dict, strict=strict)
        return self.module.load_state_dict(state_dict, strict=strict)

    def _apply(self, fn):
        """Ensure device / dtype transformations reach the wrapped module."""

        super()._apply(fn)
        self.module._apply(fn)
        return self

    def parameters(self, recurse: bool = True):
        return self.module.controller.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.module.controller.named_parameters(prefix=prefix, recurse=recurse)


class NeuralSlamMemoryCore(nn.Module):
    """Differentiable external memory for Neural SLAM."""

    def __init__(
        self,
        map_channels: int = 4,
        map_resolution: int = 64,
        read_dim: Optional[int] = None,
        shift_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.map_channels = map_channels
        self.map_resolution = map_resolution
        if read_dim is None:
            read_dim = map_channels
        elif int(read_dim) != int(map_channels):
            raise ValueError(
                "NeuralSlamMemoryCore requires read_dim to match map_channels for content addressing"
            )
        self.read_dim = int(read_dim)
        self.shift_kernel_size = shift_kernel_size

        self.register_buffer("uniform_weights", torch.ones(1, map_resolution, map_resolution))
        self.uniform_weights = self.uniform_weights / float(map_resolution * map_resolution)

        self.reset(1, torch.device("cpu"))

    def _apply_patch(
        self,
        weights: torch.Tensor,
        patch: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a spatial write patch weighted by the addressing distribution."""

        if patch.dim() == 3:
            patch = patch.unsqueeze(0)
        if patch.dim() != 4:
            raise ValueError("Patches must have shape (B, C, S, S) or (C, S, S)")

        batch, channels, size_h, size_w = patch.shape
        if size_h != size_w:
            raise ValueError("Write patches must be square")

        if weights.dim() == 2:
            weights = weights.unsqueeze(0)
        if weights.dim() != 3:
            raise ValueError("Weights must have shape (B, H, W)")

        _, height, width = weights.shape
        device = patch.device
        dtype = patch.dtype
        result = torch.zeros(batch, channels, height, width, device=device, dtype=dtype)
        center = size_h // 2

        for dh in range(size_h):
            shift_h = dh - center
            if shift_h >= 0:
                src_h = slice(0, height - shift_h)
                dst_h = slice(shift_h, height)
            else:
                src_h = slice(-shift_h, height)
                dst_h = slice(0, height + shift_h)
            if src_h.stop <= src_h.start:
                continue

            for dw in range(size_w):
                shift_w = dw - center
                if shift_w >= 0:
                    src_w = slice(0, width - shift_w)
                    dst_w = slice(shift_w, width)
                else:
                    src_w = slice(-shift_w, width)
                    dst_w = slice(0, width + shift_w)
                if src_w.stop <= src_w.start:
                    continue

                coeff = patch[:, :, dh, dw].view(batch, channels, 1, 1)
                if torch.allclose(coeff, torch.zeros_like(coeff)):
                    continue
                contrib = weights[:, src_h, src_w].unsqueeze(1) * coeff
                result[:, :, dst_h, dst_w] = result[:, :, dst_h, dst_w] + contrib

        return result

    def reset(self, batch_size: int, device: torch.device) -> None:
        h = self.map_resolution
        w = self.map_resolution
        self.map_state = torch.zeros(batch_size, self.map_channels, h, w, device=device)
        self.weight_state = self.uniform_weights.to(device).expand(batch_size, h, w).clone()
        self.read_state = torch.zeros(batch_size, self.read_dim, device=device)

    def get_last_read(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.read_state is None or self.read_state.size(0) != batch_size:
            self.reset(batch_size, device)
        return self.read_state

    def load_external_map(self, prev_map: torch.Tensor) -> None:
        self.map_state = prev_map.clone()

    def forward(self, head_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        key = head_params["key"]
        beta = head_params["beta"]
        gate = head_params["gate"]
        shift = head_params["shift"]
        sharpen = head_params["sharpen"]
        erase = head_params["erase"]
        add = head_params["add"]

        batch, _, h, w = self.map_state.shape
        device = self.map_state.device

        flat_mem = self.map_state.view(batch, self.map_channels, -1).transpose(1, 2)  # (B, HW, C)

        key_norm = F.normalize(key, p=2, dim=-1)
        mem_norm = F.normalize(flat_mem, p=2, dim=-1)
        similarity = torch.matmul(mem_norm, key_norm.unsqueeze(-1)).squeeze(-1)

        beta_pos = F.softplus(beta) + 1e-6
        content_w = torch.softmax(beta_pos * similarity, dim=-1)

        gate_sigma = torch.sigmoid(gate)
        prev_w = self.weight_state.view(batch, -1)
        gated = gate_sigma * content_w + (1.0 - gate_sigma) * prev_w

        shift_kernel = shift.view(batch, self.shift_kernel_size, self.shift_kernel_size)
        shift_kernel = torch.softmax(shift_kernel.view(batch, -1), dim=-1).view_as(shift_kernel)
        shifted = _circular_convolution(gated.view(batch, h, w), shift_kernel)

        sharpen_pos = 1.0 + F.softplus(sharpen).view(batch, 1)
        sharpened = torch.clamp(shifted, min=1e-6).view(batch, -1).pow(sharpen_pos)
        weights = sharpened / (sharpened.sum(dim=-1, keepdim=True) + 1e-6)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        weights_2d = weights.view(batch, h, w)

        add_patch_logits = head_params.get("add_patch_logits")
        erase_patch_logits = head_params.get("erase_patch_logits")

        add_patch = None
        erase_patch = None
        if add_patch_logits is not None:
            add_patch = torch.sigmoid(add_patch_logits.to(device=device, dtype=self.map_state.dtype))
            if add_patch.dim() == 3:
                add_patch = add_patch.unsqueeze(0)
        if erase_patch_logits is not None:
            erase_patch = torch.sigmoid(erase_patch_logits.to(device=device, dtype=self.map_state.dtype))
            if erase_patch.dim() == 3:
                erase_patch = erase_patch.unsqueeze(0)

        if add_patch is not None:
            add_map = self._apply_patch(weights_2d, add_patch)
        else:
            add_term = add.view(batch, self.map_channels, 1, 1)
            add_map = weights_2d.unsqueeze(1) * add_term

        if erase_patch is not None:
            erase_map = self._apply_patch(weights_2d, erase_patch).clamp_(0.0, 1.0)
        else:
            erase_term = torch.sigmoid(erase).view(batch, self.map_channels, 1, 1)
            erase_map = weights_2d.unsqueeze(1) * erase_term

        map_updated = self.map_state * (1.0 - erase_map) + add_map

        read_vector = torch.sum(map_updated * weights_2d.unsqueeze(1), dim=(2, 3))

        self.map_state = map_updated
        self.weight_state = weights_2d
        self.read_state = read_vector

        weight_sums = weights_2d.view(batch, -1).sum(dim=-1)
        ones = torch.ones_like(weight_sums)
        if not torch.allclose(weight_sums, ones, atol=1e-4, rtol=1e-3):
            raise AssertionError("Addressing weights must sum to one")

        return {
            "updated_map": map_updated,
            "weights": weights_2d,
            "read_vector": read_vector,
            "head_params": {
                "key": key,
                "beta": beta,
                "gate": gate,
                "shift": shift,
                "sharpen": sharpen,
                "erase": erase,
                "add": add,
            },
        }


class NeuralSlamPoseEstimator(nn.Module):
    """Wrapper around the pose head of :class:`NeuralSlamModule`."""

    def __init__(
        self,
        module: Optional[NeuralSlamModule] = None,
        map_channels: int = 2,
    ) -> None:
        super().__init__()
        if module is not None:
            self.__dict__["_module"] = module
            self.pose_head: Optional[_PoseHead] = None
        else:
            self.__dict__["_module"] = None
            self.pose_head = _PoseHead(map_channels=map_channels)

    @property
    def module(self) -> Optional[NeuralSlamModule]:
        return self.__dict__["_module"]

    def forward(self, prev_map: torch.Tensor, updated_map: torch.Tensor) -> torch.Tensor:
        module = self.module
        if module is not None:
            return module.pose_forward(prev_map, updated_map)
        if self.pose_head is None:
            raise RuntimeError("Pose estimator initialised without parameters")
        return self.pose_head(prev_map, updated_map)


__all__ = [
    "NeuralSlamModule",
    "NeuralSlamController",
    "NeuralSlamMemoryCore",
    "NeuralSlamPoseEstimator",
    "NeuralSlamState",
]
