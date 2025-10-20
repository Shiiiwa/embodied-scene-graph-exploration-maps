# src/components/exploration/maps/neural_slam_map.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_weighted_patch(weights: torch.Tensor, patch: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Compute the spatial write contribution of a patch weighted by an addressing map."""

    if patch.dim() == 3:
        patch = patch.unsqueeze(0)
    if patch.dim() != 4:
        raise ValueError("Patch tensor must have shape (B, C, S, S) or (C, S, S)")

    if weights.dim() == 2:
        weights = weights.unsqueeze(0)
    if weights.dim() != 3:
        raise ValueError("Weights must have shape (B, H, W) or (H, W)")

    batch_w, height_w, width_w = weights.shape
    batch_p, channels, size_h, size_w = patch.shape
    if size_h != size_w:
        raise ValueError("Patch updates require square kernels")

    if batch_p not in (1, batch_w):
        raise ValueError("Patch batch dimension must match weights or be 1")
    if batch_p == 1 and batch_w > 1:
        patch = patch.expand(batch_w, -1, -1, -1)

    center = size_h // 2

    weights_reshaped = weights.view(1, batch_w, height_w, width_w)
    kernels = patch.flip(-1, -2).contiguous().view(batch_w * channels, 1, size_h, size_w)
    result = F.conv2d(
        weights_reshaped,
        kernels,
        padding=center,
        groups=batch_w,
    )
    result = result.view(batch_w, channels, result.shape[-2], result.shape[-1])
    if result.shape[-2] != height or result.shape[-1] != width:
        extra_h = max(result.shape[-2] - height, 0)
        extra_w = max(result.shape[-1] - width, 0)
        start_h = extra_h // 2
        start_w = extra_w // 2
        result = result[:, :, start_h:start_h + height, start_w:start_w + width]

    if result.size(0) == 1:
        return result[0]
    return result


class NeuralSlamMemory(nn.Module):
    """Torch-backed differentiable memory for Neural-SLAM maps.

    Keeps the global map ``M`` and pose belief ``omega`` as buffers so that
    gradients can flow through differentiable read/write heads while allowing
    the map to be snapshot via ``state_dict``/``load_state_dict``.
    """

    def __init__(self, channels: int, height: int, width: int, *,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.C = int(channels)
        self.H = int(height)
        self.W = int(width)
        device = device or torch.device("cpu")
        map_tensor = torch.zeros(self.C, self.H, self.W, dtype=dtype, device=device)
        pose_belief = torch.full(
            (self.H, self.W),
            1.0 / float(self.H * self.W),
            dtype=dtype,
            device=device,
        )
        self.register_buffer("map", map_tensor)
        self.register_buffer("pose_belief", pose_belief)

    @property
    def device(self) -> torch.device:
        return self.map.device

    def reset(
        self,
        start_pose: Optional[Tuple[int, int]] = None,
        vision_range: Optional[float] = None,
    ):
        """Reset the map and optionally seed a Gaussian pose prior.

        Args:
            start_pose: Optional ``(i, j)`` start indices in map coordinates.
            vision_range: Optional spread of the prior **in cells**. If ``None``
                or non-positive, a uniform prior is used.
        """

        self.map.zero_()
        if start_pose is None:
            self.pose_belief.fill_(1.0 / float(self.H * self.W))
            return
        self._apply_pose_prior(start_pose, vision_range)

    def set_pose_prior(
        self,
        start_pose: Optional[Tuple[int, int]] = None,
        vision_range: Optional[float] = None,
    ) -> None:
        """Update ``pose_belief`` without touching the memory map."""

        if start_pose is None:
            self.pose_belief.fill_(1.0 / float(self.H * self.W))
            return
        self._apply_pose_prior(start_pose, vision_range)

    def _apply_pose_prior(
        self,
        start_pose: Tuple[int, int],
        vision_range: Optional[float],
    ) -> None:
        i, j = start_pose
        i = int(round(float(i)))
        j = int(round(float(j)))
        i = max(0, min(self.H - 1, i))
        j = max(0, min(self.W - 1, j))

        if vision_range is None or not math.isfinite(float(vision_range)) or float(vision_range) <= 0.0:
            sigma = 1.0
        else:
            sigma = max(float(vision_range) / 2.0, 1e-3)

        dtype = self.map.dtype
        device = self.device
        coords_i = torch.arange(self.H, dtype=dtype, device=device).unsqueeze(1)
        coords_j = torch.arange(self.W, dtype=dtype, device=device).unsqueeze(0)
        diff_i = coords_i - torch.tensor(float(i), dtype=dtype, device=device)
        diff_j = coords_j - torch.tensor(float(j), dtype=dtype, device=device)
        dist_sq = diff_i.pow(2) + diff_j.pow(2)
        gaussian = torch.exp(-0.5 * dist_sq / (sigma ** 2))
        gaussian = torch.clamp(gaussian, min=1e-12)
        normaliser = gaussian.sum()
        if normaliser.item() == 0.0:
            self.pose_belief.fill_(1.0 / float(self.H * self.W))
            return
        self.pose_belief.copy_(gaussian / normaliser)

    def address(
        self,
        key: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        shift: torch.Tensor,
        sharpen: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute differentiable addressing weights for the memory grid.

        Args:
            key:      [C] content key.
            beta:     scalar inverse temperature controlling content focus.
            gate:     scalar interpolation gate ``g`` in [0, 1].
            shift:    [Kh, Kw] shift kernel (unnormalised) ``rho``.
            sharpen:  scalar ``zeta`` controlling sharpness (>1).
            prev_weights: Optional previous belief [H, W]; defaults to the
                stored pose belief buffer.

        Returns:
            Normalised addressing weights with shape [H, W].
        """

        key = key.to(self.device, dtype=self.map.dtype).view(self.C)
        beta = beta.to(self.device, dtype=self.map.dtype)
        gate = gate.to(self.device, dtype=self.map.dtype)
        shift = shift.to(self.device, dtype=self.map.dtype)
        sharpen = sharpen.to(self.device, dtype=self.map.dtype)

        flat_mem = self.map.view(self.C, -1).transpose(0, 1)  # [HW, C]
        key_norm = F.normalize(key, dim=0, eps=1e-8)
        mem_norm = F.normalize(flat_mem, dim=1, eps=1e-8)
        content_sim = torch.matmul(mem_norm, key_norm)  # [HW]
        beta_pos = F.softplus(beta).view(1)
        w_c = F.softmax(beta_pos * content_sim, dim=0)

        gate = torch.sigmoid(gate).view(1)
        if prev_weights is None:
            prev_weights = self.pose_belief
        prev_weights = prev_weights.to(self.device, dtype=self.map.dtype).view(-1)
        w_g = gate * w_c + (1.0 - gate) * prev_weights

        shift = F.softmax(shift.reshape(-1), dim=0).reshape_as(shift)
        w_shifted = self._shift_weights(w_g.view(self.H, self.W), shift)

        sharpen = F.softplus(sharpen) + 1.0
        w_power = torch.clamp(w_shifted, min=1e-8).pow(sharpen)
        w = w_power.view(-1)
        w = w / torch.clamp(w.sum(), min=1e-6)
        return w.view(self.H, self.W)

    def _shift_weights(self, weights: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """Circular convolution (shift) of addressing weights with ``kernel``."""

        assert kernel.dim() == 2, "Shift kernel must be 2-D"
        kh, kw = kernel.shape
        center_h = kh // 2
        center_w = kw // 2
        pad_h = kh // 2
        pad_w = kw // 2
        weights_4d = weights.unsqueeze(0).unsqueeze(0)
        kernel_4d = kernel.view(1, 1, kh, kw)
        padded = F.pad(weights_4d, (pad_w, pad_w, pad_h, pad_h), mode="circular")
        convolved = F.conv2d(padded, kernel_4d)
        return convolved[0, 0]

    def _translate_weights(self, weights: torch.Tensor, shift_rows: int, shift_cols: int) -> torch.Tensor:
        """Shift weights with zero padding instead of circular wrap-around."""

        if weights.dim() != 2:
            raise ValueError("Pose belief translation expects a 2-D tensor")

        if abs(shift_rows) >= self.H or abs(shift_cols) >= self.W:
            return torch.zeros_like(weights)

        shifted = torch.roll(weights, shifts=(shift_rows, shift_cols), dims=(0, 1))

        if shift_rows > 0:
            shifted[:shift_rows, :] = 0
        elif shift_rows < 0:
            shifted[shift_rows:, :] = 0

        if shift_cols > 0:
            shifted[:, :shift_cols] = 0
        elif shift_cols < 0:
            shifted[:, shift_cols:] = 0

        return shifted

    def _fractional_translate(self, weights: torch.Tensor, shift_rows: float, shift_cols: float) -> torch.Tensor:
        """Bilinear translation with zero padding for fractional offsets."""

        if weights.dim() != 2:
            raise ValueError("Pose belief translation expects a 2-D tensor")

        if abs(shift_rows) < 1e-6 and abs(shift_cols) < 1e-6:
            return weights

        row_base = math.floor(shift_rows)
        col_base = math.floor(shift_cols)

        row_frac = float(shift_rows - row_base)
        col_frac = float(shift_cols - col_base)

        row_weights = [1.0 - row_frac, row_frac]
        col_weights = [1.0 - col_frac, col_frac]
        row_shifts = [row_base, row_base + 1]
        col_shifts = [col_base, col_base + 1]

        result = torch.zeros_like(weights)
        for r_weight, r_shift in zip(row_weights, row_shifts):
            if r_weight <= 0.0:
                continue
            for c_weight, c_shift in zip(col_weights, col_shifts):
                if c_weight <= 0.0:
                    continue
                contribution = self._translate_weights(weights, r_shift, c_shift)
                if contribution.numel() == 0:
                    continue
                result = result + contribution * (r_weight * c_weight)
        return result

    def read(self, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.to(self.device, dtype=self.map.dtype)
        read_vec = torch.sum(self.map * weights.unsqueeze(0), dim=(1, 2))
        return read_vec

    def write(
        self,
        weights: torch.Tensor,
        erase: torch.Tensor,
        add: torch.Tensor,
        *,
        erase_patch: Optional[torch.Tensor] = None,
        add_patch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weights = weights.to(self.device, dtype=self.map.dtype)
        if weights.dim() == 3:
            weights_2d = weights[0]
        elif weights.dim() == 2:
            weights_2d = weights
        else:
            raise ValueError("Weights must have shape (H, W) or (1, H, W)")

        weights_expanded = weights_2d.unsqueeze(0)

        if add_patch is not None:
            patch = add_patch.to(self.device, dtype=self.map.dtype)
            add_map = _apply_weighted_patch(weights_2d, patch, self.H, self.W)
        else:
            add_tensor = add.to(self.device, dtype=self.map.dtype).view(self.C, 1, 1)
            add_map = weights_expanded * add_tensor

        if erase_patch is not None:
            patch = erase_patch.to(self.device, dtype=self.map.dtype)
            erase_map = _apply_weighted_patch(weights_2d, patch, self.H, self.W).clamp(0.0, 1.0)
        else:
            erase_tensor = torch.sigmoid(erase.to(self.device, dtype=self.map.dtype)).view(self.C, 1, 1)
            erase_map = weights_expanded * erase_tensor

        updated = self.map * (1.0 - erase_map) + add_map
        self.map = updated
        return updated

    def read_write(
        self,
        key: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        shift: torch.Tensor,
        sharpen: torch.Tensor,
        erase: torch.Tensor,
        add: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
        *,
        erase_patch: Optional[torch.Tensor] = None,
        add_patch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.address(key, beta, gate, shift, sharpen, prev_weights)
        self.write(weights, erase, add, erase_patch=erase_patch, add_patch=add_patch)
        read_vec = self.read(weights)
        self.pose_belief = weights.detach()
        sum_weights = self.pose_belief.sum()
        target = torch.ones_like(sum_weights)
        if not torch.allclose(sum_weights, target, atol=1e-4, rtol=1e-3):
            raise AssertionError("Pose belief must sum to one")
        return read_vec, weights



class NeuralSlamMap:
    """
    Minimal-konsistente Neural-SLAM-Map für IL-Datensatzerzeugung:
    - Hält eine globale Rasterkarte (C,H,W) in Weltkoordinaten (Origin + cell_size)
    - Integriert pro Schritt eine egocentrische GT-Karte (obstacles/explored)
    - Liefert Frontier-Zellen + nearest_frontier()
    - Liefert Trainingsdaten (ego_map_gt, poses) für IL
    """

    def __init__(self, map_shape: Tuple[int, int], cell_size_cm: int = 25, vision_range_cm: int = 320):
        """
        Args:
            map_shape: (H, W) Zellen.
            cell_size_cm: Zellgröße in Zentimetern (Default 25cm = 0.25m).
            vision_range_cm: Sichtweite (z. B. 320cm = 3.2m).
        """
        self.map_shape = tuple(map(int, map_shape))
        self.H, self.W = self.map_shape
        self.cell_size_m = float(cell_size_cm) / 100.0
        self.vision_range_m = float(vision_range_cm) / 100.0

        # Welt-Koordinaten-Ursprung für Raster -> MUSS von der Env gesetzt werden
        self.origin_x: float = 0.0
        self.origin_z: float = 0.0

        # Globale Speicherkarte: Kanäle
        #  0: explored (freier/explorierter Raum)
        #  1: obstacle (Hindernis)
        #  2: agent_traj (optionales Diagnose-Feature)
        #  3: frontier_mask (wird on-the-fly berechnet, aber wir halten einen Cache)
        self.C = 4
        self.memory = NeuralSlamMemory(self.C, self.H, self.W)

        # Letzte Trainingsdaten (für ExplorationMapManager.get_training_data())
        self._last_training: Optional[Dict[str, Any]] = None
        # Debug/diagnostics: track whether the most recent update used the neural
        # controller outputs instead of falling back to GT integration.
        self._last_wrote_prediction: bool = False
        # Für Bewegungsschätzung aus aufeinanderfolgenden Beobachtungen
        self._last_agent_pose: Optional[Dict[str, float]] = None
        # Aktiviert den neuralspezifischen Update-Pfad
        self.neural_mode: bool = True
        # Precomputed coordinate grids for egocentric projections
        self._ego_fov_cache: Dict[int, Dict[str, np.ndarray]] = {}
        self._ego_integration_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        # Policy cache: lokale Karten (aktuell/previous) + Projektionen
        self._cached_local_map_current: Optional[torch.Tensor] = None
        self._cached_local_map_previous: Optional[torch.Tensor] = None
        self._cached_local_map_current_global: Optional[torch.Tensor] = None
        self._cached_local_map_previous_global: Optional[torch.Tensor] = None
        self._cached_local_map_target: Optional[torch.Tensor] = None
        # Cached coordinate grids to avoid reallocating full map linspaces each step
        self._global_grid_cache: Dict[Tuple[torch.device, torch.dtype], Dict[str, torch.Tensor]] = {}

    # ------------- Public API, von ExplorationMapManager aufgerufen -------------

    def reset(
        self,
        start_pose: Optional[Tuple[int, int]] = None,
        vision_range_m: Optional[float] = None,
        vision_range_cells: Optional[float] = None,
    ):
        pose = self._clamp_pose(start_pose)
        spread_cells = self._resolve_vision_range(vision_range_m, vision_range_cells)
        self.memory.reset(start_pose=pose, vision_range=spread_cells)
        self._last_training = None
        self._last_agent_pose = None
        self._last_wrote_prediction = False
        self._cached_local_map_current = None
        self._cached_local_map_previous = None
        self._cached_local_map_current_global = None
        self._cached_local_map_previous_global = None
        self._cached_local_map_target = None

    def set_pose_prior(
        self,
        start_pose: Optional[Tuple[int, int]] = None,
        vision_range_m: Optional[float] = None,
        vision_range_cells: Optional[float] = None,
    ) -> None:
        pose = self._clamp_pose(start_pose)
        spread_cells = self._resolve_vision_range(vision_range_m, vision_range_cells)
        self.memory.set_pose_prior(start_pose=pose, vision_range=spread_cells)

    def set_map_origin(self, world_x: float, world_z: float):
        self.origin_x = float(world_x)
        self.origin_z = float(world_z)
        self._global_grid_cache.clear()

    def set_neural_mode(self, enabled: bool) -> None:
        """Enable or disable neural write-head updates."""

        self.neural_mode = bool(enabled)

    # Wichtig: konsistent mit PrecomputedThorEnv.get_occupancy_indices()
    def world_to_map(self, x: float, z: float) -> Tuple[int, int]:
        dx = x - self.origin_x
        dz = z - self.origin_z
        j = int(dx / self.cell_size_m)
        i = int(self.H - 1 - (dz / self.cell_size_m))
        return i, j

    def _clamp_pose(
        self, start_pose: Optional[Tuple[int, int]]
    ) -> Optional[Tuple[int, int]]:
        if start_pose is None:
            return None
        i, j = start_pose
        i = int(round(float(i)))
        j = int(round(float(j)))
        i = max(0, min(self.H - 1, i))
        j = max(0, min(self.W - 1, j))
        return i, j

    def _resolve_vision_range(
        self,
        vision_range_m: Optional[float],
        vision_range_cells: Optional[float],
    ) -> Optional[float]:
        if vision_range_cells is not None:
            try:
                vr = float(vision_range_cells)
            except (TypeError, ValueError):
                vr = None
        else:
            vr = None
        if vr is None:
            if vision_range_m is None:
                vision_range_m = self.vision_range_m
            if vision_range_m is None:
                return None
            try:
                vr = float(vision_range_m) / max(self.cell_size_m, 1e-6)
            except (TypeError, ValueError):
                vr = None
        if vr is None or not math.isfinite(vr):
            return None
        if vr <= 0.0:
            return None
        return vr

    def map_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x = self.origin_x + j * self.cell_size_m
        z = self.origin_z + (self.H - 1 - i) * self.cell_size_m
        return x, z

    def get(self):
        """Rohes Feature-Tensor (C,H,W)."""
        return self.memory.map

    def get_map_for_policy(self):
        """Features + Meta (für spätere RL/Policy-Eingaben)."""
        meta = {
            "cell_size_m": self.cell_size_m,
            "origin_x": self.origin_x,
            "origin_z": self.origin_z,
        }
        feats = self.memory.map.detach().clone()
        if getattr(self, "neural_mode", True):
            extra = torch.zeros(
                4,
                self.H,
                self.W,
                dtype=feats.dtype,
                device=feats.device,
            )
            if self._cached_local_map_current_global is not None:
                cur = self._cached_local_map_current_global.to(feats.device, dtype=feats.dtype)
                channels = min(cur.shape[0], 2)
                extra[0:channels] = cur[:channels]
            if self._cached_local_map_previous_global is not None:
                prev = self._cached_local_map_previous_global.to(feats.device, dtype=feats.dtype)
                channels = min(prev.shape[0], 2)
                extra[2 : 2 + channels] = prev[:channels]
            feats = torch.cat([feats, extra], dim=0)
            meta["local_map_current"] = None if self._cached_local_map_current is None else self._cached_local_map_current.detach().cpu()
            meta["local_map_previous"] = None if self._cached_local_map_previous is None else self._cached_local_map_previous.detach().cpu()
            meta["local_map_target"] = None if self._cached_local_map_target is None else self._cached_local_map_target.detach().cpu()
        return feats, meta

    def state_dict(self) -> Dict[str, Any]:
        state = {
            "map_shape": torch.tensor([self.H, self.W], dtype=torch.long),
            "cell_size_m": torch.tensor(self.cell_size_m, dtype=torch.float32),
            "origin": torch.tensor([self.origin_x, self.origin_z], dtype=torch.float32),
            "memory": self.memory.state_dict(),
            "neural_mode": torch.tensor(bool(self.neural_mode)),
        }
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        if "map_shape" in state:
            shape = state["map_shape"]
            if isinstance(shape, torch.Tensor):
                shape = shape.tolist()
            shape = tuple(int(x) for x in shape)
            if shape != (self.H, self.W):
                raise ValueError(f"Mismatched map shape {shape} for NeuralSlamMap {self.map_shape}")
        if "cell_size_m" in state:
            csm = state["cell_size_m"]
            if isinstance(csm, torch.Tensor):
                csm = float(csm.item())
            self.cell_size_m = float(csm)
        if "origin" in state:
            origin = state["origin"]
            if isinstance(origin, torch.Tensor):
                origin = origin.detach().cpu().numpy()
            self.origin_x = float(origin[0])
            self.origin_z = float(origin[1])
        if "memory" in state:
            mem_state = {}
            for k, v in state["memory"].items():
                if isinstance(v, torch.Tensor):
                    tensor = v.to(self.memory.device, dtype=self.memory.map.dtype)
                else:
                    tensor = torch.as_tensor(v, dtype=self.memory.map.dtype, device=self.memory.device)
                mem_state[k] = tensor
            self.memory.load_state_dict(mem_state)
        if "neural_mode" in state:
            mode = state["neural_mode"]
            if isinstance(mode, torch.Tensor):
                mode = bool(mode.item())
            self.neural_mode = bool(mode)

    def apply_memory_head(
        self,
        key: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        shift: torch.Tensor,
        sharpen: torch.Tensor,
        erase: torch.Tensor,
        add: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
        *,
        erase_patch: Optional[torch.Tensor] = None,
        add_patch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper around :meth:`NeuralSlamMemory.read_write`."""

        return self.memory.read_write(
            key,
            beta,
            gate,
            shift,
            sharpen,
            erase,
            add,
            prev_weights,
            erase_patch=erase_patch,
            add_patch=add_patch,
        )

    def address_memory(
        self,
        key: torch.Tensor,
        beta: torch.Tensor,
        gate: torch.Tensor,
        shift: torch.Tensor,
        sharpen: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute addressing weights without mutating the memory."""

        return self.memory.address(key, beta, gate, shift, sharpen, prev_weights)

    def read_memory(self, weights: torch.Tensor) -> torch.Tensor:
        return self.memory.read(weights)

    def write_memory(
        self,
        weights: torch.Tensor,
        erase: torch.Tensor,
        add: torch.Tensor,
        *,
        erase_patch: Optional[torch.Tensor] = None,
        add_patch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        updated = self.memory.write(weights, erase, add, erase_patch=erase_patch, add_patch=add_patch)
        self.memory.pose_belief = weights.detach()
        weight_sum = self.memory.pose_belief.sum()
        target = torch.ones_like(weight_sum)
        if not torch.allclose(weight_sum, target, atol=1e-4, rtol=1e-3):
            raise AssertionError("Pose belief must sum to one")
        return updated

    def get_training_data(self):
        """Letzte Trainingsdaten (für IL-Dataset persistieren)."""
        return None if self._last_training is None else dict(self._last_training)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "map_shape": [self.H, self.W],
            "cell_size_m": self.cell_size_m,
            "origin": [self.origin_x, self.origin_z],
            "memory": self.memory.map.detach().cpu().numpy().astype(np.float16).tolist(),
            "pose_belief": self.memory.pose_belief.detach().cpu().numpy().astype(np.float32).tolist(),
            "neural_mode": bool(self.neural_mode),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuralSlamMap":
        H, W = int(data["map_shape"][0]), int(data["map_shape"][1])
        cell_size_m = float(data.get("cell_size_m", 0.25))
        m = cls(map_shape=(H, W), cell_size_cm=int(cell_size_m * 100.0))
        m.origin_x, m.origin_z = float(data["origin"][0]), float(data["origin"][1])
        mem = torch.tensor(data["memory"], dtype=torch.float32)
        assert mem.shape == (m.C, H, W), f"Unexpected memory shape {mem.shape} != {(m.C, H, W)}"
        m.memory.map = mem.to(m.memory.device)
        pose = data.get("pose_belief", None)
        if pose is not None:
            pose_tensor = torch.tensor(pose, dtype=torch.float32)
            if pose_tensor.shape == (H, W):
                m.memory.pose_belief = pose_tensor.to(m.memory.device)
        m.neural_mode = bool(data.get("neural_mode", True))
        return m

    def render_matplotlib(self, show=True, save_path=None):
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            # einfache Visualisierung: explored - obstacle
            img = (self.memory.map[0] - self.memory.map[1]).detach().cpu().numpy()
            ax.imshow(img, origin="upper", interpolation="nearest")
            ax.set_title("Neural-SLAM Map (explored - obstacles)")
            if save_path:
                fig.savefig(save_path, dpi=150)
            if show:
                plt.show()
            plt.close(fig)
        except Exception:
            pass

    def render_ascii(self):
        mem = self.memory.map.detach().cpu().numpy()
        chars = []
        for i in range(self.H):
            row = []
            for j in range(self.W):
                if mem[1, i, j] > 0.5:
                    row.append("#")  # obstacle
                elif mem[0, i, j] > 0.5:
                    row.append(".")  # explored
                else:
                    row.append(" ")  # unknown
            chars.append("".join(row))
        print("\n".join(chars))

    # -------------------- Hauptintegration / Update-Pfad ------------------------

    def update_from_observation(
        self,
        rgb_obs: np.ndarray,
        agent_world_pose: Dict[str, float],
        sensor_pose: Optional[Dict[str, float]] = None,
        event: Optional[Any] = None,
        action: Optional[Any] = None,
        mapper_logits: Optional[Any] = None,
        head_params: Optional[Dict[str, Any]] = None,
        odometry: Optional[Dict[str, float]] = None,
        *,
        proj_pred: Optional[Any] = None,
        fp_exp_pred: Optional[Any] = None,
        pose_delta_pred: Optional[Any] = None,
        proj_target: Optional[Any] = None,
        fp_exp_target: Optional[Any] = None,
        pose_delta_target: Optional[Any] = None,
        exploration_mask: Optional[Any] = None,
        pose_delta: Optional[Any] = None,
        prev_observation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Integrate the latest observation into the neural SLAM map."""

        neural_mode_active = bool(getattr(self, "neural_mode", True))

        # 1) Motion prior + optional neural pose refinement
        self._apply_motion_prediction(action, odometry, agent_world_pose)
        if neural_mode_active:
            self._apply_pose_delta_prediction(pose_delta_pred)

        # 2) Build GT egocentric supervision
        ego_gt = self._create_ground_truth_egocentric_map(event)
        device = self.memory.device
        dtype = self.memory.map.dtype
        gt_tensor = torch.as_tensor(ego_gt, device=device, dtype=dtype)
        gt_local_tensor = self._ensure_local_channels(gt_tensor)

        # 3) Optional neural module outputs / targets
        proj_pred_tensor = self._coerce_optional_tensor(proj_pred)
        if proj_pred_tensor is not None:
            proj_pred_tensor = torch.sigmoid(proj_pred_tensor)
            proj_pred_tensor = self._ensure_local_channels(proj_pred_tensor)

        fp_exp_pred_tensor = self._coerce_optional_tensor(fp_exp_pred)
        if fp_exp_pred_tensor is not None:
            fp_exp_pred_tensor = torch.sigmoid(fp_exp_pred_tensor)
            fp_exp_pred_tensor = self._ensure_local_channels(fp_exp_pred_tensor)

        proj_target_tensor = self._coerce_optional_tensor(proj_target)
        if proj_target_tensor is not None:
            proj_target_tensor = self._ensure_local_channels(proj_target_tensor)

        fp_exp_target_tensor = self._coerce_optional_tensor(fp_exp_target)
        if fp_exp_target_tensor is not None:
            fp_exp_target_tensor = self._ensure_local_channels(fp_exp_target_tensor)

        if neural_mode_active:
            cache_pred = proj_pred_tensor
            if cache_pred is None:
                cache_pred = fp_exp_pred_tensor
            cache_target = proj_target_tensor
            if cache_target is None:
                cache_target = fp_exp_target_tensor
            self._update_local_map_cache(
                pred_local=cache_pred,
                target_local=cache_target,
                gt_local=gt_local_tensor,
                agent_pose=agent_world_pose,
            )
        else:
            self._update_local_map_cache(
                pred_local=None,
                target_local=None,
                gt_local=None,
                agent_pose=agent_world_pose,
            )

        def _prediction_to_patch(
            local_tensor: torch.Tensor,
            floorplan_tensor: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            local_tensor = self._ensure_local_channels(local_tensor)
            size = int(local_tensor.shape[-1])
            patch = torch.zeros(self.C, size, size, dtype=dtype, device=device)
            # Channel order: 0 -> explored, 1 -> obstacle
            patch[0] = torch.clamp(local_tensor[1], 0.0, 1.0)
            patch[1] = torch.clamp(local_tensor[0], 0.0, 1.0)
            if floorplan_tensor is not None:
                floorplan_tensor = self._ensure_local_channels(floorplan_tensor)
                patch[0] = torch.maximum(patch[0], torch.clamp(floorplan_tensor[1], 0.0, 1.0))
            return patch

        add_patch_tensor: Optional[torch.Tensor] = None
        if neural_mode_active and proj_pred_tensor is not None:
            patch_map = _prediction_to_patch(proj_pred_tensor, fp_exp_pred_tensor)
            add_patch_tensor = patch_map.unsqueeze(0)
        elif neural_mode_active and fp_exp_pred_tensor is not None:
            patch_map = _prediction_to_patch(fp_exp_pred_tensor)
            add_patch_tensor = patch_map.unsqueeze(0)
        elif mapper_logits is not None:
            mapper_tensor = mapper_logits
            if not isinstance(mapper_tensor, torch.Tensor):
                mapper_tensor = torch.as_tensor(mapper_tensor, dtype=dtype, device=device)
            else:
                mapper_tensor = mapper_tensor.to(device=device, dtype=dtype)
            if mapper_tensor.dim() == 3:
                mapper_tensor = mapper_tensor.unsqueeze(0)
            add_patch_tensor = torch.sigmoid(mapper_tensor)

        head_tensors: Dict[str, torch.Tensor] = {}
        if head_params:
            for name, value in head_params.items():
                if value is None:
                    continue
                if isinstance(value, torch.Tensor):
                    head_tensors[name] = value.to(device=device, dtype=dtype)
                else:
                    head_tensors[name] = torch.as_tensor(value, dtype=dtype, device=device)

        prior_pose_belief = self.memory.pose_belief.detach().clone()
        measurement_weights: Optional[torch.Tensor] = None
        wrote_prediction = False

        if neural_mode_active and add_patch_tensor is not None and head_tensors:
            try:
                patch_map = add_patch_tensor[0]
                if patch_map.shape[0] != self.C:
                    patch_full = torch.zeros(self.C, patch_map.shape[-1], patch_map.shape[-1], device=device, dtype=dtype)
                    channels = min(self.C, patch_map.shape[0])
                    patch_full[:channels] = patch_map[:channels]
                    patch_map = patch_full

                erase_patch_tensor = None
                erase_logits = head_tensors.get("erase_patch_logits")
                if erase_logits is not None:
                    if erase_logits.dim() == 4:
                        erase_logits = erase_logits[0]
                    if erase_logits.shape[0] != self.C:
                        erase_full = torch.zeros(self.C, erase_logits.shape[-1], erase_logits.shape[-1], device=device, dtype=dtype)
                        channels = min(self.C, erase_logits.shape[0])
                        erase_full[:channels] = erase_logits[:channels]
                        erase_logits = erase_full
                    erase_patch_tensor = torch.sigmoid(erase_logits)

                def _ensure_vec(name: str, length: int) -> torch.Tensor:
                    tensor = head_tensors.get(name)
                    if tensor is None:
                        return torch.zeros(length, device=device, dtype=dtype)
                    if tensor.dim() > 1:
                        tensor = tensor[0]
                    tensor = tensor.view(-1)
                    if tensor.numel() == length:
                        return tensor
                    vec = torch.zeros(length, device=device, dtype=dtype)
                    copy = min(length, tensor.numel())
                    vec[:copy] = tensor[:copy]
                    return vec

                key = head_tensors.get("key")
                if key is None:
                    raise ValueError("Missing key for memory head")
                if key.dim() > 1:
                    key = key[0]
                key = key.view(-1)
                if key.numel() != self.C:
                    key_full = torch.zeros(self.C, device=device, dtype=dtype)
                    copy = min(self.C, key.numel())
                    key_full[:copy] = key[:copy]
                    key = key_full

                beta = head_tensors.get("beta")
                if beta is None:
                    raise ValueError("Missing beta for memory head")
                beta = beta.view(-1)[:1]

                gate = head_tensors.get("gate")
                if gate is None:
                    raise ValueError("Missing gate for memory head")
                gate = gate.view(-1)[:1]

                sharpen = head_tensors.get("sharpen")
                if sharpen is None:
                    raise ValueError("Missing sharpen for memory head")
                sharpen = sharpen.view(-1)[:1]

                shift = head_tensors.get("shift")
                if shift is None:
                    raise ValueError("Missing shift kernel for memory head")
                if shift.dim() == 3:
                    shift = shift[0]
                if shift.dim() == 1:
                    side = int(round(math.sqrt(float(shift.numel()))))
                    shift = shift.view(side, side)
                elif shift.dim() == 2:
                    pass
                else:
                    raise ValueError("Unexpected shift shape")

                add_vec = _ensure_vec("add", self.C)
                erase_vec = _ensure_vec("erase", self.C)

                _, measurement_weights = self.apply_memory_head(
                    key,
                    beta,
                    gate,
                    shift,
                    sharpen,
                    erase_vec,
                    add_vec,
                    prev_weights=self.memory.pose_belief,
                    erase_patch=erase_patch_tensor,
                    add_patch=patch_map,
                )
                wrote_prediction = True
            except Exception:
                wrote_prediction = False

        if wrote_prediction and neural_mode_active and measurement_weights is not None:
            self._blend_pose_belief(prior_pose_belief, measurement_weights)
        else:
            self.memory.pose_belief.copy_(prior_pose_belief)
            self._integrate_egocentric_into_global(
                ego_map=ego_gt,
                agent_pose=agent_world_pose,
            )

        self._last_wrote_prediction = bool(wrote_prediction)

        # Diagnose channel: agent trajectory
        ai, aj = self.world_to_map(agent_world_pose["x"], agent_world_pose["z"])
        if 0 <= ai < self.H and 0 <= aj < self.W:
            self.memory.map[2, ai, aj] = 1.0

        self._update_frontier_layer()

        stg_world = None
        nf = self.nearest_frontier((ai, aj))
        if nf is not None:
            stg_world = {
                "x": float(self.map_to_world(nf[0], nf[1])[0]),
                "z": float(self.map_to_world(nf[0], nf[1])[1]),
            }

        # 6) Trainingsdaten-Bundle für diese Observation
        prev_obs_record = None
        if isinstance(prev_observation, dict) and prev_observation:
            prev_obs_record = {}
            if "rgb" in prev_observation and prev_observation["rgb"] is not None:
                prev_obs_record["rgb"] = np.array(prev_observation["rgb"], copy=True)
            if "pose" in prev_observation and prev_observation["pose"] is not None:
                prev_obs_record["pose"] = dict(prev_observation["pose"])

        mask_record = None
        if exploration_mask is not None:
            if isinstance(exploration_mask, torch.Tensor):
                mask_record = exploration_mask.detach().cpu()
            else:
                mask_record = torch.as_tensor(exploration_mask).detach().cpu()

        pose_delta_record = None
        if pose_delta is not None:
            if isinstance(pose_delta, torch.Tensor):
                pose_delta_record = pose_delta.detach().cpu()
            else:
                pose_delta_record = torch.as_tensor(pose_delta).detach().cpu()

        training = {
            "ego_map_gt": ego_gt,
            "agent_world_pose": dict(agent_world_pose),
            "sensor_pose": None if sensor_pose is None else dict(sensor_pose),
            "short_term_goal": stg_world,
            "action": action,
            "odometry": None if odometry is None else dict(odometry),
            "mapper_logits": mapper_logits,
            "head_params": head_params,
            "proj_pred": proj_pred,
            "fp_exp_pred": fp_exp_pred,
            "pose_delta_pred": pose_delta_pred,
            "proj_target": proj_target,
            "fp_exp_target": fp_exp_target,
            "pose_delta_target": pose_delta_target,
            "exploration_mask": mask_record,
            "pose_delta": pose_delta_record,
            "prev_observation": prev_obs_record,
        }

        self._last_training = training
        self._last_agent_pose = dict(agent_world_pose)
        return training

    def last_write_used_prediction(self) -> bool:
        """Return whether the most recent update used neural predictions."""

        return bool(self._last_wrote_prediction)

    # -------------------- Bewegungsabschätzung / Pose-Belief --------------------

    def _blend_pose_belief(
        self,
        prior_pose: torch.Tensor,
        measurement: torch.Tensor,
    ) -> None:
        """Combine motion prior with addressing weights to form a posterior belief."""

        if measurement.dim() == 3:
            measurement = measurement[0]
        measurement = measurement.to(self.memory.device, dtype=self.memory.map.dtype)
        prior_pose = prior_pose.to(self.memory.device, dtype=self.memory.map.dtype)

        likelihood = torch.clamp(measurement, min=1e-9)
        posterior = prior_pose * likelihood
        total = posterior.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            self.memory.pose_belief.copy_(prior_pose)
            return
        self.memory.pose_belief.copy_(posterior / total)

    def _apply_motion_prediction(
        self,
        action: Optional[Any],
        odometry: Optional[Dict[str, float]],
        agent_world_pose: Dict[str, float],
    ) -> None:
        """Shiftet die Pose-Belief-Heatmap anhand einer Bewegungsabschätzung.

        Priorisiert Odometriedaten (dx, dz, dyaw). Falls diese fehlen, wird die
        Bewegung aus der Umgebungsaktion rekonstruiert, sofern eine
        Vorbeobachtung existiert. Nach dem Shift werden die Gewichte normalisiert
        (Summe = 1).
        """

        if self.memory.pose_belief is None:
            return

        delta_x = 0.0
        delta_z = 0.0
        delta_yaw = 0.0

        if odometry is not None:
            delta_x = float(odometry.get("dx", 0.0))
            delta_z = float(odometry.get("dz", 0.0))
            delta_yaw = float(odometry.get("dyaw", 0.0))
        else:
            motion = self._motion_from_action(action)
            if motion is not None:
                delta_x, delta_z, delta_yaw = motion
            elif self._last_agent_pose is not None:
                delta_x = float(agent_world_pose["x"] - self._last_agent_pose.get("x", 0.0))
                delta_z = float(agent_world_pose["z"] - self._last_agent_pose.get("z", 0.0))
                delta_yaw = float(
                    ((agent_world_pose["yaw"] - self._last_agent_pose.get("yaw", 0.0) + 180.0) % 360.0) - 180.0
                )

        # Keine Translation -> kein Shift notwendig (Rotation beeinflusst Belief nicht)
        if abs(delta_x) < 1e-6 and abs(delta_z) < 1e-6:
            return

        shift_rows = int(round(-delta_z / self.cell_size_m))
        shift_cols = int(round(delta_x / self.cell_size_m))

        if shift_rows == 0 and shift_cols == 0:
            return

        shifted = self.memory._translate_weights(self.memory.pose_belief, shift_rows, shift_cols)
        total = torch.clamp(shifted.sum(), min=1e-6)
        self.memory.pose_belief = shifted / total

    def _motion_from_action(self, action: Optional[Any]) -> Optional[Tuple[float, float, float]]:
        """Rekonstruiert (dx, dz, dyaw) aus Umgebungsaktionen."""

        if action is None or self._last_agent_pose is None:
            return None

        primitives: Optional[Any] = None
        if isinstance(action, dict) and "primitives" in action:
            primitives = action["primitives"]
        elif isinstance(action, (list, tuple)):
            primitives = action
        if primitives is None:
            return None

        yaw = float(self._last_agent_pose.get("yaw", 0.0))
        dx_total = 0.0
        dz_total = 0.0

        for primitive in primitives:
            if primitive == "RotateRight":
                yaw += 90.0
            elif primitive == "RotateLeft":
                yaw -= 90.0
            elif primitive == "MoveAhead":
                dx, dz = self._rotate_move(0.0, self.cell_size_m, yaw)
                dx_total += dx
                dz_total += dz
            elif primitive == "MoveBack":
                dx, dz = self._rotate_move(0.0, -self.cell_size_m, yaw)
                dx_total += dx
                dz_total += dz
            elif primitive == "MoveRight":
                dx, dz = self._rotate_move(self.cell_size_m, 0.0, yaw)
                dx_total += dx
                dz_total += dz
            elif primitive == "MoveLeft":
                dx, dz = self._rotate_move(-self.cell_size_m, 0.0, yaw)
                dx_total += dx
                dz_total += dz
            # "Pass" führt zu keiner Änderung

        delta_yaw = float(((yaw - self._last_agent_pose.get("yaw", 0.0)) + 180.0) % 360.0 - 180.0)
        return dx_total, dz_total, delta_yaw

    @staticmethod
    def _rotate_move(dx: float, dz: float, heading: float) -> Tuple[float, float]:
        """Rotation eines Bewegungsvektors für ganzzahlige 90°-Heading-Werte."""

        # Heading der Umgebung ist auf 90°-Vielfache quantisiert -> robust runden
        angle = int(round(heading / 90.0)) * 90 % 360
        if angle == 90:
            return dz, -dx
        if angle == 180:
            return -dx, -dz
        if angle == 270:
            return -dz, dx
        return dx, dz

    def _build_shift_kernel(self, shift_rows: int, shift_cols: int) -> torch.Tensor:
        kh = 2 * abs(shift_rows) + 1
        kw = 2 * abs(shift_cols) + 1
        kernel = torch.zeros(
            (kh, kw),
            dtype=self.memory.map.dtype,
            device=self.memory.device,
        )
        center_h = kh // 2
        center_w = kw // 2
        kernel[center_h + shift_rows, center_w + shift_cols] = 1.0
        return kernel

    # ---------------------- SLAM-Hilfsfunktionen -------------------------------

    def _create_ground_truth_egocentric_map(
        self, event: Any, vision_range_cells: int = 64
    ) -> np.ndarray:
        """
        Erzeugt aus dem AI2-THOR-Event eine egocentrische GT-Karte:
        Channel 0: obstacles
        Channel 1: explored (FOV)
        """
        ego = np.zeros((2, vision_range_cells, vision_range_cells), dtype=np.float32)
        if event is None:
            return ego

        agent = event.metadata["agent"]
        agent_pos = agent["position"]
        agent_rot = agent["rotation"]["y"]
        yaw = math.radians(agent_rot)

        objects = event.metadata.get("objects", [])
        center = vision_range_cells // 2
        cell = self.cell_size_m  # 0.25m
        max_range = self.vision_range_m

        cache = self._ego_fov_cache.get(vision_range_cells)
        if cache is None:
            coords = np.arange(vision_range_cells, dtype=np.float32)
            grid_i, grid_j = np.meshgrid(coords, coords, indexing="ij")
            rel_i = grid_i - float(center)
            rel_j = grid_j - float(center)
            dist = np.hypot(rel_i * cell, rel_j * cell)
            angles = np.arctan2(rel_j, rel_i)
            cache = {"dist": dist, "angles": angles}
            self._ego_fov_cache[vision_range_cells] = cache

        fov_angle = math.radians(90)
        dist = cache["dist"]
        angles = cache["angles"]
        fov_mask = (dist <= max_range) & (np.abs(angles) <= fov_angle / 2.0)
        ego[1][fov_mask] = 1.0

        # Hindernisse (sichtbare, unbewegliche/nicht-pickupbare Objekte)
        obstacle_offsets = []
        for obj in objects:
            if not obj.get("visible", False):
                continue
            if obj.get("moveable", True) or obj.get("pickupable", False):
                continue
            ox = float(obj["position"]["x"] - agent_pos["x"])
            oz = float(obj["position"]["z"] - agent_pos["z"])
            obstacle_offsets.append((ox, oz))

        if not obstacle_offsets:
            return ego

        offsets = np.asarray(obstacle_offsets, dtype=np.float32)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        rotation = np.array([[cos_y, sin_y], [-sin_y, cos_y]], dtype=np.float32)
        ego_coords = offsets @ rotation.T

        within_range = (np.abs(ego_coords[:, 0]) <= max_range) & (np.abs(ego_coords[:, 1]) <= max_range)
        if not np.any(within_range):
            return ego

        ego_coords = ego_coords[within_range]
        cols = np.trunc(center + ego_coords[:, 0] / cell).astype(np.int64)
        rows = np.trunc(center + ego_coords[:, 1] / cell).astype(np.int64)

        valid = (
            (rows >= 0)
            & (rows < vision_range_cells)
            & (cols >= 0)
            & (cols < vision_range_cells)
        )
        if not np.any(valid):
            return ego

        rows = rows[valid]
        cols = cols[valid]
        ego[0, rows, cols] = 1.0
        ego[1, rows, cols] = 1.0  # sichtbarer Bereich gilt als explored
        return ego

    def _integrate_egocentric_into_global(
        self, ego_map: np.ndarray, agent_pose: Dict[str, float]
    ):
        """
        Projiziert ego_map (2,S,S) in globale Rasterkarte (C,H,W) via Pose.
        """
        S = int(ego_map.shape[1])
        cache = self._ego_integration_cache.get(S)
        if cache is None:
            center = S // 2
            coords = torch.arange(S, dtype=torch.float32)
            grid_i, grid_j = torch.meshgrid(coords, coords, indexing="ij")
            dz_local = (grid_i - float(center)) * self.cell_size_m
            dx_local = (grid_j - float(center)) * self.cell_size_m
            cache = {
                "dx_local": dx_local.reshape(-1),
                "dz_local": dz_local.reshape(-1),
            }
            self._ego_integration_cache[S] = cache

        device = self.memory.device
        dtype = self.memory.map.dtype
        ego_tensor = torch.as_tensor(ego_map, device=device, dtype=dtype)
        obs = ego_tensor[0]
        exp = ego_tensor[1]
        update_mask = torch.logical_or(exp >= 0.5, obs >= 0.5)
        if not torch.any(update_mask):
            return

        dx_local = cache["dx_local"].to(device=device, dtype=dtype)
        dz_local = cache["dz_local"].to(device=device, dtype=dtype)
        mask_flat = update_mask.reshape(-1)
        dx_selected = dx_local[mask_flat]
        dz_selected = dz_local[mask_flat]

        yaw = math.radians(agent_pose["yaw"] % 360)
        yaw_tensor = torch.tensor(yaw, device=device, dtype=dtype)
        cos_y = torch.cos(yaw_tensor)
        sin_y = torch.sin(yaw_tensor)

        dx_world = cos_y * dx_selected - sin_y * dz_selected
        dz_world = sin_y * dx_selected + cos_y * dz_selected

        agent_x = torch.tensor(float(agent_pose["x"]), device=device, dtype=dtype)
        agent_z = torch.tensor(float(agent_pose["z"]), device=device, dtype=dtype)
        wx = agent_x + dx_world
        wz = agent_z + dz_world

        rel_x = wx - float(self.origin_x)
        rel_z = wz - float(self.origin_z)
        cell_size = torch.tensor(self.cell_size_m, device=device, dtype=dtype)
        j_float = rel_x / cell_size
        i_float = torch.tensor(self.H - 1, device=device, dtype=dtype) - (rel_z / cell_size)
        j_idx = torch.trunc(j_float).to(torch.long)
        i_idx = torch.trunc(i_float).to(torch.long)

        valid = (
            (i_idx >= 0)
            & (i_idx < self.H)
            & (j_idx >= 0)
            & (j_idx < self.W)
        )
        if not torch.any(valid):
            return

        i_idx = i_idx[valid]
        j_idx = j_idx[valid]

        exp_flat = exp.reshape(-1)[mask_flat][valid]
        obs_flat = obs.reshape(-1)[mask_flat][valid]
        explored_mask = exp_flat >= 0.5
        obstacle_mask = obs_flat >= 0.5

        if torch.any(explored_mask):
            ei = i_idx[explored_mask]
            ej = j_idx[explored_mask]
            self.memory.map[0, ei, ej] = 1.0
        if torch.any(obstacle_mask):
            oi = i_idx[obstacle_mask]
            oj = j_idx[obstacle_mask]
            self.memory.map[1, oi, oj] = 1.0
            self.memory.map[0, oi, oj] = 1.0

    # --------------------------- Helper für Neural-Update ----------------------

    def _coerce_optional_tensor(
        self,
        value: Optional[Any],
        *,
        squeeze_batch: bool = True,
    ) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.memory.device, dtype=self.memory.map.dtype)
        else:
            tensor = torch.as_tensor(value, dtype=self.memory.map.dtype, device=self.memory.device)
        if squeeze_batch and tensor.dim() > 0 and tensor.shape[0] == 1:
            tensor = tensor.view(*tensor.shape[1:])
        return tensor

    def _ensure_local_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError("Local map tensor must have shape (C, H, W) or (H, W)")
        if tensor.shape[0] == 0:
            raise ValueError("Local map tensor must have at least one channel")
        if tensor.shape[0] == 1:
            zeros = torch.zeros_like(tensor)
            tensor = torch.cat([zeros, tensor], dim=0)
        elif tensor.shape[0] > 2:
            tensor = tensor[:2]
        return tensor

    def _project_local_to_global(
        self,
        local_map: torch.Tensor,
        agent_pose: Dict[str, float],
    ) -> torch.Tensor:
        if local_map.dim() == 2:
            local_map = local_map.unsqueeze(0)
        if local_map.dim() != 3:
            raise ValueError("Projected local map must have shape (C, S, S)")
        size = int(local_map.shape[-1])
        if size <= 1:
            raise ValueError("Local map projection requires spatial size > 1")

        dtype = self.memory.map.dtype
        device = self.memory.device
        local_map = local_map.to(device=device, dtype=dtype)

        cache_key = (device, dtype)
        grid_cache = self._global_grid_cache.get(cache_key)
        if grid_cache is None:
            coords_i = torch.arange(self.H, device=device, dtype=dtype)
            coords_j = torch.arange(self.W, device=device, dtype=dtype)
            grid_i, grid_j = torch.meshgrid(coords_i, coords_j, indexing="ij")
            grid_cache = {
                "grid_i": grid_i,
                "grid_j": grid_j,
            }
            self._global_grid_cache[cache_key] = grid_cache
        grid_i = grid_cache["grid_i"]
        grid_j = grid_cache["grid_j"]

        origin_x = torch.as_tensor(self.origin_x, device=device, dtype=dtype)
        origin_z = torch.as_tensor(self.origin_z, device=device, dtype=dtype)
        cell_size = torch.as_tensor(self.cell_size_m, device=device, dtype=dtype)
        world_x = origin_x + grid_j * cell_size
        world_z = origin_z + (self.H - 1 - grid_i) * cell_size

        agent_x = torch.as_tensor(float(agent_pose["x"]), device=device, dtype=dtype)
        agent_z = torch.as_tensor(float(agent_pose["z"]), device=device, dtype=dtype)

        rel_x = world_x - agent_x
        rel_z = world_z - agent_z

        yaw = torch.as_tensor(math.radians(agent_pose["yaw"] % 360), device=device, dtype=dtype)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        dx_local = cos_y * rel_x + sin_y * rel_z
        dz_local = -sin_y * rel_x + cos_y * rel_z

        center = (size - 1) / 2.0
        clamped_cell = torch.clamp(cell_size, min=1e-6)

        j_local = center + dx_local / clamped_cell
        i_local = center + dz_local / clamped_cell

        normaliser = float(max(size - 1, 1))
        inv_norm = 1.0 / normaliser
        j_norm = 2.0 * (j_local * inv_norm) - 1.0
        i_norm = 2.0 * (i_local * inv_norm) - 1.0

        grid = torch.stack([j_norm, i_norm], dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            local_map.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled[0]

    def _update_local_map_cache(
        self,
        *,
        pred_local: Optional[torch.Tensor],
        target_local: Optional[torch.Tensor],
        gt_local: Optional[torch.Tensor],
        agent_pose: Dict[str, float],
    ) -> None:
        if not getattr(self, "neural_mode", True):
            self._cached_local_map_previous = None
            self._cached_local_map_previous_global = None
            self._cached_local_map_current = None
            self._cached_local_map_current_global = None
            self._cached_local_map_target = None
            return

        base_local = pred_local
        if base_local is None:
            base_local = target_local
        if base_local is None:
            base_local = gt_local

        prev_local = None if self._cached_local_map_current is None else self._cached_local_map_current.detach().clone()
        prev_global = (
            None
            if self._cached_local_map_current_global is None
            else self._cached_local_map_current_global.detach().clone()
        )

        self._cached_local_map_previous = prev_local
        self._cached_local_map_previous_global = prev_global

        if base_local is None:
            self._cached_local_map_current = None
            self._cached_local_map_current_global = None
        else:
            ensured = self._ensure_local_channels(base_local)
            self._cached_local_map_current = ensured.detach().clone()
            self._cached_local_map_current_global = self._project_local_to_global(ensured, agent_pose)

        if target_local is not None:
            ensured_target = self._ensure_local_channels(target_local)
            self._cached_local_map_target = ensured_target.detach().clone()
        elif gt_local is not None:
            ensured_gt = self._ensure_local_channels(gt_local)
            self._cached_local_map_target = ensured_gt.detach().clone()
        else:
            self._cached_local_map_target = None

    def _apply_pose_delta_prediction(self, pose_delta: Optional[Any]) -> None:
        if not getattr(self, "neural_mode", True):
            return
        tensor = self._coerce_optional_tensor(pose_delta, squeeze_batch=True)
        if tensor is None:
            return
        tensor = tensor.view(-1)
        if tensor.numel() < 2:
            return

        dx = float(tensor[0].item())
        dz = float(tensor[1].item())

        if abs(dx) < 1e-9 and abs(dz) < 1e-9:
            return

        dx_cells = dx / max(self.cell_size_m, 1e-6)
        dz_cells = dz / max(self.cell_size_m, 1e-6)

        belief = self.memory.pose_belief.to(self.memory.device, dtype=self.memory.map.dtype)
        shifted = self._fractional_translate(belief, dz_cells, -dx_cells)
        total = shifted.sum()
        if not torch.isfinite(total) or float(total.item()) <= 1e-9:
            return
        self.memory.pose_belief.copy_(shifted / total)

    def _update_frontier_layer(self):
        """Frontier = unbekannte Zelle, die an explored grenzt und nicht obstacle ist."""
        self.memory.map[3].zero_()
        explored = self.memory.map[0] >= 0.5
        obstacle = self.memory.map[1] >= 0.5
        unknown = ~explored

        # 4-Nachbarschaft der explored-Zellen
        neigh = torch.zeros_like(explored, dtype=torch.bool)
        neigh[:-1, :] |= explored[1:, :]
        neigh[1:, :] |= explored[:-1, :]
        neigh[:, :-1] |= explored[:, 1:]
        neigh[:, 1:] |= explored[:, :-1]

        frontier = neigh & unknown & (~obstacle)
        self.memory.map[3] = frontier.to(self.memory.map.dtype)

    def nearest_frontier(self, agent_cell: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Finde nächste Frontier-Zelle in Gitterkoordinaten (i,j)."""
        ai, aj = agent_cell
        mask = self.memory.map[3] >= 0.5
        if not bool(mask.any()):
            return None

        # BFS / Ring-Suche (Manhattan)
        # einfache, deterministische Suche
        best = None
        best_d = None
        for i in range(self.H):
            for j in range(self.W):
                if not mask[i, j]:
                    continue
                d = abs(i - ai) + abs(j - aj)
                if best_d is None or d < best_d:
                    best = (i, j)
                    best_d = d
        return best
