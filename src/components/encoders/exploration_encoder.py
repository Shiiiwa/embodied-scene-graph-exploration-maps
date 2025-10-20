import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class ExplorationMapEncoder(nn.Module):

    def __init__(
        self,
        output_dim=64,
        target_size=(64, 64),
        include_walkable=True,
        include_object_score=True,
        include_agent=False,
    ):
        super().__init__()
        self.target_h, self.target_w = target_size
        self.include_walkable = include_walkable
        self.include_object_score = include_object_score
        self.include_agent = include_agent

        # There will be these channels:
        #   0: visited mask (0/1)
        #   1: walkable mask (0/1)  [optional]
        #   2: object score (max visibility in cell, 0..1) [optional]
        #   3: agent position (one-hot) [optional]
        in_channels = (
            1
            + (1 if include_walkable else 0)
            + (1 if include_object_score else 0)
            + (1 if include_agent else 0)
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(64, output_dim)

        self.output_dim = output_dim
        self.in_channels = in_channels

    @torch.no_grad()
    def _rasterize_one(self, map_dict: dict, show_raster=False) -> torch.Tensor:
        """
        Convert a single sparse map dict into a tensor grid [C, H, W].
        Handles:
            - None/invalid maps → zeros
            - New format with {"types", "instances"} under "objects"
            - Legacy flat "objects" or legacy dense 2D maps
        """
        if map_dict is None or not isinstance(map_dict, dict):
            return torch.zeros(self.in_channels, self.target_h, self.target_w)
        if "visited" in map_dict:
            entries = map_dict["visited"]

            # --- Resolve H, W robustly ---
        H, W = self.target_h, self.target_w
        shape = map_dict.get("map_shape") or map_dict.get("shape") or map_dict.get("size")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            H, W = int(shape[0]), int(shape[1])
        elif isinstance(shape, dict):
            H = int(shape.get("H") or shape.get("h") or shape.get("height", H))
            W = int(shape.get("W") or shape.get("w") or shape.get("width", W))

        # --- Allocate channels ---
        visited = torch.zeros(1, H, W)
        channels = [visited]
        if self.include_walkable:
            walk = torch.zeros(1, H, W)
            channels.append(walk)
        if self.include_object_score:
            obj = torch.zeros(1, H, W)
            channels.append(obj)
        agent = None
        if self.include_agent:
            agent = torch.zeros(1, H, W)
            channels.append(agent)

        # --- Helper to update a single cell ---
        def _update_cell(i: int, j: int, cell: dict):
            visited[0, i, j] = 1.0

            if self.include_walkable:
                walkable = cell.get("walkable")
                walk[0, i, j] = 1.0 if walkable is True else 0.0

            if self.include_object_score:
                objs = cell.get("objects", {}) or {}
                type_scores = {}

                if isinstance(objs, dict):
                    # Newer structured form: {"types": {...}, "instances": {...}}
                    if "types" in objs or "instances" in objs:
                        if isinstance(objs.get("types"), dict):
                            for k, v in objs["types"].items():
                                if isinstance(v, (int, float)):
                                    type_scores[k] = float(v)
                        if isinstance(objs.get("instances"), dict):
                            for inst in objs["instances"].values():
                                if isinstance(inst, dict):
                                    t = inst.get("type", "Unknown")
                                    v = float(inst.get("vis", 0.0) or 0.0)
                                    v = max(0.0, min(1.0, v))  # clamp
                                    type_scores[t] = max(type_scores.get(t, 0.0), v)
                    else:
                        # Legacy flat dict: {object_type: visibility}
                        for k, v in objs.items():
                            if isinstance(v, (int, float)):
                                type_scores[k] = float(v)

                max_vis = max(type_scores.values(), default=0.0)
                obj[0, i, j] = max_vis

        agent_index = None
        if self.include_agent and isinstance(map_dict, dict):
            raw_idx = map_dict.get("agent_index", map_dict.get("map_index"))

            if isinstance(raw_idx, torch.Tensor):
                raw_idx = raw_idx.detach().cpu().tolist()

            if isinstance(raw_idx, dict):
                agent_index = (
                    raw_idx.get("i", raw_idx.get("row")),
                    raw_idx.get("j", raw_idx.get("col")),
                )
            elif isinstance(raw_idx, (list, tuple)) and len(raw_idx) >= 2:
                agent_index = (raw_idx[0], raw_idx[1])
            elif isinstance(raw_idx, (int, float)):
                agent_index = (raw_idx, None)

            if isinstance(agent_index, tuple) and len(agent_index) >= 2:
                try:
                    ai = int(agent_index[0])
                    aj = int(agent_index[1]) if agent_index[1] is not None else None
                except (TypeError, ValueError):
                    ai = aj = None
                if aj is not None and 0 <= ai < H and 0 <= aj < W:
                    agent[0, ai, aj] = 1.0

        # --- Get cell entries (v1/v2/fallback) ---
        entries = map_dict.get("visited")
        if entries is None:
            entries = map_dict.get("cells")
        if entries is None:
            entries = map_dict.get("map")

        # --- Fill grid from entries ---
        if isinstance(entries, list) and entries:
            # Case A: list of dict cells
            if isinstance(entries[0], dict):
                for entry in entries:
                    i = entry.get("i", entry.get("row"))
                    j = entry.get("j", entry.get("col"))
                    if i is None or j is None:
                        continue
                    i, j = int(i), int(j)
                    if 0 <= i < H and 0 <= j < W:
                        _update_cell(i, j, entry)

            # Case B: legacy dense 2D list entries[H][W] of dicts
            elif isinstance(entries[0], list):
                for i in range(min(H, len(entries))):
                    row = entries[i]
                    for j in range(min(W, len(row))):
                        cell = row[j]
                        if isinstance(cell, dict) and (cell.get("visited", 0) or cell.get("vis", 0)):
                            _update_cell(i, j, cell)

        # --- Stack channels ---
        grid = torch.cat(channels, dim=0)  # [C, H, W]

        # --- Optional debug snapshot ---
        if show_raster:
            import uuid
            self.debug_raster(map_dict, f"debug_{uuid.uuid4().hex[:6]}.png")

        # --- Resize to target ---
        if (H, W) != (self.target_h, self.target_w):
            grid = grid.unsqueeze(0)  # [1, C, H, W]
            grid = nn.functional.interpolate(grid, size=(self.target_h, self.target_w), mode="nearest")
            grid = grid.squeeze(0)

        return grid


    def debug_raster(self, map_dict, save_path="map_debug.png"):
        """
        Rasterize one map_dict and save the channels as images.
        """
        encoder = ExplorationMapEncoder(output_dim=64, target_size=(64, 64))
        grid = encoder._rasterize_one(map_dict)  # [C,H,W], no grad
        C, H, W = grid.shape

        fig, axs = plt.subplots(1, C, figsize=(4 * C, 4))
        if C == 1:
            axs = [axs]

        for c in range(C):
            axs[c].imshow(grid[c].cpu().numpy(), cmap="viridis", origin="lower")
            axs[c].set_title(f"Channel {c}")
            axs[c].axis("off")

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"[DEBUG] Saved raster visualization with {C} channels to {save_path}")

    @torch.no_grad()
    def rasterize_batch(self, map_list):
        """
        map_list: List[dict or None] of length N (flattened over [B,T])
        Returns: Tensor [N, C, H, W]
        """
        grids = [self._rasterize_one(m) for m in map_list]
        return torch.stack(grids, dim=0)

    def forward(self, map_list):
        """
        map_list: List[dict or None] of length N (flattened over [B,T]).
        Returns: Tensor [N, output_dim]
        """
        x = self.rasterize_batch(map_list).to(next(self.parameters()).device)  # [N, C, H, W]
        if len(map_list) > 1 and False:
            unique_maps = []
            for m in map_list:
                grid = self._rasterize_one(m)
                unique_maps.append(grid.flatten())
            unique_maps = torch.stack(unique_maps)
            pairwise_dist = torch.cdist(unique_maps, unique_maps).mean().item()
            print(f"[RASTER DIVERSITY] Avg pairwise distance: {pairwise_dist:.4f}")
        feat_map = self.encoder(x)  # [N, 64, 32, 32]

        attn_logits = self.spatial_attention[0](feat_map)  # [N, 1, 32, 32] (before sigmoid)

        attn_weights = F.softmax(attn_logits.view(attn_logits.size(0), -1), dim=-1)  # [N, 1024]
        attn_weights = attn_weights.view(attn_logits.size(0), 1, 32, 32)  # [N, 1, 32, 32]

        weighted = feat_map * attn_weights
        pooled = weighted.sum(dim=(-2, -1))  # [N, 64]; no normalization needed

        out = self.fc(pooled)

        # if out.size(0) > 1:
            # feature_std = out.std(dim=0).mean().item()
            # print(f"[FEATURE DIVERSITY] Mean std across batch: {feature_std:.4f}")

        return out

class MetricSemanticMapV2Encoder(nn.Module):
    """
    Dense metric-semantic v2 encoder with:
    - Spatial attention similar to v1
    - Reduced, high-signal channels
    - Stable class embeddings instead of dynamic slot mapping
    - Batch normalization
    - Multi-scale feature extraction
    - No meta-vector shortcuts
    """

    def __init__(
            self,
            output_dim=64,
            target_size=(64, 64),
            include_agent=True,
            num_class_channels=8,  # Reduced from 16
            use_class_embeddings=True,  # New: embeddings instead of one-hot
            class_vocab=None,
    ):
        super().__init__()
        self.target_h, self.target_w = target_size
        self.include_agent = include_agent
        self.num_class_channels = num_class_channels
        self.use_class_embeddings = use_class_embeddings

        # Feature selection focuses on the most informative channels:
        # 1. visited (binary) - whether the cell was visited
        # 2. p_free (0..1) - walkability probability
        # 3. confidence (0..1) - observation confidence
        # 4. frontier (binary) - boundary to unexplored space
        # 5. time_since_visit_norm (0..1) - age of information
        # 6. visit_count_norm (0..1) - visit frequency
        # 7-14. class_heat (top-K object heatmaps)
        # 15. agent_position (optional)

        self.static_feature_channels = 6  # visited, p_free, conf, frontier, tsv, vcnt
        self.base_channels = self.static_feature_channels + self.num_class_channels

        in_channels = self.base_channels + (1 if include_agent else 0)

        # Class embeddings instead of dynamic slots
        if use_class_embeddings:
            self.class_vocab = {}  # name -> fixed idx
            if class_vocab:
                for idx, name in enumerate(class_vocab):
                    self.class_vocab[name] = idx
            self.max_vocab_size = 200  # Fixed vocabulary size
            self.class_embedding_dim = 32
            self.class_embedding = nn.Embedding(self.max_vocab_size, self.class_embedding_dim)
            # Projection: class_heat channels × embedding → num_class_channels
            self.class_proj = nn.Linear(self.class_embedding_dim, 1)

        # Multi-scale CNN encoder reminiscent of ResNet blocks
        self.encoder = nn.Sequential(
            # Block 1: 64×64 → 32×32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → 32×32

            # Block 2: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # → 16×16

            # Block 3: 16×16 (keep resolution for attention)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        # Spatial attention block
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            # Sigmoid is applied later in forward()
        )

        # Output projection
        self.fc = nn.Linear(128, output_dim)
        self.output_dim = output_dim

    def forward(self, map_entries, agent_indices=None):
        """
        map_entries: List[dict or tuple]; each entry has (features, meta) or {dense: ...}
        agent_indices: List[tuple] with (i, j) agent positions
        Returns: [N, output_dim]
        """
        if not map_entries:
            device = next(self.parameters()).device
            return torch.zeros(0, self.output_dim, device=device)

        if agent_indices is None:
            agent_indices = [None] * len(map_entries)
        elif len(agent_indices) < len(map_entries):
            agent_indices = list(agent_indices) + [None] * (len(map_entries) - len(agent_indices))

        device = next(self.parameters()).device
        grids = []

        for entry, agent_idx in zip(map_entries, agent_indices):
            # Parse entry → NumPy array [C, H, W]
            feats_np, meta = self._parse_entry(entry)

            # Feature Engineering: bessere Normalisierung
            feats_np = self._normalize_features(feats_np)

            # Class Heatmaps: Embedding-basiert statt Slot-Mapping
            if self.use_class_embeddings and "class_index" in meta:
                feats_np = self._encode_class_semantics(feats_np, meta)
            else:
                feats_np = self._ensure_channel_count(feats_np)

            tensor = torch.from_numpy(feats_np.astype(np.float32))

            if tensor.ndim != 3:
                raise ValueError(f"Expected [C,H,W], got {tensor.shape}")

            H, W = tensor.shape[1], tensor.shape[2]

            # Agent Position als separater Kanal
            if self.include_agent:
                agent_layer = torch.zeros(1, H, W, dtype=tensor.dtype)
                if isinstance(agent_idx, (tuple, list)) and len(agent_idx) >= 2:
                    ai, aj = agent_idx[:2]
                    try:
                        ai, aj = int(ai), int(aj)
                        if 0 <= ai < H and 0 <= aj < W:
                            agent_layer[0, ai, aj] = 1.0
                    except (TypeError, ValueError):
                        pass
                tensor = torch.cat([tensor, agent_layer], dim=0)

            # Resize to target
            if (H, W) != (self.target_h, self.target_w):
                tensor = tensor.unsqueeze(0)
                tensor = F.interpolate(tensor, size=(self.target_h, self.target_w), mode="bilinear",
                                       align_corners=False)
                tensor = tensor.squeeze(0)

            grids.append(tensor)

        grid_tensor = torch.stack(grids, dim=0).to(device)  # [N, C, H, W]

        # --- CNN Feature Extraction ---
        feat_map = self.encoder(grid_tensor)  # [N, 128, H', W']

        attn_logits = self.spatial_attention(feat_map)  # [N, 1, H', W']

        # Softmax over spatial dimensions (Sigmoid would break normalization)
        N, _, H_feat, W_feat = attn_logits.shape
        temperature = 0.5  # Lower temperature sharpens the attention focus
        attn_weights = F.softmax(attn_logits.view(N, -1) / temperature, dim=-1)  # [N, H'×W']
        attn_weights = attn_weights.view(N, 1, H_feat, W_feat)  # [N, 1, H', W']

        # Weighted pooling
        weighted_feat = feat_map * attn_weights  # [N, 128, H', W']
        pooled = weighted_feat.sum(dim=(-2, -1))  # [N, 128]

        # --- Output ---
        out = self.fc(pooled)  # [N, output_dim]

        return out

    def _parse_entry(self, entry):
        if entry is None:
            return np.zeros((self.base_channels, self.target_h, self.target_w), dtype=np.float32), {}

        if isinstance(entry, (list, tuple)):
            feats = entry[0]
            meta = entry[1] if len(entry) > 1 else {}
            feats = self._canonicalize_feature_order(feats)
            return np.asarray(feats, dtype=np.float32), dict(meta or {})

        elif isinstance(entry, dict):
            dense = entry.get("dense") if entry else None
            if dense is not None:
                return self._features_from_dense(dense)
            feats = entry.get("features")
            meta = entry.get("meta", {})
            feats = self._canonicalize_feature_order(feats)
            return np.asarray(feats, dtype=np.float32), dict(meta or {})

        # Fallback
        feats = np.zeros((self.base_channels, self.target_h, self.target_w), dtype=np.float32)
        return feats, {}

    def _features_from_dense(self, dense):
        """
        Convert the dense map representation into a feature array.

        Returns:
            feats: np.ndarray [C, H, W] where C = static_channels + class_channels
            meta: dict containing global statistics
        """
        # Extract raw tensors
        pf = np.asarray(dense.get("p_free", 0.5), dtype=np.float32)
        conf = np.asarray(dense.get("confidence", 0.0), dtype=np.float32)
        visited = np.asarray(dense.get("visited_mask", 0), dtype=np.float32)
        visit_cnt = np.asarray(dense.get("visit_cnt", 0.0), dtype=np.float32)
        last_visit = np.asarray(dense.get("last_visit", -1), dtype=np.float32)
        frontier = np.asarray(dense.get("frontier", 0), dtype=np.float32)
        class_heat = np.asarray(dense.get("class_heat", 0.0), dtype=np.float32)
        timestamp = float(dense.get("t", 0))

        # Enforce shape consistency ([H, W] for most channels, [K, H, W] for class_heat)
        if visited.ndim == 3:
            visited = visited.squeeze(0)
        if pf.ndim == 3:
            pf = pf.squeeze(0)
        if conf.ndim == 3:
            conf = conf.squeeze(0)
        if frontier.ndim == 3:
            frontier = frontier.squeeze(0)
        if visit_cnt.ndim == 3:
            visit_cnt = visit_cnt.squeeze(0)
        if last_visit.ndim == 3:
            last_visit = last_visit.squeeze(0)

        if class_heat.ndim == 2:
            class_heat = class_heat[None, ...]  # Upgrade [1, H, W] to [K, H, W]

        H, W = visited.shape  # Reference shape

        # Adaptive normalization for time_since_visit
        tsv = np.where(last_visit >= 0, timestamp - last_visit, timestamp)
        # Exponential decay: older cells approach 1, recent cells approach 0
        tsv = 1.0 - np.exp(-tsv / 100.0)  # 100 steps ≈ 0.63, 500 ≈ 0.99
        tsv = tsv.astype(np.float32)

        # Smooth log scaling for visit_count
        vcnt = visit_cnt.astype(np.float32)
        vcnt = np.tanh(np.log1p(vcnt) / 3.0)  # 1→0.31, 10→0.63, 100→0.93

        # Stack six static channels with an explicit order
        base_stack = np.stack([
            visited,  # Ch0: Binary mask (0/1)
            pf,  # Ch1: Occupancy probability (0..1)
            conf,  # Ch2: Confidence (0..1)
            frontier,  # Ch3: Frontier mask (0/1)
            tsv,  # Ch4: Time since visit (0..1)
            vcnt  # Ch5: Visit count normalized (0..1)
        ], axis=0)  # Shape: [6, H, W]

        # Sanity check
        assert base_stack.shape == (6, H, W), f"Base stack shape mismatch: {base_stack.shape}"

        # Process class heatmaps
        expected_class_channels = self.num_class_channels

        if class_heat.ndim != 3:
            class_heat = np.zeros((expected_class_channels, H, W), dtype=np.float32)
        else:
            class_heat = class_heat.astype(np.float32)
            current_channels = class_heat.shape[0]

            if current_channels < expected_class_channels:
                pad_shape = (expected_class_channels - current_channels, H, W)
                pad = np.zeros(pad_shape, dtype=np.float32)
                class_heat = np.concatenate([class_heat, pad], axis=0)

            elif current_channels > expected_class_channels:
                channel_importance = class_heat.reshape(current_channels, -1).max(axis=1)
                top_k_indices = np.argsort(channel_importance)[-expected_class_channels:]
                top_k_indices = np.sort(top_k_indices)  # Preserve original ordering
                class_heat = class_heat[top_k_indices]

        feats = np.concatenate([base_stack, class_heat], axis=0)  # [6+K, H, W]

        meta = {
            "coverage": float(visited.mean()) if visited.size else 0.0,
            "mean_confidence_visited": float(conf[visited > 0.5].mean()) if np.any(visited > 0.5) else 0.0,
            "num_frontiers": float(frontier.sum()),
            "class_index": dense.get("class_index", {}),
        }

        return feats, meta

    def _normalize_features(self, feats):
        """
        Post-processing: channel-wise clipping and smoothing
        """
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 2:
            feats = feats[None, ...]

        C, H, W = feats.shape

        # Ensure all base features remain within [0, 1]
        for c in range(min(C, self.static_feature_channels)):
            feats[c] = np.clip(feats[c], 0.0, 1.0)

        # Class heatmaps: apply gamma correction for better contrast
        if C > self.static_feature_channels:
            class_start = self.static_feature_channels
            for c in range(class_start, C):
                # Gamma = 0.7 amplifies high activations while keeping range [0, 1]
                feats[c] = np.clip(feats[c], 0.0, 1.0) ** 0.7

        return feats

    def _canonicalize_feature_order(self, feats: np.ndarray) -> np.ndarray:
        """
        Ensure the first six channels follow the canonical order:
        [visited, p_free, confidence, frontier, tsv, vcnt].
        If the input follows the legacy schema ([pf, conf, visited, tsv, frontier, vcnt]),
        deterministically permute it into the canonical layout.
        """
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 2:
            feats = feats[None, ...]
        if feats.ndim != 3 or feats.shape[0] < 6:
            return feats  # Nothing to do

        C, H, W = feats.shape

        # Detect pf-first ordering (pf ≈ 0.5 everywhere, not binary)
        ch0 = feats[0]
        ch0_mean = float(ch0.mean())
        ch0_nonzero = float((ch0 > 0).mean())
        ch0_unique = np.unique(ch0.round(3))
        looks_like_pfree_first = (0.45 <= ch0_mean <= 0.55) and (ch0_nonzero > 0.95) and (ch0_unique.size > 2)

        if looks_like_pfree_first:
            # Legacy -> canonical mapping: [pf, conf, visited, tsv, frontier, vcnt] -> [visited, pf, conf, frontier, tsv, vcnt]
            perm = [2, 0, 1, 4, 3, 5]
            base = feats[perm, ...]
            rest = feats[6:, ...] if C > 6 else np.zeros((0, H, W), dtype=np.float32)
            feats = np.concatenate([base, rest], axis=0)

        return feats

    def _ensure_channel_count(self, feats):
        """Pad or truncate to base_channels."""
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 2:
            feats = feats[None, ...]
        if feats.ndim != 3:
            raise ValueError(f"Expected 3D array, got {feats.shape}")

        c, h, w = feats.shape
        if c < self.base_channels:
            pad = np.zeros((self.base_channels - c, h, w), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        elif c > self.base_channels:
            feats = feats[: self.base_channels]
        return feats

    def _encode_class_semantics(self, feats, meta):
        """
        Encode class heatmaps with a stable vocabulary of class embeddings.
        """
        C, H, W = feats.shape
        class_index = meta.get("class_index", {})

        if not class_index or C <= self.static_feature_channels:
            return self._ensure_channel_count(feats)

        # Base features (first six static channels)
        base_feats = feats[: self.static_feature_channels]
        class_heat = feats[self.static_feature_channels:]

        new_classes = []
        for name in class_index.keys():
            if name not in self.class_vocab and len(self.class_vocab) < self.max_vocab_size:
                self.class_vocab[name] = len(self.class_vocab)
                new_classes.append(name)

        # Aggregate heatmaps via the embedding slots
        aggregated = np.zeros((self.num_class_channels, H, W), dtype=np.float32)

        for class_name, map_idx in class_index.items():
            if class_name not in self.class_vocab:
                continue  # Skip unknown classes

            vocab_idx = self.class_vocab[class_name]
            if map_idx >= class_heat.shape[0]:
                continue

            heat = class_heat[map_idx]  # [H, W]

            max_vis = heat.max()
            if max_vis > 0:
                slot = min(vocab_idx, self.num_class_channels - 1)
                aggregated[slot] = np.maximum(aggregated[slot], heat)

        return np.concatenate([base_feats, aggregated], axis=0)

class NeuralMapEncoder(nn.Module):

    def __init__(self, output_dim=64, obs_dim=512, read_dim=64):
        super().__init__()
        self.obs_dim  = obs_dim
        self.read_dim = read_dim
        self.output_dim = output_dim

        self.C = 192
        self._mem_shape = None

        # persistent buffers for incremental usage
        self.register_buffer("memory", torch.zeros(1, self.C, 1, 1))
        self.register_buffer("memory_mask", torch.zeros(1, 1, 1, 1))

        # Global Read CNN
        self.global_cnn = nn.Sequential(
            nn.Conv2d(self.C, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.global_proj = nn.Linear(128, self.read_dim)

        # Context Read (content-based)
        self.key_proj   = nn.Conv2d(self.C, 128, 1)
        self.val_proj   = nn.Conv2d(self.C, 128, 1)
        self.query_proj = nn.Linear(self.obs_dim, 128)
        self.ctx_out    = nn.Linear(128, self.read_dim)

        self.write_fc = nn.Sequential(
            nn.Linear(self.obs_dim + self.read_dim + self.C, 2*self.C),
            nn.ReLU(inplace=True),
            nn.Linear(2*self.C, 2*self.C)
        )

        self.out_proj = nn.Linear(self.read_dim*2, self.output_dim)
        self.readout_proj = nn.Linear(self.read_dim, self.output_dim)
        self._readout_input_adapter = None
        self.env_proj = nn.Conv2d(4, self.C, kernel_size=1, bias=False)  # 4->192 anchor
        self.env_blend = 0.15

    def _extract_map_meta(self, map_dict):
        H = W = 64
        env_memory = None
        if isinstance(map_dict, dict):
            shape = map_dict.get("map_shape") or map_dict.get("shape") or map_dict.get("size")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                H, W = int(shape[0]), int(shape[1])
            elif isinstance(shape, dict):
                H = int(shape.get("H") or shape.get("h") or shape.get("height", H))
                W = int(shape.get("W") or shape.get("w") or shape.get("width", W))

            raw_memory = map_dict.get("memory")
            env_memory = self._coerce_env_memory_tensor(raw_memory, torch.device("cpu"))
            if env_memory is not None:
                H = int(env_memory.size(-2))
                W = int(env_memory.size(-1))

        return H, W, env_memory

    def _coerce_env_memory_tensor(self, env_memory, device):
        """Return a tensorised memory snapshot or ``None`` if unsupported."""

        if env_memory is None:
            return None

        if isinstance(env_memory, dict):
            local = env_memory.get("local")
            global_ = env_memory.get("global")
            if local is not None and global_ is not None:
                try:
                    local_t = torch.as_tensor(local, dtype=torch.float32, device=device)
                    global_t = torch.as_tensor(global_, dtype=torch.float32, device=device)
                except Exception:
                    return None

                if local_t.dim() == 3:
                    local_t = local_t.unsqueeze(0)
                if global_t.dim() == 3:
                    global_t = global_t.unsqueeze(0)

                if local_t.dim() != 4 or global_t.dim() != 4:
                    return None

                if local_t.size(-2) != global_t.size(-2) or local_t.size(-1) != global_t.size(-1):
                    return None

                return torch.cat([local_t, global_t], dim=1)

        try:
            tensor = torch.as_tensor(env_memory, dtype=torch.float32, device=device)
        except Exception:
            return None

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 4:
            if tensor.size(0) != 1:
                return None
        else:
            return None

        channels = tensor.size(1)
        if channels not in (4, 8):
            return None

        return tensor

    def _project_env_seed(self, env_tensor: torch.Tensor) -> torch.Tensor:
        """Project environment memory seeds for both 4- and 8-channel formats."""

        if env_tensor.dim() != 4 or env_tensor.size(0) != 1:
            raise ValueError(f"Expected env tensor with shape [1,C,H,W], got {tuple(env_tensor.shape)}")

        channels = env_tensor.size(1)
        if channels == 4:
            return self.env_proj(env_tensor)
        if channels == 8:
            local = env_tensor[:, :4]
            global_ = env_tensor[:, 4:]
            local_proj = self.env_proj(local)
            global_proj = self.env_proj(global_)
            return 0.5 * (local_proj + global_proj)

        raise ValueError(f"Unsupported environment seed with {channels} channels")

    def _init_memory(self, B, map_list, device):
        H, W = 64, 64
        env_seed = None
        map_iter = map_list or []
        for batch_seq in map_iter:
            if not batch_seq:
                continue
            for m in batch_seq:
                if not isinstance(m, dict):
                    continue
                H, W, mem = self._extract_map_meta(m)
                if mem is not None:
                    env_seed = mem.to(device)
                    break
            if env_seed is not None:
                break

        M = torch.zeros(B, self.C, H, W, device=device)
        if env_seed is not None:
            projected = self._project_env_seed(env_seed)
            M = M + projected.expand(B, -1, -1, -1)
        self._mem_shape = (H, W)
        return M

    def reset_state(self, map_dict=None):
        """Reset persistent memory buffers, optionally seeding from a map snapshot."""
        device = next(self.parameters()).device
        H, W, env_memory = self._extract_map_meta(map_dict)
        memory = torch.zeros(1, self.C, H, W, device=device)
        mask = torch.zeros(1, 1, H, W, device=device)

        if env_memory is not None:
            env_tensor = env_memory.to(device)
            projected = self._project_env_seed(env_tensor)
            memory = memory + self.env_blend * projected
            if env_tensor.size(1) > 0:
                mask = mask + (env_tensor.abs().sum(dim=1, keepdim=True) > 0).float()

        self.memory = memory
        self.memory_mask = mask
        self._mem_shape = (H, W)

    def get_state(self):
        if self.memory is None or self.memory.numel() == 0 or self._mem_shape is None:
            return None
        return {
            "map_shape": tuple(self._mem_shape),
            "memory": self.memory.detach().clone().cpu(),
            "mask": self.memory_mask.detach().clone().cpu(),
        }

    def _ensure_state(self, map_dict=None):
        device = next(self.parameters()).device
        if self.memory is None or self.memory.numel() == 0 or self._mem_shape is None:
            self.reset_state(map_dict)
        else:
            if map_dict is not None:
                H, W, _ = self._extract_map_meta(map_dict)
                if (H, W) != tuple(self._mem_shape or (None, None)):
                    self.reset_state(map_dict)
                else:
                    self.memory = self.memory.to(device)
                    self.memory_mask = self.memory_mask.to(device)
            else:
                self.memory = self.memory.to(device)
                self.memory_mask = self.memory_mask.to(device)

    def _global_read(self, M):
        g = self.global_cnn(M).flatten(1)       # [B,128]
        return self.global_proj(g)              # [B,read_dim]

    def _context_read(self, M, obs_feat):
        K = self.key_proj(M)                    # [B,128,H,W]
        V = self.val_proj(M)                    # [B,128,H,W]
        B, D, H, W = K.shape
        Q = self.query_proj(obs_feat)           # [B,128]
        Kf = K.view(B, D, H*W)                  # [B,128,HW]
        Vf = V.view(B, D, H*W)                  # [B,128,HW]
        attn = torch.bmm(Q.unsqueeze(1), Kf) / (D**0.5)  # [B,1,HW]
        attn = F.softmax(attn, dim=-1)
        ctx  = torch.bmm(attn, Vf.transpose(1,2)).squeeze(1)  # [B,128]
        return self.ctx_out(ctx)                 # [B,read_dim]

    def _write(self, M, obs_feat, g_read, ij_idx, mask=None, inplace=False):
        """ij_idx: LongTensor [B, 2] with (i, j); -1 marks a skipped write."""
        B, C, H, W = M.shape
        device = M.device

        i_idx = ij_idx[:, 0].long()
        j_idx = ij_idx[:, 1].long()
        valid = (i_idx >= 0) & (j_idx >= 0) & (i_idx < H) & (j_idx < W)

        if not valid.any():
            return (M, mask) if mask is not None else M

        rows = torch.arange(B, device=device)[valid]
        local = M[rows, :, i_idx[valid], j_idx[valid]]  # [Bv, C]

        x = torch.cat([obs_feat[valid], g_read[valid], local], dim=1)  # [Bv, obs+read+C]
        out = self.write_fc(x)  # [Bv, 2C]
        cand, gate_pre = torch.split(out, C, dim=1)
        gate = torch.sigmoid(gate_pre)
        blended = gate * cand + (1.0 - gate) * local  # [Bv, C]

        # Avoid in-place writes on tensors that participate in autograd by
        # building a boolean mask of the cells that should be updated and then
        # blending a freshly allocated tensor with the previous memory. This
        # keeps the computation graph intact while still applying the same
        # gated update as the original in-place implementation.
        write_mask = torch.zeros_like(M, dtype=torch.bool)
        write_mask[rows, :, i_idx[valid], j_idx[valid]] = True

        blended_full = torch.zeros_like(M)
        blended_full[rows, :, i_idx[valid], j_idx[valid]] = blended

        target_M = torch.where(write_mask, blended_full, M)

        if mask is None:
            return target_M

        mask_updates = torch.zeros_like(mask)
        mask_updates[rows, :, i_idx[valid], j_idx[valid]] = 1.0
        target_mask = torch.where(mask_updates.bool(), mask_updates, mask)
        return target_M, target_mask

    def forward_step(self, obs_feat_1, ij_idx_1, map_meta=None, read_vector=None):
        """
        Online step with persistent state.
        obs_feat_1: [1, D_obs], ij_idx_1: Long[1, 2], map_dict: metadata for map size.
        returns: [1, output_dim]
        """
        if read_vector is not None:
            rv = read_vector if isinstance(read_vector, torch.Tensor) else torch.as_tensor(
                read_vector, dtype=torch.float32, device=obs_feat_1.device
            )
            rv = rv.to(obs_feat_1.device)
            proj = self._project_read_vectors(rv)
            if proj.dim() == 1:
                proj = proj.unsqueeze(0)
            return proj

        self._ensure_state(map_meta)

        B, _, _, _ = self.memory.shape
        assert B == 1, "forward_step is defined for batch size 1."

        g = self._global_read(self.memory)  # [1,read_dim]
        c = self._context_read(self.memory, obs_feat_1)  # [1,read_dim]
        self.memory, self.memory_mask = self._write(
            self.memory, obs_feat_1, g, ij_idx_1, mask=self.memory_mask, inplace=True
        )
        out = self.out_proj(torch.cat([g, c], dim=-1))  # [1, output_dim]
        return out

    def _set_state(self, state):
        """Load a cached state (memory + mask) if provided."""

        if state is None:
            return False

        memory = state.get("memory") if isinstance(state, dict) else None
        mask = state.get("mask") if isinstance(state, dict) else None
        map_shape = state.get("map_shape") if isinstance(state, dict) else None

        if memory is None:
            return False

        device = next(self.parameters()).device
        memory = memory if isinstance(memory, torch.Tensor) else torch.as_tensor(
            memory, dtype=torch.float32, device=device
        )
        memory = memory.to(device)

        if memory.dim() != 4:
            return False

        if mask is None:
            mask = torch.zeros(memory.size(0), 1, memory.size(-2), memory.size(-1), device=device)
        else:
            mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(
                mask, dtype=torch.float32, device=device
            )
            mask = mask.to(device)

        if mask.dim() != 4:
            return False

        if map_shape is None:
            map_shape = (memory.size(-2), memory.size(-1))

        self.memory = memory
        self.memory_mask = mask
        self._mem_shape = (int(map_shape[0]), int(map_shape[1]))
        return True

    def forward_seq(self, obs_feat_seq, ij_seq, map_list, read_seq=None, state=None):
        """
        obs_feat_seq: [B, T, D_obs]
        ij_seq:       [B, T, 2] (Long; -1/-1 when no index)
        map_list:     [B, T]   (used to initialize H, W)
        returns:      [B, T, output_dim]
        """
        device = next(self.parameters()).device
        if read_seq is not None:
            read = read_seq if isinstance(read_seq, torch.Tensor) else torch.as_tensor(
                read_seq, dtype=torch.float32, device=device
            )
            if read.dim() == 2:
                read = read.unsqueeze(1)
            proj = self._project_read_vectors(read)
            return proj, None

        B, T, _ = obs_feat_seq.shape

        seed_map = None
        if map_list:
            for seq in map_list:
                for item in seq:
                    if isinstance(item, dict):
                        seed_map = item
                        break
                if seed_map is not None:
                    break

        if state is None or not self._set_state(state):
            self.reset_state(seed_map)

        H, W = self._mem_shape

        base_memory = self.memory.to(device)
        base_mask = self.memory_mask.to(device)

        mem_batch = base_memory.size(0)
        if mem_batch == B:
            M = base_memory.clone()
        elif mem_batch == 1:
            M = base_memory.expand(B, -1, -1, -1).clone()
        else:
            # Evaluation loaders often yield a final batch that is smaller than
            # the training batch size.  Instead of erroring out when the
            # persistent cache still holds the previous batch dimension we fall
            # back to the first cached element and expand it to the requested
            # batch size.  This mirrors the behaviour of the historical code
            # path which effectively only kept a single memory entry anyway.
            M = base_memory[:1].expand(B, -1, -1, -1).clone()

        mask_batch = base_mask.size(0)
        if mask_batch == B:
            mask = base_mask.clone()
        elif mask_batch == 1:
            mask = base_mask.expand(B, -1, -1, -1).clone()
        else:
            mask = base_mask[:1].expand(B, -1, -1, -1).clone()

        outs = []
        for t in range(T):
            obs_t = obs_feat_seq[:, t, :]        # [B,D_obs]
            ij_t  = ij_seq[:, t, :]              # [B,2]
            g = self._global_read(M)             # [B,read_dim]
            c = self._context_read(M, obs_t)
            M, mask = self._write(M, obs_t, g, ij_t, mask=mask, inplace=False)
            out_t = self.out_proj(torch.cat([g, c], dim=-1)) # [B,output_dim]
            outs.append(out_t.unsqueeze(1))

        output = torch.cat(outs, dim=1)            # [B,T,output_dim]

        cached_state = {
            "map_shape": (H, W),
            "memory": M.detach().clone(),
            "mask": mask.detach().clone(),
        }

        self.memory = cached_state["memory"]
        self.memory_mask = cached_state["mask"]
        self._mem_shape = (H, W)

        return output, cached_state

    def _ensure_readout_adapter(self, in_dim: int):
        """Ensure we can project external read vectors to ``self.read_dim``.

        Historic rollouts sometimes store read vectors with a different
        dimensionality than the ``read_dim`` that the encoder was
        initialised with.  When such vectors are replayed during IL
        training we adapt them on-the-fly via a small linear projection so
        they can still be consumed by the current model weights.
        """

        target_dim = self.readout_proj.in_features
        if in_dim == target_dim:
            return None

        needs_new = (
            self._readout_input_adapter is None
            or self._readout_input_adapter.in_features != in_dim
            or self._readout_input_adapter.out_features != target_dim
        )

        if needs_new:
            self._readout_input_adapter = nn.Linear(in_dim, target_dim)
            self._readout_input_adapter.to(self.readout_proj.weight.device)

        return self._readout_input_adapter

    def _project_read_vectors(self, read_tensor: torch.Tensor) -> torch.Tensor:
        """Project arbitrary read vectors to the encoder output space."""

        if read_tensor.dim() == 1:
            read_tensor = read_tensor.unsqueeze(0)

        orig_shape = read_tensor.shape[:-1]
        in_dim = read_tensor.shape[-1]
        flat = read_tensor.reshape(-1, in_dim)

        adapter = self._ensure_readout_adapter(in_dim)
        if adapter is not None:
            flat = adapter(flat)

        proj = self.readout_proj(flat)
        return proj.view(*orig_shape, -1)

    def encode_read_sequence(self, read_seq):
        device = next(self.parameters()).device
        read = read_seq if isinstance(read_seq, torch.Tensor) else torch.as_tensor(
            read_seq, dtype=torch.float32, device=device
        )
        if read.dim() == 2:
            read = read.unsqueeze(1)
        proj = self._project_read_vectors(read)
        return proj
