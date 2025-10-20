import json
import math
import os
import re
from itertools import zip_longest

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch_geometric.data import Data, HeteroData
from torchvision.models import resnet18, ResNet18_Weights

from src.components.encoders.exploration_encoder import (
    ExplorationMapEncoder,
    MetricSemanticMapV2Encoder,
    NeuralMapEncoder,
)
from src.components.models.graph_encoder import NodeEdgeHGTEncoder


class FeatureEncoder(nn.Module):
    """
    Encodes the multimodal agent state including:
    - RGB image via ResNet18
    - Last action via embedding
    - Occupancy map via CNN
    - Local and global scene graphs via LSTM **or** Transformer (configurable)
    Combines all features into a single state vector.
    """

    def __init__(
        self,
        num_actions,
        rgb_dim=512,
        map_dim=64,
        action_dim=32,
        sg_dim=256,
        obj_embedding_dim=128,
        max_object_types=1000,
        rel_embedding_dim=64,
        max_relation_types=50,
        use_transformer=False,
        use_map=True,
        mapping_path=None,
        exploration_mode="raster",
        use_scene_graph=True,
        use_rgb=True,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.rgb_encoder = ResNetFeatureExtractor(rgb_dim)
        self.action_emb = ActionEmbedding(num_actions, action_dim)

        self.sg_dim = sg_dim
        self.use_transformer = use_transformer
        self.use_map = use_map
        self.use_scene_graph = use_scene_graph
        self.use_rgb = use_rgb

        SGEncoderClass = SceneGraphTransformerEncoder if use_transformer else SceneGraphLSTMEncoder

        self.lssg_encoder = SGEncoderClass(input_dim=sg_dim, hidden_dim=sg_dim)
        self.gssg_encoder = SGEncoderClass(input_dim=sg_dim, hidden_dim=sg_dim)
        self.node_att_vector = nn.Parameter(torch.randn(int(sg_dim / 2)))
        self.edge_att_vector = nn.Parameter(torch.randn(int(sg_dim / 2)))

        self.object_to_idx = {}
        self.relation_to_idx = {}  # New mapping for relations

        self.max_object_types = max_object_types
        self.max_relation_types = max_relation_types
        self.obj_type_embedding = nn.Embedding(max_object_types, obj_embedding_dim)
        self.rel_type_embedding = nn.Embedding(max_relation_types, rel_embedding_dim)  # New embedding layer

        self.mapping_path = mapping_path
        if mapping_path and os.path.exists(os.path.join(mapping_path, "object_types.json")):
            self.load_mappings(mapping_path)

        relation_types = list(self.relation_to_idx.keys())
        graph_encoder_in_channels = 4 + obj_embedding_dim  # Node features: visibility + pos (x, y, z) + obj_embedding
        self.graph_feature_extractor = NodeEdgeHGTEncoder(
            in_channels=graph_encoder_in_channels,
            edge_in_channels=rel_embedding_dim,
            hidden_channels=128,
            out_channels=int(sg_dim / 2),
            relation_types=relation_types,
        )

        self.map_dim = map_dim
        self.exploration_mode = exploration_mode # raster | neural
        print(f"[INFO] Exploration mode: {self.exploration_mode}")
        obs_base_dim = rgb_dim + action_dim + 2 * sg_dim  #  Features without Map

        if self.use_map:
            if self.exploration_mode == "neural":
                self.map_encoder = NeuralMapEncoder(output_dim=map_dim, obs_dim=obs_base_dim,
                                                    read_dim=min(map_dim, 128))
            elif self.exploration_mode == "raster_v2":
                self.map_encoder = MetricSemanticMapV2Encoder(output_dim=map_dim, include_agent=True)
            else:
                self.map_encoder = ExplorationMapEncoder(output_dim=map_dim, include_agent=True)
        else:
            self.map_encoder = None


        self.object_count = len(self.object_to_idx)
        self.relation_count = len(self.relation_to_idx)
        self._map_hidden_state = None

    @staticmethod
    def _normalize_map_index(idx):
        if idx is None:
            return None

        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().tolist()

        if isinstance(idx, np.ndarray):
            idx = idx.tolist()

        if isinstance(idx, dict):
            idx = (
                idx.get("i", idx.get("row")),
                idx.get("j", idx.get("col")),
            )

        if isinstance(idx, (list, tuple)):
            if len(idx) < 2:
                return None
            candidate = (idx[0], idx[1])
        else:
            candidate = idx

        if not isinstance(candidate, tuple):
            return None

        try:
            i = int(candidate[0])
            j = int(candidate[1])
        except (TypeError, ValueError):
            return None

        if i < 0 or j < 0:
            return None

        return i, j

    @staticmethod
    def _inject_map_indices(map_sequences, index_sequences):
        if not map_sequences or index_sequences is None:
            return

        for map_seq, idx_seq in zip_longest(map_sequences, index_sequences, fillvalue=None):
            if not map_seq:
                continue

            for t, map_entry in enumerate(map_seq):
                if not isinstance(map_entry, dict):
                    continue

                idx = None
                if idx_seq is not None:
                    try:
                        idx = idx_seq[t]
                    except (IndexError, TypeError):
                        idx = None

                norm_idx = FeatureEncoder._normalize_map_index(idx)
                if norm_idx is None:
                    map_entry.pop("map_index", None)
                    map_entry.pop("agent_index", None)
                else:
                    map_entry["map_index"] = norm_idx
                    map_entry["agent_index"] = norm_idx

    @staticmethod
    def preprocess_rgb(rgb_list):
        """
        Convert a list of RGB inputs (np/PIL/Tensor) to a normalized tensor
        [N, 3, 224, 224]. Replaces padded frames with zeros
        """
        transform = T.Compose(
            [
                T.ToTensor(),  # Converts np.ndarray/PIL → Tensor [0,1]
                T.Resize((224, 224)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        processed = []
        for rgb in rgb_list:
            if rgb is None or (isinstance(rgb, int) and rgb == 0):
                # Padding
                processed.append(torch.zeros(3, 224, 224))
            elif isinstance(rgb, torch.Tensor):
                img = rgb
                if img.ndim == 3:
                    processed.append(transform(img))
                else:
                    processed.append(img)
            else:
                if isinstance(rgb, np.ndarray):
                    rgb = np.ascontiguousarray(rgb)
                elif hasattr(rgb, "copy"):  # e.g. PIL Image
                    rgb = np.array(rgb).copy()
                processed.append(transform(rgb))
        return torch.stack(processed)  # [N, 3, H, W]

    def create_hgt_data(self, sg, device):
        if sg is None or (isinstance(sg, int) and sg == 0) or not sg.nodes:
            return None

        data = HeteroData()

        node_id_map = {node_id: i for i, node_id in enumerate(sg.nodes)}

        # --- Node Features ---
        node_positions, object_type_indices, visibilities = [], [], []
        for node in sg.nodes.values():
            node_positions.append(node.position)
            obj_type_idx = self.object_to_idx.setdefault(node.name, len(self.object_to_idx))
            object_type_indices.append(obj_type_idx)
            visibilities.append(getattr(node, "visibility", 1.0))
        obj_indices_tensor = torch.tensor(object_type_indices, dtype=torch.long, device=device)
        obj_embeddings = self.obj_type_embedding(obj_indices_tensor)
        pos_tensor = torch.tensor(node_positions, dtype=torch.float32, device=device)
        vis_tensor = torch.tensor(visibilities, dtype=torch.float32, device=device).unsqueeze(1)
        x = torch.cat([pos_tensor, vis_tensor, obj_embeddings], dim=1)
        data["object"].x = x

        # --- Edge Features ---
        if not self.relation_to_idx:
            # No known relations -> just return node-only graph
            return data

        for rel_type in self.relation_to_idx:
            sources, targets, edge_attr_idx = [], [], []
            for edge in sg.edges:
                if edge.relation == rel_type:
                    if edge.source in node_id_map and edge.target in node_id_map:
                        sources.append(node_id_map[edge.source])
                        targets.append(node_id_map[edge.target])
                        idx = self.relation_to_idx[rel_type]
                        edge_attr_idx.append(idx)

            edge_type = ("object", rel_type, "object")
            if sources:
                data[edge_type].edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
                edge_attr_tensor = self.rel_type_embedding(torch.tensor(edge_attr_idx, dtype=torch.long, device=device))
                data[edge_type].edge_attr = edge_attr_tensor
            else:
                # --- make sure empty edges exist ---
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                data[edge_type].edge_attr = torch.empty((0, self.rel_type_embedding.embedding_dim), device=device)

        return data

    def _create_gat_data(self, sg, device):
        """
        Legacy homogeneous graph conversion for GAT-style models. Produces
        Data(x, edge_index, edge_attr) with object and relation embeddings.
        """

        if sg is None or (isinstance(sg, int) and sg == 0) or not sg.nodes:
            return None

        node_id_map = {node_id: i for i, node_id in enumerate(sg.nodes)}

        # --- 1. Create Node Features (x) ---
        node_positions = []
        object_type_indices = []
        visibilities = []
        for node in sg.nodes.values():
            node_positions.append(node.position)
            # Add new object types to the mapping if they don't exist
            obj_type_idx = self.object_to_idx.setdefault(node.name, len(self.object_to_idx))
            assert obj_type_idx < self.max_object_types, "Exceeded max_object_types!"
            object_type_indices.append(obj_type_idx)
            visibilities.append(node.visibility if hasattr(node, "visibility") else 1.0)

        # Get embeddings for all object types in the graph at once
        obj_indices_tensor = torch.tensor(object_type_indices, dtype=torch.long, device=device)
        obj_embeddings = self.obj_type_embedding(obj_indices_tensor)
        pos_tensor = torch.tensor(node_positions, dtype=torch.float32, device=device)
        vis_tensor = torch.tensor(visibilities, dtype=torch.float32, device=device).unsqueeze(1)

        # Node features: pos (3) + vis (1) + obj_embedding (d)
        x = torch.cat([pos_tensor, vis_tensor, obj_embeddings], dim=1)

        # --- 2. Create Edge Index and Edge Attributes ---
        source_nodes, target_nodes, relation_indices = [], [], []
        for edge in sg.edges:
            if edge.source in node_id_map and edge.target in node_id_map:
                source_nodes.append(node_id_map[edge.source])
                target_nodes.append(node_id_map[edge.target])

                # Get or create an integer index for the relation type
                rel_idx = self.relation_to_idx.setdefault(edge.relation, len(self.relation_to_idx))
                assert rel_idx < self.max_relation_types, "Exceeded max_relation_types!"
                relation_indices.append(rel_idx)

        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long, device=device)

        # Get embeddings for all relation types in the graph
        rel_indices_tensor = torch.tensor(relation_indices, dtype=torch.long, device=device)
        edge_attr = self.rel_type_embedding(rel_indices_tensor)  # This creates the edge features

        # Update mappings if new objects or relations were added
        if self.mapping_path and (len(self.object_to_idx) > self.object_count or len(self.relation_to_idx) > self.relation_count):
            self.save_mappings(self.mapping_path)
            self.object_count = len(self.object_to_idx)
            self.relation_count = len(self.relation_to_idx)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def attention_pooling(self, features, att_vector):
        """
        Attention-weighted pooling over node/edge features using a learnable
        context vector.

        Returns a single pooled vector; zeros if no features
        """
        if features.size(0) == 0:
            return torch.zeros_like(att_vector)
        scores = torch.matmul(features, att_vector)
        weights = torch.softmax(scores, dim=0)
        pooled = torch.sum(weights.unsqueeze(-1) * features, dim=0)
        return pooled

    def get_graph_features(self, sg_list: list):
        """
        Encode a list of scene graphs via HGT and attention pooling to obtain one
        sg_dim-sized embedding per graph.

        Returns (embeddings, valid_indices)
        """

        device = next(self.parameters()).device
        data_list, valid_indices = [], []

        for i, sg in enumerate(sg_list):
            data = self.create_hgt_data(sg, device)
            if data is None:
                continue

            # Skip if no nodes at all
            if "object" not in data or not hasattr(data["object"], "x") or data["object"].x.size(0) == 0:
                continue

            # If no edges: we still add it, but with empty edges
            if len(data.edge_types) == 0:
                # create an empty edge type so PyG doesn't crash
                rel_type = ("object", "no_edge", "object")
                data[rel_type].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                data[rel_type].edge_attr = torch.empty((0, self.rel_type_embedding.embedding_dim), device=device)

            data_list.append(data)
            valid_indices.append(i)

        if not data_list:
            pooled_features = torch.zeros((0, self.sg_dim), device=device)
            return pooled_features, torch.tensor([], dtype=torch.long, device=device)

        graph_embeds = []
        for d in data_list:
            try:
                node_out, edge_out = self.graph_feature_extractor(d)
            except KeyError:
                # fallback if edge dict is still empty
                node_out, edge_out = torch.zeros((0, int(self.sg_dim / 2)), device=device), torch.zeros(
                    (0, int(self.sg_dim / 2)), device=device)

            node_pooled = (
                self.attention_pooling(node_out, self.node_att_vector)
                if node_out.shape[0] > 0 else torch.zeros_like(self.node_att_vector)
            )
            edge_pooled = (
                self.attention_pooling(edge_out, self.edge_att_vector)
                if edge_out.shape[0] > 0 else torch.zeros_like(self.edge_att_vector)
            )
            graph_embeds.append(torch.cat([node_pooled, edge_pooled], dim=-1))

        pooled_features = torch.stack(graph_embeds, dim=0)  # [n_valid, sg_dim]
        return pooled_features, torch.tensor(valid_indices, dtype=torch.long, device=device)

    @staticmethod
    def _extract_map(observation, *, prefer_policy: bool = False):
        """
        Gibt eine Map-Repräsentation (dict oder kompatible Struktur) zurück.
        prefer_policy=True NUR für metric_semantic_v2 verwenden.
        """
        exp_map = None
        # 1) aus state[3], wenn vorhanden
        if hasattr(observation, "state") and len(observation.state) > 3:
            exp_map = observation.state[3]

        # 2) optional policy_map bevorzugen (nur raster_v2)
        if prefer_policy:
            info = getattr(observation, "info", {}) or {}
            policy_map = info.get("policy_map")
            if policy_map is not None:
                if isinstance(policy_map, tuple) and len(policy_map) == 2:
                    exp_map = {"dense": policy_map[0], "meta": policy_map[1]}
                else:
                    exp_map = policy_map

        # 3) ansonsten auf exploration_map zurückfallen
        if not isinstance(exp_map, dict):
            info = getattr(observation, "info", {}) or {}
            if exp_map is None:  # nur wenn noch nichts Sinnvolles da ist
                exp_map = info.get("exploration_map")

        # 4) ggf. to_dict() (nur wenn Objekt das kann; Listen/Tuples intakt lassen)
        if not isinstance(exp_map, dict) and hasattr(exp_map, "to_dict"):
            try:
                exp_map = exp_map.to_dict()
            except Exception:
                pass

        return exp_map

    def obs_to_dict(self, obs):
        """
        Converts an Observation or a list of Observations to a dict for feature extraction.
        Handles both single observations and temporal sequences.
        Returns a dict matching the input structure expected by forward_seq.
        """
        if self.use_map and self.exploration_mode == "neural":
            if isinstance(obs, list):

                rgb = [o.state[0] for o in obs]
                lssg = [o.state[1] for o in obs]
                gssg = [o.state[2] for o in obs]
                prefer = (self.exploration_mode == "raster_v2")
                exp_map = [self._extract_map(o, prefer_policy=prefer) for o in obs]

                agent_pos = [o.info.get("agent_pos", None) for o in obs]
                map_index = [o.info.get("map_index", None) for o in obs]
                policy_map = [o.info.get("policy_map", None) for o in obs]
                return {
                    "rgb": [rgb],
                    "lssg": [lssg],
                    "gssg": [gssg],
                    "map": [exp_map],
                    "agent_pos": [agent_pos],
                    "map_index": [map_index],
                    "map_policy": [policy_map],
                }
            else:
                # Single Observation (batch=1, T=1)
                rgb = obs.state[0]
                lssg = obs.state[1]
                gssg = obs.state[2]
                prefer = (self.exploration_mode == "raster_v2")
                exp_map = self._extract_map(obs, prefer_policy=prefer)

                # Convert map managers into serializable payloads when needed.
                if not isinstance(exp_map, dict) and hasattr(exp_map, "to_dict"):
                    exp_map = exp_map.to_dict()

                agent_pos = obs.info.get("agent_pos", None)
                map_index = obs.info.get("map_index", None)

                policy_map = obs.info.get("policy_map", None)
                return {
                    "rgb": [[rgb]],
                    "lssg": [[lssg]],
                    "gssg": [[gssg]],
                    "map": [[exp_map]],
                    "agent_pos": [[agent_pos]],
                    "map_index": [[map_index]],
                    "map_policy": [[policy_map]],
                }
        else:
            if isinstance(obs, list):
                # Sequence of Observations (single batch, T steps)
                rgb = [o.state[0] for o in obs]
                lssg = [o.state[1] for o in obs]
                gssg = [o.state[2] for o in obs]
                prefer = (self.exploration_mode == "raster_v2")
                exp_map = [self._extract_map(o, prefer_policy=prefer) for o in obs]

                agent_pos = [o.info.get("agent_pos", None) for o in obs]
                map_index = [o.info.get("map_index", None) for o in obs]
                policy_map = [o.info.get("policy_map", None) for o in obs]
                return {
                    "rgb": [rgb],  # [B=1, T]
                    "lssg": [lssg],  # [B=1, T]
                    "gssg": [gssg],  # [B=1, T]
                    "map": [exp_map],  # [B=1, T]
                    "agent_pos": [agent_pos],  # [B=1, T]
                    "map_index": [map_index],
                    "map_policy": [policy_map],
                }
            else:
                # Single Observation (batch=1, T=1)
                rgb = obs.state[0]
                lssg = obs.state[1]
                gssg = obs.state[2]
                prefer = (self.exploration_mode == "raster_v2")
                exp_map = self._extract_map(obs, prefer_policy=prefer)

                # If the manager returns an object, serialize before packing.
                if not isinstance(exp_map, dict) and hasattr(exp_map, "to_dict"):
                    exp_map = exp_map.to_dict()

                agent_pos = obs.info.get("agent_pos", None)
                map_index = obs.info.get("map_index", None)
                policy_map = obs.info.get("policy_map", None)
                return {
                    "rgb": [[rgb]],
                    "lssg": [[lssg]],
                    "gssg": [[gssg]],
                    "map": [[exp_map]],
                    "agent_pos": [[agent_pos]],
                    "map_index": [[map_index]],
                    "map_policy": [[policy_map]],
                }  # [B=1, T=1]

    def forward(self, obs, last_action, lssg_hidden=None, gssg_hidden=None):
        """
        Forward pass for a single observation or a sequence.
        Unifies preprocessing so that RL and IL use the same code path.
        obs: Observation or list of Observations
        last_action: int, list[int], or LongTensor [T] or [B,T]
        """
        # 1. Convert obs to batch_dict format
        batch_dict = self.obs_to_dict(obs) if not isinstance(obs, dict) else obs

        # 2. Normalize last_action to shape [B, T]
        device = next(self.parameters()).device
        if isinstance(last_action, int):
            last_action = torch.tensor([[last_action]], dtype=torch.long, device=device)
        elif isinstance(last_action, torch.Tensor):
            if last_action.ndim == 1:
                last_action = last_action.unsqueeze(0)  # [T] -> [1,T]
            elif last_action.ndim == 0:
                last_action = last_action.view(1, 1)
            last_action = last_action.to(device)
        elif isinstance(last_action, list):
            last_action = torch.tensor([last_action], dtype=torch.long, device=device)
        else:
            raise ValueError("last_action must be int, Tensor, or list of int.")

        # 3. Feature extraction via forward_seq (handles [B,T,...])
        return self.forward_seq(batch_dict, last_action, lssg_hidden=lssg_hidden, gssg_hidden=gssg_hidden)  # [B, T, D_total]

    def forward_seq(
        self,
        batch_dict,
        last_actions,
        pad_mask=None,
        lssg_hidden=None,
        gssg_hidden=None,
        neural_map_read=None,
        return_map_features=False,
        map_override=None,
    ):
        """
        Preprocess and encode batch of sequences. Inputs: "raw" batch from seq_collate.
        batch_dict keys: 'map', 'rgb', 'lssg', 'gssg', 'agent_pos'
        """
        device = next(self.parameters()).device
        B, T = len(batch_dict["rgb"]), len(batch_dict["rgb"][0])
        total_steps = B * T

        # 1) RGB + Actions
        rgb_flat = [im for seq in batch_dict["rgb"] for im in seq]
        rgb_tensor = self.preprocess_rgb(rgb_flat).to(device)
        if self.use_rgb:
            rgb_feat = self.rgb_encoder(rgb_tensor)  # [N, rgb_dim]
        else:
            rgb_feat = torch.zeros(rgb_tensor.size(0), self.rgb_encoder.output_dim, device=device, dtype=rgb_tensor.dtype)
        act_flat = last_actions.view(-1).to(device)
        act_feat = self.action_emb(act_flat)  # [N, action_dim]

        # 2) Scene graphs → HGT + attention pooling (optional)
        if self.use_scene_graph:
            lssg_flat = [sg for seq in batch_dict["lssg"] for sg in seq]
            gssg_flat = [sg for seq in batch_dict["gssg"] for sg in seq]
            lssg_embeds, lssg_valid = self.get_graph_features(lssg_flat)  # [n_valid, sg_dim]
            gssg_embeds, gssg_valid = self.get_graph_features(gssg_flat)

            lssg_feat_full = torch.zeros(total_steps, self.sg_dim, device=device)
            gssg_feat_full = torch.zeros(total_steps, self.sg_dim, device=device)
            lssg_feat_full[lssg_valid] = lssg_embeds
            gssg_feat_full[gssg_valid] = gssg_embeds

            lssg_seq = lssg_feat_full.view(B, T, -1)
            gssg_seq = gssg_feat_full.view(B, T, -1)

            if self.use_transformer:
                if "lssg_mask" in batch_dict:
                    lssg_mask = torch.tensor(batch_dict["lssg_mask"], dtype=torch.bool, device=device)
                    gssg_mask = torch.tensor(batch_dict["gssg_mask"], dtype=torch.bool, device=device)
                    lssg_feat = self.lssg_encoder(lssg_seq, pad_mask=~lssg_mask)
                    gssg_feat = self.gssg_encoder(gssg_seq, pad_mask=~gssg_mask)
                else:
                    lssg_feat = self.lssg_encoder(lssg_seq, pad_mask=pad_mask)
                    gssg_feat = self.gssg_encoder(gssg_seq, pad_mask=pad_mask)
            else:
                lssg_feat, lssg_hidden = self.lssg_encoder(lssg_seq, lssg_hidden)
                gssg_feat, gssg_hidden = self.gssg_encoder(gssg_seq, gssg_hidden)

            lssg_feat = lssg_feat.reshape(total_steps, -1)
            gssg_feat = gssg_feat.reshape(total_steps, -1)
        else:
            lssg_feat = torch.zeros(total_steps, self.sg_dim, device=device)
            gssg_feat = torch.zeros(total_steps, self.sg_dim, device=device)
            lssg_hidden = None
            gssg_hidden = None

        base_components = [act_feat, rgb_feat, lssg_feat, gssg_feat]
        base_feat_flat = torch.cat(base_components, dim=-1)  # [N, obs_base_dim]
        base_feat_seq = base_feat_flat.view(B, T, -1)  # [B,T,obs_base_dim]

        # Exploration Map
        map_feats = None
        if self.use_map:
            if map_override is not None:
                if isinstance(map_override, torch.Tensor):
                    map_feats = map_override.to(device)
                else:
                    map_feats = torch.as_tensor(map_override, dtype=torch.float32, device=device)
                if map_feats.dim() == 2:
                    map_feats = map_feats.unsqueeze(0)
                map_feats = map_feats.view(B * T, self.map_dim)
            elif self.exploration_mode == "neural":
                if neural_map_read is not None:
                    if isinstance(neural_map_read, torch.Tensor):
                        read_seq = neural_map_read
                    else:
                        read_seq = torch.as_tensor(neural_map_read, device=device, dtype=torch.float32)

                    if read_seq.dim() == 2:
                        read_seq = read_seq.unsqueeze(1).expand(B, T, -1)
                    elif read_seq.dim() == 3 and read_seq.size(1) == 1 and read_seq.size(1) != T:
                        read_seq = read_seq.expand(B, T, -1)

                    map_feat_seq = self.map_encoder.encode_read_sequence(read_seq)
                    map_feats = map_feat_seq.reshape(B * T, self.map_dim)
                    self._map_hidden_state = None
                else:
                    ij_flat = []
                    for seq in batch_dict.get("map_index", [[None] * T] * B):
                        for idx in seq:
                            if idx is None or idx == (-1, -1):
                                ij_flat.append([-1, -1])
                            else:
                                i, j = (int(idx[0]), int(idx[1])) if isinstance(idx, (list, tuple)) and len(idx) >= 2 else (
                                    -1, -1)
                                ij_flat.append([i, j])

                    ij_tensor = torch.tensor(ij_flat, dtype=torch.long, device=device).view(B, T, 2)
                    map_list = batch_dict["map"]

                    if B == 1 and T == 1:
                        # >>> Online: persistente Karte benutzen
                        obs_feat_1 = base_feat_seq.view(1, -1)  # [1, D_obs]
                        ij_1 = ij_tensor.view(1, 2)  # [1, 2]
                        map_dict = map_list[0][0] if map_list and map_list[0] else {}
                        map_feat_1 = self.map_encoder.forward_step(obs_feat_1, ij_1, map_dict)  # [1, map_dim]
                        map_feats = map_feat_1  # [1, map_dim]
                        map_feats = map_feats.view(B * T, self.map_dim)  # [1, map_dim]
                        if hasattr(self.map_encoder, "get_state"):
                            self._map_hidden_state = self.map_encoder.get_state()
                        else:
                            self._map_hidden_state = None
                    else:
                        # >>> Training: Sequenzverarbeitung wie bisher
                        cached_state = None
                        if self._map_hidden_state is not None:
                            def _coerce_cached_tensor(value):
                                if value is None:
                                    return None
                                tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                                if tensor.dim() == 0:
                                    return None
                                batch_dim = tensor.size(0)
                                if batch_dim not in (1, B):
                                    return None
                                return tensor.detach().clone()

                            memory_tensor = _coerce_cached_tensor(self._map_hidden_state.get("memory"))
                            mask_tensor = _coerce_cached_tensor(self._map_hidden_state.get("mask"))

                            if memory_tensor is not None and (mask_tensor is not None or self._map_hidden_state.get("mask") is None):
                                cached_state = {
                                    "map_shape": self._map_hidden_state.get("map_shape"),
                                    "memory": memory_tensor,
                                }
                                if mask_tensor is not None:
                                    cached_state["mask"] = mask_tensor
                            else:
                                cached_state = None
                        map_feat_seq, next_state = self.map_encoder.forward_seq(
                            base_feat_seq, ij_tensor, map_list, state=cached_state
                        )  # [B,T,map_dim]
                        map_feats = map_feat_seq.reshape(B * T, self.map_dim)
                        if next_state is not None:
                            def _prepare_tensor_for_cache(value):
                                if value is None:
                                    return None
                                tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                                if tensor.dim() == 0:
                                    return None
                                if tensor.size(0) > 1:
                                    tensor = tensor[:1]
                                return tensor.detach().clone().cpu()

                            memory_to_store = _prepare_tensor_for_cache(next_state.get("memory"))
                            mask_to_store = _prepare_tensor_for_cache(next_state.get("mask"))

                            if memory_to_store is not None:
                                self._map_hidden_state = {
                                    "map_shape": next_state.get("map_shape"),
                                    "memory": memory_to_store,
                                    "mask": mask_to_store,
                                }
                            else:
                                self._map_hidden_state = None
                        else:
                            self._map_hidden_state = None
            elif self.exploration_mode == "raster_v2":
                def _has_any_map(seq):
                    return bool(seq) and any(any(e is not None for e in s) for s in seq)

                maps_pol = batch_dict.get("map_policy")
                maps_exp = batch_dict.get("map")
                if _has_any_map(maps_pol):
                    map_sequences = maps_pol
                elif _has_any_map(maps_exp):
                    map_sequences = maps_exp
                else:
                    map_sequences = []

                index_sequences = batch_dict.get("map_index") or []

                map_flat = []
                idx_flat = []
                for seq_idx, map_seq in enumerate(map_sequences):
                    idx_seq = index_sequences[seq_idx] if seq_idx < len(index_sequences) else None
                    if map_seq is None:
                        map_seq = []
                    for t, entry in enumerate(map_seq):
                        map_flat.append(entry)
                        idx = None
                        if idx_seq is not None:
                            try:
                                idx = idx_seq[t]
                            except (IndexError, TypeError):
                                idx = None
                        idx_flat.append(self._normalize_map_index(idx))

                if not map_flat:
                    map_feats = torch.zeros(B * T, self.map_dim, device=device)
                else:
                    map_feats = self.map_encoder(map_flat, agent_indices=idx_flat)
                    if map_feats.dim() == 1:
                        map_feats = map_feats.view(1, -1)
                    map_feats = map_feats.view(len(map_flat), -1)
                    map_feats = map_feats.to(device)
                    if map_feats.size(0) != B * T:
                        pad = B * T - map_feats.size(0)
                        if pad > 0:
                            pad_tensor = torch.zeros(pad, self.map_dim, device=device, dtype=map_feats.dtype)
                            map_feats = torch.cat([map_feats, pad_tensor], dim=0)
                        else:
                            map_feats = map_feats[: B * T]
                self._map_hidden_state = None
            else:
                # raster cnn
                map_sequences = batch_dict.get("map") or []
                self._inject_map_indices(map_sequences, batch_dict.get("map_index"))
                map_flat = [m for seq in map_sequences for m in seq]
                map_feats = self.map_encoder(map_flat)
                self._map_hidden_state = None
        else:
            self._map_hidden_state = None

        if self.use_map and map_feats is not None and False:
            map_norm = torch.norm(map_feats, dim=-1).mean().item()
            map_std = map_feats.std().item()
            map_min = map_feats.min().item()
            map_max = map_feats.max().item()
            print(f"[MAP TO POLICY] norm={map_norm:.3f}, std={map_std:.4f}, "
                  f"range=[{map_min:.2f}, {map_max:.2f}], shape={map_feats.shape}")

        # Concatenate all features
        feats = [act_feat, rgb_feat, lssg_feat, gssg_feat]
        if self.use_map and map_feats is not None:
            map_norm = torch.norm(map_feats, dim=-1).mean().item()
            map_std = map_feats.std().item()
            feats.append(map_feats)

        out = torch.cat(feats, dim=-1).view(B, T, -1)
        # print(f"[FINAL STATE] shape={out.shape}, map_included={self.use_map}")

        if return_map_features:
            if map_feats is None:
                map_tensor = torch.zeros(B, T, self.map_dim, device=device)
            else:
                map_tensor = map_feats.view(B, T, self.map_dim)
            return out.view(B, T, -1), lssg_hidden, gssg_hidden, map_tensor

        return out.view(B, T, -1), lssg_hidden, gssg_hidden

    def reset_map_state(self, map_dict=None):
        self._map_hidden_state = None
        if self.use_map and self.map_encoder is not None and self.exploration_mode == "neural":
            self.map_encoder.reset_state(map_dict)

    def save_mappings(self, path: str):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "object_types.json"), "w") as f:
            json.dump(self.object_to_idx, f)
        with open(os.path.join(path, "relation_types.json"), "w") as f:
            json.dump(self.relation_to_idx, f)

    def load_mappings(self, path: str):
        object_types_file = os.path.join(path, "object_types.json")
        relation_types_file = os.path.join(path, "relation_types.json")
        if os.path.exists(object_types_file):
            with open(object_types_file, "r") as f:
                self.object_to_idx = json.load(f)
                self.object_to_idx = {k: int(v) for k, v in self.object_to_idx.items()}
        if os.path.exists(relation_types_file):
            with open(relation_types_file, "r") as f:
                self.relation_to_idx = json.load(f)
                self.relation_to_idx = {k: int(v) for k, v in self.relation_to_idx.items()}

    def save_model(self, path):

        filename = (
            f"feature_encoder_{self.num_actions}_{self.rgb_encoder.output_dim}_{self.action_emb.embedding.embedding_dim}_"
            f"{self.map_dim}_{self.sg_dim}_{self.use_transformer}_{self.use_map}_{self.use_scene_graph}_{self.use_rgb}.pth"
        )
        os.makedirs(path, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "meta": {
                "version": "enc_v1",
                "use_map": bool(getattr(self, "use_map", False)),
                "exploration_mode": str(getattr(self, "exploration_mode", "raster")),  # "raster" | "neural"
                "map_dim": int(getattr(self, "map_dim", 0)) if getattr(self, "use_map", False) else 0,
                "num_actions": int(getattr(self, "num_actions", 0)),
                "rgb_dim": int(self.rgb_encoder.output_dim),
                "action_dim": int(self.action_emb.embedding.embedding_dim),
                "sg_dim": int(getattr(self, "sg_dim", getattr(self.lssg_encoder, "lstm",
                                                              getattr(self.lssg_encoder, "output_dim",
                                                                      0)).hidden_size if hasattr(self.lssg_encoder,
                                                                                                 "lstm") else 0)),
                "use_transformer": bool(getattr(self, "use_transformer", False)),
                "use_scene_graph": bool(getattr(self, "use_scene_graph", False)),
                "use_rgb": bool(getattr(self, "use_rgb", False)),
            }
        }
        torch.save(payload, os.path.join(path, filename))

    @classmethod
    def create_and_load_model(cls, model_path, mapping_path=None, device="cpu"):
        """
        Load a FeatureEncoder by parsing hyperparameters from the filename and then loading weights.

        Supported filename patterns:
          new: feature_encoder_{num_actions}_{rgb_dim}_{action_dim}_{map_dim}_{sg_dim}_{True|False}_{True|False}_{True|False}_{True|False}.pth
               e.g. feature_encoder_45_512_32_64_256_True_False_True_True.pth
          old: feature_encoder_{num_actions}_{rgb_dim}_{action_dim}_{map_dim}_{sg_dim}.pth
               e.g. feature_encoder_45_512_32_64_256.pth
        """
        basename = os.path.basename(model_path)
        payload_meta = {}
        if os.path.exists(model_path):
            try:
                payload = torch.load(model_path, map_location=device)
                if isinstance(payload, dict):
                    payload_meta = payload.get("meta", {}) or {}
                del payload
            except Exception:
                payload_meta = {}

        pattern_full = r"feature_encoder_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(True|False)_(True|False)_(True|False)_(True|False)\.pth"
        m = re.match(pattern_full, basename)

        if m:
            num_actions = int(m.group(1))
            rgb_dim = int(m.group(2))
            action_dim = int(m.group(3))
            map_dim = int(m.group(4))
            sg_dim = int(m.group(5))
            use_transformer = (m.group(6) == "True")
            use_map = (m.group(7) == "True")
            use_scene_graph = (m.group(8) == "True")
            use_rgb = (m.group(9) == "True")
        else:
            pattern_partial = r"feature_encoder_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(True|False)_(True|False)\.pth"
            m = re.match(pattern_partial, basename)

            if m:
                num_actions = int(m.group(1))
                rgb_dim = int(m.group(2))
                action_dim = int(m.group(3))
                map_dim = int(m.group(4))
                sg_dim = int(m.group(5))
                use_transformer = (m.group(6) == "True")
                use_map = (m.group(7) == "True")
                use_scene_graph = True
                use_rgb = True
            else:
                # Backward-compatible fallback (no boolean flags present)
                pat_old = r"feature_encoder_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.pth"
                m_old = re.match(pat_old, basename)
                if not m_old:
                    raise ValueError(f"Filename {basename} does not match expected patterns.")
                num_actions = int(m_old.group(1))
                rgb_dim = int(m_old.group(2))
                action_dim = int(m_old.group(3))
                map_dim = int(m_old.group(4))
                sg_dim = int(m_old.group(5))
                use_transformer = False
                use_map = False
                use_scene_graph = True
                use_rgb = True

        meta_use_scene_graph = payload_meta.get("use_scene_graph")
        if meta_use_scene_graph is not None:
            use_scene_graph = bool(meta_use_scene_graph)

        meta_use_rgb = payload_meta.get("use_rgb")
        if meta_use_rgb is not None:
            use_rgb = bool(meta_use_rgb)

        exploration_mode = payload_meta.get("exploration_mode", "raster")

        model = cls(
            num_actions=num_actions,
            rgb_dim=rgb_dim,
            action_dim=action_dim,
            map_dim=map_dim,
            sg_dim=sg_dim,
            use_transformer=use_transformer,
            use_map=use_map,
            mapping_path=mapping_path,
            exploration_mode=exploration_mode,
            use_scene_graph=use_scene_graph,
            use_rgb=use_rgb,
        )

        # Load weights
        model.load_weights(model_path, device)
        return model

    def load_weights(self, model_path, device="cpu", strict=None):
        """
        Loads model weights into an existing FeatureEncoder instance.
        """
        payload = torch.load(model_path, map_location=device)
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload

        # ------------------------------------------------------------------
        # Backwards compatibility: older checkpoints were trained without the
        # optional agent-position channel in the exploration map encoder. When
        # that channel is enabled (include_agent=True) the first convolution
        # expects one additional input channel.  If we try to load the old
        # weights as-is, ``load_state_dict`` raises a size-mismatch error.  To
        # keep the old checkpoints usable we detect this situation and pad the
        # weight tensor with zeros for the new channel.
        # ------------------------------------------------------------------
        map_conv_key = "map_encoder.encoder.0.weight"
        if (
            self.use_map
            and hasattr(self, "map_encoder")
            and self.map_encoder is not None
            and map_conv_key in state_dict
        ):
            target_weight = self.map_encoder.encoder[0].weight
            loaded_weight = state_dict[map_conv_key]

            if (
                loaded_weight.ndim == target_weight.ndim
                and loaded_weight.shape[0] == target_weight.shape[0]
                and loaded_weight.shape[2:] == target_weight.shape[2:]
                and loaded_weight.shape[1] < target_weight.shape[1]
            ):
                missing_channels = target_weight.shape[1] - loaded_weight.shape[1]
                if missing_channels > 0:
                    pad_shape = (
                        loaded_weight.shape[0],
                        missing_channels,
                        loaded_weight.shape[2],
                        loaded_weight.shape[3],
                    )
                    pad = loaded_weight.new_zeros(pad_shape)
                    state_dict[map_conv_key] = torch.cat([loaded_weight, pad], dim=1)

        self.load_state_dict(state_dict, strict=strict)
        self.to(device)
        return payload.get("meta", {})




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNetFeatureExtractor(nn.Module):
    """
    Extracts visual features from the RGB input using pretrained ResNet18.
    Outputs a flat feature vector of size [B, 512].
    """

    def __init__(self, output_dim=512):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove FC layer
        self.resnet = nn.Sequential(*modules)
        self.projection = nn.Linear(512, output_dim) if output_dim != 512 else nn.Identity()
        self.output_dim = output_dim

    def forward(self, x):  # x: [B, 3, H, W]
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # [B, 512]
        return self.projection(x)  # [B, output_dim]


class ActionEmbedding(nn.Module):
    """
    Embeds the last discrete action into a learnable vector space.
    Index -1 is reserved for the 'START' state before any action is taken.
    """

    def __init__(self, num_actions, emb_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_actions + 1, emb_dim)  # 1 additional action for START (index: -1)

    def forward(self, action_idx):
        # Replace padding (-100) with 0 (arbitrary, due to masking later)
        safe_idx = action_idx.clone()
        safe_idx[safe_idx == -100] = 0
        return self.embedding(safe_idx + 1)  # [B, 32], plus 1 because of START action


class SceneGraphLSTMEncoder(nn.Module):
    """
    Encodes a sequence of scene graph embeddings using a two-layer LSTM.
    Supports both full sequences [B, T, D] and single steps [B, D].
    Optionally accepts/retruns LSTM hidden state for streaming/online inference.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=self.num_layers)

    def forward(self, x_seq, hidden=None):
        # Accepts [B, T, D] or [B, D]
        if x_seq.dim() == 2:
            # Single step: add time dimension
            x_seq = x_seq.unsqueeze(1)  # [B, 1, D]
        elif x_seq.dim() != 3:
            raise ValueError(f"Expected input [B, T, D] or [B, D], got {x_seq.shape}")

        # Pass through LSTM (optionally with hidden state)
        output, (hn, cn) = self.lstm(x_seq, hidden)  # output: [B, T, H], hn: [num_layers, B, H]

        return output, (hn, cn)


class SceneGraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1, max_len=256):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = input_dim

    def forward(self, x_seq, pad_mask=None):
        x_seq = self.pos_encoder(x_seq)
        if pad_mask is not None:
            return self.transformer(x_seq, src_key_padding_mask=pad_mask)
        else:
            return self.transformer(x_seq)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        return x + self.pe[:, : x.size(1)]
