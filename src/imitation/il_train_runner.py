import os
import datetime
import shutil

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


class ILTrainRunner:
    def __init__(self, agent, dataset, device=None, lr=1e-4, batch_size=8, val_split=0.15, seed=42, topk=(1, 2, 3)):
        """
        Set up IL training: split dataset (train/val), build DataLoaders with sequence padding,
        create optimizer, and compute class-weighted CrossEntropy (from action frequencies).

        Also stores Top-k settings for validation metrics and handles Neural SLAM training.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = agent.to(self.device)
        self.is_neural_slam = getattr(agent, 'is_neural_slam', False)
        self._debug_neural_slam = False  # Debug flag

        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        torch.manual_seed(seed)
        train_set, val_set = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=self.seq_collate)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=self.seq_collate)

        # Create optimizer with parameter groups to ensure joint optimisation
        param_groups = []
        neural_slam_config = getattr(self.agent, 'neural_slam_config', {}) if self.is_neural_slam else {}
        weight_decay = neural_slam_config.get('weight_decay', 1e-5)

        if self.is_neural_slam and hasattr(self.agent, 'neural_slam_networks') and self.agent.neural_slam_networks:
            mapper_lr = neural_slam_config.get('mapper_lr', lr)
            pose_lr = neural_slam_config.get('pose_estimator_lr', lr)
            memory_lr = neural_slam_config.get('memory_core_lr', mapper_lr)

            controller = (
                self.agent.neural_slam_networks['controller']
                if 'controller' in self.agent.neural_slam_networks
                else None
            )
            memory_core = (
                self.agent.neural_slam_networks['memory_core']
                if 'memory_core' in self.agent.neural_slam_networks
                else None
            )
            pose_est = (
                self.agent.neural_slam_networks['pose_estimator']
                if 'pose_estimator' in self.agent.neural_slam_networks
                else None
            )

            if controller is not None:
                params = list(controller.parameters())
                if params:
                    param_groups.append({
                        'params': params,
                        'lr': mapper_lr,
                        'weight_decay': weight_decay,
                    })
            if memory_core is not None:
                params = list(memory_core.parameters())
                if params:
                    param_groups.append({
                        'params': params,
                        'lr': memory_lr,
                        'weight_decay': weight_decay,
                    })
            if pose_est is not None:
                params = list(pose_est.parameters())
                if params:
                    param_groups.append({
                        'params': params,
                        'lr': pose_lr,
                        'weight_decay': weight_decay,
                    })

        base_params = []
        for name, param in self.agent.named_parameters():
            if not name.startswith('neural_slam_networks.'):
                base_params.append(param)
        if base_params:
            param_groups.append({'params': base_params, 'lr': lr, 'weight_decay': weight_decay})

        self.optimizer = optim.Adam(param_groups, weight_decay=0.0)

        # --- class weights (ignoring padding index -1) -------------------
        action_counts = np.zeros(self.agent.num_actions)
        for batch_data in self.train_loader:
            if self.is_neural_slam:
                x_batch, last_act, tgt_act, lengths, neural_slam_data = batch_data
            else:
                x_batch, last_act, tgt_act, lengths = batch_data

            valid = tgt_act != -100
            for a in tgt_act[valid]:
                action_counts[a.item()] += 1

        class_w = 1.0 / (action_counts + 1e-5)
        class_w = class_w / class_w.sum() * self.agent.num_actions
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32).to(self.device))

        self.topk = topk

    def run(self, num_epochs=50, save_folder=None):
        """
        Main training loop: iterate epochs, train on padded batches, pack sequences for loss,
        evaluate on validation set, and manage checkpoints (periodic + best-by-Top1).
        Includes Neural SLAM training (optional).
        """
        run_start = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        last_checkpoint = None
        best_val_score = -float("inf")
        best_checkpoint = None

        print(f"=== DATASET DEBUG INFO ===")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Number of unique actions in dataset: {self.agent.num_actions}")

        for i, batch in enumerate(self.train_loader):
            if i >= 3:
                break
            print(f"Train batch {i} shape info:")
            if len(batch) >= 4:
                obs, last_act, tgt_act, lengths = batch[:4]
                print(f"  Batch size: {len(obs)}, Seq lengths: {lengths}")
                print(f"  Sample target actions: {tgt_act[0][:5] if len(tgt_act[0]) > 0 else 'empty'}")
        print("=========================")

        for epoch in range(1, num_epochs + 1):
            self.agent.train()
            if self.is_neural_slam:
                self.agent.set_neural_slam_training_mode(True)

            tot_loss = 0.0
            tot_slam_loss = 0.0
            mapper_loss_sum = 0.0
            mapper_loss_count = 0
            pose_loss_sum = 0.0
            pose_loss_count = 0
            train_iter = tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)

            for batch_data in train_iter:
                if self.is_neural_slam:
                    x_batch, last_act, tgt_act, lengths, neural_slam_data = batch_data
                else:
                    x_batch, last_act, tgt_act, lengths = batch_data
                    neural_slam_data = None

                # Move tensors to device (dict may contain lists; leave them – encoder/processing handle that)
                for k, v in x_batch.items():
                    if isinstance(v, torch.Tensor):
                        x_batch[k] = v.to(self.device)
                last_act = last_act.to(self.device)
                tgt_act = tgt_act.to(self.device)

                # --- SLAM Batch-Vorbereitung (t-1 -> t) ---
                processed_neural_slam_data = None
                if neural_slam_data:
                    processed_neural_slam_data = self._process_neural_slam_batch_data(
                        neural_slam_data, x_batch, lengths
                    )
                    if hasattr(self, '_debug_neural_slam') and not self._debug_neural_slam:
                        print("[DEBUG] Neural SLAM raw keys:", list(neural_slam_data.keys()))
                        for key, value in neural_slam_data.items():
                            if isinstance(value, list):
                                t0 = type(value[0]) if value else "empty"
                                print(f"[DEBUG] {key}: list(len={len(value)}), first={t0}")
                            else:
                                print(f"[DEBUG] {key}: {type(value)}")
                        if processed_neural_slam_data:
                            print("[DEBUG] Neural SLAM processed shapes:",
                                  {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else type(v))
                                   for k, v in processed_neural_slam_data.items()})
                        self._debug_neural_slam = True

                # --- Forward ---
                value_pred = None
                if self.is_neural_slam and processed_neural_slam_data:
                    outputs = self.agent.forward(x_batch, last_act, processed_neural_slam_data)
                else:
                    outputs = self.agent.forward(x_batch, last_act)

                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                    value_pred = outputs.get("value")
                    neural_slam_outputs = outputs.get("neural_slam")
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                    neural_slam_outputs = outputs[1] if len(outputs) > 1 else None
                else:
                    logits = outputs
                    neural_slam_outputs = None

                # Cross-entropy loss computed on packed sequences
                pack_logits = nn.utils.rnn.pack_padded_sequence(
                    logits, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                pack_tgt = nn.utils.rnn.pack_padded_sequence(
                    tgt_act, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                il_loss = self.criterion(pack_logits.data, pack_tgt.data)

                # Optional Neural SLAM loss
                slam_losses = {}
                slam_loss_tensor = None
                if self.is_neural_slam and neural_slam_outputs and processed_neural_slam_data:
                    slam_targets = self._prepare_neural_slam_targets(processed_neural_slam_data)
                    slam_losses = self.agent.compute_neural_slam_loss(neural_slam_outputs, slam_targets)
                    if slam_losses:
                        slam_loss_tensor = sum(slam_losses.values())

                # Combine imitation and SLAM losses
                slam_weight = float(getattr(self.agent, "neural_slam_config", {}).get("loss_weight", 1.0)) \
                    if self.is_neural_slam else 0.0
                slam_loss_val = (slam_loss_tensor.item() if isinstance(slam_loss_tensor, torch.Tensor) else 0.0)
                total_loss = il_loss + slam_weight * (slam_loss_tensor if slam_loss_tensor is not None else 0.0)

                # Log mapper and pose diagnostics when available
                mapper_val = None
                if "mapper_loss" in slam_losses:
                    mapper_val = slam_losses["mapper_loss"].detach().item()
                    mapper_loss_sum += mapper_val
                    mapper_loss_count += 1

                pose_val = None
                if "pose_loss" in slam_losses:
                    pose_val = slam_losses["pose_loss"].detach().item()
                    pose_loss_sum += pose_val
                    pose_loss_count += 1

                mapper_display = None
                if mapper_val is not None:
                    mapper_display = mapper_val
                elif mapper_loss_count:
                    mapper_display = mapper_loss_sum / mapper_loss_count

                pose_display = None
                if pose_val is not None:
                    pose_display = pose_val
                elif pose_loss_count:
                    pose_display = pose_loss_sum / pose_loss_count
                postfix = {
                    "IL": f"{il_loss.item():.3f}",
                    "SLAM": f"{slam_loss_val:.3f}" if self.is_neural_slam else "n/a",
                    "Total(w)": f"{total_loss.item():.3f}",
                }
                if self.is_neural_slam:
                    postfix["mapper"] = (
                        f"{mapper_display:.3f}" if mapper_display is not None else "0.000"
                    )
                    postfix["pose"] = (
                        f"{pose_display:.3f}" if pose_display is not None else "0.000"
                    )
                train_iter.set_postfix(postfix)

                # Backpropagation step
                self.optimizer.zero_grad()

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 5.0)
                if self.is_neural_slam and self.agent.neural_slam_networks:
                    for net in self.agent.neural_slam_networks.values():
                        if hasattr(net, "parameters"):
                            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)

                self.optimizer.step()
                

                tot_loss += il_loss.item()
                tot_slam_loss += slam_loss_val

            # --- Epoch-Ende ---
            denom = max(1, len(self.train_loader))
            avg_il_loss = tot_loss / denom
            avg_slam_loss = (tot_slam_loss / denom) if self.is_neural_slam else 0.0

            print(f"[Epoch {epoch}] IL Loss={avg_il_loss:.4f}", end="")
            if self.is_neural_slam:
                print(f", SLAM Loss={avg_slam_loss:.4f}")
            else:
                print()

            avg_loss, acc_dict = self.evaluate()
            curr_val_score = acc_dict[1]

            # --- Checkpointing ---
            if save_folder and epoch % 25 == 0:
                checkpoint_name = f"{run_start}_imitation_agent_epoch{epoch}.pth"
                checkpoint_path = os.path.join(save_folder, checkpoint_name)
                os.makedirs(save_folder, exist_ok=True)
                if last_checkpoint and os.path.exists(last_checkpoint):
                    os.remove(last_checkpoint)
                self.save_model(checkpoint_path)
                last_checkpoint = checkpoint_path

            if save_folder and curr_val_score > best_val_score:
                best_val_score = curr_val_score
                if best_checkpoint and os.path.exists(best_checkpoint):
                    if os.path.isdir(best_checkpoint):
                        shutil.rmtree(best_checkpoint)
                    else:
                        os.remove(best_checkpoint)
                best_name = f"best_{epoch}_acc_{curr_val_score:.2f}".replace(".", "_")
                best_checkpoint = os.path.join(save_folder, best_name)
                self.agent.save_model(best_checkpoint)
                print(f"[INFO] New best model saved: {best_checkpoint} (acc: {curr_val_score:.2f}%)")

            torch.cuda.empty_cache()

        # --- final save ---
        if save_folder:
            if last_checkpoint and os.path.exists(last_checkpoint):
                os.remove(last_checkpoint)
            self.agent.save_model(save_folder)

    def evaluate(self):
        """
        Validation pass without gradients: forward on val loader, pack sequences to ignore padding,
        compute CrossEntropy loss and Top-k accuracies, and return (avg_loss, acc_dict).
        """
        self.agent.eval()
        if self.is_neural_slam:
            self.agent.set_neural_slam_training_mode(False)

        total_loss = 0.0
        all_preds = []
        all_targets = []

        val_iter = tqdm(self.val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch_data in val_iter:
                if self.is_neural_slam:
                    x_batch, last_act, tgt_act, lengths, neural_slam_data = batch_data
                else:
                    x_batch, last_act, tgt_act, lengths = batch_data

                for key in x_batch:
                    if isinstance(x_batch[key], torch.Tensor):
                        x_batch[key] = x_batch[key].to(self.device)

                target_actions = tgt_act.to(self.device)
                last_act = last_act.to(self.device)

                # Forward pass (no Neural SLAM training in validation)
                outputs = self.agent.forward(x_batch, last_act)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Pack -> remove pads for CE-Loss
                pack_logits = nn.utils.rnn.pack_padded_sequence(logits, lengths.cpu(), batch_first=True,
                                                                enforce_sorted=False)
                pack_tgt = nn.utils.rnn.pack_padded_sequence(target_actions, lengths.cpu(), batch_first=True,
                                                             enforce_sorted=False)

                loss = self.criterion(pack_logits.data, pack_tgt.data)
                total_loss += loss.item()
                val_iter.set_postfix(loss=loss.item())

                all_preds.append(pack_logits.data.cpu().detach())
                all_targets.append(pack_tgt.data.cpu().detach())

        avg_loss = total_loss / len(self.val_loader)
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        acc_dict = self.compute_metrics(preds, targets)

        print(f"  Validation Loss: {avg_loss:.4f}, Top-1 Acc: {acc_dict[1]:.2f}%", end="")
        for k in self.topk:
            if k != 1:
                print(f", Top-{k} Acc: {acc_dict[k]:.2f}%", end="")
        print()
        return avg_loss, acc_dict

    def _process_neural_slam_batch_data(self, neural_slam_data, x_batch, lengths):
        """Extract ``t-1`` to ``t`` pairs needed for Neural SLAM supervision."""
        if not neural_slam_data or "neural_slam_training" not in neural_slam_data:
            return None

        training = neural_slam_data.get("neural_slam_training")
        if not isinstance(training, dict):
            return None

        mask = training.get("mask")
        rgb_prev = training.get("rgb_prev")
        rgb_curr = training.get("rgb_curr")
        pose_delta = training.get("pose_delta")
        fp_proj = training.get("fp_proj_gt")
        fp_explored = training.get("fp_explored_gt")

        required = [mask, rgb_prev, rgb_curr, pose_delta, fp_proj]
        if any(not isinstance(t, torch.Tensor) for t in required):
            return None

        lengths = lengths.tolist()
        B, T = mask.shape
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        def _normalise_rgb(t: torch.Tensor) -> torch.Tensor:
            tensor = t.detach().float()
            if tensor.dim() == 3 and tensor.size(0) != 3 and tensor.size(-1) == 3:
                tensor = tensor.permute(2, 0, 1)
            if tensor.dim() != 3:
                raise ValueError("Expected 3D RGB tensor")
            if tensor.numel() > 0 and float(tensor.max()) > 1.0:
                tensor = tensor / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze(0)
            return (tensor - mean) / std

        def _prepare_map(t: torch.Tensor) -> torch.Tensor:
            tensor = t.detach().float()
            if tensor.dim() == 3:
                tensor = tensor[0]
            if tensor.dim() != 2:
                raise ValueError("Expected 2D map tensor")
            return tensor.clamp(0.0, 1.0)

        def _prepare_pose(t: torch.Tensor) -> torch.Tensor:
            tensor = t.detach().float().view(-1)
            if tensor.numel() < 3:
                tensor = torch.nn.functional.pad(tensor, (0, 3 - tensor.numel()))
            return tensor[:3]

        rgb_prev_list, rgb_curr_list = [], []
        pose_list, proj_list, explored_list = [], [], []
        valid_indices = []

        for b in range(B):
            max_t = min(int(lengths[b]), T)
            idx = None
            for t in range(max_t - 1, -1, -1):
                if mask[b, t]:
                    idx = t
                    break
            if idx is None:
                continue

            try:
                rgb_prev_list.append(_normalise_rgb(rgb_prev[b, idx]))
                rgb_curr_list.append(_normalise_rgb(rgb_curr[b, idx]))
                pose_list.append(_prepare_pose(pose_delta[b, idx]))
                proj_list.append(_prepare_map(fp_proj[b, idx]))
                if isinstance(fp_explored, torch.Tensor):
                    explored_list.append(_prepare_map(fp_explored[b, idx]))
                else:
                    explored_list.append(torch.ones_like(proj_list[-1]))
                valid_indices.append(b)
            except Exception:
                continue

        if not rgb_curr_list:
            return None

        device = self.device
        processed = {
            "rgb_prev": torch.stack(rgb_prev_list, dim=0).to(device),
            "rgb_curr": torch.stack(rgb_curr_list, dim=0).to(device),
            "pose_delta": torch.stack(pose_list, dim=0).to(device),
            "fp_proj_gt": torch.stack(proj_list, dim=0).to(device),
            "fp_explored_gt": torch.stack(explored_list, dim=0).to(device),
            "batch_indices": torch.tensor(valid_indices, dtype=torch.long, device=device),
        }

        return processed

    def _prepare_neural_slam_targets(self, ns_data):
        """Build target tensors for the Neural SLAM auxiliary heads."""
        targets = {}

        occupancy = ns_data.get("fp_proj_gt")
        if isinstance(occupancy, torch.Tensor):
            targets["occupancy_target"] = occupancy.float().clamp(0.0, 1.0).to(self.device)

        explored = ns_data.get("fp_explored_gt")
        if isinstance(explored, torch.Tensor):
            targets["exploration_mask"] = explored.float().clamp(0.0, 1.0).to(self.device)

        pose = ns_data.get("pose_delta")
        if isinstance(pose, torch.Tensor):
            targets["pose_delta_gt"] = pose.float().to(self.device)

        return targets

    def compute_metrics(self, logits, targets):
        """
        Compute Top-k accuracies from flat logits/targets (after packing).
        Returns a dict {k: accuracy_percent}.
        """
        accs = {}
        max_k = max(self.topk)
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))

        for k in self.topk:
            acc = correct[:, :k].any(dim=1).float().mean().item() * 100
            accs[k] = acc
        return accs

    def seq_collate(self, batch):
        """Pad variable-length sequences and organise optional Neural SLAM extras."""
        if len(batch[0]) >= 5:
            obs_lists, last_list, tgt_list, lengths, extras = zip(*batch)
        else:
            obs_lists, last_list, tgt_list, lengths = zip(*batch)
            extras = None

        max_T = max(lengths)

        # Map (sparse/dict form) und optionale dichte Policy-Map (MetricSemantic v2)
        pad_map = []
        pad_policy_map = []
        for obs_seq in obs_lists:
            map_seq = [o.state[3] for o in obs_seq]
            pad_map.append(map_seq + [None] * (max_T - len(map_seq)))

            policy_seq = [o.info.get("policy_map", None) for o in obs_seq]
            pad_policy_map.append(policy_seq + [None] * (max_T - len(policy_seq)))

        # RGB
        pad_rgb = []
        for obs_seq in obs_lists:
            rgb_seq = [o.state[0] for o in obs_seq]
            pad_rgb.append(rgb_seq + [None] * (max_T - len(rgb_seq)))

        # SGs + Masken
        pad_lssg, pad_gssg, pad_lssg_mask, pad_gssg_mask = [], [], [], []
        for obs_seq in obs_lists:
            lssg_seq = [o.state[1] for o in obs_seq]
            gssg_seq = [o.state[2] for o in obs_seq]
            pad_len = max_T - len(lssg_seq)
            pad_lssg.append(lssg_seq + [None] * pad_len)
            pad_lssg_mask.append([1] * len(lssg_seq) + [0] * pad_len)
            pad_gssg.append(gssg_seq + [None] * pad_len)
            pad_gssg_mask.append([1] * len(gssg_seq) + [0] * pad_len)

        last_act = torch.tensor([list(la) + [-100] * (max_T - len(la)) for la in last_list], dtype=torch.long)
        tgt_act = torch.stack([
            torch.cat([torch.tensor(ta, dtype=torch.long),
                       torch.full((max_T - len(ta),), -100, dtype=torch.long)])
            for ta in tgt_list
        ])

        # optionale Zusatzinfos
        pad_agent_pos = []
        for obs_seq in obs_lists:
            agent_seq = [o.info.get("agent_pos", None) for o in obs_seq]
            pad_agent_pos.append(agent_seq + [None] * (max_T - len(agent_seq)))

        pad_map_index = []
        for obs_seq in obs_lists:
            idx_seq = [o.info.get("map_index", None) for o in obs_seq]
            pad_map_index.append(idx_seq + [None] * (max_T - len(idx_seq)))

        x_batch = {
            "map": pad_map,
            "map_policy": pad_policy_map,
            "rgb": pad_rgb,
            "lssg": pad_lssg,
            "lssg_mask": pad_lssg_mask,
            "gssg": pad_gssg,
            "gssg_mask": pad_gssg_mask,
            "agent_pos": pad_agent_pos,
            "map_index": pad_map_index
        }

        # Only return Neural SLAM extras when the feature is active
        if self.is_neural_slam and extras is not None:
            neural_slam_data = ILTrainRunner._collate_neural_slam_data(extras, max_T)
            return x_batch, last_act, tgt_act, torch.tensor(lengths), neural_slam_data
        else:
            return x_batch, last_act, tgt_act, torch.tensor(lengths)

    @staticmethod
    def _collate_neural_slam_data(neural_slam_list, max_T):
        """Collate Neural SLAM training data across batch and time dimensions."""
        if not neural_slam_list:
            return None

        import numpy as np

        all_keys = set()
        for sample in neural_slam_list:
            if sample:
                all_keys.update(sample.keys())

        if not all_keys:
            return None

        def pad_sequence(seq):
            if isinstance(seq, (list, tuple)):
                seq_list = list(seq)
            elif seq is None:
                seq_list = []
            else:
                seq_list = [seq]
            seq_list = seq_list[:max_T]
            if len(seq_list) < max_T:
                seq_list.extend([None] * (max_T - len(seq_list)))
            return seq_list

        def _tensor_from_value(val, *, dtype=torch.float32):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                return val.detach().clone().to(dtype)
            try:
                return torch.as_tensor(val, dtype=dtype)
            except Exception:
                return None

        def _format_rgb_tensor(val):
            tensor = _tensor_from_value(val, dtype=torch.float32)
            if tensor is None:
                return None
            if tensor.dim() == 3:
                if tensor.size(0) != 3 and tensor.size(-1) == 3:
                    tensor = tensor.permute(2, 0, 1)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            else:
                return None
            if tensor.numel() > 0 and float(tensor.max()) > 1.0:
                tensor = tensor / 255.0
            return tensor

        def _format_map_tensor(val):
            tensor = _tensor_from_value(val, dtype=torch.float32)
            if tensor is None:
                return None
            if tensor.dim() == 2:
                return tensor
            if tensor.dim() == 3:
                if tensor.size(0) == 1:
                    return tensor.squeeze(0)
                return tensor[0]
            return None

        def _format_pose_tensor(val):
            tensor = _tensor_from_value(val, dtype=torch.float32)
            if tensor is None:
                return None
            if tensor.dim() == 0:
                tensor = tensor.unsqueeze(0)
            return tensor.view(-1)

        def collate_neural_tuples(seq_batch):
            first = None
            for seq in seq_batch:
                for item in seq:
                    if isinstance(item, (list, tuple)) and len(item) == 5:
                        first = item
                        break
                if first is not None:
                    break
            if first is None:
                return None

            rgb_prev_ex = _format_rgb_tensor(first[0])
            rgb_curr_ex = _format_rgb_tensor(first[1])
            pose_ex = _format_pose_tensor(first[2])
            proj_ex = _format_map_tensor(first[3])
            explored_ex = _format_map_tensor(first[4])
            if None in (rgb_prev_ex, rgb_curr_ex, pose_ex, proj_ex, explored_ex):
                return None

            B = len(seq_batch)
            T = len(seq_batch[0]) if seq_batch else 0
            rgb_prev = torch.zeros((B, T) + tuple(rgb_prev_ex.shape), dtype=torch.float32)
            rgb_curr = torch.zeros((B, T) + tuple(rgb_curr_ex.shape), dtype=torch.float32)
            pose_delta = torch.zeros((B, T, pose_ex.numel()), dtype=torch.float32)
            fp_proj = torch.zeros((B, T) + tuple(proj_ex.shape), dtype=torch.float32)
            fp_explored = torch.zeros((B, T) + tuple(explored_ex.shape), dtype=torch.float32)
            mask = torch.zeros(B, T, dtype=torch.bool)

            for b, seq in enumerate(seq_batch):
                for t, item in enumerate(seq):
                    if not (isinstance(item, (list, tuple)) and len(item) == 5):
                        continue
                    rgb_p = _format_rgb_tensor(item[0])
                    rgb_c = _format_rgb_tensor(item[1])
                    pose = _format_pose_tensor(item[2])
                    proj = _format_map_tensor(item[3])
                    explored = _format_map_tensor(item[4])
                    if None in (rgb_p, rgb_c, pose, proj, explored):
                        continue
                    rgb_prev[b, t] = rgb_p
                    rgb_curr[b, t] = rgb_c
                    pose_delta[b, t] = pose
                    fp_proj[b, t] = proj
                    fp_explored[b, t] = explored
                    mask[b, t] = True

            return {
                "rgb_prev": rgb_prev,
                "rgb_curr": rgb_curr,
                "pose_delta": pose_delta,
                "fp_proj_gt": fp_proj,
                "fp_explored_gt": fp_explored,
                "mask": mask,
            }

        def dict_pose_to_tensor(val):
            if not isinstance(val, dict):
                return None
            keys = set(val.keys())
            if {"x", "z", "yaw"}.issubset(keys):
                return torch.tensor([
                    float(val.get("x", 0.0)),
                    float(val.get("z", 0.0)),
                    float(val.get("yaw", 0.0)),
                ], dtype=torch.float32)
            if {"dx", "dz", "dyaw"}.issubset(keys):
                return torch.tensor([
                    float(val.get("dx", 0.0)),
                    float(val.get("dz", 0.0)),
                    float(val.get("dyaw", 0.0)),
                ], dtype=torch.float32)
            if {"x", "z"}.issubset(keys) and len(keys) <= 2:
                return torch.tensor([
                    float(val.get("x", 0.0)),
                    float(val.get("z", 0.0)),
                ], dtype=torch.float32)
            return None

        def value_to_tensor(val, *, dtype=None, shape=None):
            if val is None:
                return None
            if isinstance(val, torch.Tensor):
                tensor = val.detach().cpu()
            elif isinstance(val, np.ndarray):
                tensor = torch.from_numpy(val)
            elif isinstance(val, (list, tuple)):
                try:
                    tensor = torch.as_tensor(val)
                except Exception:
                    return None
            elif isinstance(val, (float, np.floating)):
                target_dtype = dtype if dtype is not None else torch.float32
                tensor = torch.tensor(float(val), dtype=target_dtype)
            elif isinstance(val, (int, np.integer, bool)):
                target_dtype = dtype if dtype is not None else torch.long
                tensor = torch.tensor(int(val), dtype=target_dtype)
            else:
                tensor = dict_pose_to_tensor(val)
                if tensor is None:
                    return None

            tensor = tensor.clone().detach()
            if dtype is not None:
                tensor = tensor.to(dtype)
            if shape is not None and tuple(tensor.shape) != tuple(shape):
                try:
                    tensor = tensor.view(shape)
                except Exception:
                    return None
            return tensor

        def stack_sequence(seq_batch, key):
            first_item = None
            for seq in seq_batch:
                for item in seq:
                    if item is not None:
                        first_item = item
                        break
                if first_item is not None:
                    break
            if first_item is None:
                return None

            template = value_to_tensor(first_item)
            if template is None:
                return None

            dtype = template.dtype
            shape = tuple(template.shape)
            B = len(seq_batch)
            T = len(seq_batch[0]) if seq_batch else 0
            data_shape = (B, T) + shape if shape else (B, T)
            data = torch.zeros(data_shape, dtype=dtype)
            mask = torch.zeros(B, T, dtype=torch.bool)

            for b, seq in enumerate(seq_batch):
                for t, item in enumerate(seq):
                    if item is None:
                        continue
                    tensor_item = value_to_tensor(item, dtype=dtype, shape=shape)
                    if tensor_item is None:
                        return None
                    data[(b, t)] = tensor_item
                    mask[b, t] = True

            return {"data": data, "mask": mask}

        def collate_sequence(seq_batch, key):
            stacked = stack_sequence(seq_batch, key)
            if stacked is not None:
                return stacked

            first_item = None
            for seq in seq_batch:
                for item in seq:
                    if item is not None:
                        first_item = item
                        break
                if first_item is not None:
                    break
            if first_item is None:
                return None

            if isinstance(first_item, dict):
                subkeys = set()
                for seq in seq_batch:
                    for item in seq:
                        if isinstance(item, dict):
                            subkeys.update(item.keys())

                result = {}
                for subkey in sorted(subkeys):
                    sub_seq_batch = []
                    for seq in seq_batch:
                        sub_seq = []
                        for item in seq:
                            if isinstance(item, dict):
                                sub_seq.append(item.get(subkey))
                            else:
                                sub_seq.append(None)
                        sub_seq_batch.append(sub_seq)
                    sub_collated = collate_sequence(sub_seq_batch, f"{key}.{subkey}")
                    if sub_collated is not None:
                        result[subkey] = sub_collated
                return result if result else None

            return seq_batch

        final_collated = {}
        for key in sorted(all_keys):
            seq_batch = []
            for sample in neural_slam_list:
                value = sample.get(key) if sample else None
                seq_batch.append(pad_sequence(value))

            if key == "neural_slam_training":
                collated_value = collate_neural_tuples(seq_batch)
            else:
                collated_value = collate_sequence(seq_batch, key)
            if collated_value is not None:
                final_collated[key] = collated_value

        return final_collated if final_collated else None

    def save_model(self, path):
        """Delegate model persistence to the underlying agent."""
        self.agent.save_model(path)
