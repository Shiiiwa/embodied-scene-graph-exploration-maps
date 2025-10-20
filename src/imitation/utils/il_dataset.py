import os
import pickle
from glob import glob

import torch
from torch.utils.data import Dataset

import sys, types

from src.components.graph.global_graph import GlobalSceneGraph
from src.components.graph.scene_graph import SceneGraph
import src.components.graph.scene_graph as _sg
import src.components.graph.global_graph as _gg
import src.components.utils.observation as _obs
import src.components.exploration.exploration_map_manager as _exm
import src.components.exploration.maps.metric_semantic_map as _msm

def _make_pkg(name: str):
    """Create a package-like module (has __path__) and put in sys.modules."""
    mod = types.ModuleType(name)
    mod.__path__ = []  # <- macht es zum Paket
    sys.modules[name] = mod
    return mod

def _inject_pkg(alias_root: str):
    # Pakethierarchie anlegen: alias_root, alias_root.graph, alias_root.utils, alias_root.exploration, alias_root.exploration.maps
    root = _make_pkg(alias_root)
    graph = _make_pkg(f"{alias_root}.graph")
    utils = _make_pkg(f"{alias_root}.utils")
    exploration = _make_pkg(f"{alias_root}.exploration")
    maps = _make_pkg(f"{alias_root}.exploration.maps")

    # Untermodul-Aliasse auf deine echten Module mappen
    sys.modules[f"{alias_root}.graph.scene_graph"] = _sg
    sys.modules[f"{alias_root}.graph.global_graph"] = _gg
    sys.modules[f"{alias_root}.utils.observation"] = _obs
    sys.modules[f"{alias_root}.exploration.exploration_map_manager"] = _exm

    # WICHTIG: beide historischen Namen auf das gleiche echte Modul zeigen lassen
    sys.modules[f"{alias_root}.exploration.maps.metric_semantic_map"] = _msm
    sys.modules[f"{alias_root}.exploration.maps.metric_semantic_v1"] = _msm



_inject_pkg("App")
_inject_pkg("app")

def list_all_pkl_files(data_dir):
    pattern = os.path.join(data_dir, "**", "*.pkl")
    return sorted(glob(pattern, recursive=True))


class ImitationLearningDataset(Dataset):
    """
    loading pre-recorded imitation learning episodes.

    Splits each episode into fixed-length windows, reconstructs
    scene graph objects from saved dictionaries, and returns
    sequences of observations, previous actions, and target actions
    for policy training
    """

    def __init__(self, data_dir, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.windows = []  # list of {"path": ..., "start": ...}
        self.num_actions = None

        for file_path in list_all_pkl_files(data_dir):
            with open(file_path, "rb") as f:
                ep = pickle.load(f)
            num_steps = len(ep)
            if self.num_actions is None:
                self.num_actions = ep[0].get("num_actions", None)
            for t in range(0, num_steps, seq_len):
                self.windows.append({"path": file_path, "start_idx": t, "length": min(seq_len, num_steps - t)})


    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        entry = self.windows[idx]
        with open(entry["path"], "rb") as f:
            ep = pickle.load(f)
        chunk = ep[entry["start_idx"]: entry["start_idx"] + entry["length"]]

        obs, last_act, tgt_act = [], [], []

        base_extra_keys = {
            "pose_gt",
            "sensor_pose",
            "ego_map_gt",
            "coverage_m2",
            "short_term_goal",
        }
        extras_lists = {key: [] for key in base_extra_keys}
        extras_lists["neural_slam_training"] = []
        dynamic_extra_keys = set()

        for s in chunk:
            obs_copy = s["obs"]
            step_idx = len(obs)

            # ----- SLAM-Extras einsammeln (sofern vorhanden) -----
            extras_lists["pose_gt"].append(s.get("pose_gt", None))
            extras_lists["coverage_m2"].append(s.get("coverage_m2", None))
            extras_lists["short_term_goal"].append(s.get("short_term_goal", None))

            # Benennung harmonisieren: wir wollen "sensor_pose" (nicht _noisy)
            sp = s.get("sensor_pose_noisy", s.get("sensor_pose", None))
            extras_lists["sensor_pose"].append(sp)

            eg = s.get("ego_map_gt", None)
            # In Tensors umwandeln erst später (Collate/Processing), hier nur durchreichen
            extras_lists["ego_map_gt"].append(eg)

            neural_training = s.get("neural_slam_training")
            tuple_sample = None
            neural_training_dict = {}
            if isinstance(neural_training, dict):
                neural_training_dict = neural_training
                maybe_tuple = neural_training.get("neural_slam_training")
                if isinstance(maybe_tuple, (list, tuple)) and len(maybe_tuple) == 5:
                    tuple_sample = tuple(maybe_tuple)
            elif isinstance(neural_training, (list, tuple)) and len(neural_training) == 5:
                tuple_sample = tuple(neural_training)

            extras_lists["neural_slam_training"].append(tuple_sample)

            if not neural_training_dict and isinstance(neural_training, dict):
                neural_training_dict = neural_training

            # Überschreibe Basiswerte, wenn Neural-SLAM-Daten sie bereitstellen
            for key in base_extra_keys.intersection(neural_training_dict.keys()):
                extras_lists[key][-1] = neural_training_dict.get(key)

            dynamic_present = set()
            if neural_training_dict:
                for key, value in neural_training_dict.items():
                    if key in base_extra_keys:
                        continue
                    if key not in extras_lists:
                        extras_lists[key] = [None] * step_idx
                    extras_lists[key].append(value)
                    dynamic_extra_keys.add(key)
                    dynamic_present.add(key)

            for key in dynamic_extra_keys:
                if key not in dynamic_present:
                    extras_lists[key].append(None)

            # --- SGs rekonstruieren (wie gehabt) ---
            if isinstance(obs_copy.state[1], dict):
                obs_copy.state[1] = SceneGraph.from_dict(obs_copy.state[1])
            if isinstance(obs_copy.state[2], dict):
                obs_copy.state[2] = GlobalSceneGraph.from_dict(obs_copy.state[2])

            # exploration_map in state[3] einstecken
            exp_map = s.get("exploration_map", None)
            st = list(obs_copy.state) if isinstance(obs_copy.state, (list, tuple)) else [obs_copy.state]
            if len(st) < 4:
                st += [None] * (4 - len(st))
            st[3] = exp_map
            obs_copy.state = st

            obs.append(obs_copy)
            last_act.append(s["last_action"])
            tgt_act.append(torch.tensor(s["obs"].info["action"], dtype=torch.long))

        extras = {
            key: values
            for key, values in extras_lists.items()
            if any(value is not None for value in values)
        }

        return obs, last_act, tgt_act, len(chunk), extras

