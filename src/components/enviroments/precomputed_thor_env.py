import math
import os
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pickle

from src.components.exploration.exploration_map_manager import ExplorationMapManager
from src.components.graph.global_graph import GlobalSceneGraph
from src.components.graph.gt_graph import GTGraph
from src.components.graph.local_graph_builder import LocalSceneGraphBuilder
from src.components.utils.paths import DATA_DIR, TRANSITION_TABLES, GT_GRAPHS_DIR
from src.scripts.generate_gt_graphs import generate_gt_scene_graphs
from src.components.utils.observation import Observation


class PrecomputedThorEnv:
    def __init__(
        self,
        rho=0.02,
        scene_number=None,
        render=False,
        grid_size=0.25,
        transition_tables_path=None,
        max_actions=40,
        map_version=None
    ):
        self.rho = rho
        self.scene_number = 1 if scene_number is None else scene_number
        self.max_actions = max_actions
        self.render = render
        self.grid_size = grid_size
        if grid_size != 0.25:
            raise ValueError("PrecomputedThorEnv only supports grid_size of 0.25")

        if transition_tables_path is None:
            self.transition_tables_path = TRANSITION_TABLES
        else:
            p = Path(transition_tables_path)
            self.transition_tables_path = p if p.is_absolute() else (DATA_DIR / p)
        self.transition_tables_path = Path(self.transition_tables_path)

        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.state = None
        self.gt_graph = None
        self.viewpoints = defaultdict(set)
        self.last_score = 0.0
        self.step_count = 0
        self.scene_number = 1 if scene_number is None else scene_number
        self.map_origin = None

        self.exploration_map = None
        self.exploration_map_version = map_version

        self.occupancy_map = None
        self.num_orientations = 4
        self.stop_index = self.get_actions().index(["Pass", "Pass"])
        self.last_event = None

        self.pending_action: Optional[Dict[str, Any]] = None
        self.prev_agent_pose: Optional[Dict[str, float]] = None
        self.pending_mapper_logits: Optional[Any] = None
        self.pending_head_params: Optional[Any] = None
        self.pending_exploration_mask: Optional[Any] = None
        self.pending_pose_delta: Optional[Any] = None
        self.pending_prev_observation: Optional[Dict[str, Any]] = None
        self._pending_mapper_step: Optional[int] = None
        self._previous_rgb_frame: Optional[np.ndarray] = None
        self._previous_action_payload: Optional[Dict[str, Any]] = None


        table_path = self.transition_tables_path / f"FloorPlan{self.scene_number}.pkl"
        if not table_path.exists():
            raise FileNotFoundError(f"\n[ERROR] Required file not found: {table_path}")

        # Load precomputed mapping: dict {(x,z,rotation): event or None}
        with open(table_path, "rb") as f:
            data = pickle.load(f)
        self.mapping = data["table"]

        # current agent state in world coords
        self.current_pos = None  # tuple (x, z)
        self.current_rot = None  # int degrees (0,90,180,270)

    def get_action_dim(self):
        return len(self.get_actions())

    def get_actions(self):
        agent_rotations = ["RotateRight", "RotateLeft", "Pass"]
        movements = ["MoveAhead", "MoveRight", "MoveLeft", "MoveBack", "Pass"]
        return [[move, rot1] for move in movements for rot1 in agent_rotations]

    def set_mapper_outputs(
        self,
        mapper_logits=None,
        head_params=None,
        *,
        exploration_mask=None,
        pose_delta=None,
        prev_event=None,
    ):
        """Store mapper outputs that will be consumed on the next observation update."""
        self.pending_mapper_logits = mapper_logits
        self.pending_head_params = head_params
        self.pending_exploration_mask = exploration_mask
        self.pending_pose_delta = pose_delta

        prev_obs = None
        if prev_event is not None:
            prev_rgb = getattr(prev_event, "frame", None)
            metadata = getattr(prev_event, "metadata", {}) or {}
            pose_info = metadata.get("agent", {})
            position = pose_info.get("position", {})
            rotation = pose_info.get("rotation", {})
            prev_pose = {
                "x": float(position.get("x", 0.0)),
                "z": float(position.get("z", 0.0)),
                "yaw": float(rotation.get("y", 0.0)),
            }
            prev_obs = {
                "rgb": None if prev_rgb is None else np.array(prev_rgb, copy=True),
                "pose": prev_pose,
            }

        self.pending_prev_observation = prev_obs
        self._pending_mapper_step = getattr(self, "step_count", 0)

    def get_state_dim(self):
        warnings.warn(
            "The state dimension cannot be reliably determined from the environment. " "This method should not be used.", UserWarning
        )
        return [3, 128, 256]

    def reset(self, scene_number=None, random_start=False, start_position=None, start_rotation=None):
        """
        Resets the whole enviroment, so new episode can start. Creating fresh graphs,
        loads startingposition & loading scene data
        """
        if scene_number is not None:
            self.scene_number = scene_number

        with open(os.path.join(self.transition_tables_path, f"FloorPlan{self.scene_number}.pkl"), "rb") as f:
            data = pickle.load(f)
        self.mapping = data["table"]

        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.step_count = 0
        self.last_score = 0.0
        self.viewpoints.clear()
        self.gt_graph = self.get_ground_truth_graph(f"FloorPlan{self.scene_number}")
        self.pending_action = None
        self.pending_mapper_logits = None
        self.pending_head_params = None
        self.pending_exploration_mask = None
        self.pending_pose_delta = None
        self.pending_prev_observation = None
        self.prev_agent_pose = None
        self._pending_mapper_step = None
        self._previous_rgb_frame = None
        self._previous_action_payload = None

        # choose start state
        if random_start:
            # pick a random key where an event exists
            valid = [k for k, evt in self.mapping.items() if evt is not None]
            x, z, rot = random.choice(valid)
        elif start_position is not None and start_rotation is not None:
            # round to grid
            x = round(start_position[0] / self.grid_size) * self.grid_size
            z = round(start_position[1] / self.grid_size) * self.grid_size
            rot = start_rotation % 360
            if (x, z, rot) not in self.mapping or self.mapping[(x, z, rot)] is None:
                raise ValueError("Invalid start state: no event at this position/rotation")
        else:
            raise ValueError("Either random_start or start_position/start_rotation must be given")

        self.current_pos = (x, z)
        self.current_rot = rot
        event = self.mapping[(x, z, rot)]
        self.last_event = event

        # first observation builds map origin etc.
        obs = self._build_observation(event, reset=True)
        self._compute_reward(obs)
        return obs

    def _build_observation(self, event, reset=False):
        rgb = event.frame
        local_sg = self.builder.build_from_metadata(event.metadata)
        self.global_sg.add_local_sg(local_sg)

        # Build exploration map and occupancy
        bounds = event.metadata["sceneBounds"]
        size = bounds["size"]
        agent_view = event.metadata["agent"]
        agent_x = agent_view["position"]["x"]
        agent_z = agent_view["position"]["z"]
        agent_rot = agent_view["rotation"]["y"]

        current_pose = {"x": float(agent_x), "z": float(agent_z), "yaw": float(agent_rot)}
        prev_rgb_frame = None if self._previous_rgb_frame is None else np.array(self._previous_rgb_frame, copy=True)
        odometry = None
        if self.prev_agent_pose is not None:
            dx = current_pose["x"] - self.prev_agent_pose["x"]
            dz = current_pose["z"] - self.prev_agent_pose["z"]
            dyaw = ((current_pose["yaw"] - self.prev_agent_pose["yaw"] + 180.0) % 360.0) - 180.0
            odometry = {"dx": float(dx), "dz": float(dz), "dyaw": float(dyaw)}
        self.prev_agent_pose = current_pose

        viewpoint = (
            round(agent_x / self.grid_size) * self.grid_size,
            round(agent_z / self.grid_size) * self.grid_size,
            round(agent_rot / 90) * 90,
        )
        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(viewpoint)

        if reset:
            map_width = math.ceil((size["x"] * 2) / self.grid_size)
            map_height = math.ceil((size["z"] * 2) / self.grid_size)
            map_width += 1 if map_width % 2 == 0 else 0
            map_height += 1 if map_height % 2 == 0 else 0

            self.map_origin = (agent_x - (map_width // 2) * self.grid_size,
                               agent_z - (map_height // 2) * self.grid_size)

            self.exploration_map = ExplorationMapManager(
                map_shape=(map_height, map_width),
                map_type=self.exploration_map_version,
                cell_size_cm=int(self.grid_size * 100),  # Convert to cm
                vision_range_cm=5000  # 3.2m vision in paper -> 50m because scene graph
            )
            self.exploration_map.reset()

            if self.exploration_map_version == "neural_slam":
                self.exploration_map.set_map_origin(self.map_origin[0], self.map_origin[1])

            self.occupancy_map = np.zeros((map_height, map_width, self.num_orientations), dtype=np.float32)

        self._update_occupancy(event)

        i, j, _ = self.get_occupancy_indices(event)

        walkable = any(self.try_action(a) for a in ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"])


        action_payload = deepcopy(self.pending_action) if self.pending_action is not None else None
        mapper_logits = self.pending_mapper_logits
        head_params = self.pending_head_params
        if self._pending_mapper_step is not None and self._pending_mapper_step != self.step_count:
            mapper_logits = None
            head_params = None

        self.exploration_map.update(
            (i, j),
            event,
            local_sg,
            walkable,
            action=action_payload,
            exploration_mask=self.pending_exploration_mask,
            pose_delta=self.pending_pose_delta,
            mapper_logits=mapper_logits,
            head_params=head_params,
            odometry=odometry,
            prev_observation=self.pending_prev_observation,
        )
        self.pending_action = None
        self.pending_mapper_logits = None
        self.pending_head_params = None
        self.pending_exploration_mask = None
        self.pending_pose_delta = None
        self.pending_prev_observation = None
        self._pending_mapper_step = None
        self._previous_action_payload = deepcopy(action_payload) if action_payload is not None else None
        self._previous_rgb_frame = np.array(rgb, copy=True)

        exploration_map_dict = (
            deepcopy(self.exploration_map.to_dict()) if self.exploration_map else None
        )

        policy_map = (
            self.exploration_map.get_map_for_policy() if self.exploration_map else None
        )
        if isinstance(policy_map, tuple) and len(policy_map) == 2:
            features, meta = policy_map
        else:
            features, meta = policy_map, None

        self.state = [rgb, local_sg, self.global_sg, exploration_map_dict]

        i, j, _ = self.get_occupancy_indices(event)
        info = {
            "event": event,
            "exploration_map": exploration_map_dict,
            "policy_map": policy_map,
            "map_index": (i, j),
        }
        map_type = getattr(self, "exploration_map_version", None)

        if map_type == "neural_slam":
            training_bundle = (
                self.exploration_map.get_training_data() if self.exploration_map else None
            )
            if training_bundle:
                ego_map_gt = training_bundle.get("ego_map_gt")
                if isinstance(ego_map_gt, np.ndarray):
                    ego_map_gt = np.array(ego_map_gt, copy=True)
                odom_payload = None if odometry is None else dict(odometry)
                info["neural_slam_training"] = {
                    "rgb_prev": prev_rgb_frame,
                    "rgb_curr": np.array(rgb, copy=True),
                    "odometry": odom_payload,
                    "ego_map_gt": ego_map_gt,
                }
        else:
            assert "neural_slam_training" not in info

        return Observation(
            state=self.state,
            info=info,
        )

        return Observation(state=self.state, info={"event": event})

    def get_ground_truth_graph(self, floorplan_name: str):
        """
        Loads or generates and returns the full ground-truth scene graph for a given floorplan.
        If no saved graph exists, it will be generated and saved automatically.
        """
        save_path = GT_GRAPHS_DIR / f"{floorplan_name}.json"

        # Generate if not exists
        if not os.path.exists(save_path):
            print(f"⚠️ GT Graph for {floorplan_name} not found. Generating...")
            generate_gt_scene_graphs(floorplans=[floorplan_name])

        return GTGraph().load_from_file(save_path)

    def _rotate_move(self, dx: float, dz: float, heading: int):
        """Rotate a movement vector by the agent heading in multiples of 90 degrees."""
        angle = heading % 360
        if angle == 90:
            return dz, -dx
        if angle == 180:
            return -dx, -dz
        if angle == 270:
            return -dz, dx
        return dx, dz

    def transition_step(self, action_str):
        # ensure env initialized
        if self.current_pos is None:
            raise ValueError("Call reset() before stepping.")

        x, z = self.current_pos
        rot = self.current_rot
        new_x, new_z, new_rot = x, z, rot
        success = True

        # rotation primitives
        if action_str == "RotateRight":
            new_rot = (rot + 90) % 360
        elif action_str == "RotateLeft":
            new_rot = (rot - 90) % 360
        elif action_str.startswith("Move"):
            # compute translation based on current orientation
            if action_str == "MoveAhead":
                dx, dz = 0, self.grid_size
            elif action_str == "MoveBack":
                dx, dz = 0, -self.grid_size
            elif action_str == "MoveRight":
                dx, dz = self.grid_size, 0
            elif action_str == "MoveLeft":
                dx, dz = -self.grid_size, 0
            else:
                dx, dz = 0, 0
            # rotate translation by agent heading
            # since moves are axis-aligned, for multiples of 90°, swap signs
            dx, dz = self._rotate_move(dx, dz, rot)
            new_x, new_z = x + dx, z + dz
        else:
            # Pass or unknown
            pass

        # lookup event
        key = (round(new_x, 2), round(new_z, 2), new_rot)
        event = self.mapping.get(key, None)
        if event is None:
            # invalid transition -> stay in place
            event = self.mapping.get((round(x, 2), round(z, 2), new_rot))
            new_x, new_z = x, z
            success = False

        # update agent pose and event metadata
        self.current_pos = (new_x, new_z)
        self.current_rot = new_rot
        event.metadata["lastActionSuccess"] = success

        self.last_event = event
        return event

    def step(self, action):
        actions = self.get_actions()[action]
        self.pending_action = {"index": action, "primitives": list(actions)}
        all_success = True
        for primitive_action in actions:
            event = self.transition_step(primitive_action)
            if not event.metadata.get("lastActionSuccess", True):
                all_success = False
        if self.pending_action is not None:
            self.pending_action["all_success"] = all_success

        obs = self._build_observation(event)

        previous_info = dict(obs.info) if obs.info is not None else {}
        preserved_policy_map = previous_info.get("policy_map")

        self.step_count += 1

        truncated = action == self.stop_index or self.step_count >= self.max_actions
        terminated = (
            len([k for k, n in self.global_sg.nodes.items() if n.visibility >= 0.8]) == len(self.gt_graph.nodes)
            and action == self.stop_index
        )

        if terminated:
            truncated = False
        obs.terminated = terminated
        obs.truncated = truncated
        score, recall_node, recall_edge = self.compute_score(obs)

        i, j, _ = self.get_occupancy_indices(event)
        obs.info = {
            "event": event,
            "score": score,
            "recall_node": recall_node,
            "recall_edge": recall_edge,
            "exploration_map": deepcopy(self.exploration_map.to_dict()) if self.exploration_map else None,
            "action": action,
            "agent_pos": self.current_pos,
            "allActionsSuccess": all_success,
            "max_steps_reached": self.step_count >= self.max_actions,
            "map_index": (i, j),
            "num_actions": self.get_action_dim()
        }

        if getattr(self, "exploration_map_version", None) != "neural_slam":
            assert "neural_slam_training" not in obs.info

        if "policy_map" in previous_info:
            obs.info["policy_map"] = preserved_policy_map

        for key, value in previous_info.items():
            if key not in obs.info:
                obs.info[key] = value

        obs.reward = self._compute_reward(obs)
        return obs

    def compute_score(self, obs):
        """
        Computes score based on discovered objects and termination status.
        Also returns recall for nodes and edges.
        """
        num_gt_objects = len(self.gt_graph.nodes)
        discovered_nodes = [n for n in self.global_sg.nodes.values() if n.visibility >= 0.8]
        num_discovered = len(discovered_nodes)
        # Recall for nodes
        recall_node = num_discovered / num_gt_objects if num_gt_objects > 0 else 0.0

        # Compute edge recall
        num_gt_edges = len(self.gt_graph.edges)
        num_discovered_edges = len(self.global_sg.edges) if hasattr(self.global_sg, "edges") else 0
        recall_edge = num_discovered_edges / num_gt_edges if num_gt_edges > 0 else 0.0

        termination_bonus = 0.0 if obs.terminated else 0.0
        score = recall_node + termination_bonus

        return score, recall_node, recall_edge

    def get_occupancy_indices(self, event):
        # Extract agent's current position and orientation
        pos = event.metadata["agent"]["position"]
        rot_y = event.metadata["agent"]["rotation"]["y"]

        x, z = pos["x"], pos["z"]

        # Compute offset from map origin
        dx = x - self.map_origin[0]
        dz = z - self.map_origin[1]

        i = self.occupancy_map.shape[0] - 1 - int(dz / self.grid_size)
        j = int(dx / self.grid_size)

        # Quantize rotation
        rot_idx = int(round(rot_y / 90.0)) % self.num_orientations

        return i, j, rot_idx

    def _update_occupancy(self, event):
        i, j, rot_idx = self.get_occupancy_indices(event)

        assert 0 <= rot_idx < self.num_orientations, f"Invalid rotation index: {rot_idx}"
        assert 0 <= i < self.occupancy_map.shape[0], f"Invalid i index: {i}"
        assert 0 <= j < self.occupancy_map.shape[1], f"Invalid j index: {j}"

        self.occupancy_map[i, j, rot_idx] = 1.0

        return i, j

    def _compute_reward(self, obs):
        """
        Compute the step reward for the agent based on scene graph discovery progress.

        The reward is defined as the change in a similarity-based score between the
        predicted global scene graph and the ground truth graph. The score combines:
            - Node recall (ratio of discovered objects to all ground truth objects),
            - Edge recall (ratio of discovered relations to all ground truth relations),
            - Node precision (average visibility of discovered objects),
            - Diversity (unique viewpoints from which objects have been observed),
            - A step penalty to discourage unnecessarily long episodes.

        Returns: The dense reward signal (difference in score since the previous step).
        """

        # Parameters as set in the original paper
        lambda_node = 0.1
        lambda_p = 0.5
        lambda_d = 0.001
        rho = self.rho

        # Recall for nodes and edges, extracted from the current global scene graph
        Rnode = obs.info.get("recall_node", 0.0)  # Recall for nodes
        Redge = obs.info.get("recall_edge", 0.0)  # Recall for edges

        if hasattr(self.global_sg, "nodes") and self.global_sg.nodes:
            Pnode = np.mean([n.visibility for n in self.global_sg.nodes.values()])
        else:
            Pnode = 0.0

        Pedge = 1.0

        # Diversity: sum of unique viewpoints for all objects
        diversity = sum(len(v) for v in self.viewpoints.values())

        # Compute similarity score between generated and ground truth scene graph
        sim = lambda_node * (Rnode + lambda_p * Pnode) + Redge + lambda_p * Pedge

        # Overall score at the current step, as in the paper
        score = sim + lambda_d * diversity - rho * self.step_count

        # Reward is the change in score since the last step (dense reward)
        reward = score - self.last_score
        self.last_score = score

        return reward

    def get_agent_state(self):
        ag = self.last_event.metadata["agent"]
        return {"position": (ag["position"]["x"], ag["position"]["z"]), "rotation": ag["rotation"]["y"]}

    def restore_agent_state(self, state):
        self.current_pos, self.current_rot = state["position"], state["rotation"]

    def get_env_state(self):
        return {
            "state": deepcopy(self.state),
            "global_sg": deepcopy(self.global_sg),
            #"exploration_map": deepcopy(self.exploration_map),
            "viewpoints": deepcopy(self.viewpoints),
            "last_score": self.last_score,
            "step_count": self.step_count,
        }

    def restore_env_state(self, env_state):
        self.state = deepcopy(env_state["state"])
        self.global_sg = deepcopy(env_state["global_sg"])

        if env_state.get("exploration_map"):
            if self.exploration_map is None:
                shape = tuple(env_state["exploration_map"]["map_shape"])
                # this will crash -> should not happen
                self.exploration_map = ExplorationMapManager(map_shape=shape, map_type="basic")
            self.exploration_map.from_dict(env_state["exploration_map"])

        self.viewpoints = deepcopy(env_state["viewpoints"])
        self.last_score = env_state["last_score"]
        self.step_count = env_state["step_count"]
        self.pending_action = None
        self.pending_mapper_logits = None
        self.pending_head_params = None
        self.prev_agent_pose = None

    def try_action(self, action_str, pos=None, rot=None):
        """Check if an action from a given pose leads to a valid event (not None)."""
        # determine base pose
        bx, bz = pos if pos is not None else self.current_pos
        brot = rot if rot is not None else self.current_rot
        nx, nz, nrot = bx, bz, brot

        # simulate primitive
        if action_str == "RotateRight":
            nrot = (brot + 90) % 360
        elif action_str == "RotateLeft":
            nrot = (brot - 90) % 360
        elif action_str.startswith("Move"):
            if action_str == "MoveAhead":
                dx, dz = 0, self.grid_size
            elif action_str == "MoveBack":
                dx, dz = 0, -self.grid_size
            elif action_str == "MoveRight":
                dx, dz = self.grid_size, 0
            elif action_str == "MoveLeft":
                dx, dz = -self.grid_size, 0
            else:
                dx, dz = 0, 0
            dx, dz = self._rotate_move(dx, dz, brot)
            nx, nz = bx + dx, bz + dz

        # lookup
        return self.mapping.get((round(nx, 2), round(nz, 2), nrot)) is not None

    def get_top_down_view(self):
        return None

    def visualize_shortest_path(self, start, target):
        return None

    def close(self):
        pass
