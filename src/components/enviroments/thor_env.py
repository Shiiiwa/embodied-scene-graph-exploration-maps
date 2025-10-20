import math
import os
import platform
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from src.components.exploration.exploration_map_manager import ExplorationMapManager
from src.components.graph.global_graph import GlobalSceneGraph
from src.components.graph.gt_graph import GTGraph
from src.components.graph.local_graph_builder import LocalSceneGraphBuilder
from src.components.utils.paths import GT_GRAPHS_DIR
from src.scripts.generate_gt_graphs import generate_gt_scene_graphs
from src.components.utils.observation import Observation


warnings.filterwarnings("ignore", message="could not connect to X Display*", category=UserWarning)


class ThorEnv:
    def __init__(
        self,
        rho=0.02,
        scene_number=None,
        render=False,
        grid_size=0.25,
        max_actions=40,
        map_version="metric_semantic_v1",
    ):
        super().__init__()
        self.rho = rho
        self.grid_size = grid_size
        self.visibilityDistance = 50  # high value so objects in the frame are always visible, visibility deals with far objects
        self.max_actions = max_actions
        # On Linux, use the specified 'render' flag; on other platforms, always set render=True (no headless mode support)
        self.render = render if platform.system() == "Linux" else True
        if self.render:
            self.controller = Controller(moveMagnitude=self.grid_size, grid_size=self.grid_size, visibilityDistance=self.visibilityDistance)
        else:
            self.controller = Controller(
                moveMagnitude=self.grid_size, grid_size=self.grid_size, visibilityDistance=self.visibilityDistance, platform=CloudRendering
            )
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
        self.occupancy_map = None
        self.exploration_map_version = map_version or "metric_semantic_v1"
        self.num_orientations = 4
        self.stop_index = self.get_actions().index(["Pass", "Pass"])
        self.agent_state = None
        self.pending_action: Optional[Dict[str, Any]] = None
        self.pending_mapper_logits: Optional[Any] = None
        self.pending_head_params: Optional[Any] = None
        self._pending_mapper_step: Optional[int] = None
        self.prev_agent_pose: Optional[Dict[str, float]] = None
        self._previous_rgb_frame: Optional[np.ndarray] = None
        self._previous_action_payload: Optional[Dict[str, Any]] = None

    def get_action_dim(self):
        return len(self.get_actions())

    def get_actions(self):
        agent_rotations = ["RotateRight", "RotateLeft", "Pass"]
        movements = ["MoveAhead", "MoveRight", "MoveLeft", "MoveBack", "Pass"]
        return [[move, rot1] for move in movements for rot1 in agent_rotations]

    def set_mapper_outputs(self, mapper_logits=None, head_params=None):
        """Store mapper outputs that will be consumed on the next observation update."""
        self.pending_mapper_logits = mapper_logits
        self.pending_head_params = head_params
        self._pending_mapper_step = getattr(self, "step_count", 0)

    def get_state_dim(self):
        warnings.warn(
            "The state dimension cannot be reliably determined from the environment. " "This method should not be used.", UserWarning
        )
        return [3, 128, 256]

    def reset(
        self,
        scene_number=None,
        random_start=False,
        start_position=None,
        start_rotation=None,
        map_version=None,
    ):
        if scene_number is not None:
            self.scene_number = scene_number
        if map_version is not None:
            self.exploration_map_version = map_version or "metric_semantic_v1"
        self.controller.reset(
            scene=f"FloorPlan{self.scene_number}",
            moveMagnitude=self.grid_size,
            grid_size=self.grid_size,
            visibilityDistance=self.visibilityDistance,
        )
        self.builder = LocalSceneGraphBuilder()
        self.global_sg = GlobalSceneGraph()
        self.step_count = 0
        self.last_score = 0.0
        self.viewpoints.clear()
        self.pending_action = None
        self.pending_mapper_logits = None
        self.pending_head_params = None
        self._pending_mapper_step = None
        self.prev_agent_pose = None
        self._previous_rgb_frame = None
        self._previous_action_payload = None

        if random_start:
            reachable = self.safe_step(action="GetReachablePositions").metadata["actionReturn"]
            pos = random.choice(reachable)
            rot = {"x": 0, "y": random.choice([0, 90, 180, 270]), "z": 0}
        elif start_position and start_rotation:
            pos = start_position
            rot = start_rotation
        else:
            pos = None

        if pos is not None:
            self.safe_step(action="Teleport", position=pos, rotation=rot)

        event = self.safe_step(action="Pass")
        rgb = event.frame
        local_sg = self.builder.build_from_metadata(event.metadata)

        agent_view = event.metadata["agent"]
        agent_pos = agent_view["position"]
        agent_rot = agent_view["rotation"]["y"]

        viewpoint = (
            round(agent_pos["x"] / self.grid_size) * self.grid_size,
            round(agent_pos["z"] / self.grid_size) * self.grid_size,
            round(agent_rot / 90) * 90,
        )
        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(viewpoint)
        self.global_sg.add_local_sg(local_sg)

        # Get scene size information (used for estimating a generous map size)
        bounds = event.metadata["sceneBounds"]
        size = bounds["size"]

        # Get agent's starting position
        start_x = agent_pos["x"]
        start_z = agent_pos["z"]

        # Compute extended map dimensions: 2x scene width and height (in grid units)
        map_width = math.ceil((size["x"] * 2) / self.grid_size)
        map_height = math.ceil((size["z"] * 2) / self.grid_size)

        # Force odd size for perfect centering
        if map_width % 2 == 0:
            map_width += 1
        if map_height % 2 == 0:
            map_height += 1

        # Set the map origin so that the agent starts roughly at the center of the occupancy map
        self.map_origin = (start_x - (map_width // 2) * self.grid_size, start_z - (map_height // 2) * self.grid_size)

        # Initialize the occupancy map with zeros (unvisited); shape: [H, W, rotations]
        # self.exploration_map = ExplorationMap(grid_size=self.grid_size, map_width=map_width, map_height=map_height, origin=self.map_origin)
        # self.exploration_map.update_from_event(event)
        map_shape = (map_height, map_width)

        map_kwargs = {}
        if self.exploration_map_version == "neural_slam":
            map_kwargs = {
                "cell_size_cm": int(self.grid_size * 100),
                "vision_range_cm": 5000,
            }
        self.exploration_map = ExplorationMapManager(
            map_shape=map_shape,
            map_type=self.exploration_map_version,
            **map_kwargs,
        )
        if hasattr(self.exploration_map, "reset"):
            self.exploration_map.reset()
        if self.exploration_map_version == "neural_slam":
            self.exploration_map.set_map_origin(self.map_origin[0], self.map_origin[1])

        agent_pos = event.metadata["agent"]["position"]
        x, z = agent_pos["x"], agent_pos["z"]
        current_pose = {
            "x": float(x),
            "z": float(z),
            "yaw": float(agent_view["rotation"]["y"]),
        }
        prev_rgb_frame = None if self._previous_rgb_frame is None else np.array(self._previous_rgb_frame, copy=True)

        # Convert world coordinate to grid cells
        i = map_height - 1 - int((z - self.map_origin[1]) / self.grid_size)
        j = int((x - self.map_origin[0]) / self.grid_size)
        self.exploration_map.update(
            (i, j),
            event,
            local_sg,
            action=None,
            mapper_logits=None,
            head_params=None,
            odometry=None,
        )
        self.prev_agent_pose = current_pose
        self._previous_action_payload = None
        self._previous_rgb_frame = np.array(rgb, copy=True)

        self.occupancy_map = np.zeros((map_height, map_width, self.num_orientations), dtype=np.float32)
        agent_x, agent_z = self._update_occupancy(event)

        exploration_map_dict = deepcopy(self.exploration_map.to_dict()) if self.exploration_map else None
        policy_map = self.exploration_map.get_map_for_policy() if self.exploration_map else None
        self.state = [rgb, local_sg, self.global_sg, exploration_map_dict]

        info = {"event": event, "exploration_map": exploration_map_dict, "policy_map": policy_map}
        if self.exploration_map_version == "neural_slam":
            training_bundle = (
                self.exploration_map.get_training_data() if self.exploration_map else None
            )
            if training_bundle:
                ego_map_gt = training_bundle.get("ego_map_gt")
                if isinstance(ego_map_gt, np.ndarray):
                    ego_map_gt = np.array(ego_map_gt, copy=True)
                info["neural_slam_training"] = {
                    "rgb_prev": prev_rgb_frame,
                    "rgb_curr": np.array(rgb, copy=True),
                    "odometry": None,
                    "ego_map_gt": ego_map_gt,
                }
        else:
            assert "neural_slam_training" not in info
        obs = Observation(
            state=self.state,
            info=info,
        )
        self._compute_reward(obs)

        self.gt_graph = self.get_ground_truth_graph(f"FloorPlan{self.scene_number}")

        # --- Add Top-Down Camera after reset ---
        bounds = event.metadata["sceneBounds"]
        center = bounds["center"]
        size = bounds["size"]
        top_camera_height = size["y"] - 0.5

        evt = self.safe_step(
            action="AddThirdPartyCamera",
            rotation=dict(x=90, y=0, z=0),
            position=dict(x=center["x"], y=top_camera_height, z=center["z"]),
            fieldOfView=2.25,
            orthographic=True,
        )

        return obs

    def get_ground_truth_graph(self, floorplan_name: str):
        """
        Loads or generates and returns the full ground-truth scene graph for a given floorplan.
        If no saved graph exists, it will be generated and saved automatically.
        """
        save_path = GT_GRAPHS_DIR / (floorplan_name + ".json")

        # Generate if not exists
        if not os.path.exists(save_path):
            print(f"⚠️ GT Graph for {floorplan_name} not found. Generating...")
            generate_gt_scene_graphs(floorplans=[floorplan_name])

        return GTGraph().load_from_file(save_path)

    def safe_step(self, *args, **kwargs):
        try:
            self.agent_state = self.get_agent_state()
            return self.controller.step(*args, **kwargs)
        except TimeoutError as e:
            print(f"[TIMEOUT] Action '{kwargs.get('action', 'unknown')}' timed out. Restarting environment.")
            self.reset_hard()
            return self.controller.step(*args, **kwargs)

    def reset_hard(self):
        try:
            self.controller.stop()
        except Exception as e:
            print(f"[WARN] Failed to stop controller cleanly: {e}")

        if self.render:
            self.controller = Controller(moveMagnitude=self.grid_size, grid_size=self.grid_size, visibilityDistance=self.visibilityDistance)
        else:
            self.controller = Controller(
                moveMagnitude=self.grid_size, grid_size=self.grid_size, visibilityDistance=self.visibilityDistance, platform=CloudRendering
            )
        self.reset(scene_number=self.scene_number)
        self.restore_agent_state(self.agent_state)

    def step(self, action):
        actions = self.get_actions()[action]
        self.pending_action = {"index": action, "primitives": list(actions)}
        error_msgs: Dict[str, Any] = {}
        all_success = True
        for primitive_action in actions:
            if "Move" in primitive_action:
                event = self.safe_step(action=primitive_action, moveMagnitude=self.grid_size)
            else:
                event = self.safe_step(action=primitive_action)
            success = event.metadata["lastActionSuccess"]
            if not success:
                error_msgs[primitive_action] = event.metadata["errorMessage"]
                all_success = False
        if self.pending_action is not None:
            self.pending_action["all_success"] = all_success
        event = self.safe_step(action="Pass")

        # self.exploration_map.update_from_event(event)
        # self.exploration_map.mark_discoveries(event, self.global_sg)
        # agent_x, agent_z = self._update_occupancy(event)

        prev_rgb_frame = None if self._previous_rgb_frame is None else np.array(self._previous_rgb_frame, copy=True)

        agent_view = event.metadata["agent"]
        agent_pos = agent_view["position"]
        agent_rot = agent_view["rotation"]["y"]
        current_pose = {
            "x": float(agent_pos["x"]),
            "z": float(agent_pos["z"]),
            "yaw": float(agent_rot),
        }
        odometry = None
        if self.prev_agent_pose is not None:
            dx = current_pose["x"] - self.prev_agent_pose["x"]
            dz = current_pose["z"] - self.prev_agent_pose["z"]
            dyaw = ((current_pose["yaw"] - self.prev_agent_pose["yaw"] + 180.0) % 360.0) - 180.0
            odometry = {"dx": float(dx), "dz": float(dz), "dyaw": float(dyaw)}
        self.prev_agent_pose = current_pose

        x, z = agent_pos["x"], agent_pos["z"]
        i = self.occupancy_map.shape[0] - 1 - int((z - self.map_origin[1]) / self.grid_size)
        j = int((x - self.map_origin[0]) / self.grid_size)

        agent_x, agent_z = self._update_occupancy(event)

        rgb = event.frame
        local_sg = self.builder.build_from_metadata(event.metadata)
        self.global_sg.add_local_sg(local_sg)

        walkable = None
        if self.exploration_map:
            walkable = any(self.try_action(a) for a in ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight"])

        if self.exploration_map:
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
                mapper_logits=mapper_logits,
                head_params=head_params,
                odometry=odometry,
            )
        else:
            action_payload = None

        self.pending_action = None
        self.pending_mapper_logits = None
        self.pending_head_params = None
        self._pending_mapper_step = None
        self._previous_action_payload = deepcopy(action_payload) if action_payload is not None else None
        self._previous_rgb_frame = np.array(rgb, copy=True)

        exploration_map_dict = deepcopy(self.exploration_map.to_dict()) if self.exploration_map else None
        policy_map = self.exploration_map.get_map_for_policy() if self.exploration_map else None
        self.state = [rgb, local_sg, self.global_sg, exploration_map_dict]

        viewpoint = (
            round(agent_pos["x"] / self.grid_size) * self.grid_size,
            round(agent_pos["z"] / self.grid_size) * self.grid_size,
            round(agent_rot / 90) * 90,
        )

        # Update viewpoint tracking
        for node in local_sg.nodes.values():
            self.viewpoints[node.object_id].add(viewpoint)

        self.step_count += 1

        truncated = action == self.stop_index or self.step_count >= self.max_actions
        terminated = (
            len([k for k, n in self.global_sg.nodes.items() if n.visibility >= 0.8]) == len(self.gt_graph.nodes)
            and action == self.stop_index
        )

        if terminated:
            truncated = False
        obs = Observation(state=self.state, truncated=truncated, terminated=terminated)

        score, recall_node, recall_edge = self.compute_score(obs)

        info = {
            "event": event,
            "score": score,
            "recall_node": recall_node,
            "recall_edge": recall_edge,
            "action": action,
            "agent_pos": (agent_x, agent_z),
            "map_index": (i, j),
            "exploration_map": exploration_map_dict,
            "policy_map": policy_map,
            "allActionsSuccess": all_success,
            "errorMessages": error_msgs,
            "max_steps_reached": self.step_count >= self.max_actions,
        }
        if self.exploration_map_version == "neural_slam":
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

        obs.info = info

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
        agent = self.controller.last_event.metadata["agent"]
        return {"position": agent["position"], "rotation": agent["rotation"]}

    def restore_agent_state(self, agent_state):
        self.controller.step(action="Teleport", position=agent_state["position"], rotation=agent_state["rotation"], horizon=0)
        self.controller.step(action="Pass")

    def get_env_state(self):
        return {
            "state": deepcopy(self.state),
            "global_sg": deepcopy(self.global_sg),
            "exploration_map": deepcopy(self.exploration_map),
            "viewpoints": deepcopy(self.viewpoints),
            "last_score": self.last_score,
            "step_count": self.step_count,
        }

    def restore_env_state(self, env_state):
        self.state = deepcopy(env_state["state"])
        self.global_sg = deepcopy(env_state["global_sg"])
        self.exploration_map = deepcopy(env_state["exploration_map"])
        self.viewpoints = deepcopy(env_state["viewpoints"])
        self.last_score = env_state["last_score"]
        self.step_count = env_state["step_count"]

    def try_action(self, action, agent_pos=None, agent_rot=None):
        env_state = self.get_env_state()
        agent_state = self.get_agent_state()
        if agent_pos is not None and agent_rot is not None:
            event = self.safe_step(action="Teleport", position=agent_pos, rotation=dict(x=0, y=agent_rot, z=0))
        event = self.safe_step(action=action)
        self.restore_env_state(env_state)
        self.restore_agent_state(agent_state)
        return event.metadata["lastActionSuccess"]

    def get_top_down_view(self):
        """
        Returns the current top-down view image from the third-party camera as a numpy array (H,W,3).
        """
        event = self.safe_step(action="Pass")
        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            return event.third_party_camera_frames[0]
        else:
            raise RuntimeError("No third-party camera frames found.")

    def visualize_shortest_path(self, start, target):
        """
        Visualizes the shortest path from start to goal on the current top-down view.
        - start, goal: dicts with 'x', 'y', 'z'
        """
        if len(target) == 2:
            target = {"x": target["x"], "y": start["y"], "z": target["z"]}
        event = self.safe_step(action="GetShortestPathToPoint", position=start, target=target)
        path = event.metadata["actionReturn"]["corners"]  # list of dicts

        event = self.safe_step(action="VisualizePath", positions=path, grid=False, endText="Target")
        if hasattr(event, "third_party_camera_frames") and event.third_party_camera_frames:
            arr = event.third_party_camera_frames[0]
            return Image.fromarray(arr)
        return None

    def close(self):
        self.controller.stop()
