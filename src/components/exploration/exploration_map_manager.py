from typing import Optional, Tuple

from src.components.exploration.maps.metric_map_v2 import MetricSemanticMapV2
from src.components.exploration.maps.metric_semantic_map import MetricSemanticMap
from src.components.exploration.maps.neural_slam_map import NeuralSlamMap


class ExplorationMapManager:
    def __init__(self, map_shape, map_type="metric_semantic_v1", **kwargs):
        """Initialize the map manager.

        Args:
            map_shape: Tuple ``(H, W)`` measured in cells using agent-centric
                coordinates.
            map_type: One of ``"metric_semantic_v1"``, ``"metric_semantic_v2"``,
                or ``"neural_slam"``.
            kwargs: Additional map parameters such as ``cell_size_cm`` or
                ``vision_range_cm`` (used by the Neural SLAM map).
        """
        self.map_shape = map_shape
        self.map_type = map_type
        self.kwargs = kwargs
        self._last_training_data = None
        self._pending_pose_prior: Optional[Tuple[Tuple[int, int], Optional[float]]] = None
        self._init_map()

    def _init_map(self):
        if self.map_type == "metric_semantic_v1":
            self.map = MetricSemanticMap(self.map_shape)
        elif self.map_type == "metric_semantic_v2":
            self.map = MetricSemanticMapV2(self.map_shape)
        elif self.map_type == "neural_slam":
            cell_size_cm = int(self.kwargs.get("cell_size_cm", 25))
            vision_range_cm = int(self.kwargs.get("vision_range_cm", 320))
            self.map = NeuralSlamMap(
                map_shape=self.map_shape,
                cell_size_cm=cell_size_cm,
                vision_range_cm=vision_range_cm
            )
        else:
            raise NotImplementedError(f"Map type '{self.map_type}' unknown.")

    def reset(
        self,
        spawn_cell: Optional[Tuple[int, int]] = None,
        vision_range: Optional[float] = None,
    ):
        if spawn_cell is None and self._pending_pose_prior is not None:
            spawn_cell, vision_range = self._pending_pose_prior
        if self.map_type == "neural_slam":
            self.map.reset(start_pose=spawn_cell, vision_range_m=vision_range)
        else:
            self.map.reset()
        self._last_training_data = None
        self._pending_pose_prior = None

    def set_pose_prior(
        self,
        spawn_cell: Optional[Tuple[int, int]],
        vision_range: Optional[float] = None,
    ) -> None:
        """Hook to seed a pose prior at the beginning of an episode."""

        if spawn_cell is None:
            self._pending_pose_prior = None
            if self.map_type == "neural_slam":
                self.map.set_pose_prior(None, vision_range_m=vision_range)
            return

        pose_tuple = tuple(int(round(float(x))) for x in spawn_cell)
        self._pending_pose_prior = (pose_tuple, vision_range)
        if self.map_type == "neural_slam":
            self.map.set_pose_prior(start_pose=pose_tuple, vision_range_m=vision_range)

    def update(
        self,
        agent_position,
        event,
        local_sg,
        walkable=None,
        **kwargs,
    ):
        """Update the active map based on the latest observation.

        Args:
            agent_position: ``(i, j)`` cell coordinates (not world meters).
            walkable: Optional boolean indicating whether the current cell is
                traversable.
        """
        action = kwargs.pop("action", None)
        mapper_logits = kwargs.pop("mapper_logits", None)
        head_params = kwargs.pop("head_params", None)
        odometry = kwargs.pop("odometry", None)
        exploration_mask = kwargs.pop("exploration_mask", None)
        pose_delta = kwargs.pop("pose_delta", None)
        prev_observation = kwargs.pop("prev_observation", None)

        if kwargs:
            raise TypeError(
                "Unexpected keyword arguments for ExplorationMapManager.update: "
                f"{', '.join(kwargs.keys())}"
            )

        if self.map_type == "neural_slam":
            # Neural SLAM requires the simulator event and the pose metadata.
            rgb_obs = event.frame  # (H, W, 3) NumPy array
            agent_metadata = event.metadata["agent"]
            agent_world_pose = {
                "x": float(agent_metadata["position"]["x"]),
                "z": float(agent_metadata["position"]["z"]),
                "yaw": float(agent_metadata["rotation"]["y"])
            }
            training_data = self.map.update_from_observation(
                rgb_obs=rgb_obs,
                agent_world_pose=agent_world_pose,
                sensor_pose=None,  # Placeholder for future noisy-sensor input.
                event=event,  # Required for ego_map_gt generation.
                action=action,
                mapper_logits=mapper_logits,
                head_params=head_params,
                odometry=odometry,
                exploration_mask=exploration_mask,
                pose_delta=pose_delta,
                prev_observation=prev_observation,
            )
            self._last_training_data = training_data
        else:
            # Metric semantic maps keep their existing API.
            if walkable is None:
                self.map.update(agent_position, event, local_sg)
            else:
                self.map.update(agent_position, event, local_sg, walkable)

    def get_map(self):
        """Return the raw map object (cells dict for v1/v2; internal for SLAM)."""
        return self.map.get()

    def to_dict(self):
        """Return a serializable representation (e.g., for pickle)."""
        return self.map.to_dict()

    def render_matplotlib(self, show=True, save_path=None):
        self.map.render_matplotlib(show=show, save_path=save_path)

    def render_ascii(self):
        if hasattr(self.map, "render_ascii"):
            self.map.render_ascii()

    def print_visited_with_objects(self):
        if hasattr(self.map, "print_visited_with_objects"):
            self.map.print_visited_with_objects()

    def _world_to_cell(self, x, z):
        """Convert world-space meters to cell coordinates when supported.

        Neural SLAM exposes ``world_to_map`` directly. The v1 implementation
        provides ``_world_to_cell``; v2 usually operates in cell space already,
        but this fallback keeps the call robust across map types.
        """
        if self.map_type == "neural_slam":
            return self.map.world_to_map(x, z)
        # Provide a best-effort fallback for the v1/v2 implementations.
        if hasattr(self.map, "_world_to_cell"):
            return self.map._world_to_cell(x, z)
        raise NotImplementedError(
            "world_to_cell is not available for this map type. "
            "Pass (i, j) cell coordinates directly to update()."
        )

    def from_dict(self, data):
        """Reconstruct a map from a serialized dictionary payload."""
        if self.map_type == "metric_semantic_v1":
            self.map = MetricSemanticMap.from_dict(data)
        elif self.map_type == "metric_semantic_v2":
            self.map = MetricSemanticMapV2.from_dict(data)
        elif self.map_type == "neural_slam":
            self.map = NeuralSlamMap.from_dict(data)
        else:
            raise NotImplementedError(f"Map type '{self.map_type}' unknown.")

    def set_map_origin(self, world_x, world_z):
        """Set the world-space origin for Neural SLAM maps."""
        if self.map_type == "neural_slam":
            self.map.set_map_origin(world_x, world_z)

    def get_training_data(self):
        """Return Neural SLAM training data when available; otherwise ``None``."""
        if self.map_type == "neural_slam":
            return self.map.get_training_data()
        return None

    def get_map_for_policy(self):
        """Return the policy-ready representation for the active map type."""
        if self.map_type == "neural_slam":
            return self.map.get_map_for_policy()
        if self.map_type == "metric_semantic_v2":
            feats, meta = self.map.get_map_for_policy()
            return feats, meta
        if self.map_type == "metric_semantic_v1":
            # Returns a serialized structure with map_shape and visited cells.
            return self.map.to_dict()
        return self.get_map()
