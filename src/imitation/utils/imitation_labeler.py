import ast
import copy
import math
import random
import re

from collections import namedtuple, deque


class ImitationLabeler:
    def __init__(self, env, mode = "viewpoint", success_recall_threshold=0.98,
             soft_recall_threshold=0.90,
             max_stagnation_slam=12,
             rotation_sweep_turns=3):
        self.env = env
        self.mode = mode # "viewpoint" | "slam"
        self.last_short_term_goal = None

        self.success_recall_threshold = float(success_recall_threshold)
        self.soft_recall_threshold = float(soft_recall_threshold)
        self.max_stagnation_slam = int(max_stagnation_slam)
        self.rotation_sweep_turns = int(rotation_sweep_turns)
        self._rotation_sweep_remaining = 0
        self._last_visible_count = None
        self._stagnation_ctr = 0

    def compute_score(self, env, visibility_before: dict, alpha=0.8):
        event = env.last_event
        visibility_after = {k: n.visibility for k, n in self.env.global_sg.nodes.items()}

        score = 0.0
        for obj_id, vis_after in visibility_after.items():
            vis_before = visibility_before.get(obj_id, 0.0)
            updated_vis = 1 - (1 - vis_before) * (1 - alpha * vis_after)
            delta_vis = updated_vis - vis_before
            score += delta_vis
            if vis_before < 0.8 <= updated_vis:
                score += 1

        i, j, rot_idx = self.env.get_occupancy_indices(event)
        occupancy_bonus = 0.9 if self.env.occupancy_map[i][j][rot_idx] == 0 else 0.0

        return score + occupancy_bonus

    def recover_missing_viewpoints(self, viewpoints, threshold=0.2):
        """
        If some objects are not yet sufficiently visible and all viewpoints have been explored,
        reintroduce viewpoints that help cover the missing objects.
        """
        global_seen = {k for k, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8}
        all_nodes = set(self.env.gt_graph.nodes.keys())

        missing = all_nodes - global_seen
        if not missing:
            return  # All done

        # Search for viewpoints that see these objects
        recovered_viewpoints = {}
        v2o = self.env.gt_graph.viewpoint_to_objects

        for vp_key, obj_list in v2o.items():
            recovered = []
            for obj in obj_list:
                for obj_id, vis in obj.items():
                    if obj_id in missing and vis >= threshold:
                        recovered.append(obj_id)
            if recovered:
                recovered_viewpoints[vp_key] = recovered

        if recovered_viewpoints:
            print(f"Recovered {len(recovered_viewpoints)} viewpoints for missing objects: {missing}")
            viewpoints.update(recovered_viewpoints)

    def _prune_viewpoints(self, viewpoints):
        if not viewpoints:
            return

        seen = {k for k, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8}
        to_delete = []
        for vp_key, obj_ids in list(viewpoints.items()):
            filtered = [obj_id for obj_id in obj_ids if obj_id not in seen]
            if filtered:
                viewpoints[vp_key] = filtered
            else:
                to_delete.append(vp_key)

        for vp_key in to_delete:
            del viewpoints[vp_key]

    def _norm_pos(self, pos):
        # tuple/list -> dict, dict -> dict{x,z}
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            return {"x": float(pos[0]), "z": float(pos[1])}
        if isinstance(pos, dict):
            return {"x": float(pos["x"]), "z": float(pos["z"])}
        raise TypeError(f"Unsupported position type: {type(pos)}")

    def _norm_rot(self, rot):
        # dict{'y':...} | int | float -> int in [0,360)
        if isinstance(rot, dict):
            rot = rot.get("y", rot.get("yaw", rot.get("rotation", 0)))
        if isinstance(rot, (int, float)):
            return int(round(rot)) % 360
        raise TypeError(f"Unsupported rotation type: {type(rot)}")

    def get_next_move_action(self, agent_pos, agent_rot, viewpoints, tol: float = 0.2):
        """
        Wählt eine primitive Move-Action (Ahead/Back/Left/Right), die den Agenten
        zum nächsten Viewpoint bringt. Arbeitet 100% offline mit PrecomputedThorEnv.
        """
        # --- Normalize inputs ---
        agent_pos = self._norm_pos(agent_pos)
        agent_rot = self._norm_rot(agent_rot)

        # Fertige Viewpoints abarbeiten: wenn am VP-Ort, evtl. nur drehen (Pass signalisiert Rotationsphase)
        while viewpoints:
            vp_key = next(iter(viewpoints))
            vp_pos, vp_rot = self.deserialize_viewpoint(vp_key)  # vp_pos ist dict{x,z}, vp_rot int
            if abs(agent_pos["x"] - vp_pos["x"]) < tol and abs(agent_pos["z"] - vp_pos["z"]) < tol:
                if agent_rot == vp_rot:
                    del viewpoints[vp_key]  # fertig mit diesem VP
                    continue
                else:
                    return "Pass"  # Rotationsschritt wird extern umgesetzt
            else:
                break

        # Falls nichts übrig: recovery versuchen
        if not viewpoints:
            self.recover_missing_viewpoints(viewpoints)
            if not viewpoints:
                raise ValueError("No viewpoints left to explore (even after recovery)")

        # Nächsten VP wählen (Manhattan-Distanz)
        vp_key = next(iter(viewpoints))
        vp_pos, _ = self.deserialize_viewpoint(vp_key)

        # Vektor zum Ziel (Weltkoordinaten)
        dx = vp_pos["x"] - agent_pos["x"]
        dz = vp_pos["z"] - agent_pos["z"]

        # In Agenten-Koordinaten drehen
        theta = math.radians(agent_rot)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        rel_forward = sin_t * dx + cos_t * dz
        rel_right = cos_t * dx - sin_t * dz

        # Kandidaten-Aktionen nach Beitrag sortieren
        actions = [
            ("MoveAhead" if rel_forward > 0 else "MoveBack", abs(rel_forward)),
            ("MoveRight" if rel_right > 0 else "MoveLeft", abs(rel_right)),
        ]
        actions.sort(key=lambda x: -x[1])

        # Erstbeste valide Aktion (gegen Mapping) wählen
        base_pos_tuple = (agent_pos["x"], agent_pos["z"])
        for action, val in actions:
            if val > 0 and self.env.try_action(action, pos=base_pos_tuple, rot=agent_rot):
                return action

        # Fallback, falls beide Moves blockiert sind: versuchen wir, nur zu drehen (Pass triggert Rotationsphase oben)
        return "Pass"

    def _create_ground_truth_egocentric_map(self, event, vision_range_cells: int = 64, cell_size_m: float = 0.05):
        """
        Create ground truth egocentric map from AI2THOR event metadata.
        This creates the training target for the Mapper network.

        Args:
            event: AI2THOR event with metadata
            vision_range_cells: Size of egocentric map (64x64 like in paper)
            cell_size_m: Size of each cell in meters (0.05m = 5cm like in paper)

        Returns:
            egocentric_map: (2, vision_range_cells, vision_range_cells) numpy array
                           Channel 0: obstacles, Channel 1: explored areas
        """
        import numpy as np
        import math

        # Initialize egocentric map
        ego_map = np.zeros((2, vision_range_cells, vision_range_cells), dtype=np.float32)

        # Get agent pose
        agent = event.metadata["agent"]
        agent_pos = agent["position"]
        agent_rot = agent["rotation"]["y"]

        # Convert rotation to radians
        agent_yaw = np.radians(agent_rot)

        # Get objects in the scene
        objects = event.metadata.get("objects", [])

        # Process each object to determine if it's an obstacle
        center = vision_range_cells // 2
        max_range = vision_range_cells * cell_size_m / 2  # Maximum vision range in meters

        for obj in objects:
            if not obj.get("visible", False):
                continue

            obj_pos = obj["position"]
            # Calculate relative position
            rel_x = obj_pos["x"] - agent_pos["x"]
            rel_z = obj_pos["z"] - agent_pos["z"]

            # Rotate to agent's reference frame
            cos_yaw, sin_yaw = np.cos(-agent_yaw), np.sin(-agent_yaw)
            ego_x = cos_yaw * rel_x - sin_yaw * rel_z
            ego_z = sin_yaw * rel_x + cos_yaw * rel_z

            # Skip if object is outside vision range
            if abs(ego_x) > max_range or abs(ego_z) > max_range:
                continue

            # Convert to cell coordinates
            cell_j = int(center + ego_x / cell_size_m)
            cell_i = int(center + ego_z / cell_size_m)

            # Check bounds
            if 0 <= cell_i < vision_range_cells and 0 <= cell_j < vision_range_cells:
                # Mark as explored (channel 1)
                ego_map[1, cell_i, cell_j] = 1.0

                # Mark as obstacle if not moveable and has collision
                if not obj.get("moveable", True) and not obj.get("pickupable", False):
                    # These are likely obstacles (walls, furniture, etc.)
                    ego_map[0, cell_i, cell_j] = 1.0

        # Mark the area immediately in front of agent as explored (field of view)
        fov_angle = np.radians(90)  # Field of view
        for i in range(vision_range_cells):
            for j in range(vision_range_cells):
                rel_i = i - center
                rel_j = j - center

                # Check if within reasonable distance and angle
                distance = np.sqrt(rel_i ** 2 + rel_j ** 2) * cell_size_m
                if distance <= 3.2:  # 3.2m max range as in paper
                    angle = np.arctan2(rel_j, rel_i)
                    if abs(angle) <= fov_angle / 2:
                        ego_map[1, i, j] = 1.0

        return ego_map

    def _add_sensor_noise(self, true_pose: dict, noise_std: dict = None) -> dict:
        """
        Add realistic sensor noise to agent pose for pose estimation training.

        Args:
            true_pose: Ground truth pose {"x": float, "z": float, "yaw": float}
            noise_std: Standard deviations for noise {"x": float, "z": float, "yaw": float}

        Returns:
            Noisy sensor pose
        """
        import numpy as np

        if noise_std is None:
            # Default noise values based on typical robot odometry
            noise_std = {"x": 0.02, "z": 0.02, "yaw": 2.0}  # 2cm translation, 2° rotation

        noisy_pose = {
            "x": true_pose["x"] + np.random.normal(0, noise_std["x"]),
            "z": true_pose["z"] + np.random.normal(0, noise_std["z"]),
            "yaw": true_pose["yaw"] + np.random.normal(0, noise_std["yaw"])
        }

        # Normalize yaw to [0, 360)
        noisy_pose["yaw"] = noisy_pose["yaw"] % 360

        return noisy_pose

    def select_best_action(self, viewpoints, planning_steps=3, replan_each_step=True, beam_width=3,
                           use_beam_width=False):
        """
        Updated to support Neural SLAM data collection with ground truth generation.
        """
        if self.mode == "neural_slam":
            event = self.env.last_event
            pose = self._agent_pose_from_event(event)

            # Fortschritt messen (Stagnation)
            vis_now = self._count_visible_nodes()
            if self._last_visible_count is None:
                self._last_visible_count = vis_now
            else:
                if vis_now <= self._last_visible_count:
                    self._stagnation_ctr += 1
                else:
                    self._stagnation_ctr = 0
                self._last_visible_count = vis_now

            total_nodes = len(self.env.gt_graph.nodes)
            recall_now = vis_now / total_nodes if total_nodes > 0 else 0.0

            # 1) Harte Erfolgsschwelle -> Stop
            if recall_now >= self.success_recall_threshold:
                return [self.env.stop_index]

            # GT-Egomap + Noisy-Sensor für IL-Supervision
            self.last_egocentric_map_gt = self._create_ground_truth_egocentric_map(event)
            self.last_sensor_pose = self._add_sensor_noise(pose)

            # 2) Frontier bestimmen
            target = self._nearest_frontier_world(pose)
            self.last_short_term_goal = target

            # 2a) Keine Frontier -> wenn soft-recall erreicht ODER stagnation -> Stop, sonst vorsichtig drehen
            if target is None:
                if recall_now >= self.soft_recall_threshold or self._stagnation_ctr >= self.max_stagnation_slam:
                    return [self.env.stop_index]
                # Fallback: geplante Viewpoints nutzen, sofern vorhanden
                if viewpoints:
                    self._prune_viewpoints(viewpoints)
                if viewpoints:
                    try:
                        move_action = self.get_next_move_action(
                            agent_pos={"x": pose["x"], "z": pose["z"]},
                            agent_rot=pose["yaw"],
                            viewpoints=viewpoints
                        )
                        action_indices = self.get_valid_action_indices(move_action)
                        if action_indices:
                            self._rotation_sweep_remaining = 0
                            return [action_indices[0]]
                    except ValueError:
                        pass
                # kleiner Dreh-Impuls, um evtl. versteckte Sichtlinien zu öffnen
                ridx = self._find_action_index("Pass", "RotateRight")
                return [ridx]

            # 3) Rotation-Sweep, wenn am Ziel
            if self._dist_world(pose, target) <= (self.env.grid_size * 0.6):
                if self._rotation_sweep_remaining <= 0:
                    self._rotation_sweep_remaining = self.rotation_sweep_turns
                ridx = self._find_action_index("Pass", "RotateRight")
                self._rotation_sweep_remaining -= 1
                return [ridx]

            # 4) Normale Fortbewegung zum Frontier-Ziel
            idx = self._choose_primitive_action_towards_local(
                agent_pos={"x": pose["x"], "z": pose["z"]},
                agent_yaw_deg=pose["yaw"],
                target_pos=target
            )
            return [idx]

        # Rest of the method stays the same for viewpoint mode
        env = self.env
        env_state = env.get_env_state()
        agent_state = env.get_agent_state()

        visibility_before = {k: n.visibility for k, n in env.global_sg.nodes.items()}
        node_before = [k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8]
        total_node_count = len(env.gt_graph.nodes)

        if len(node_before) == total_node_count:
            return [env.stop_index]

        ActionSeq = namedtuple("ActionSeq", ["seq", "score", "positions", "rotations", "viewpoints"])
        queue = deque()

        try:
            move_action = self.get_next_move_action(agent_state["position"], agent_state["rotation"], viewpoints)
        except ValueError as e:
            raise

        valid_action_indices = self.get_valid_action_indices(move_action)

        for i in valid_action_indices:
            env.restore_env_state(env_state)
            env.restore_agent_state(agent_state)
            obs_new = env.step(i)
            score = self.compute_score(env, visibility_before)
            agent_pos = obs_new.info["event"].metadata["agent"]["position"]
            agent_rot = obs_new.info["event"].metadata["agent"]["rotation"]
            queue.append(ActionSeq([i], score, [agent_pos], [agent_rot], [copy.deepcopy(viewpoints)]))

        queue = deque(sorted(queue, key=lambda x: x.score, reverse=True)[:beam_width] if use_beam_width else
                      sorted(queue, key=lambda x: x.score, reverse=True))

        for _ in range(1, planning_steps):
            candidates = []
            for action_seq in queue:
                vp_copy = {k: v[:] for k, v in viewpoints.items()} if replan_each_step else viewpoints
                try:
                    move_action = self.get_next_move_action(action_seq.positions[-1],
                                                            action_seq.rotations[-1], vp_copy)
                except ValueError:
                    continue

                valid_action_indices = self.get_valid_action_indices(move_action)
                for action in valid_action_indices:
                    env.restore_env_state(env_state)
                    env.restore_agent_state(agent_state)
                    visibility_before = {k: n.visibility for k, n in env.global_sg.nodes.items()}
                    step_score = 0

                    for i, a in enumerate(action_seq.seq):
                        env.step(a)
                        partial_score = self.compute_score(env, visibility_before)
                        step_score += partial_score
                        visibility_before = {k: n.visibility for k, n in env.global_sg.nodes.items()}

                    obs = env.step(action)

                    current_nodes = [k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8]
                    if len(current_nodes) == total_node_count:
                        return action_seq.seq + [action, env.stop_index]

                    final_score = self.compute_score(env, visibility_before)
                    total_score = step_score + final_score
                    combined_pos = action_seq.positions + [obs.info["event"].metadata["agent"]["position"]]
                    combined_rot = action_seq.rotations + [obs.info["event"].metadata["agent"]["rotation"]]
                    combined_vp = action_seq.viewpoints + [copy.deepcopy(vp_copy)]
                    candidates.append(ActionSeq(action_seq.seq + [action], total_score,
                                                combined_pos, combined_rot, combined_vp))

            queue = deque(sorted(candidates, key=lambda x: x.score, reverse=True)[:beam_width]
                          if use_beam_width else sorted(candidates, key=lambda x: x.score, reverse=True))

        if queue:
            best_seq = queue[0]
        else:
            best_seq = [random.choice(valid_action_indices)]

        return [best_seq.seq[0]] if replan_each_step else best_seq.seq

    def extract_navmesh_positions_from_error(self, error_msg):
        """
        Extracts the 'closest navmesh positions' for both start and target from an error message.
        Returns two dicts (start, target) or (None, None) if parsing fails.
        """
        pattern = r"closest navmesh position \((-?\d+\.\d+), [\d\.]+, (-?\d+\.\d+)\)"
        matches = re.findall(pattern, error_msg)

        if len(matches) >= 2:
            start_pos = {"x": float(matches[0][0]), "y": 0.900999, "z": float(matches[0][1])}
            target_pos = {"x": float(matches[1][0]), "y": 0.900999, "z": float(matches[1][1])}
            return start_pos, target_pos
        return None, None

    def get_shortest_path_to_point(self, initial_position, target_position, tolerance=0.2):
        if "y" not in initial_position:
            initial_position = {**initial_position, "y": 0.900999}
        if "y" not in target_position:
            target_position = {**target_position, "y": 0.900999}

        try:
            event = self.env.controller.step(
                action="GetShortestPathToPoint", position=initial_position, target=target_position, raise_for_failure=True
            )
            return event.metadata["actionReturn"]["corners"]
        except Exception as e:
            error_msg = str(e)
            snapped_start, snapped_target = self.extract_navmesh_positions_from_error(error_msg)
            if snapped_start is None or snapped_target is None:
                raise ValueError(f"Path failed and no usable navmesh correction found: {error_msg}")

            dx_start = abs(snapped_start["x"] - initial_position["x"])
            dz_start = abs(snapped_start["z"] - initial_position["z"])
            dx_target = abs(snapped_target["x"] - target_position["x"])
            dz_target = abs(snapped_target["z"] - target_position["z"])

            if dx_start <= tolerance and dz_start <= tolerance and dx_target <= tolerance and dz_target <= tolerance:
                retry_event = self.env.controller.step(
                    action="GetShortestPathToPoint", position=snapped_start, target=snapped_target, raise_for_failure=True
                )
                return retry_event.metadata["actionReturn"]["corners"]
            else:
                raise ValueError(
                    f"Navmesh snap too far from original positions. dx_start={dx_start}, dz_start={dz_start}, "
                    f"dx_target={dx_target}, dz_target={dz_target}. Error was: {error_msg}"
                )

    def get_valid_action_indices(self, move_action):
        actions = self.env.get_actions()
        return [i for i, a in enumerate(actions) if i != self.env.stop_index and move_action == a[0]]

    def _choose_primitive_action_towards_local(self, agent_pos: dict, agent_yaw_deg: float, target_pos: dict):
        import math
        dx = float(target_pos["x"]) - float(agent_pos["x"])
        dz = float(target_pos["z"]) - float(agent_pos["z"])
        theta = math.radians(agent_yaw_deg % 360)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        rel_forward = sin_t * dx + cos_t * dz
        rel_right   = cos_t * dx - math.sin(theta) * dz

        actions = self.env.get_actions()
        candidates = []
        if rel_forward >= 0:
            candidates.append("MoveAhead")
        else:
            candidates.append("MoveBack")
        if rel_right >= 0:
            candidates.append("MoveRight")
        else:
            candidates.append("MoveLeft")
        candidates.append("Pass")

        for a in candidates:
            idxs = [i for i, aa in enumerate(actions) if i != self.env.stop_index and a == aa[0]]
            for idx in idxs:
                if self.env.try_action(a, pos=(agent_pos["x"], agent_pos["z"]), rot=agent_yaw_deg):
                    return idx
        valid = [i for i in range(len(actions)) if i != self.env.stop_index]
        return valid[0] if valid else self.env.stop_index

    def _agent_pose_from_event(self, event) -> dict:
        ag = event.metadata["agent"]
        pos = ag["position"]
        rot = ag["rotation"]["y"]
        return {"x": float(pos["x"]), "z": float(pos["z"]), "yaw": float(rot)}

    def _nearest_frontier_world(self, agent_pos: dict):
        """
        Nimmt die nächste Frontier-Zelle im Kartenraster und rechnet sie
        mit der Env-Definition (map_origin + grid_size) in Weltkoordinaten um.
        -> passt 1:1 zu get_occupancy_indices() deiner Env.
        """
        # Map (Manager -> konkrete Map)
        m = self.env.exploration_map
        if hasattr(m, "map"):
            m = m.map

        if not (hasattr(m, "nearest_frontier") and hasattr(m, "map_shape")):
            return None

        # Agentenzelle aus der Env (konsistent mit Occupancy/Origin)
        ai, aj, _ = self.env.get_occupancy_indices(self.env.last_event)
        H, W = m.map_shape
        if not (0 <= ai < H and 0 <= aj < W):
            return None

        # Nächste Frontier im Raster
        res = m.nearest_frontier((ai, aj))
        if res is None:
            return None
        fi, fj = res

        # Raster -> Welt mit Env-Logik
        grid = float(self.env.grid_size)
        origin_x, origin_z = self.env.map_origin
        # Achtung: in get_occupancy_indices gilt i = H-1 - int(dz/grid)
        x_world = origin_x + fj * grid
        z_world = origin_z + (H - 1 - fi) * grid

        return {"x": float(x_world), "z": float(z_world)}

    @classmethod
    def deserialize_viewpoint(cls, s: str):
        try:
            dict_part, rotation = s.split("_")
            pos_dict = ast.literal_eval(dict_part)
            return pos_dict, int(rotation)
        except Exception as e:
            raise ValueError(f"Failed to deserialize viewpoint: {s} ({e})")

    def _count_visible_nodes(self):
        return sum(1 for _, n in self.env.global_sg.nodes.items() if n.visibility >= 0.8)

    def _find_action_index(self, move: str, rot: str):
        actions = self.env.get_actions()
        for i, (m, r) in enumerate(actions):
            if m == move and r == rot:
                return i
        return self.env.stop_index

    def _dist_world(self, pose: dict, target: dict):
        dx = float(target["x"]) - float(pose["x"])
        dz = float(target["z"]) - float(pose["z"])
        return (dx * dx + dz * dz) ** 0.5
