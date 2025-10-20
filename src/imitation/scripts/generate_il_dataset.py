from pathlib import Path
import argparse
import copy
import pickle
import random
import warnings
from tqdm import tqdm
from math import hypot
from typing import Dict, List, Optional

from src.components.utils.aco_tsp import SolveTSPUsingACO
from src.components.enviroments.precomputed_thor_env import PrecomputedThorEnv
from src.components.utils.paths import TRANSITION_TABLES, IL_DATASET_DIR, CONFIG_DIR
from src.components.utils.config_loading import (
    load_normalized_config,
    derive_experiment_tag,
)
from src.imitation.utils.imitation_labeler import ImitationLabeler


def get_unique_starts(env, k=5):
    """Get unique starting positions from the precomputed environment mapping"""
    # Get all valid positions from the precomputed mapping
    valid_keys = [(x, z, rot) for (x, z, rot), evt in env.mapping.items() if evt is not None]
    random.shuffle(valid_keys)

    unique = set()
    starts = []

    for x, z, rot in valid_keys:
        pos_key = (round(x, 2), round(z, 2))
        if pos_key in unique:
            continue

        # Check if this is a valid starting position by testing movement
        if is_valid_start_precomputed(env, (x, z), rot):
            unique.add(pos_key)
            start_pos = (x, z)  # Changed from dict to tuple
            start_rot = rot  # Changed from dict to int
            starts.append((start_pos, start_rot))

        if len(starts) >= k:
            break

    env.close()
    return starts


def is_valid_start_precomputed(env, position, rotation):
    """Check if a position/rotation is valid by testing if movement actions are possible"""
    x, z = position
    directions = ["MoveAhead", "MoveLeft", "MoveRight", "MoveBack"]

    for action in directions:
        if env.try_action(action, pos=(x, z), rot=rotation):
            return True
    return False


def aggregate_visibility(global_vis: float, local_vis: float, alpha: float = 0.8) -> float:
    return 1 - (1 - global_vis) * (1 - alpha * local_vis)


def compute_minimal_viewpoint_cover(
        viewpoint_to_objects: Dict[str, List[Dict[str, float]]], threshold: float = 0.8, alpha: float = 0.8
) -> Dict[str, List[str]]:
    # All unique object IDs
    all_objects = set()
    for obj_list in viewpoint_to_objects.values():
        for obj_dict in obj_list:
            all_objects.update(obj_dict.keys())

    # Initialize visibility for all objects
    visibility = {obj_id: 0.0 for obj_id in all_objects}
    selected_viewpoints = {}
    remaining_viewpoints = set(viewpoint_to_objects.keys())

    while any(v < threshold for v in visibility.values()):
        best_vp = None
        best_gain = 0
        best_new_vis = None

        for vp in remaining_viewpoints:
            temp_vis = copy.deepcopy(visibility)
            gain = 0.0
            for obj_dict in viewpoint_to_objects[vp]:
                for obj_id, local_vis in obj_dict.items():
                    if temp_vis[obj_id] >= threshold:
                        continue
                    updated_vis = aggregate_visibility(temp_vis[obj_id], local_vis, alpha)
                    gain += max(0.0, updated_vis - temp_vis[obj_id])
                    temp_vis[obj_id] = updated_vis

            if gain > best_gain:
                best_gain = gain
                best_vp = vp
                best_new_vis = temp_vis

        if best_vp is None:
            # No improvement possible anymore
            break

        visibility = best_new_vis
        selected_viewpoints[best_vp] = [list(d.keys())[0] for d in viewpoint_to_objects[best_vp]]
        remaining_viewpoints.remove(best_vp)

    return selected_viewpoints


def update_viewpoints(env, viewpoints):
    """
    Remove all objects from each viewpoint that are already seen in the global scene graph.
    Optionally, remove viewpoints with empty object lists.
    """
    seen = set([k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8])
    to_delete = []
    for vp, objs in viewpoints.items():
        filtered = [obj for obj in objs if obj not in seen]
        if filtered:
            viewpoints[vp] = filtered
        else:
            to_delete.append(vp)
    for vp in to_delete:
        del viewpoints[vp]


def get_shortest_viewpoint_path(start_x, start_z, viewpoints, use_aco=True):
    """
    Computes a short tour through all viewpoints using either ACO or greedy heuristic.
    The starting position (start_x, start_z) is NOT included in the TSP computation.
    Instead, the resulting path is rotated so that it starts at the viewpoint closest to the start position.
    """
    vp_keys = list(viewpoints.keys())
    vp_positions = {vp: ImitationLabeler.deserialize_viewpoint(vp)[0] for vp in vp_keys}
    vp_coords = {vp: (pos["x"], pos["z"]) for vp, pos in vp_positions.items()}

    if use_aco:
        points = [vp_coords[vp] for vp in vp_keys]
        tsp_solver = SolveTSPUsingACO(
            mode="MaxMin", colony_size=max(10, len(vp_keys)), steps=max(200, 20 * len(vp_keys)), nodes=points,
            labels=vp_keys
        )
        _, _ = tsp_solver.run()
        tour = tsp_solver.global_best_tour
        ordered_vps = [vp_keys[i] for i in tour]
    else:
        # Greedy TSP
        unvisited = set(vp_keys)
        ordered_vps = []
        curr_vp = vp_keys[0]  # arbitrary start
        curr_x, curr_z = vp_coords[curr_vp]
        ordered_vps.append(curr_vp)
        unvisited.remove(curr_vp)

        while unvisited:
            next_vp = min(unvisited, key=lambda vp: hypot(vp_coords[vp][0] - curr_x, vp_coords[vp][1] - curr_z))
            ordered_vps.append(next_vp)
            curr_x, curr_z = vp_coords[next_vp]
            unvisited.remove(next_vp)

    # Rotate path to start from viewpoint nearest to (start_x, start_z)
    distances_to_start = [hypot(vp_coords[vp][0] - start_x, vp_coords[vp][1] - start_z) for vp in ordered_vps]
    closest_idx = distances_to_start.index(min(distances_to_start))
    rotated_path = ordered_vps[closest_idx:] + ordered_vps[:closest_idx]

    return rotated_path


def has_valid_path_precomputed(env, start, target):
    """
    For precomputed environment, we approximate path validity by checking
    if both start and target positions exist in the mapping
    """
    start_key = (round(start["x"], 2), round(start["z"], 2), 0)  # rotation doesn't matter for existence check
    target_key = (round(target["x"], 2), round(target["z"], 2), 0)

    # Check if both positions have valid events in any rotation
    start_valid = any(env.mapping.get((start_key[0], start_key[1], rot)) is not None for rot in [0, 90, 180, 270])
    target_valid = any(env.mapping.get((target_key[0], target_key[1], rot)) is not None for rot in [0, 90, 180, 270])

    return start_valid and target_valid

def save_dataset(scene_id: int, data: list, start_x: float, start_z: float, rot_y: float, group_name: str):
    scene_name = f"FloorPlan{scene_id}"

    scene_dir = IL_DATASET_DIR / group_name / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{scene_name}_px_{start_x}_pz_{start_z}_ry_{rot_y}.pkl"
    save_path = scene_dir / filename

    with open(save_path, "wb") as f:
        pickle.dump(data, f)

    return str(save_path)

def build_sample(env, obs, step_idx, last_action_idx, start_pos, start_rot, labeler, config):
    # Coverage
    visited_any = (env.occupancy_map > 0).any(axis=2) if env.occupancy_map is not None else None
    coverage_m2 = float(visited_any.sum()) * (env.grid_size ** 2) if visited_any is not None else None

    # Save-Obs ohne fettes Event
    save_obs = copy.deepcopy(obs)
    save_obs.info["event"] = {}
    save_obs.state[1] = save_obs.state[1].to_dict()
    save_obs.state[2] = save_obs.state[2].to_dict()

    base = {
        "scene": env.scene_number,
        "start_position": start_pos,
        "start_rotation": start_rot,
        "step": step_idx,
        "obs": save_obs,
        "last_action": last_action_idx,
        "num_actions": env.get_action_dim(),
        "exploration_map": copy.deepcopy(obs.info.get("exploration_map")),
    }

    if config['exploration']['map_version'] == "neural_slam":
        # Achtung: Pose/Shortgoal aus dem *echten* obs (nicht save_obs)
        pose_gt = None
        if obs and obs.info and "event" in obs.info and obs.info["event"]:
            ag = obs.info["event"].metadata["agent"]
            pose_gt = {"x": ag["position"]["x"], "z": ag["position"]["z"], "yaw": ag["rotation"]["y"]}

        base.update({
            "pose_gt": pose_gt,
            "coverage_m2": coverage_m2,
            "short_term_goal": getattr(labeler, "last_short_term_goal", None),
            "ego_map_gt": getattr(labeler, "last_egocentric_map_gt", None),
            "sensor_pose_noisy": getattr(labeler, "last_sensor_pose", None),
            "neural_slam_training": env.exploration_map.get_training_data() if hasattr(env, "exploration_map") else None,
        })
    else:
        base.update({"coverage_m2": coverage_m2})

    return base



def generate_dataset(max_steps=80, planning_steps=2, num_starts=10, max_stagnation=35, scene_numbers=None,
                     visualize_path=False, config_path: Optional[str] = None, experiment_tag: Optional[str] = None,
                     dataset_tag: Optional[str] = None):
    cfg_path = Path(config_path) if config_path else CONFIG_DIR / "config.json"
    config = load_normalized_config(cfg_path)

    derived_tag = derive_experiment_tag(cfg_path, config)
    experiment_tag = experiment_tag or derived_tag
    dataset_tag = dataset_tag or experiment_tag

    SUCCESS_RECALL = 0.96
    SOFT_RECALL = 0.90

    print(f"[INFO] Loaded config from: {cfg_path}")
    print(f"[INFO] Experiment tag: {experiment_tag}")
    print(f"[INFO] Dataset tag   : {dataset_tag}")
    print(f"[INFO] Using exploration: {config['exploration']['active']}")
    print(f"[INFO] Using exploration map: {config['exploration']['map_version']}")

    if scene_numbers is not None:
        scene_ids = sorted(scene_numbers)
        scene_iter = scene_ids if len(scene_ids) == 1 else tqdm(scene_ids, desc="Scenes")
    else:
        scene_ids = list(range(1, 31))
        scene_iter = tqdm(scene_ids, desc="Scenes")

    for scene_id in scene_iter:
        env = PrecomputedThorEnv(scene_number=scene_id, transition_tables_path=TRANSITION_TABLES, map_version=config['exploration']['map_version'])
        all_starts = get_unique_starts(env, k=num_starts * 4)  # generate more than needed
        env.close()

        successful = 0
        attempts = 0

        with tqdm(total=num_starts, desc=f"Startpositions Scene {scene_id}", leave=True) as outer_bar:
            while successful < num_starts and attempts < len(all_starts):
                start_pos, start_rot = all_starts[attempts]
                attempts += 1

                env = PrecomputedThorEnv(scene_number=scene_id, transition_tables_path=TRANSITION_TABLES, map_version=config['exploration']['map_version'])
                labeler_mode = "neural_slam" if config['exploration']['map_version'] == "neural_slam" else "viewpoint"
                labeler_kwargs = {}
                if labeler_mode == "neural_slam":
                    # Require perfect recall before allowing the labeler to emit STOP so that
                    # dataset episodes only terminate once all objects have been recovered.
                    labeler_kwargs.update(
                        success_recall_threshold=1.0,
                        soft_recall_threshold=1.0,
                    )

                labeler = ImitationLabeler(env, mode=labeler_mode, **labeler_kwargs)

                try:
                    obs = env.reset(scene_number=scene_id, start_position=start_pos, start_rotation=start_rot)
                    event = obs.info["event"]
                    real_start_pos = event.metadata["agent"]["position"]
                    real_start_rot = event.metadata["agent"]["rotation"]

                    rounded_start_pos = (round(start_pos[0], 2), round(start_pos[1], 2))
                    rounded_real_pos = (round(real_start_pos["x"], 2), round(real_start_pos["z"], 2))
                    rounded_start_rot = round(start_rot, 2)
                    rounded_real_rot = round(real_start_rot["y"], 2)

                    if not rounded_start_pos == rounded_real_pos or not rounded_start_rot == rounded_real_rot:
                        warnings.warn(
                            f"Start position and rotation do not match: {rounded_start_pos} vs {rounded_real_pos}, {rounded_start_rot} vs {rounded_real_rot}"
                        )

                    start_x = round(real_start_pos["x"], 2)
                    start_z = round(real_start_pos["z"], 2)
                    rot_y = round(real_start_rot["y"], 1)

                    data = []
                    steps = 0
                    last_action = -1
                    total_nodes = len(env.gt_graph.nodes)
                    stagnation_counter = 0

                    inner_bar = tqdm(total=total_nodes, desc=f"Scene {scene_id} Start {successful}", leave=False)

                    all_viewpoints = env.gt_graph.viewpoint_to_objects
                    filtered_viewpoints = {
                        vp: objs
                        for vp, objs in all_viewpoints.items()
                        if has_valid_path_precomputed(
                            env, start={"x": start_x, "z": start_z},
                            target=ImitationLabeler.deserialize_viewpoint(vp)[0]
                        )
                    }

                    viewpoints = compute_minimal_viewpoint_cover(filtered_viewpoints)
                    path = get_shortest_viewpoint_path(start_x, start_z, viewpoints)
                    viewpoints = {vp: viewpoints[vp] for vp in path}

                    # Note: visualize_path functionality would need to be adapted for PrecomputedThorEnv
                    # as it doesn't have the same visualization methods as ThorEnv

                    while not obs.terminated and steps < max_steps and stagnation_counter < max_stagnation:
                        inner_bar.n = len([k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8])
                        inner_bar.refresh()
                        best_actions = labeler.select_best_action(viewpoints, planning_steps)

                        for action in best_actions:
                            prev_node_count = len([k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8])

                            obs = env.step(action)
                            slam_training = env.exploration_map.get_training_data() if hasattr(env,
                                                                                               "exploration_map") else None
                            current_node_count = len([k for k, n in env.global_sg.nodes.items() if n.visibility >= 0.8])

                            update_viewpoints(env, viewpoints)

                            stagnation_counter = 0 if current_node_count > prev_node_count else (stagnation_counter + 1)

                            # normales Step-Sample
                            sample = build_sample(
                                env=env, obs=obs, step_idx=steps,
                                last_action_idx=last_action, start_pos=start_pos, start_rot=start_rot,
                                labeler=labeler, config=config
                            )
                            data.append(sample)

                            # Recall checken
                            current_recall = current_node_count / total_nodes if total_nodes > 0 else 0.0

                            # 1) Harte Schwelle → Stop-Step erzeugen & beenden
                            if current_recall >= SUCCESS_RECALL and not obs.terminated:
                                stop_obs = env.step(env.stop_index)
                                stop_sample = build_sample(
                                    env=env, obs=stop_obs, step_idx=steps + 1,
                                    last_action_idx=env.stop_index, start_pos=start_pos, start_rot=start_rot,
                                    labeler=labeler, config=config
                                )
                                data.append(stop_sample)
                                obs = stop_obs
                                steps += 2  # wir haben zwei Samples hinzugefügt (aktueller + Stop)
                                break

                            # 2) Weiche Schwelle + Stagnation → Stop-Step & beenden
                            if stagnation_counter >= max_stagnation and current_recall >= SOFT_RECALL and not obs.terminated:
                                stop_obs = env.step(env.stop_index)
                                stop_sample = build_sample(
                                    env=env, obs=stop_obs, step_idx=steps + 1,
                                    last_action_idx=env.stop_index, start_pos=start_pos, start_rot=start_rot,
                                    labeler=labeler, config=config
                                )
                                data.append(stop_sample)
                                obs = stop_obs
                                steps += 2
                                break

                            # 3) 100% Nodes -> Stop-Step & beenden
                            if current_node_count == total_nodes and not obs.terminated:
                                stop_obs = env.step(env.stop_index)
                                stop_sample = build_sample(
                                    env=env, obs=stop_obs, step_idx=steps + 1,
                                    last_action_idx=env.stop_index, start_pos=start_pos, start_rot=start_rot,
                                    labeler=labeler, config=config
                                )
                                data.append(stop_sample)
                                obs = stop_obs
                                steps += 2
                                break

                            # normalen Fortschritt verbuchen
                            last_action = action
                            steps += 1

                            if obs.terminated or steps >= max_steps or stagnation_counter >= max_stagnation:
                                if steps >= max_steps:
                                    print(f"Max steps reached: {steps}")
                                if stagnation_counter >= max_stagnation:
                                    print(f"Max stagnation reached: {stagnation_counter}")
                                break

                    if obs.terminated:
                        save_path = save_dataset(scene_id, data, start_x, start_z, rot_y, dataset_tag)
                        outer_bar.update(1)
                        tqdm.write(f"Saved to {save_path}")
                        successful += 1

                    inner_bar.close()
                    env.close()

                except ValueError as e:
                    print(f"Skipping start position {start_pos} / {start_rot} due to error: {e}")
                    env.close()
                    continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate imitation learning datasets for THOR scenes.")
    parser.add_argument(
        "--scenes",
        type=int,
        nargs="*",
        default=None,
        help="Scene numbers to generate datasets for (e.g., --scenes 1 2 3). Defaults to all scenes (1-30).",
    )
    parser.add_argument(
        "--visualize-path",
        action="store_true",
        help="Visualize planned paths during dataset generation.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to the config file to use (e.g. a generated Slurm config). Defaults to configs/config.json.",
    )
    parser.add_argument(
        "--experiment-tag",
        type=str,
        default=None,
        help="Override the experiment tag derived from the config path.",
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        default=None,
        help="Override the dataset folder name under data/il_dataset/.",
    )

    args = parser.parse_args()

    default_scenes = list(range(1, 31))
    scene_numbers = args.scenes if args.scenes else default_scenes

    generate_dataset(
        scene_numbers=scene_numbers,
        visualize_path=args.visualize_path,
        config_path=args.config_path,
        experiment_tag=args.experiment_tag,
        dataset_tag=args.dataset_tag,
    )
