import argparse
import os
import pickle
import platform

from tqdm import tqdm

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from src.components.utils.paths import TRANSITION_TABLES


def _normalize_floorplans(floorplans: list) -> list:
    normalized = []
    for scene in floorplans:
        if isinstance(scene, int):
            normalized.append(f"FloorPlan{scene}")
        elif isinstance(scene, str):
            scene = scene.strip()
            if scene.startswith("FloorPlan"):
                normalized.append(scene)
            elif scene.isdigit():
                normalized.append(f"FloorPlan{scene}")
            else:
                raise ValueError(f"Unsupported scene identifier: {scene}")
        else:
            raise ValueError(f"Unsupported scene identifier type: {type(scene)}")
    return normalized


def generate_transition_tables(floorplans: list = None):
    """
    Precompute and save transition tables for given AI2-THOR floorplans.
    Each table maps (x, z, rotation) -> event (or None if unreachable).
    """
    # Ensure output directory exists
    os.makedirs(TRANSITION_TABLES, exist_ok=True)

    if floorplans is None:
        floorplans = [f"FloorPlan{i}" for i in list(range(1, 31))]
    else:
        floorplans = _normalize_floorplans(floorplans)

    # Initialize AI2-THOR controller in cloud rendering mode
    if platform.system() == "Linux":
        controller = Controller(platform=CloudRendering, visibilityDistance=50.0)
    else:
        controller = Controller(visibilityDistance=50.0)
    rotations = [0, 90, 180, 270]

    for scene in floorplans:
        print(f"\nGenerating transition table for: {scene}")
        controller.reset(scene=scene)

        # Query reachable positions
        reachable = controller.step("GetReachablePositions").metadata["actionReturn"]
        total_tasks = len(reachable) * len(rotations)

        mapping = {}
        # Iterate over all reachable positions and orientations
        with tqdm(total=total_tasks, desc=f"Exploring {scene}") as pbar:
            for pos in reachable:
                x = round(pos["x"], 2)
                z = round(pos["z"], 2)
                for rot in rotations:
                    # Teleport agent to exact position and rotation
                    controller.step(
                        action="Teleport",
                        position={"x": pos["x"], "y": pos.get("y", 0), "z": pos["z"]},
                        rotation={"x": 0, "y": rot, "z": 0},
                        horizon=0,
                        standing=True,
                        forceAction=True,
                    )
                    # Perform a pass action to get the event
                    event = controller.step("Pass")
                    # Store event in mapping (key: world coords and rotation)
                    mapping[(x, z, rot)] = event
                    pbar.update(1)

        # Save mapping (and grid_size) to disk
        output_path = os.path.join(TRANSITION_TABLES, f"{scene}.pkl")
        data = {"table": mapping, "grid_size": 0.25}
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Saved transition table: {output_path}")

    controller.stop()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate transition tables for THOR scenes.")
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Scene numbers or names to generate tables for (e.g., --scenes 1 2 3 or FloorPlan1 FloorPlan2). Defaults to a preset list.",
    )

    args = parser.parse_args()

    generate_transition_tables(floorplans=args.scenes)
