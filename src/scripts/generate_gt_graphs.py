import argparse
import os

from ai2thor.controller import Controller
from tqdm import tqdm

from src.components.graph.gt_graph import GTGraph
from src.components.graph.local_graph_builder import LocalSceneGraphBuilder
from src.components.utils.paths import GT_GRAPHS_DIR


def _normalize_floorplans(floorplans: list) -> list:
    normalized = []
    for scene in floorplans:
        if isinstance(scene, int):
            normalized.append(f"FloorPlan{scene}")
        elif isinstance(scene, str):
            normalized.append(scene if scene.startswith("FloorPlan") else f"FloorPlan{scene}")
        else:
            raise ValueError(f"Unsupported scene identifier type: {type(scene)}")
    return normalized


def generate_gt_scene_graphs(floorplans: list = None):
    save_dir = GT_GRAPHS_DIR
    os.makedirs(save_dir, exist_ok=True)

    if floorplans is None:
        floorplans = [f"FloorPlan{i}" for i in range(1, 31)]
    else:
        floorplans = _normalize_floorplans(floorplans)

    controller = Controller(visibilityDistance=2.0)
    builder = LocalSceneGraphBuilder()

    rotations = [0, 90, 180, 270]

    for scene in floorplans:
        print(f"\nCreating GT scene graph for: {scene}")
        controller.reset(scene=scene)

        # Get reachable positions
        reachable_positions = controller.step("GetReachablePositions").metadata["actionReturn"]

        total_steps = len(reachable_positions) * len(rotations)

        with tqdm(total=total_steps, desc=f"Exploring {scene}") as pbar:
            gt_graph = GTGraph()
            for pos in reachable_positions:
                for rot in rotations:
                    controller.step(
                        action="Teleport", position=pos, rotation={"x": 0, "y": rot, "z": 0}, horizon=0, standing=True, forceAction=True
                    )

                    vp = {"position": {"x": round(pos["x"], 2), "z": round(pos["z"], 2)}, "rotation": rot}
                    viewpoint = f"{vp['position']}_{vp['rotation']}"

                    event = controller.step("Pass")
                    local_sg = builder.build_from_metadata(event.metadata)
                    gt_graph.add_local_sg(local_sg)
                    gt_graph.add_viewpoint(viewpoint, local_sg.nodes)

                    pbar.update(1)

        # Save graph to file
        output_path = os.path.join(save_dir, f"{scene}.json")
        gt_graph.save_to_file(output_path)

        print(f"✅ Saved: {output_path}")

    controller.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GT scene graphs for THOR scenes.")
    parser.add_argument(
        "--scenes",
        type=int,
        nargs="*",
        default=None,
        help="Scene numbers to generate GT graphs for (e.g., --scenes 1 2 3). Defaults to all scenes (1-30).",
    )

    args = parser.parse_args()

    default_scenes = list(range(1, 31))
    scene_numbers = args.scenes if args.scenes else default_scenes
    floorplans = _normalize_floorplans(scene_numbers)

    generate_gt_scene_graphs(floorplans=floorplans)
