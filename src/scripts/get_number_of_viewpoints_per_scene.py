import glob
import json
import os

import numpy as np


def get_number_of_viewpoints(gt_path="data/gt_graphs"):
    """Report viewpoint counts per scene and summary statistics."""
    json_files = glob.glob(os.path.join(gt_path, "FloorPlan*.json"))
    print(f"Found {len(json_files)} GT files.")
    all_viewpoints = []
    for path in json_files:
        with open(path, "r") as f:
            graph = json.load(f)
            viewpoints = graph.get("viewpoint_to_objects", [])
            all_viewpoints.append(len(viewpoints))
            graph_name = path.split("/")[-1].split(".")[0]
            print(f"Found {len(viewpoints)} viewpoints in {graph_name}")

    print(f"Mean viewpoints: {np.mean(all_viewpoints)}")


if __name__ == "__main__":
    base_path = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs")
    get_number_of_viewpoints(base_path)
