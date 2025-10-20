import json
import os


def collect_unique_object_types(gt_graphs_path):
    """Print a sorted list of unique object names from the ground-truth graphs."""
    unique_objects = set()
    total_files = 0

    for fname in os.listdir(gt_graphs_path):
        if fname.endswith(".json"):
            total_files += 1
            path = os.path.join(gt_graphs_path, fname)
            with open(path, "r") as f:
                data = json.load(f)

            for node in data.get("nodes", []):
                obj_type = node.get("name")
                if obj_type:
                    unique_objects.add(obj_type)

    print(f"\nParsed {total_files} files.")
    print(f"Found {len(unique_objects)} unique object types:\n")
    for obj in sorted(unique_objects):
        print(f"• {obj}")


if __name__ == "__main__":
    gt_graphs_path = os.path.join(os.path.dirname(__file__), "..", "data", "gt_graphs")
    collect_unique_object_types(gt_graphs_path)
