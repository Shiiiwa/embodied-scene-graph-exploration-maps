import copy
from typing import Dict, List


def aggregate_visibility(global_vis: float, local_vis: float, alpha: float = 0.8) -> float:
    return 1 - (1 - global_vis) * (1 - alpha * local_vis)

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
        best_gain = -1
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