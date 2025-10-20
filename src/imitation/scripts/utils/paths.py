from math import hypot
from multiprocessing import get_context

from src.imitation.utils.imitation_labeler import ImitationLabeler
from src.components.utils.aco_tsp import SolveTSPUsingACO

import time as tm

def _greedy_tour(vp_keys, vp_coords):
    unvisited = set(vp_keys)
    ordered = []
    curr = vp_keys[0]
    cx, cz = vp_coords[curr]
    ordered.append(curr)
    unvisited.remove(curr)
    while unvisited:
        nxt = min(unvisited, key=lambda v: hypot(vp_coords[v][0] - cx, vp_coords[v][1] - cz))
        ordered.append(nxt)
        cx, cz = vp_coords[nxt]
        unvisited.remove(nxt)
    return ordered


def _two_opt(order, coords, max_iter=200):
    if len(order) < 4:
        return order
    def dist(a, b):
        ax, az = coords[a]; bx, bz = coords[b]
        return hypot(ax - bx, az - bz)
    n = len(order)
    improved, it = True, 0
    while improved and it < max_iter:
        improved, it = False, it + 1
        for i in range(1, n - 2):
            for k in range(i + 1, n - 1):
                a1, a2 = order[i - 1], order[i]
                b1, b2 = order[k], order[k + 1]
                if dist(a1, a2) + dist(b1, b2) > dist(a1, b1) + dist(a2, b2) + 1e-12:
                    order[i:k + 1] = reversed(order[i:k + 1])
                    improved = True
    return order

def _aco_worker(points, labels, colony_size, steps, mode, out_q):
    try:
        tsp = SolveTSPUsingACO(
            mode=mode,
            colony_size=colony_size,
            steps=steps,
            nodes=points,
            labels=labels,
        )
        _, _ = tsp.run()
        if getattr(tsp, "global_best_tour", None):
            out_q.put(("ok", [labels[i] for i in tsp.global_best_tour]))
        else:
            out_q.put(("none", None))
    except Exception as e:
        out_q.put(("err", str(e)))

def get_shortest_viewpoint_path(
    start_x,
    start_z,
    viewpoints,
    *,
    scene_id=None,
    use_aco=True,
    aco_timeout_s=3.0,
    aco_nodes_threshold=75,
    aco_blacklist_scenes=None,
    aco_mode="MaxMin"
):
    """
    Bestimme eine kurze Tour über Viewpoints. Falls ACO aktiv ist, läuft es in
    einem Subprozess mit hartem Timeout. Fallback ist Greedy + 2-Opt.
    Der Agent-Start (start_x, start_z) ist NICHT Teil des TSP; wir rotieren
    die Tour anschließend zum nächstgelegenen Viewpoint.
    """
    vp_keys = list(viewpoints.keys())
    if len(vp_keys) <= 1:
        return vp_keys[:]

    vp_positions = {vp: ImitationLabeler.deserialize_viewpoint(vp)[0] for vp in vp_keys}
    vp_coords = {vp: (pos["x"], pos["z"]) for vp, pos in vp_positions.items()}

    aco_blacklist_scenes = set(aco_blacklist_scenes or [])
    n = len(vp_keys)
    skip_aco = (not use_aco) or (n > aco_nodes_threshold) or (scene_id in aco_blacklist_scenes)

    if skip_aco:
        ordered = _two_opt(_greedy_tour(vp_keys, vp_coords), vp_coords, max_iter=200)
    else:
        # dynamische Parameter (konservativ)
        steps = min(600, max(120, 15 * n))
        colony = max(10, min(40, (n + 1) // 2))

        ctx = get_context("spawn")  # robust auf macOS
        q = ctx.Queue()
        p = ctx.Process(target=_aco_worker, args=(
            [vp_coords[vp] for vp in vp_keys],
            vp_keys, colony, steps, aco_mode, q
        ))
        p.start()
        p.join(timeout=aco_timeout_s)

        if p.is_alive():
            p.terminate()
            p.join()
            ordered = _two_opt(_greedy_tour(vp_keys, vp_coords), vp_coords, max_iter=200)
        else:
            try:
                status, payload = q.get_nowait()
            except Exception:
                status, payload = ("none", None)
            if status == "ok" and payload:
                ordered = payload
            else:
                ordered = _two_opt(_greedy_tour(vp_keys, vp_coords), vp_coords, max_iter=200)


    dists = [hypot(vp_coords[v][0] - start_x, vp_coords[v][1] - start_z) for v in ordered]
    k = dists.index(min(dists))
    return ordered[k:] + ordered[:k]


def has_valid_path(controller, start, target):
    """
    Checks whether a path exists from `start` to `target` by calling AI2-THOR's internal path planner.
    Returns True only if the path is valid and both endpoints are close enough to the NavMesh.
    """

    if "y" not in start:
        start = {**start, "y": 0.900999}
    if "y" not in target:
        target = {**target, "y": 0.900999}

    try:
        event = controller.step(action="GetShortestPathToPoint", position=start, target=target, raise_for_failure=True)
        return True
    except Exception as e:
        return False