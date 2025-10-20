import random


def get_unique_starts(env, k=5):
    reachable = env.controller.step("GetReachablePositions").metadata["actionReturn"]
    random.shuffle(reachable)
    unique = set()
    starts = []

    for pos in reachable:
        pos_key = (round(pos["x"], 2), round(pos["z"], 2))
        if pos_key in unique:
            continue

        rot = {"x": 0, "y": random.choice([0, 90, 180, 270]), "z": 0}
        success, real_pos, real_rot = is_valid_start(env.controller, pos, rot)
        if success:
            rounded_pos_key = (round(real_pos["x"], 2), round(real_pos["z"], 2))
            if rounded_pos_key not in unique:
                unique.add(rounded_pos_key)
                starts.append((real_pos, real_rot))

        if len(starts) >= k:
            break

    return starts


def is_valid_start(controller, position, rotation):
    controller.step(action="Teleport", position=position, rotation=rotation, forceAction=True)
    controller.step("Pass")

    directions = ["MoveAhead", "MoveLeft", "MoveRight", "MoveBack"]
    for action in directions:
        result = controller.step(action)
        controller.step("Pass")
        if result.metadata.get("lastActionSuccess", False):
            agent_pos = result.metadata["agent"]["position"]
            agent_rot = result.metadata["agent"]["rotation"]
            return True, agent_pos, agent_rot

    return False, None, None