from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.components.enviroments.precomputed_thor_env import PrecomputedThorEnv


class TrackingMapping(dict):
    """Dictionary that records the last key requested via get."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_key = None

    def get(self, key, default=None):  # pragma: no cover - trivial wrapper
        self.last_key = key
        return super().get(key, default)


class DummyEvent:
    def __init__(self, key):
        self.metadata = {"key": key, "lastActionSuccess": True}


def _build_test_env():
    env = PrecomputedThorEnv.__new__(PrecomputedThorEnv)
    env.grid_size = 0.25
    env.mapping = TrackingMapping()
    env.current_pos = (0.0, 0.0)
    env.current_rot = 0
    env.last_event = None
    return env


def run_check() -> None:
    env = _build_test_env()
    base = (0.0, 0.0)
    headings = [0, 90, 180, 270]
    movements = {
        "MoveAhead": (0.0, env.grid_size),
        "MoveBack": (0.0, -env.grid_size),
        "MoveRight": (env.grid_size, 0.0),
        "MoveLeft": (-env.grid_size, 0.0),
    }

    for heading in headings:
        env.mapping[(round(base[0], 2), round(base[1], 2), heading)] = DummyEvent(
            (round(base[0], 2), round(base[1], 2), heading)
        )
        for move_vec in movements.values():
            dx, dz = env._rotate_move(*move_vec, heading)
            target_key = (round(base[0] + dx, 2), round(base[1] + dz, 2), heading)
            env.mapping[target_key] = DummyEvent(target_key)

    for heading in headings:
        for action, move_vec in movements.items():
            env.current_pos = base
            env.current_rot = heading

            env.mapping.last_key = None
            assert env.try_action(action), f"{action} should be valid from heading {heading}"
            predicted_key = env.mapping.last_key

            dx, dz = env._rotate_move(*move_vec, heading)
            expected_key = (round(base[0] + dx, 2), round(base[1] + dz, 2), heading)
            assert predicted_key == expected_key, "try_action predicted incorrect target"

            env.mapping.last_key = None
            env.transition_step(action)
            actual_key = env.mapping.last_key
            assert actual_key == expected_key, "transition_step fetched incorrect target"

            expected_pos = (base[0] + dx, base[1] + dz)
            assert math.isclose(env.current_pos[0], expected_pos[0])
            assert math.isclose(env.current_pos[1], expected_pos[1])
            assert env.current_rot == heading

            env.current_pos = base
            env.current_rot = heading

    print("PrecomputedThorEnv motion helpers check passed ✅")


if __name__ == "__main__":
    run_check()
