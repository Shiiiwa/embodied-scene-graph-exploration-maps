import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.components.agents.abstract_agent import AbstractAgent
from src.components.enviroments.precomputed_thor_env import PrecomputedThorEnv
from src.components.utils.observation import Observation
from src.imitation.models.neural_slam_il import NeuralSlamController, NeuralSlamMemoryCore


def _dummy_event():
    class DummyEvent:
        def __init__(self):
            self.frame = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
            self.metadata = {
                "agent": {
                    "position": {"x": 0.0, "z": 0.0},
                    "rotation": {"y": 0.0},
                }
            }

    return DummyEvent()


def _make_stub_env():
    env = PrecomputedThorEnv.__new__(PrecomputedThorEnv)
    env.pending_action = None
    env.pending_mapper_logits = None
    env.pending_head_params = None
    env.pending_exploration_mask = None
    env.pending_pose_delta = None
    env.pending_prev_observation = None
    return env


def _make_stub_agent(env, exploration_mode="neural"):
    agent = AbstractAgent.__new__(AbstractAgent)
    nn.Module.__init__(agent)
    agent.env = env
    agent.device = torch.device("cpu")
    agent.exploration_mode = exploration_mode
    agent.is_neural_slam = exploration_mode == "neural"

    if exploration_mode == "neural":
        networks = nn.ModuleDict({
            "controller": NeuralSlamController(map_channels=2, map_resolution=64, hidden_size=256, read_dim=2),
            "memory_core": NeuralSlamMemoryCore(map_channels=2, map_resolution=64, read_dim=2),
        })
        networks.to(agent.device)
        agent.neural_slam_networks = networks
    else:
        agent.neural_slam_networks = None

    agent._neural_state = {}
    agent._reset_neural_map_state()
    agent._neural_mapper_needs_reset = True
    return agent


def test_prepare_neural_map_populates_pending_head_params():
    env = _make_stub_env()
    agent = _make_stub_agent(env, exploration_mode="neural")

    rgb = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    obs = Observation(state=(rgb, None, None, None), info={"event": _dummy_event()})

    result = agent.prepare_neural_map_update(obs, env=env)

    assert result is not None
    assert env.pending_mapper_logits is not None
    assert env.pending_head_params is not None
    assert env.pending_prev_observation is not None
    assert env.pending_prev_observation.get("pose") is not None
    if env.pending_prev_observation.get("rgb") is not None:
        assert env.pending_prev_observation["rgb"].shape[:2] == (64, 64)
    assert env.pending_mapper_logits.shape[0] == 1
    assert "key" in env.pending_head_params
    assert env.pending_head_params["key"].shape[0] == 1


def test_prepare_neural_map_skips_for_non_neural_mode():
    env = _make_stub_env()
    agent = _make_stub_agent(env, exploration_mode="metric_semantic_v1")

    rgb = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    obs = Observation(state=(rgb, None, None, None), info={"event": _dummy_event()})

    agent.prepare_neural_map_update(obs, env=env)

    assert env.pending_mapper_logits is None
    assert env.pending_head_params is None
