import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.reinforcement.a2c_agent import A2CAgent


class DummyA2CAgent(A2CAgent):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.exploration_mode = "raster"
        self.is_neural_slam = False
        self.neural_slam_loss_weights = {}
        param = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.register_parameter("_dummy_param", param)
        self.optimizer = torch.optim.SGD([param], lr=0.1)

    def _get_update_values(self):
        actions = torch.tensor([[0, 1]], dtype=torch.long, device=self.device)
        returns = torch.tensor([[1.0, 0.5]], dtype=torch.float32, device=self.device)
        last_actions = torch.tensor([[0, 0]], dtype=torch.long, device=self.device)
        return {
            "actions": actions,
            "returns": returns,
            "last_actions": last_actions,
        }

    def forward_update(self, batch):
        base = self._dummy_param
        logits = torch.stack(
            [
                torch.stack([base, -base]),
                torch.stack([base, -base]),
            ]
        )
        values = torch.stack([base, base])
        return logits, values

    def reset(self):
        pass


def test_non_neural_update_matches_baseline_formula():
    agent = DummyA2CAgent()
    result = agent.update()

    expected = (
        result["policy_loss"]
        + agent.value_coef * result["value_loss"]
        - agent.entropy_coef * result["entropy"]
    )
    assert math.isclose(result["loss"], expected, rel_tol=1e-6)
    assert "mapper_loss" not in result
