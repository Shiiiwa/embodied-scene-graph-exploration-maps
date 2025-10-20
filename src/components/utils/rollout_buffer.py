# utils/rollout_buffer.py

from typing import Any, Optional, Tuple

import torch


NeuralTuple = Tuple[Any, Any, Any, Any, Any]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RolloutBuffer:

    def __init__(self, n_steps: int):
        self.n_steps = n_steps
        self.state_rgb = []
        self.state_lssg = []
        self.state_gssg = []
        self.state_occ = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.last_actions = []
        self.agent_positions = []

        self.map_indices = []
        self.neural_slam_training: list[Optional[NeuralTuple]] = []

        # Hidden states at the beginning of the rollout
        self.initial_lssg_hidden = None
        self.initial_gssg_hidden = None
        self.initial_policy_hidden = None

        self.is_first_add = True

    # -------- lifecycle --------
    def clear(self):
        # Per-step data
        self.state_rgb = []
        self.state_lssg = []
        self.state_gssg = []
        self.state_occ = []       # <— Map/Occupancy steckt hier drin (das reicht)
        self.actions = []
        self.rewards = []
        self.dones = []
        self.last_actions = []

        # Optional (nur falls wirklich genutzt — sonst auskommentieren)
        self.agent_positions = []
        self.map_indices = []
        self.neural_slam_training = []

        # Hidden states (RNN) am Anfang des Rollouts
        self.initial_lssg_hidden = None
        self.initial_gssg_hidden = None
        self.initial_policy_hidden = None

        self.is_first_add = True

    # -------- add transitions --------
    def add(self, state, action, reward, done, hiddens, last_action,
            agent_position=None, map_index=None, neural_slam=None):
        """
        state: Tuple(rgb, lssg, gssg, occ) — occ enthält deine Map-Features.
        hiddens: (lssg_h, gssg_h, policy_h)
        """
        if self.is_first_add:
            self.initial_lssg_hidden, self.initial_gssg_hidden, self.initial_policy_hidden = hiddens
            self.is_first_add = False

        s_rgb, s_lssg, s_gssg, s_occ = state
        self.state_rgb.append(s_rgb)
        self.state_lssg.append(s_lssg)
        self.state_gssg.append(s_gssg)
        self.state_occ.append(s_occ)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.last_actions.append(last_action)

        # Optional mitschleppen (nur für Auswertung/Debug; Training ignoriert’s)
        self.agent_positions.append(agent_position)
        self.map_indices.append(map_index)
        self.neural_slam_training.append(self._coerce_neural_sample(neural_slam))

    def add_batch(self, states, actions, rewards, dones, hiddens, last_actions, agent_pos, map_indices=None,
                  neural_slam_batch=None):
        """
        Batch-Variante, funktionsgleich zum Referenz-Repo, nur mit occ im State.
        """
        if self.is_first_add and hiddens:
            self.initial_lssg_hidden = hiddens[0][0]
            self.initial_gssg_hidden = hiddens[0][1]
            self.initial_policy_hidden = hiddens[0][2]
            self.is_first_add = False

        if states:
            states_rgb, states_lssg, states_gssg, states_occ = zip(*states)
            self.state_rgb.extend(states_rgb)
            self.state_lssg.extend(states_lssg)
            self.state_gssg.extend(states_gssg)
            self.state_occ.extend(states_occ)

        self.actions.extend(actions)
        self.rewards.extend([float(r) for r in rewards])
        self.dones.extend([bool(d) for d in dones])
        self.last_actions.extend(last_actions)
        self.agent_positions.extend(agent_pos)
        if map_indices is not None:
            self.map_indices.extend(map_indices)
        else:
            self.map_indices.extend([None] * len(actions))
        if neural_slam_batch is not None:
            self.neural_slam_training.extend(
                [self._coerce_neural_sample(sample) for sample in neural_slam_batch]
            )
        else:
            self.neural_slam_training.extend([None] * len(actions))

    # -------- helpers --------
    def is_ready(self):
        return len(self.rewards) >= self.n_steps

    def _compute_returns_mc(self, gamma: float):
        """
        Monte-Carlo Returns wie im Referenz-Repo:
        G_t = r_t + gamma * G_{t+1}; bei done: G = 0.
        """
        returns = []
        G = 0.0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                G = 0.0
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    # -------- export batch --------
    def get(self, gamma: float):
        returns = self._compute_returns_mc(gamma)

        # Konsistenzprüfungen (früh failen statt still falsche Tensors zu bauen)
        n = len(self.actions)
        assert all(len(lst) == n for lst in [
            self.state_rgb, self.state_lssg, self.state_gssg, self.state_occ,
            self.rewards, self.dones, self.last_actions
        ]), "RolloutBuffer: inkonsistente Längen."

        batch = {
            "rgb": self.state_rgb,
            "lssg": self.state_lssg,
            "gssg": self.state_gssg,
            "occ": self.state_occ,  # <— Map fließt hier hinein
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones, dtype=torch.bool, device=device),
            "last_actions": torch.tensor(self.last_actions, dtype=torch.long, device=device),
            "initial_lssg_hidden": self.initial_lssg_hidden,
            "initial_gssg_hidden": self.initial_gssg_hidden,
            "initial_policy_hidden": self.initial_policy_hidden,
            # Optional nur für Analyse:
            "agent_positions": self.agent_positions,
            "map_index": self.map_indices,
            "neural_slam": self.neural_slam_training,
        }
        return batch

    @staticmethod
    def _coerce_neural_sample(sample: Any) -> Optional[NeuralTuple]:
        if sample is None:
            return None

        if isinstance(sample, tuple) and len(sample) == 5:
            return sample  # type: ignore[return-value]
        if isinstance(sample, list) and len(sample) == 5:
            return tuple(sample)  # type: ignore[return-value]

        if isinstance(sample, dict):
            keys = ("rgb_prev", "rgb_curr", "pose_delta", "fp_proj_gt", "fp_explored_gt")
            if all(k in sample for k in keys):
                return tuple(sample[k] for k in keys)  # type: ignore[return-value]
            inner = sample.get("neural_slam_training")
            if isinstance(inner, (list, tuple)) and len(inner) == 5:
                return tuple(inner)  # type: ignore[return-value]

        return None
