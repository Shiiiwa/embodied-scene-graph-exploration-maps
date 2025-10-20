from dataclasses import dataclass

@dataclass
class LengthObjectiveConfig:
    enabled = True
    target = 25         # steps for no penalty
    soft_cap = 30       # secound boundary
    min_factor = 0.9  # factor for soft_cap
    shaping_coef = 0.0
    success_only = False  # boolean for decision, if only penalty if episode was successful
    step_penalty = None

class LengthObjective:
    def __init__(self, cfg: LengthObjectiveConfig = LengthObjectiveConfig()):
        self.cfg = cfg

    @property
    def enabled(self):
        return bool(self.cfg.enabled)

    def factor(self, steps):
        if not self.enabled:
            return 1.0
        t, s, mf = self.cfg.target, self.cfg.soft_cap, self.cfg.min_factor
        if steps <= t:
            return 1.0
        if steps <= s:
            span = max(1, s - t)
            drop = (steps - t) / span  # 0..1
            return 1.0 - (1.0 - mf) * drop
        return 0.0

    def adjusted_score(self, score, steps, success):
        if not self.enabled:
            return score
        if self.cfg.success_only and success is False:
            return score
        return float(score) * self.factor(steps)

    def terminal_penalty(self, steps, success):
        if not self.enabled or self.cfg.shaping_coef <= 0.0:
            return 0.0
        if self.cfg.success_only and success is False:
            return 0.0

        fac = self.factor(steps)
        if steps <= self.cfg.soft_cap:
            return float(self.cfg.shaping_coef * (1.0 - fac))
        return float(self.cfg.shaping_coef)

    def per_step_penalty(self):
        if self.enabled and self.cfg.step_penalty is not None and self.cfg.step_penalty > 0:
            return -float(self.cfg.step_penalty)
        return 0.0
