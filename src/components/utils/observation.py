class Observation:
    def __init__(self, state, reward=None, terminated=None, truncated=None, info=None):
        self.state = state
        self.info = info if info is not None else {}
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
