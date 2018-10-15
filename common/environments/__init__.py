

class Environment:

    def __init__(self, actions, dynamics):
        self.states = None
        self.actions = actions
        self.dynamics = dynamics

    def step(self, state, action):
        raise NotImplementedError(f"step has not been NotImplemented")
