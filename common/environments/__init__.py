

class Environment:

    def __init__(self, states, dynamics):
        """ Generic init method for an environment.
            Requires a list of states and the dynamics of the MDP, everything else depends on the specific case

        Parameters
        ----------
        states : type
            List of states (a state can be an object, integer, etc)
        dynamics : type
            Dictionary specifying 'rewards' and 'transition_probs' for each state - action pair

        """
        self.states = states
        self.dynamics = dynamics

    def step(self, state, action):
        raise NotImplementedError(f"step has not been implemented")

    def get_valid_actions(self, state):
        raise NotImplementedError(f"'get_valid_actions' has not been implemented")
