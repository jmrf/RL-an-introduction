import numpy as np
from typing import Dict, List, Tuple


class GridWorld:

    def __init__(self, size: int, actions: List[Tuple[int, int]], dynamics: Dict):
        """ size: defines the grid dimensions as in size x world_size
            actions: list of actions as in possible movement. E.g.: (x, y) = [0, 1] -> move 1 position to the right
            dynamics: Dictionary specifying rewards and transition probabilities for each state - action pair
        """
        self.size = size
        self.states = range(size * size)    # states as a flat list [0..size^2]
        self.actions = actions
        self.dynamics = dynamics
        # init state values
        self.current_state = self.states[0]

    @staticmethod
    def state_2_coordinates(state: int, size: int) -> Tuple[int, int]:
        return np.array([state // size, state % size])

    @staticmethod
    def coordinates_2_state(coords: Tuple[int, int], size: int) -> int:
        return coords[0] * size + coords[1]

    @staticmethod
    def get_max_or_break_tie(transition_probs: Dict[int, float]) -> int:
        """ Get the key with max value. In case of a tie breaks randomly """
        probs = list(transition_probs.values())
        # solve probability ties
        max_p = np.max(probs)
        tie_states = [s for s, p in transition_probs.items() if p == max_p]
        # select accordingly
        if (len(tie_states) > 1):
            # select randomly between the ties
            transition = np.random.choice(tie_states)
        else:
            return tie_states[0]

    def get_next_state(self, action: int) -> int:
        """ Get next state given an action and the internal grid world current state """
        # state-transition probabilities when taking action 'a' in state 's'
        next_state = self.get_max_or_break_tie(self.dynamics[self.current_state][action]['transition_probs'])
        return next_state

    def step(self, action: int) -> Tuple[int, float]:
        """ Perform one step move in the grid world """
        if (action < 0 or action > len(self.actions)):
            raise ValueError(f"Action '{action}' is an invalid action. Valid actions are: {self.actions}")

        next_state = self.get_next_state(action)  # move to next state
        reward = self.dynamics[self.current_state][action]['rewards'][next_state]  # obtain reward for the step
        self.current_state = next_state  # update current state
        return next_state, reward
