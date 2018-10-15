import numpy as np
from typing import Dict, List, Tuple

from common import logger
from common.environments import Environment


class GridWorld(Environment):

    """
        MDP for grid world.
        Example 3.8 of the book
    """

    def __init__(self, size: int, actions: List[Tuple[int, int]], dynamics: Dict) -> None:
        """
        Parameters
        ----------
        size : int
            defines the grid dimensions as in size x world_size
        actions : List[Tuple[int, int]]
            list of actions as in possible movement. E.g.: (x, y) = [0, 1] -> move 1 position to the right

        """
        states = range(size * size)
        super().__init__(states, dynamics)  # states as a flat list [0..size^2]
        self.actions = actions
        self.size = size

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
            logger.warning("Selectin state randomly")
            return np.random.choice(tie_states)
        else:
            return tie_states[0]

    def get_next_state(self, state, action: int) -> int:
        """ Get next state given an action and the internal grid world current state """
        # state-transition probabilities when taking action 'a' in state 's'
        return self.get_max_or_break_tie(self.dynamics[state][action]['transition_probs'])

    def step(self, state: int, action: int) -> Tuple[int, float]:
        """ Perform one step move in the grid world """
        if (action < 0 or action > len(self.actions)):
            raise ValueError(f"Action '{action}' is an invalid action. Valid actions are: {self.actions}")

        next_state = self.get_next_state(state, action)  # move to next state
        reward = self.dynamics[state][action]['rewards'][next_state]  # obtain reward for the step
        return next_state, reward

    def get_valid_actions(self, state):
        return self.actions
