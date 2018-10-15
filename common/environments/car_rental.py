import numpy as np
from typing import Dict, List, Tuple

from common import logger
from common.environments import Environment


class CarRental(Environment):

    """
        MDP for the car rental problem.
        Example 4.2 of the book
    """

    def __init__(self, dynamics, max_c1=20, max_c2=20):
        # type: (Dict, int, int) -> None
        states = self._generate_states(max_c1, max_c2)
        super().__init__(states, dynamics)

    def _generate_states(self, max_c1: int, max_c2: int):
        return [
            (c1, c2)
            for c1 in range(max_c1)
            for c2 in range(max_c2)
        ]

    def get_valid_actions(self, state: int) -> List:
        cars_s1, cars_s2 = self.states[state]
        return [
            (c1, c2)
            for c1 in range(cars_s1)
            for c2 in range(cars_s2)
        ]

    def step(self, state: int, action: int) -> Tuple[int, float]:
        pass
