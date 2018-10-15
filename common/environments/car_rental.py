import numpy as np
from typing import Dict, List, Tuple

from common import logger
from common.environments import Environment


class CarRental(Environment):

    """
        MDP for the car rental problem.
        Example 4.2 of the book
    """

    def __init__(self, dynamics, **kwargs):
        # type: (Dict, int, int) -> None

        """
            Requires the following additional parameters apart from the Environment ones:
            max_cX: maximum cars at station X
            sX_req_lambda: lambda parameter of a Poisson distribution for car requests at station X
            sX_ret_lambda: lambda parameter of a Poisson distribution for car returns at station X

            with X in [1, 2]
        """
        super().__init__(self._generate_states(kwargs['max_c1'], kwargs['max_c2']), dynamics)

        # There are two stations with different request and return possison probabilities
        self.max_c1 = kwargs['max_c1']
        self.max_c2 = kwargs['max_c2']
        self.s1_req_lambda = kwargs['s1_req_lambda']
        self.s1_ret_lambda = kwargs['s1_ret_lambda']
        self.s2_req_lambda = kwargs['s2_req_lambda']
        self.s1_ret_lambda = kwargs['s1_ret_lambda']

    def _generate_states(self, max_c1: int, max_c2: int) -> List:
        return [
            (c1, c2)
            for c1 in range(max_c1)
            for c2 in range(max_c2)
        ]

    def get_valid_actions(self, state_idx: int) -> List:
        cars_s1, cars_s2 = self.states[state_idx]

        return [
            (c1, c2)
            for c1 in range(cars_s1)
            for c2 in range(cars_s2)
        ]

    def step(self, state_idx: int, action_idx: int) -> Tuple[int, float]:
        """ Perform one step in the MDP """
        actions = self.get_valid_actions(state_idx)
        if (action_idx < 0 or action_idx > len(actions)):
            raise ValueError(f"Invalid index action '{action}'")

        # next state is the result of moving the cars between stations
        # and the requests / returns
        s1_returns = np.random.poisson(self.s1_req_lambda)
        s1_requests = np.random.poisson(self.s1_ret_lambda)
        s2s1_requests = np.random.poisson(self.s2_req_lambda)
        s2_returns = np.random.poisson(self.s1_ret_lambda)

        # TODO: review restrictions as this migh yield invalid configurations!
        logger.debug(action_idx)
        logger.debug(actions)
        to_s2, to_s1 = actions[action_idx]
        s1 = min(max(self.max_s1, s1_returns - s1_requests - to_s2 + to_s1), 0)
        s2 = min(max(self.max_s2, s2_returns - s2_requests - to_s1 + to_s2), 0)
        next_state = self.states.index((s1, s2))

        # the reward is the earnings per car (10) - the cost of moving cars (2)
        reward = -2 * (to_s2 + to_s1) + 10 * (s1_requests + s2_requests)
