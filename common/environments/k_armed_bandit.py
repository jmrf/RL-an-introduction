import random
import typing
import numpy as np
from common import logger


class kArmedBandit:

    def __init__(self, k, mu=0, var=1, stationary=True):
        # number of arms
        self.k = k
        self.mu = mu
        self.var = var
        # stationary flag
        self.stationary = stationary
        # initialize each arm real action value as a normal dist. centered at 0 with variance 1
        self.real_values = [
            np.random.normal(self.mu, self.var)
            for _ in range(self.k)
        ]
        a = self.get_optimal_action()
        logger.info(f"Best action = '{a}' - q* = {self.get_real_action_value(a)}")

    def get_real_action_value(self, a: int) -> float:
        if (a < 0 or a > self.k - 1):
            raise ValueError(f"Invalid action '{a}'. The action index must be between 0 and {self.k}")

        if (not self.stationary):
            # make the distribution change over time
            self.real_values[a] = np.random.normal(self.mu, self.var)

        return self.real_values[a]

    def get_optimal_action(self):
        return np.argmax(self.real_values)

    def step(self, a: int) -> float:
        # returns a reward value centered at q*(a) with variance 1
        return np.random.normal(self.get_real_action_value(a), self.var)
