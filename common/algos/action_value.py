import numpy as np
import typing
from typing import Tuple, Any, Dict

from common import logger
from common.environments.k_armed_bandit import kArmedBandit
from common import incremental_average, incremental_weighted_average


def action_value_sampling(bandit, k, steps, epsilon, alpha, initial_value):
    # type: (kArmedBandit, int, int, float, float, int) -> Tuple[np.ndarray, np.ndarray]
    # action values and rewards over time
    action_values = np.full(k, initial_value)
    num_selections = np.ones(k)
    rewards = np.zeros(steps)
    is_optimal = np.zeros(steps)

    for s in range(steps):
        # select an action (greedy with prob=epsilon)
        explore = epsilon > np.random.rand()
        action_idx = np.random.randint(k) if explore else np.argmax(action_values)
        num_selections[action_idx] += 1
        # execute action
        rewards[s] = bandit.step(action_idx)
        is_optimal[s] = int(bandit.get_optimal_action() == action_idx)
        # update the average
        if alpha < 1:
            action_values[action_idx] = incremental_weighted_average(
                rewards[s],
                action_values[action_idx],
                num_selections[action_idx],
                alpha
            )
        else:
            action_values[action_idx] = incremental_average(
                rewards[s],
                action_values[action_idx],
                num_selections[action_idx]
            )

    return rewards, is_optimal


def policy_evaluation(environment, policy, gamma=0.9, epsilon=1e-4):
    # type: (Any, Dict, float, float) -> np.ndarray
    iters = 0
    action_values = np.zeros(len(environment.states))
    while True:
        iters += 1
        new_values = np.zeros(action_values.shape)
        for s in environment.states:
            for a_idx, _ in enumerate(environment.actions):
                next_state, reward = environment.step(s, a_idx)
                # Bellman equation for value function
                new_values[s] += policy[s][a_idx] * (reward + gamma * action_values[next_state])

        # termination criteria: check for convergence
        diff = np.sum(np.abs(new_values - action_values))
        if diff < epsilon:
            logger.info(f"Policy evaluation converged. Diff={diff:.6f}")
            break

        # update previous computed action values
        action_values = new_values

    return action_values


def policy_improvement(environment):
    policy_stable = True
    for s in environment.states:
        for a_idx, _ in enumerate(environment.actions):
            raise NotImplemented(f"policy iteration is not NotImplemented yet")


def policy_iteration():
    raise NotImplemented(f"policy iteration is not NotImplemented yet")
