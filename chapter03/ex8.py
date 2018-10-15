import numpy as np
import typing
import collections
from matplotlib import pyplot as plt

from common import logger
from common.environments.grid_world import GridWorld


"""
    Chapter 3 has no programming exercises, so this is not an exercise as such.
    This is just as a review of the classic grid world problem and how to solve the bellman optimality equation
    as given in the Example 3.8 of chapter 3
"""


def init_world(world_size=5):
    """ Grid world dynamics as per Example 3.8 """

    # up, right, down, left
    actions = [
        np.array([-1, 0]),
        np.array([0, 1]),
        np.array([1, 0]),
        np.array([0, -1])]

    # build this specific world grid dynamics
    dynamics = {}
    for s in range(world_size**2):
        dynamics[s] = {}

        # layout the dynamics structure
        for a in range(len(actions)):
            dynamics[s][a] = {
                'transition_probs': {},
                'rewards': {}
            }
        # populate
        for a in range(len(actions)):
            # at a given state a given action takes us where we expect
            s_prime = GridWorld.coordinates_2_state(
                actions[a] + GridWorld.state_2_coordinates(s, world_size),
                world_size
            )
            reward = 0
            if s == 1:  # special case from state A -> A'
                reward = 10     # state A: (0, 1)
                s_prime = GridWorld.coordinates_2_state([4, 1], world_size)      # state A': (4, 1)
            elif s == 3:  # special case from state B -> B'
                reward = 5      # state B: (0, 3)
                s_prime = GridWorld.coordinates_2_state([2, 3], world_size)      # state B': (2, 3)
            elif s_prime < 0 or s_prime > world_size**2 - 1:
                reward = -1     # going out of the grid
                s_prime = s     # stay in the same grid cell

            # logger.debug(f"s:{s} + a:{a}({actions[a]}) --> s'={s_prime} | r={reward}")

            # always go where the action should take us
            dynamics[s][a]['transition_probs'][s_prime] = 1  # deterministic environment
            dynamics[s][a]['rewards'][s_prime] = reward

        # logger.debug("-" * 10)

    # init the world
    return GridWorld(world_size, actions, dynamics)


def compute_value_function(world, gamma=0.9, epsilon=1e-3):
    iters = 0
    p_action = 1 / len(world.actions)     # quiprobable random policy
    action_values = np.zeros(len(world.states))
    while True:
        iters += 1
        new_values = np.zeros(action_values.shape)
        for s in world.states:
            for a_idx, _ in enumerate(world.actions):
                next_state, reward = world.step(s, a_idx)
                # logger.info(f"state={s} + action={a_idx} ==> state'={next_state} | reward={reward}")
                # Bellman equation for value function
                new_values[s] += p_action * (reward + gamma * action_values[next_state])

        # termination criteria: check for converge
        diff = np.sum(np.abs(new_values - action_values))
        if diff < epsilon:
            logger.info(f"Stoping value function iterations due to convergence. diff={diff:.6f}")
            break

        # update previous computed action values
        action_values = new_values

    np.set_printoptions(precision=3, suppress=True)
    print(f"Computed Action values (#iterations = {iters}):\n{action_values.reshape(world.size, world.size)}")


if __name__ == "__main__":

    world = init_world()
    compute_value_function(world)
