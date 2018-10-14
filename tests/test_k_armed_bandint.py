from common.problems.k_armed_bandit import kArmedBandit

import pytest
import numpy as np

@pytest.mark.parametrize('k', [10])
def test_k_arm_bandit_stationary(k):
    # init the k armed bandit
    bandit = kArmedBandit(k)

    # test normal distribution of a valid action
    action_idx = 0
    r = bandit.get_real_action_value(action_idx)     # real value of the first action
    rewards = [
        bandit.execute_action(action_idx)
        for _ in range(int(1e6))
    ]
    assert np.isclose(r, np.mean(rewards), rtol=1.e-2)


@pytest.mark.parametrize('k', [10])
def test_invalid_actions(k):
    bandit = kArmedBandit(k)
    # exception for invalid values
    with pytest.raises(ValueError):
        bandit.execute_action(k+1)
    with pytest.raises(ValueError):
        bandit.execute_action(-1)
