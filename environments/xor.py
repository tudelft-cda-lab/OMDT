import numpy as np

from omdt.mdp import MarkovDecisionProcess

from itertools import product

from tqdm import tqdm


def generate_mdp(n_states=200, seed=0):
    random_state = np.random.RandomState(seed=seed)
    observations = random_state.rand(n_states, 2)

    feature_names = ["X", "Y"]
    action_names = ["not_xor", "xor"]

    n_states = len(observations)
    n_actions = len(action_names)

    R = np.zeros((n_states, n_states, n_actions))
    T = np.zeros((n_states, n_states, n_actions))

    T[:, :, :] = 1 / len(observations)

    threshold = 0.5

    for s, observation in tqdm(enumerate(observations), total=len(observations)):
        if observation[0] >= threshold:
            if observation[1] >= threshold:
                R[s, :, 1] = -1
                R[s, :, 0] = 1
            else:
                R[s, :, 1] = 1
                R[s, :, 0] = -1
        else:
            if observation[1] >= threshold:
                R[s, :, 1] = 1
                R[s, :, 0] = -1
            else:
                R[s, :, 1] = -1
                R[s, :, 0] = 1

    # The start state is chosen uniformly at random
    initial_state_p = np.ones(n_states)
    initial_state_p /= initial_state_p.sum()

    return MarkovDecisionProcess(
        trans_probs=T,
        rewards=R,
        initial_state_p=initial_state_p,
        observations=observations,
        feature_names=feature_names,
        action_names=action_names,
    )
