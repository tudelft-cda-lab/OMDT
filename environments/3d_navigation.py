import numpy as np

from omdt.mdp import MarkovDecisionProcess


def generate_mdp(size=5, seed=0):
    """
    3D navigation with a chance of disappearing.

    Inspired by Scott Sanner's (2D) navigation with a chance of disappearing.
    """

    observations = []
    for x in range(size):
        for y in range(size):
            for z in range(size):
                observations.append((x, y, z))

    feature_names = ["x", "y", "z"]

    action_names = [
        "right",
        "left",
        "up",
        "down",
        "forward",
        "backward",
    ]

    n_states = len(observations)
    n_actions = len(action_names)

    R = np.zeros((n_states, n_states, n_actions))
    T = np.zeros((n_states, n_states, n_actions))

    observation_to_s = {obs: s for s, obs in enumerate(observations)}

    random_state = np.random.RandomState(seed)

    for s, (x, y, z) in enumerate(observations[:-1]):
        if s == 0:
            p_disappear = 0
        else:
            p_disappear = random_state.rand()

        p_move = 1 - p_disappear

        # When trying to move outside of the bounds we stay
        # in the same state.

        # action right
        if x < size - 1:
            T[s, observation_to_s[(x + 1, y, z)], 0] = p_move
        else:
            T[s, s, 0] = p_move

        # action left
        if x > 0:
            T[s, observation_to_s[(x - 1, y, z)], 1] = p_move
        else:
            T[s, s, 1] = p_move

        # action up
        if y < size - 1:
            T[s, observation_to_s[(x, y + 1, z)], 2] = p_move
        else:
            T[s, s, 2] = p_move

        # action down
        if y > 0:
            T[s, observation_to_s[(x, y - 1, z)], 3] = p_move
        else:
            T[s, s, 3] = p_move

        # action forward
        if z < size - 1:
            T[s, observation_to_s[(x, y, z + 1)], 4] = p_move
        else:
            T[s, s, 4] = p_move

        # action backward
        if z > 0:
            T[s, observation_to_s[(x, y, z - 1)], 5] = p_move
        else:
            T[s, s, 5] = p_move

        # There is always a random chance of going back to the start
        T[s, 0, :] += p_disappear

    # Get 1 reward after reaching the goal state
    R[:-1, -1, :] = 1

    # Then we go back to the start
    T[-1, 0, :] = 1

    observations = np.array(observations)

    # We always start with an empty elevator and no waiting people
    initial_state_p = np.zeros(n_states)
    initial_state_p[0] = 1

    return MarkovDecisionProcess(
        trans_probs=T,
        rewards=R,
        initial_state_p=initial_state_p,
        observations=observations,
        feature_names=feature_names,
        action_names=action_names,
    )
