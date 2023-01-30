import numpy as np

from omdt.mdp import MarkovDecisionProcess


def generate_mdp(grid_size=5, diff_features=False):
    observations = []
    for antelope_x in range(grid_size):
        for antelope_y in range(grid_size):
            for tiger_x in range(grid_size):
                for tiger_y in range(grid_size):
                    observations.append(
                        (
                            antelope_x,
                            antelope_y,
                            tiger_x,
                            tiger_y,
                        )
                    )

    # Add terminal state that loops to itself
    observations.append((-1, -1, -1, -1))

    observations = np.array(observations)

    feature_names = ["antelope_x", "antelope_y", "tiger_x", "tiger_y"]
    action_names = ["up", "right", "down", "left", "wait"]

    n_states = len(observations)
    n_actions = len(action_names)

    R = np.zeros((n_states, n_states, n_actions))
    T = np.zeros((n_states, n_states, n_actions))

    for s, (antelope_x, antelope_y, tiger_x, tiger_y) in enumerate(observations[:-1]):
        # If the antelope and tiger are at the same position then
        # receive a reward on any action and go to the terminal state
        if antelope_x == tiger_x and antelope_y == tiger_y:
            R[s, -1, :] = 1
            T[s, -1, :] = 1
            continue

        # We generate the next possible states when the antelope moves,
        # it never crosses the walls (at 0 and grid size) and never
        # steps directly into the tiger.
        next_antelope_states = []
        if antelope_x > 0 and (tiger_x != antelope_x - 1 or tiger_y != antelope_y):
            next_antelope_states.append(s - grid_size**3)
        if antelope_x < grid_size - 1 and (
            tiger_x != antelope_x + 1 or tiger_y != antelope_y
        ):
            next_antelope_states.append(s + grid_size**3)
        if antelope_y > 0 and (tiger_y != antelope_y - 1 or tiger_x != antelope_x):
            next_antelope_states.append(s - grid_size**2)
        if antelope_y < grid_size - 1 and (
            tiger_y != antelope_y + 1 or tiger_x != antelope_x
        ):
            next_antelope_states.append(s + grid_size**2)

        for antelope_state in next_antelope_states:
            if tiger_x > 0:
                T[s, antelope_state - grid_size, 3] += 1 / len(next_antelope_states)
            else:
                T[s, antelope_state, 3] += 1 / len(next_antelope_states)

            if tiger_x < grid_size - 1:
                T[s, antelope_state + grid_size, 1] += 1 / len(next_antelope_states)
            else:
                T[s, antelope_state, 1] += 1 / len(next_antelope_states)

            if tiger_y > 0:
                T[s, antelope_state - 1, 2] += 1 / len(next_antelope_states)
            else:
                T[s, antelope_state, 2] += 1 / len(next_antelope_states)

            if tiger_y < grid_size - 1:
                T[s, antelope_state + 1, 0] += 1 / len(next_antelope_states)
            else:
                T[s, antelope_state, 0] += 1 / len(next_antelope_states)

            T[s, antelope_state, 4] += 1 / len(next_antelope_states)

    # You stay in the terminal state forever
    T[-1, -1, :] = 1

    if diff_features:
        old_observations = observations
        observations = np.empty((len(observations), 6))
        observations[:, :4] = old_observations
        observations[:, 4] = observations[:, 0] - observations[:, 2]
        observations[:, 5] = observations[:, 1] - observations[:, 3]

        feature_names = [
            "antelope_x",
            "antelope_y",
            "tiger_x",
            "tiger_y",
            "antelope_x - tiger_x",
            "antelope_y - tiger_y",
        ]

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
