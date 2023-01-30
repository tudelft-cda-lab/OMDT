import numpy as np

from omdt.mdp import MarkovDecisionProcess

from itertools import product

from functools import reduce

from tqdm import tqdm

# The topology defines the nodes that have edges coming into this node
TREE_TOPOLOGY = {
    0: [],
    1: [0],
    2: [0],
    3: [1],
    4: [1],
    5: [2],
    6: [2],
}

RANDOM_1_TOPOLOGY = {
    0: [2, 6],
    1: [0],
    2: [0, 4],
    3: [2, 6],
    4: [5],
    5: [1, 6, 7],
    6: [4],
    7: [0],
}

RANDOM_2_TOPOLOGY = {
    0: [3, 6, 7],
    1: [0, 1, 2, 4],
    2: [1, 2, 6, 7],
    3: [1, 2],
    4: [1, 2, 3, 4],
    5: [1, 4],
    6: [4, 5, 7],
    7: [3, 4, 5],
}


def probability(observation, new_observation, action, topology):
    probabilities_new = []
    for i, (on, new_on) in enumerate(zip(observation, new_observation)):
        if action == i:
            # Upon reboot action the computer always starts up
            if new_on:
                probabilities_new.append(1.0)
            else:
                probabilities_new.append(0.0)
        else:
            on_probability = 0.05 + 0.9 * on

            n_neighbors = len(topology[i])
            if n_neighbors > 0:
                ratio_neighbors_on = (
                    sum(observation[neighbor] for neighbor in topology[i]) / n_neighbors
                )
                on_probability *= ratio_neighbors_on

            if new_on:
                probabilities_new.append(on_probability)
            else:
                probabilities_new.append(1 - on_probability)

    return reduce(lambda x, y: x * y, probabilities_new, 1.0)


def generate_mdp(topology="tree"):
    if topology == "random1":
        topology = RANDOM_1_TOPOLOGY
    elif topology == "random2":
        topology = RANDOM_2_TOPOLOGY
    elif topology == "tree":
        topology = TREE_TOPOLOGY
    else:
        raise ValueError(f"Unkwown topology {topology}")

    n_computers = max(key for key in topology.keys()) + 1

    # We observe what computers are broken / running
    observations = []
    for on_off in product([0, 1], repeat=n_computers):
        observations.append(on_off)
    observations = np.array(observations)

    feature_names = [f"computer_{i}_running" for i in range(n_computers)]
    action_names = [f"reboot_computer_{i}" for i in range(n_computers)] + ["wait"]

    n_states = len(observations)
    n_actions = len(action_names)

    R = np.zeros((n_states, n_states, n_actions))
    T = np.zeros((n_states, n_states, n_actions))

    for s, observation in tqdm(enumerate(observations), total=len(observations)):
        # The reward is the number of running computers
        R[s, :, :] = observation.sum()

        # Rebooting any computer costs 0.45
        R[s, :, :-1] -= 0.45

        for a in range(len(action_names)):
            for s_prime, new_observation in enumerate(observations):
                probabilities_new = []
                for i, (on, new_on) in enumerate(zip(observation, new_observation)):
                    if a == i:
                        # Upon reboot action the computer always starts up
                        if new_on:
                            probabilities_new.append(1.0)
                        else:
                            probabilities_new.append(0.0)
                    else:
                        on_probability = 0.05 + 0.9 * on

                        n_neighbors = len(topology[i])
                        if n_neighbors > 0:
                            ratio_neighbors_on = (
                                sum(observation[neighbor] for neighbor in topology[i])
                                / n_neighbors
                            )
                            on_probability *= ratio_neighbors_on

                        if new_on:
                            probabilities_new.append(on_probability)
                        else:
                            probabilities_new.append(1 - on_probability)

                T[s, s_prime, a] = reduce(lambda x, y: x * y, probabilities_new, 1.0)

    # We always start with all computers on
    initial_state_p = np.zeros(n_states)
    initial_state_p[-1] = 1

    return MarkovDecisionProcess(
        trans_probs=T,
        rewards=R,
        initial_state_p=initial_state_p,
        observations=observations,
        feature_names=feature_names,
        action_names=action_names,
    )
