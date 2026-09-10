from pathlib import Path

from omdt.mdp import MarkovDecisionProcess

import numpy as np


def convert_mdp_to_single_initial_state(mdp: MarkovDecisionProcess):
    n_states = mdp.n_states_ + 1
    n_actions = mdp.n_actions_

    new_trans_probs = np.zeros((n_states, n_states, n_actions))
    new_trans_probs[1:, 1:] = mdp.trans_probs

    new_rewards = np.zeros((n_states, n_states, n_actions))
    new_rewards[1:, 1:] = mdp.rewards

    new_observations = np.zeros((n_states, mdp.observations.shape[1]))
    new_observations[1:] = mdp.observations

    new_initial_state_p = np.zeros(n_states)

    # Define a single initial state that directs to the actual multiple initial states
    # we might later choose a specific observation value
    new_initial_state_p[0] = 1.0
    new_observations[0] = 0.0
    new_trans_probs[0, 1:] = mdp.initial_state_p[:, np.newaxis]

    return MarkovDecisionProcess(new_trans_probs, new_rewards, new_initial_state_p, new_observations, mdp.feature_names, mdp.action_names)

def write_as_drn(file: Path, mdp: MarkovDecisionProcess):
    initial_states = np.nonzero(mdp.initial_state_p)[0]
    if len(initial_states) != 1:
        mdp = convert_mdp_to_single_initial_state(mdp)

    initial_state = np.nonzero(mdp.initial_state_p)[0][0]

    with open(file, "w") as f:
        f.write("// Exported by OMDT\n")
        f.write("@type: MDP\n")
        f.write(f"@nr_states\n{mdp.n_states_}\n")
        f.write(f"@nr_choices\n{mdp.n_states_ * mdp.n_actions_}\n")
        f.write("@reward_models\nreward\n")
        f.write("@model\n")

        for state in range(mdp.n_states_):
            if state == initial_state:
                f.write(f"state {state} init\n")
            else:
                f.write(f"state {state}\n")

            observations = mdp.observations[state]
            obs = []
            for feature in range(len(observations)):
                obs.append(f"{mdp.feature_names[feature]}={int(observations[feature])}")
            f.write(f"//[{' & '.join(obs)}]\n")

            for action in range(mdp.n_actions_):
                rewards = mdp.rewards[state, :, action]
                probs = mdp.trans_probs[state, :, action]
                expected_reward = rewards.dot(probs)
                action_name = mdp.action_names[action].replace(' ', '_').replace("'", "")
                f.write(f"\taction {action_name} [{expected_reward}]\n")
                for next_state in range(mdp.n_states_):
                    probability = mdp.trans_probs[state, next_state, action]
                    if probability > 1e-8:
                        f.write(
                            f"\t\t{next_state} : {probability}\n"
                        )
