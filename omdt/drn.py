from pathlib import Path

from omdt.mdp import MarkovDecisionProcess


def write_as_drn(file: Path, mdp: MarkovDecisionProcess):
    with open(file, "w") as f:
        f.write("// Exported by OMDT\n")
        f.write("@type: MDP\n")
        f.write(f"@nr_states\n{mdp.n_states_}\n")
        f.write(f"@nr_choices\n{mdp.n_states_ * mdp.n_actions_}\n")
        f.write("@reward_models\nreward\n")
        f.write("@model\n")

        for state in range(mdp.n_states_):
            if state == 0:
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
