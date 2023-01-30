import numpy as np

from omdt.mdp import MarkovDecisionProcess


class ValueIterationTeacher:
    def __init__(self, mdp: MarkovDecisionProcess, gamma=0.99, max_iter=1000):
        # First solve to get state values using value iteration
        state_values = mdp.value_iteration(gamma, max_iter)

        # Building a Q value table from state values.
        # See: https://www.cs.cmu.edu/~mgormley/courses/10601-s17/slides/lecture26-ri.pdf
        self.q_values = np.empty((mdp.n_states_, mdp.n_actions_))
        for state in range(mdp.n_states_):
            for action in range(mdp.n_actions_):
                trans_probs = mdp.trans_probs[state, :, action]
                rewards = mdp.rewards[state, :, action]
                self.q_values[state, action] = np.sum(
                    trans_probs * (rewards + gamma * state_values)
                )

        self.obs_to_state = {}
        for state, obs in enumerate(mdp.observations):
            self.obs_to_state[tuple(obs)] = state

    def predict(self, obs):
        pred_actions = []

        # obs is a 2d array containing multiple observations
        for observation in obs:
            # Find the index of the state belonging to this observation
            state = self.obs_to_state[tuple(observation)]

            pred_actions.append(np.argmax(self.q_values[state, :]))

        return np.array(pred_actions)

    def predict_q(self, obs):
        pred_q_values = []

        # obs is a 2d array containing multiple observations
        for observation in obs:
            # Find the index of the state belonging to this observation
            state = self.obs_to_state[tuple(observation)]

            pred_q_values.append(self.q_values[state, :])

        return np.array(pred_q_values)
