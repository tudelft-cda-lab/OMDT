import pickle
from typing import Callable

import gym
import numpy as np
from gym import spaces


class MarkovDecisionProcess:
    def __init__(
        self,
        trans_probs: np.ndarray,
        rewards: np.ndarray,
        initial_state_p: np.ndarray,
        observations: np.ndarray,
        feature_names=None,
        action_names=None,
    ):
        self.trans_probs = trans_probs
        self.rewards = rewards
        self.initial_state_p = initial_state_p
        self.observations = observations

        # trans_probs and rewards have axes: state, next_state, action
        # observations has axes: state, feature
        assert len(trans_probs.shape) == 3
        assert np.all(np.isclose(trans_probs.sum(axis=1), 1))
        assert np.all((trans_probs >= 0) & (trans_probs <= 1))

        self.n_states_ = trans_probs.shape[0]
        self.n_actions_ = trans_probs.shape[2]

        assert trans_probs.shape[1] == self.n_states_
        assert trans_probs.shape == rewards.shape
        assert initial_state_p.shape == (self.n_states_,)
        assert len(observations.shape) == 2
        assert observations.shape[0] == self.n_states_

        self.n_features_in_ = observations.shape[1]

        if feature_names:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"x[{i}]" for i in range(self.n_features_in_)]

        if action_names:
            self.action_names = action_names
        else:
            self.action_names = [f"action {i}" for i in range(self.n_actions_)]

        self.__remove_unreachable_states()

    def __remove_unreachable_states(self):
        # Do a depth first search to find all states reachable from the non-zero
        # probability initial states
        visited = set()
        stack = list(np.nonzero(self.initial_state_p)[0])
        state_state_probs = self.trans_probs.sum(axis=2)
        while stack:
            state = stack.pop()
            visited.add(state)
            for next_state in np.nonzero(state_state_probs[state])[0]:
                if next_state not in visited:
                    stack.append(next_state)

        if len(visited) == self.trans_probs.shape[0]:
            return

        print("Removed states:", self.trans_probs.shape[0] - len(visited))

        # Remove any unreached states from the MDP
        visited = list(visited)
        self.trans_probs = self.trans_probs[visited][:, visited]
        self.rewards = self.rewards[visited][:, visited]
        self.initial_state_p = self.initial_state_p[visited]
        self.observations = self.observations[visited]

        self.n_states_ = self.trans_probs.shape[0]
        self.n_actions_ = self.trans_probs.shape[2]

    def export(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    def optimal_return(self, gamma, max_iter=1000000000, minimize=False):
        """Determine the return that an optimal policy would have"""
        V = self.value_iteration(gamma, max_iter, minimize)
        return np.dot(self.initial_state_p, V)

    def random_return(self, gamma, max_iter=1000000000):
        """Determine the return that a random policy would have"""
        states = np.arange(self.n_states_)

        delta = 1e-10  # Error tolerance
        V = np.zeros(len(states))  # Initialize values with zeroes

        fixed_trans_probs = np.mean(self.trans_probs, axis=2)
        fixed_rewards = np.mean(self.rewards, axis=2)

        for _ in range(max_iter):
            V_new = np.sum(
                fixed_trans_probs * (fixed_rewards + gamma * V[np.newaxis, :]),
                axis=1,
            )

            max_diff = np.max(np.abs(V_new - V))

            # Update value functions
            V = V_new

            # If diff smaller than threshold delta for all states, algorithm terminates
            if max_diff < delta:
                break

        return np.dot(V, self.initial_state_p)

    def value_iteration(
        self,
        gamma,
        max_iter=1000000000,
        minimize=False,
    ):
        states = np.arange(self.n_states_)

        delta = 1e-10  # Error tolerance
        V = np.zeros(len(states))  # Initialize values with zeroes

        for _ in range(max_iter):
            state_action_values = np.sum(
                self.trans_probs
                * (self.rewards + gamma * V[np.newaxis, :, np.newaxis]),
                axis=1,
            )

            if minimize:
                best_actions = np.argmin(state_action_values, axis=1)
            else:
                best_actions = np.argmax(state_action_values, axis=1)

            V_new = state_action_values[states, best_actions]

            max_diff = np.max(np.abs(V_new - V))

            # Update value functions
            V = V_new

            # If diff smaller than threshold delta for all states, algorithm terminates
            if max_diff < delta:
                break

        return V

    def evaluate_policy(self, policy: Callable, gamma, max_iter):
        """Evaluate the given policy. policy should be callable with an
        observation and should return an action"""
        states = np.arange(self.n_states_)

        delta = 1e-10  # Error tolerance
        V = np.zeros(len(states))  # Initialize values with zeroes

        fixed_trans_probs = np.empty((self.n_states_, self.n_states_))
        fixed_rewards = np.empty((self.n_states_, self.n_states_))
        for i, obs in enumerate(self.observations):
            action = policy(obs)
            fixed_trans_probs[i, :] = self.trans_probs[i, :, action]
            fixed_rewards[i, :] = self.rewards[i, :, action]

        for _ in range(max_iter):
            V_new = np.sum(
                fixed_trans_probs * (fixed_rewards + gamma * V[np.newaxis, :]),
                axis=1,
            )

            max_diff = np.max(np.abs(V_new - V))

            # Update value functions
            V = V_new

            # If diff smaller than threshold delta for all states, algorithm terminates
            if max_diff < delta:
                break

        return np.dot(V, self.initial_state_p)


class MarkovDecisionProcessEnv(gym.Env):
    def __init__(self, mdp: MarkovDecisionProcess, step_limit=1000, random_seed=None):
        super(MarkovDecisionProcessEnv, self).__init__()

        self.action_space = spaces.Discrete(mdp.n_actions_)

        obs_low = np.min(mdp.observations, axis=0)
        obs_high = np.max(mdp.observations, axis=0)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(mdp.observations.shape[1],),
            dtype=mdp.observations.dtype,
        )

        self.mdp = mdp
        self.step_limit = step_limit
        self.random_seed = random_seed

        self.random_state_ = np.random.RandomState(random_seed)

    def step(self, action):
        self.step_ += 1

        # Shape: state, next_state, action
        trans_probs = self.mdp.trans_probs[self.state_, :, action]

        next_state = self.random_state_.choice(self.mdp.n_states_, p=trans_probs)

        reward = self.mdp.rewards[self.state_, next_state, action]
        observation = self.mdp.observations[next_state]

        self.state_ = next_state

        if self.done:
            pass
        elif np.all(self.mdp.trans_probs[next_state, next_state, :] == 1) and np.all(
            self.mdp.rewards[next_state, next_state, :] == 0
        ):
            # If all next transitions lead to this same state and they give
            # no reward then we are in a terminal state
            self.done = True
        elif self.step_ == self.step_limit:
            self.done = True

        return observation, reward, self.done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state_ = self.random_state_.choice(
            self.mdp.n_states_, p=self.mdp.initial_state_p
        )

        self.step_ = 0
        self.done = False

        return self.mdp.observations[self.state_]

    def render(self, mode="human", close=False):
        raise NotImplementedError()
