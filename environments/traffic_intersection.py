import numpy as np

from scipy.stats import poisson

from omdt.mdp import MarkovDecisionProcess


def generate_mdp(
    max_cars=5,
    max_wait_time=5,
    p_new_car_left=0.1,
    p_new_car_right=0.5,
    switch_cost=2,
    car_reward=1,
    wait_exp_cost=0.1,
):
    # Cars per time interval is always 1

    action_names = ["switch_light", "wait"]
    feature_names = ["green_side", "cars_left", "cars_right", "waiting_time"]

    observations = []
    for green_side in range(2):
        for cars_left in range(max_cars + 1):
            for cars_right in range(max_cars + 1):
                for waiting_time in range(max_wait_time + 1):
                    observations.append(
                        (
                            green_side,
                            cars_left,
                            cars_right,
                            waiting_time,
                        )
                    )

    obs_to_s = {obs: s for s, obs in enumerate(observations)}
    observations = np.array(observations)

    n_states = len(observations)
    n_actions = len(action_names)

    R = np.zeros((n_states, n_states, n_actions))
    T = np.zeros((n_states, n_states, n_actions))

    for s, observation in enumerate(observations):
        (
            green_side,
            cars_left,
            cars_right,
            waiting_time,
        ) = observation

        # Switch traffic light action
        R[s, :, 0] = -switch_cost

        next_green_side = 1 - green_side

        if green_side == 0:
            if cars_left >= 1:
                R[s, :, 0] += (
                    car_reward - cars_right * wait_exp_cost * 2**waiting_time
                )

                for next_cars_left in range(cars_left - 1, max_cars + 1):
                    for next_cars_right in range(cars_right, max_cars + 1):
                        extra_left = next_cars_left - cars_left + 1
                        extra_right = next_cars_right - cars_right

                        probability = poisson.pmf(
                            extra_left, p_new_car_left
                        ) * poisson.pmf(extra_right, p_new_car_right)

                        # One more car goes through in the meantime
                        next_observation = (
                            next_green_side,
                            next_cars_left,
                            next_cars_right,
                            1,
                        )
                        T[s, obs_to_s[next_observation], 0] = probability
            else:
                R[s, :, 0] += cars_right * -wait_exp_cost * 2**waiting_time

                for next_cars_left in range(cars_left, max_cars + 1):
                    for next_cars_right in range(cars_right, max_cars + 1):
                        extra_left = next_cars_left - cars_left
                        extra_right = next_cars_right - cars_right

                        probability = poisson.pmf(
                            extra_left, p_new_car_left
                        ) * poisson.pmf(extra_right, p_new_car_right)

                        # One more car goes through in the meantime
                        next_observation = (
                            next_green_side,
                            next_cars_left,
                            next_cars_right,
                            1,
                        )
                        T[s, obs_to_s[next_observation], 0] = probability

        elif green_side == 1:
            if cars_right >= 1:
                R[s, :, 0] += car_reward - cars_left * wait_exp_cost * 2**waiting_time

                for next_cars_left in range(cars_left, max_cars + 1):
                    for next_cars_right in range(cars_right - 1, max_cars + 1):
                        extra_left = next_cars_left - cars_left
                        extra_right = next_cars_right - cars_right + 1

                        probability = poisson.pmf(
                            extra_left, p_new_car_left
                        ) * poisson.pmf(extra_right, p_new_car_right)

                        # One more car goes through in the meantime
                        next_observation = (
                            next_green_side,
                            next_cars_left,
                            next_cars_right,
                            1,
                        )
                        T[s, obs_to_s[next_observation], 0] = probability
            else:
                R[s, :, 0] += cars_left * -wait_exp_cost * 2**waiting_time

                for next_cars_left in range(cars_left, max_cars + 1):
                    for next_cars_right in range(cars_right, max_cars + 1):
                        extra_left = next_cars_left - cars_left
                        extra_right = next_cars_right - cars_right

                        probability = poisson.pmf(
                            extra_left, p_new_car_left
                        ) * poisson.pmf(extra_right, p_new_car_right)

                        # One more car goes through in the meantime
                        next_observation = (
                            next_green_side,
                            next_cars_left,
                            next_cars_right,
                            1,
                        )
                        T[s, obs_to_s[next_observation], 0] = probability

        # Wait action
        if green_side == 0:
            R[s, :, 1] -= cars_right * wait_exp_cost * 2**waiting_time
            if cars_left > 0:
                R[s, :, 1] += car_reward
        elif green_side == 1:
            R[s, :, 1] -= cars_left * wait_exp_cost * 2**waiting_time
            if cars_right > 0:
                R[s, :, 1] += car_reward

        if green_side == 0:
            if cars_left >= 1:
                for next_cars_left in range(cars_left - 1, max_cars + 1):
                    for next_cars_right in range(cars_right, max_cars + 1):
                        extra_left = next_cars_left - cars_left + 1
                        extra_right = next_cars_right - cars_right

                        probability = poisson.pmf(
                            extra_left, p_new_car_left
                        ) * poisson.pmf(extra_right, p_new_car_right)

                        # One more car goes through in the meantime
                        next_observation = (
                            green_side,
                            next_cars_left,
                            next_cars_right,
                            min(waiting_time + 1, max_wait_time),
                        )
                        T[s, obs_to_s[next_observation], 1] = probability
            else:
                for next_cars_left in range(cars_left, max_cars + 1):
                    for next_cars_right in range(cars_right, max_cars + 1):
                        extra_left = next_cars_left - cars_left
                        extra_right = next_cars_right - cars_right

                        probability = poisson.pmf(
                            extra_left, p_new_car_left
                        ) * poisson.pmf(extra_right, p_new_car_right)

                        # One more car goes through in the meantime
                        next_observation = (
                            green_side,
                            next_cars_left,
                            next_cars_right,
                            min(waiting_time + 1, max_wait_time),
                        )
                        T[s, obs_to_s[next_observation], 1] = probability

        elif green_side == 1:
            if cars_right >= 1:
                for next_cars_left in range(cars_left, max_cars + 1):
                    for next_cars_right in range(cars_right - 1, max_cars + 1):
                        extra_left = next_cars_left - cars_left
                        extra_right = next_cars_right - cars_right + 1

                        probability = poisson.pmf(
                            extra_left, p_new_car_left
                        ) * poisson.pmf(extra_right, p_new_car_right)

                        # One more car goes through in the meantime
                        next_observation = (
                            green_side,
                            next_cars_left,
                            next_cars_right,
                            min(waiting_time + 1, max_wait_time),
                        )
                        T[s, obs_to_s[next_observation], 1] = probability
            else:
                for next_cars_left in range(cars_left, max_cars + 1):
                    for next_cars_right in range(cars_right, max_cars + 1):
                        extra_left = next_cars_left - cars_left
                        extra_right = next_cars_right - cars_right

                        probability = poisson.pmf(
                            extra_left, p_new_car_left
                        ) * poisson.pmf(extra_right, p_new_car_right)

                        # One more car goes through in the meantime
                        next_observation = (
                            green_side,
                            next_cars_left,
                            next_cars_right,
                            min(waiting_time + 1, max_wait_time),
                        )
                        T[s, obs_to_s[next_observation], 1] = probability

    # Normalize the transition probabilities (the values can sum up to a value just under 1)
    T /= T.sum(axis=1)[:, np.newaxis, :]

    # We always start in the first round:
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
