import numpy as np

from scipy.stats import poisson

from itertools import product

from tqdm import tqdm

from omdt.mdp import MarkovDecisionProcess

ALTERNATIVE_PARAMETERS = {
    "n": 100,
    "k": 5,
    "c": 2,
    "h": 2,
    "p": 3,
    "lam": 8,
}


def generate_mdp(
    n=100,  # Maximum size of the inventory (storage)
    k=10,  # Fixed cost for ordering any amount
    c=2,  # Cost per unit bought
    h=1,  # Cost per unit for holding it in storage
    p=4,  # Profit per unit sold
    lam=15,  # Poisson dist. parameter for the demand (expected demand per day)
):
    states = np.arange(n + 1)
    actions = np.arange(n)
    demands = np.arange(
        lam * 5
    )  # limit demands to 5 * lambda as higher demands are very unlikely

    # Indexes: state, next_state, action
    rewards = np.zeros((n + 1, n + 1, n))
    trans_probs = np.zeros((n + 1, n + 1, n))

    total = len(states) * len(actions) * len(demands)
    for state, action, demand in tqdm(product(states, actions, demands), total=total):
        next_state = max(min(state + action, n) - demand, 0)

        cost = 0
        if action > 0:
            # if buying anything we incur a cost (fixed price + cost per unit)
            # this rule is slightly different from Example 1.1 in Algorithms for
            # Reinforcement Learning by Csaba Szepesvari (2010)
            # https://sites.ualberta.ca/~szepesva/RLBook.html since we pay for the
            # full amount ordered, not only the amount that is available in our inventory.
            # this makes a bit more sense as then the agent buys exactly what they order.
            # the old code would be: cost += k + c * max(min(state + action, n) - state, 0)
            cost += k + c * action

        # if there is anything in the warehouse we incur a cost (cost per unit)
        cost += h * state

        # we sell anything we have for a price per unit if there is demand
        income = p * max(min(state + action, n) - next_state, 0)

        # compute the probability that this demand occurs (poisson distribution)
        probability = poisson.pmf(demand, lam)

        rewards[state, next_state, action] += probability * (income - cost)
        trans_probs[state, next_state, action] += probability

    # Normalize the transition probabilities (the values can sum up to a value just under 1)
    trans_probs /= trans_probs.sum(axis=1)[:, np.newaxis, :]

    # We always start with an empty warehouse
    initial_state_p = np.zeros(len(states))
    initial_state_p[0] = 1

    # The observations are just the number of items in the warehouse
    observations = states.reshape(-1, 1)
    feature_names = ["inventory"]

    # The actions are "don't buy" or buy a certain amount
    action_names = ["Don't buy"] + [f"Buy {i}" for i in range(1, len(actions))]

    return MarkovDecisionProcess(
        trans_probs=trans_probs,
        rewards=rewards,
        initial_state_p=initial_state_p,
        observations=observations,
        feature_names=feature_names,
        action_names=action_names,
    )
