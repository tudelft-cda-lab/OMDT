import importlib
import os

import numpy as np
import pandas as pd

from omdt.mdp import MarkovDecisionProcess
from omdt.solver import OmdtSolver

output_dir = "out/"

environments = [
    "3d_navigation",
    "blackjack",
    "frozenlake_4x4",
    "frozenlake_8x8",
    "frozenlake_12x12",
    "inventory_management",
    "system_administrator_tree",
    "system_administrator_1",
    "system_administrator_2",
    "tictactoe_vs_random",
    "tiger_vs_antelope",
    "traffic_intersection",
    "xor",
]

output_filename = output_dir + "mdps.csv"
if os.path.exists(output_filename):
    data_df = pd.read_csv(output_filename)
else:
    data = []
    for env_name in environments:
        print(f"Generating MDP for {env_name}...")
        environment = importlib.import_module(f"environments.{env_name}")
        mdp: MarkovDecisionProcess = environment.generate_mdp()

        min_score = mdp.optimal_return(gamma=0.99, max_iter=1000000000, minimize=True)
        max_score = mdp.optimal_return(gamma=0.99, max_iter=1000000000)
        rand_score = mdp.random_return(gamma=0.99, max_iter=1000000000)

        # Compute the (naive) number of possible trees of depth 3
        depth = 3
        n_thresholds = 0
        for feature_values in mdp.observations.T:
            n_thresholds += len(np.unique(feature_values)) - 1
        n_trees = round(
            np.log10(
                float(n_thresholds) ** (2**depth - 1)
                * float(mdp.n_actions_) ** (2**depth)
            )
        )
        n_trees_str = f"$10^{{{n_trees}}}$"

        # Create the solver with the only_build_milp flag to skip solving
        solver = OmdtSolver(depth=3, only_build_milp=True)
        solver.solve(mdp)
        n_vars = solver.model_.NumVars
        n_constraints = solver.model_.NumConstrs

        data.append(
            (
                env_name,
                mdp.n_states_,
                mdp.n_actions_,
                min_score,
                max_score,
                rand_score,
                n_trees_str,
                n_vars,
                n_constraints,
            )
        )

        data_df = pd.DataFrame(
            data,
            columns=[
                "mdp",
                "states",
                "actions",
                "min_return",
                "max_return",
                "rand_return",
                "n_possible_trees",
                "variables",
                "constraints",
            ],
        )
        data_df.to_csv(output_filename, index=False)

# Print latex code for easier editing
print(
    data_df[
        ["mdp", "states", "actions", "variables", "constraints", "n_possible_trees"]
    ]
    .to_latex(escape=False, index=False)
    .replace("_", "\\_")
)
