import argparse
import importlib
import os
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Run on a given environment")
parser.add_argument("algorithm", type=str, help="the algorithm to train with")
parser.add_argument("env_name", type=str, help="the environment (MDP) to train on")
parser.add_argument(
    "--max_depth",
    default=None,
    type=int,
    help="maximum decision tree depth, trees will be trained with depth 1...max_depth. Ignored by dtcontrol",
)
parser.add_argument(
    "--time_limit",
    default=None,
    type=int,
    help="time limit for solving one tree in seconds, by default train until optimality",
)
parser.add_argument(
    "--n_cpus", default=1, type=int, help="number of CPU cores to train on"
)
parser.add_argument(
    "--gamma",
    default=0.99,
    type=float,
    help="discount factor for future states (prevents infinite MDP problems)",
)
parser.add_argument(
    "--verbose",
    default=0,
    type=int,
    help="verbosity level, 0 (no prints), 1 (solver logs), 2 (all logs)",
)
parser.add_argument(
    "--output_dir",
    default="out/",
    type=str,
    help="base directory for outputting files, files are created under out/environment/",
)
parser.add_argument(
    "--max_iter",
    default=1000,
    type=int,
    help="maximum number of value iteration iterations",
)
parser.add_argument(
    "--delta",
    default=1e-10,
    type=float,
    help="stop value iteration after value updates get this small",
)
parser.add_argument("--seed", default=0, type=int, help="random seed for the solver")
parser.add_argument(
    "--record_progress",
    default=False,
    type=bool,
    help="record solver objective and bounds over time",
)
parser.add_argument(
    "--export_graphviz",
    default=True,
    type=bool,
    help="visualize the learned trees with graphviz, needs dot command in path",
)

args = parser.parse_args()

# Create an output directory if it does not yet exist
mdp_output_dir = f"{args.output_dir}{args.env_name}/"
Path(mdp_output_dir).mkdir(parents=True, exist_ok=True)

# Generate the mdp, possibly with extra arguments
print(f"Generating MDP for {args.env_name}...")
environment = importlib.import_module(f"environments.{args.env_name}")
mdp = environment.generate_mdp()

print("Solving...")

if args.algorithm.lower() == "omdt":
    if args.max_depth is None:
        raise ValueError("OMDT is only supposed to be run with max_depth")
    else:
        # If max_depth is given we want to fit a tree of that depth with maximal return
        from omdt.solver import OmdtSolver

        solver = OmdtSolver(
            depth=args.max_depth,
            gamma=args.gamma,
            max_iter=args.max_iter,
            delta=args.delta,
            n_cpus=args.n_cpus,
            verbose=args.verbose,
            time_limit=args.time_limit,
            output_dir=mdp_output_dir,
            seed=args.seed,
        )
        method_name = (
            f"omdt_depth_{args.max_depth}_seed_{args.seed}_timelimit_{args.time_limit}"
        )
elif args.algorithm.lower() == "dtcontrol":
    if args.max_depth is not None:
        raise ValueError("dtcontrol is only supposed to be run with max_depth")

    from dtcontrol.solver import DtControlSolver

    solver = DtControlSolver(
        output_dir=mdp_output_dir,
        verbose=args.verbose,
    )
    method_name = "dtcontrol"
elif args.algorithm.lower() == "viper":
    if args.max_depth is None:
        raise ValueError("viper is only supposed to be run without max_depth")

    from viper.solver import ViperSolver

    solver = ViperSolver(
        max_depth=args.max_depth,
        output_dir=mdp_output_dir,
        verbose=args.verbose,
        random_seed=args.seed,
    )
    method_name = f"viper_depth_{args.max_depth}_seed_{args.seed}"
else:
    raise ValueError(f"Algorithm {args.algorithm} not known")

start_time = time.time()

solver.solve(mdp)

runtime = time.time() - start_time

objective = mdp.evaluate_policy(solver.act, args.gamma, 10000)

optimal = solver.optimal_
bound = solver.bound_

n_nodes = solver.tree_policy_.count_nodes()
depth = solver.tree_policy_.count_depth()

print("Writing result files...")

# Write a .dot file to visualize the learned decision tree and also
# export to PNG and PDF.
if args.export_graphviz:
    import pydot

    integer_features = np.all(
        np.isclose(mdp.observations % np.round(mdp.observations), 0), axis=0
    )

    dot_string = solver.tree_policy_.to_graphviz(
        mdp.feature_names, mdp.action_names, integer_features
    )
    graph = pydot.graph_from_dot_data(dot_string)[0]

    filename = f"{mdp_output_dir}{method_name}_visualized_policy"
    graph.write_png(f"{filename}.png")
    graph.write_pdf(f"{filename}.pdf")
    graph.write_dot(f"{filename}.dot")

result_filename = f"{args.output_dir}results.csv"

if os.path.exists(result_filename):
    write_header = False
else:
    write_header = True

# Append a new line to the result file with the results of this run.
# Optionally write a header first.
with open(result_filename, "a") as file:
    if write_header:
        file.write(
            "method,mdp,seed,time_limit,max_depth,runtime,objective,bound,n_nodes,depth,optimal\n"
        )

    depth_str = args.max_depth if args.max_depth else ""
    file.write(
        f"{args.algorithm},{args.env_name},{args.seed},{args.time_limit},{depth_str},{runtime},{objective},{bound},{n_nodes},{depth},{optimal}\n"
    )
