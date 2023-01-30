import textwrap

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .tree import Tree, TreeLeaf, TreeNode


class OmdtSolver:
    def __init__(
        self,
        depth=3,
        gamma=0.99,
        max_iter=1_000_000,
        delta=1e-6,
        n_cpus=1,
        verbose=False,
        time_limit=None,
        output_dir="",
        record_progress=False,
        seed=0,
        fixed_depth=True,
        only_build_milp=False,
    ):
        self.depth = depth
        self.gamma = gamma
        self.max_iter = max_iter
        self.delta = delta
        self.n_cpus = n_cpus
        self.verbose = verbose
        self.time_limit = time_limit
        self.output_dir = output_dir
        self.record_progress = record_progress
        self.seed = seed
        self.fixed_depth = fixed_depth
        self.only_build_milp = only_build_milp

        if time_limit:
            if not fixed_depth:
                # Distribute the time limit over all depths that we train
                self.time_limit_ = time_limit / depth
            else:
                self.time_limit_ = time_limit

        self.optimal_ = False

    def __ancestors(self, node: int):
        A_l = []
        A_r = []
        while node > 1:
            if node % 2 == 0:
                A_l.append(node // 2)
            else:
                A_r.append(node // 2)
            node //= 2
        return A_l, A_r

    def __solve_depth(
        self,
        depth,
        mdp,
        old_model=None,
    ):
        states = np.arange(mdp.n_states_)
        actions = np.arange(mdp.n_actions_)
        nodes = range(1, 2**depth)
        leaves = range(2**depth, 2 ** (depth + 1))

        self.thresholds_ = [
            np.sort(np.unique(mdp.observations[:, j]))
            for j in range(mdp.observations.shape[1])
        ]

        all_thresholds = []
        for feature_i in range(mdp.observations.shape[1]):
            for threshold in self.thresholds_[feature_i]:
                all_thresholds.append((feature_i, threshold))

        self.model_ = gp.Model("MDP")

        upper_bound = 1 / (1 - self.gamma)
        pi = self.model_.addVars(
            states, actions, name="pi", lb=0, ub=upper_bound, vtype=GRB.CONTINUOUS
        )

        threshold = self.model_.addVars(
            nodes, len(all_thresholds), name="threshold", vtype=GRB.BINARY
        )
        pred_action = self.model_.addVars(
            leaves, actions, name="pred_action", vtype=GRB.BINARY
        )
        path = self.model_.addVars(
            mdp.observations.shape[0], nodes, name="path", vtype=GRB.BINARY
        )
        takes_action = self.model_.addVars(
            states, actions, name="takes_action", vtype=GRB.BINARY
        )

        for node in nodes:
            self.model_.addConstr(
                gp.quicksum(threshold[node, i] for i in range(len(all_thresholds))) == 1
            )

        # Force deterministic policy
        for leaf in leaves:
            self.model_.addConstr(
                gp.quicksum(pred_action[leaf, action] for action in actions) == 1
            )

        M = 1 / (1 - self.gamma)

        for i, (state, observation) in enumerate(zip(states, mdp.observations)):
            threshold_coefficients = []
            for feat, thres in all_thresholds:
                if observation[feat] <= thres:
                    threshold_coefficients.append(0)
                else:
                    threshold_coefficients.append(1)

            for node in nodes:
                # The coefficient of the chosen threshold decides if the sample
                # goes left (0) or right (1)
                self.model_.addConstr(
                    gp.quicksum(
                        coef * threshold[node, threshold_i]
                        for threshold_i, coef in enumerate(threshold_coefficients)
                    )
                    == path[i, node]
                )

            self.model_.addConstr(
                gp.quicksum(takes_action[state, action] for action in actions) == 1
            )

            for leaf in leaves:
                A_l, A_r = self.__ancestors(leaf)

                for action in actions:
                    self.model_.addConstr(
                        gp.quicksum((1 - path[i, node]) for node in A_l)
                        + gp.quicksum(path[i, node] for node in A_r)
                        + pred_action[leaf, action]
                        - len(A_l + A_r)
                        <= takes_action[state, action]
                    )

            for action in actions:
                self.model_.addConstr(
                    pi[state, action] <= M * takes_action[state, action]
                )

        for state in states:
            # Precomputing non zeros helps a lot because trans_probs are
            # often very sparse matrices
            nonzero_indices = np.argwhere(mdp.trans_probs[:, state, :])

            self.model_.addConstr(
                gp.quicksum(pi[state, action] for action in actions)
                - gp.quicksum(
                    self.gamma
                    * mdp.trans_probs[other_state, state, action]
                    * pi[other_state, action]
                    for other_state, action in nonzero_indices
                )
                == mdp.initial_state_p[state]
            )

        self.model_.setObjective(
            gp.quicksum(
                pi[s, a] * np.sum(mdp.trans_probs[s, :, a] * mdp.rewards[s, :, a])
                for s in states
                for a in actions
            ),
            GRB.MAXIMIZE,
        )

        self.model_.setParam("OutputFlag", self.verbose)
        self.model_.setParam("Seed", self.seed)
        self.model_.setParam("Threads", self.n_cpus)

        if self.time_limit:
            self.model_.setParam("TimeLimit", self.time_limit_)

        self.model_.update()

        # If the old model is given, then use it to initialize a warm start
        if old_model and old_model.SolCount > 0:
            # Initialize all nodes to a useless split that sends
            # all samples to the left
            for node in nodes:
                for threshold_i in range(len(all_thresholds) - 1):
                    threshold[node, threshold_i].Start = 0
                threshold[node, len(all_thresholds) - 1].Start = 1

            for var in old_model.getVars():
                if "pred_action" in var.VarName:
                    leaf_i, action_i = var.VarName.split("[")[1][:-1].split(",")
                    leaf_i = int(leaf_i)
                    action_i = int(action_i)

                    leaf_i *= 2

                    self.model_.getVarByName(
                        f"pred_action[{leaf_i},{action_i}]"
                    ).Start = var.X
                elif "takes_action" in var.VarName:
                    state_i, action_i = var.VarName.split("[")[1][:-1].split(",")
                    state_i = int(state_i)
                    action_i = int(action_i)

                    self.model_.getVarByName(
                        f"takes_action[{state_i},{action_i}]"
                    ).Start = var.X
                else:
                    self.model_.getVarByName(var.VarName).Start = var.X

        if self.only_build_milp:
            return

        # Solve the model. If record progress is True then keep track of the
        # objective and bound over time
        if self.record_progress:
            objectives = []
            bounds = []
            runtimes = []

            def callback(model, where):
                if where == GRB.Callback.MIP:
                    objectives.append(model.cbGet(GRB.Callback.MIP_OBJBST))
                    bounds.append(model.cbGet(GRB.Callback.MIP_OBJBND))
                    runtimes.append(model.cbGet(GRB.Callback.RUNTIME))

            self.model_.optimize(callback)

            # Gurobi does not always add the last objective and bound
            objectives.append(self.model_.ObjVal)
            bounds.append(self.model_.ObjBound)
            runtimes.append(self.model_.Runtime)

            self.recorded_objectives_.append(objectives)
            self.recorded_bounds_.append(bounds)
            self.recorded_runtimes_.append(runtimes)
        else:
            self.model_.optimize()

        # Only export the tree if there was actually a feasible solution found
        # within the time limit
        if self.model_.SolCount > 0:
            self.__model_vars_to_tree_new(
                mdp,
                nodes,
                leaves,
                threshold,
                all_thresholds,
                pred_action,
                depth,
            )
        else:
            # In case of no model just return a leaf that always
            # predicts the same action
            self.tree_policy_ = Tree(TreeLeaf(0))
            self.optimal_ = False
            self.objective_ = mdp.evaluate_policy(
                self.tree_policy_.act, self.gamma, 1000000000
            )
            self.bound_ = self.model_.ObjBound

    def __model_vars_to_tree_new(
        self,
        mdp,
        nodes,
        leaves,
        threshold,
        all_thresholds,
        pred_action,
        depth,
    ):
        actions = np.arange(mdp.n_actions_)
        self.split_features_ = []
        self.split_thresholds_ = []
        self.leaf_actions_ = []
        for node in nodes:
            threshold_i = np.argmax(
                [threshold[node, i].x for i in range(len(all_thresholds))]
            )
            feature_i, threshold_value = all_thresholds[threshold_i]

            self.split_features_.append(feature_i)
            self.split_thresholds_.append(threshold_value)

        for leaf in leaves:
            leaf_action = np.argmax([pred_action[leaf, a].x for a in actions])
            self.leaf_actions_.append(leaf_action)

        tree_nodes = []
        for feature, threshold in zip(self.split_features_, self.split_thresholds_):
            tree_nodes.append(
                TreeNode(
                    feature,
                    threshold,
                    None,
                    None,
                )
            )
        for action in self.leaf_actions_:
            tree_nodes.append(
                TreeLeaf(
                    action,
                )
            )
        # Link all nodes together in the shape of a tree
        for node_i in nodes:
            tree_nodes[node_i - 1].left_child = tree_nodes[2 * node_i - 1]
            tree_nodes[node_i - 1].right_child = tree_nodes[2 * node_i]

        self.tree_policy_ = Tree(tree_nodes[0])
        self.tree_policy_.prune()

        if self.verbose > 0:
            print("Tree policy:")
            print(self.tree_policy_.to_string(mdp.feature_names, mdp.action_names))

            # print(f"Optimal value: {optimal_value}")
            print(
                f"Optimal decision tree (depth={int(np.log2(len(self.split_thresholds_) + 1))}) value: {self.model_.objVal}"
            )
            # print(f"Ratio: {self.model_.objVal / optimal_value:.3f}")

        optimal = self.model_.Status == GRB.OPTIMAL
        optimal_value = np.dot(mdp.initial_state_p, self.optimal_values)

        self.optimal_ = optimal
        self.objective_ = self.model_.ObjVal
        self.bound_ = self.model_.ObjBound

        self.result_summary_.append(
            {
                "objective": self.model_.objVal,
                "bound": self.model_.objBound,
                "VI objective": optimal_value,
                "optimal": optimal,
                "runtime": self.model_.Runtime,
                "depth": depth,
                "max_iter": self.max_iter,
                "delta": self.delta,
                "seed": self.seed,
            }
        )

        with open(
            f"{self.output_dir}policy_depth_{depth}_seed_{self.seed}.py", "w"
        ) as file:
            # Print information about the policy as comments
            line_width = 60
            code = ""
            code += f"{' Properties '.center(line_width, '#')}\n"
            code += f"# expected discounted reward: {self.model_.objVal}\n"
            code += f"# expected discounted reward bound: {self.model_.objBound}\n"
            code += f"# value iteration: {optimal_value}\n"
            code += f"# proven optimal: {optimal}\n"
            code += f"# runtime: {self.model_.Runtime}\n"
            code += f"{' Parameters '.center(line_width, '#')}\n"
            code += f"# depth: {depth}\n"
            code += f"# gamma: {self.gamma}\n"
            code += f"# max_iter: {self.max_iter}\n"
            code += f"# delta: {self.delta}\n"
            code += f"# seed: {self.seed}\n"
            code += f"{'#' * line_width}\n"

            # Add the decision tree as a function with if statements
            code += f"def act({', '.join(mdp.feature_names)}):\n"
            # code += f"def act(observation):\n"
            tree_code = self.tree_policy_.to_string(mdp.feature_names, mdp.action_names)
            code += f"{textwrap.indent(tree_code, '    ')}\n"

            # Write the whole string as a python file
            file.write(code)

    def solve(
        self,
        mdp,
    ):
        self.optimal_values = mdp.value_iteration(gamma=self.gamma)
        self.min_values = mdp.value_iteration(gamma=self.gamma, minimize=True)

        self.trees_ = []
        self.trees_optimal_ = []
        self.result_summary_ = []

        if self.record_progress:
            self.recorded_objectives_ = []
            self.recorded_bounds_ = []
            self.recorded_runtimes_ = []

        if self.fixed_depth:
            if self.verbose:
                print(f"Starting with fixed depth {self.depth}")

            self.__solve_depth(
                self.depth,
                mdp,
                None,
            )

            if self.only_build_milp:
                return

            self.trees_.append(self.tree_policy_)
        else:
            old_model = None
            for depth in range(1, self.depth + 1):
                if self.verbose:
                    print(f"Starting with depth {depth}")

                self.__solve_depth(
                    depth,
                    mdp,
                    old_model,
                )

                self.trees_.append(self.tree_policy_)
                self.trees_optimal_.append(self.optimal_)

                old_model = self.model_

            # At the end, set the tree_policy_ to the last feasible solution
            for tree, optimal in zip(
                reversed(self.trees_), reversed(self.trees_optimal_)
            ):
                if tree is not None:
                    self.tree_policy_ = tree
                    self.optimal_ = optimal
                    break

    def act(self, observation):
        """
        Return the next action given the observation according to the learned tree.
        """
        return self.tree_policy_.act(observation)
