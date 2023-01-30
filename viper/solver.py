import numpy as np

from omdt.mdp import MarkovDecisionProcess, MarkovDecisionProcessEnv
from omdt.tree import sklearn_to_omdt_tree

from .core.dt import DTPolicy
from .core.rl import test_policy, train_dagger
from .vi_teacher import ValueIterationTeacher


def identity_state_transformer(x):
    return x


class ViperSolver:
    def __init__(self, max_depth=3, output_dir="", verbose=False, random_seed=None):
        self.max_depth = max_depth
        self.verbose = verbose
        self.random_seed = random_seed
        self.output_dir = output_dir

        # VIPER does not prove optimality unless we know that the tree
        # gets the same objective as value iteration.
        self.optimal_ = False
        self.bound_ = float("inf")

    def solve(self, mdp: MarkovDecisionProcess, gamma=0.99):
        """Solve MDP with VIPER using a value iteration teacher"""

        optimal_return = mdp.optimal_return(gamma)
        self.bound_ = optimal_return

        teacher = ValueIterationTeacher(mdp, gamma=gamma)

        # Parameters
        n_batch_rollouts = 10
        max_samples = 200000
        max_iters = 80
        train_frac = 0.8
        is_reweight = True
        n_test_rollouts = 50

        env = MarkovDecisionProcessEnv(mdp, random_seed=self.random_seed)

        self.student = DTPolicy(self.max_depth, self.verbose, self.random_seed)

        # Train self.student
        self.student = train_dagger(
            env,
            teacher,
            self.student,
            identity_state_transformer,
            max_iters,
            n_batch_rollouts,
            max_samples,
            train_frac,
            is_reweight,
            n_test_rollouts,
            self.verbose,
            self.random_seed,
        )
        # Test self.student
        rew = test_policy(
            env, self.student, identity_state_transformer, n_test_rollouts
        )

        if self.verbose:
            print("Final reward: {}".format(rew))
            print("Number of nodes: {}".format(self.student.tree.tree_.node_count))

        self.tree_policy_ = sklearn_to_omdt_tree(self.student.tree)
        self.tree_policy_.prune()

        policy_return = mdp.evaluate_policy(
            self.tree_policy_.act, gamma=gamma, max_iter=1000000000
        )

        if np.isclose(policy_return, optimal_return):
            self.optimal_ = True

    def act(self, obs):
        return self.student.predict([obs])[0]
