import numpy as np

from omdt.mdp import MarkovDecisionProcess
from omdt.tree import Tree, TreeLeaf, TreeNode

from .dtcontrol.decision_tree.decision_tree import DecisionTree
from .dtcontrol.decision_tree.impurity.entropy import Entropy
from .dtcontrol.decision_tree.splitting.axis_aligned import AxisAlignedSplittingStrategy


def _make_set(v):
    if v is None:
        return set()
    if isinstance(v, tuple):
        return {v}
    try:
        return set(v)
    except TypeError:
        return {v}

def _get_unique_labels_from_2d(labels):
    """
    Computes unique labels of a 2d label array by mapping every unique inner array to an int. Returns the unique labels
    and the int mapping.
    """
    l = []
    int_to_label = {}
    next_unused_int = 1  # OC1 expects labels starting with 1
    label_str_to_int = {}
    for i in range(len(labels)):
        label_str = ','.join(sorted([str(i) for i in labels[i] if i != -1]))
        if label_str not in label_str_to_int:
            label_str_to_int[label_str] = next_unused_int
            int_to_label[next_unused_int] = labels[i]
            next_unused_int += 1
        new_label = label_str_to_int[label_str]
        l.append(new_label)
    return np.array(l), int_to_label

class SimpleDataset:
    """
    A simple dataset class that is limited to numerical observations and
    deterministic actions.
    """
    def __init__(self, observations: np.ndarray, predicted_actions: np.ndarray, name="unknown"):
        # We only allow numerical observations
        self.x = observations

        # We only allow a deterministic action in each state so axis 1 is size 1
        self.y = predicted_actions
        self.y = predicted_actions.reshape(-1, 1)

        self.name = name

        self.x_metadata = {"variables": None, "categorical": None, "category_names": None,
                           "min": None, "max": None, "step_size": None}
        self.y_metadata = {"categorical": [], "category_names": None, "min": None, "max": None, "step_size": None,
                           'num_rows': None, 'num_flattened': None}

        self.unique_labels_ = None

    def get_name(self):
        return self.name

    def is_deterministic(self):
        return True

    def get_single_labels(self):
        return self.y

    def get_unique_labels(self):
        """
        e.g.
        [[1  2  3 -1 -1],
         [1 -1 -1 -1 -1],
         [1  2 -1 -1 -1],
        ]

        gets mapped to

        unique_labels = [1, 2, 3]
        unique_mapping = {1: [1 2 3 -1 -1], 2: [1 -1 -1 -1 -1], 3: [1 2 -1 -1 -1]}
        """
        if self.unique_labels_ is None:
            self.unique_labels_, _ = _get_unique_labels_from_2d(self.y)
        return self.unique_labels_

    def get_numeric_x(self):
        return self.x

    def map_numeric_feature_back(self, feature):
        return feature

    def map_single_label_back(self, single_label):
        return single_label

    def index_label_to_actual(self, index_label):
        return index_label

    def compute_accuracy(self, y_pred):
        num_correct = 0
        for i in range(len(y_pred)):
            pred = y_pred[i]
            if pred is None:
                return None
            if set.issubset(_make_set(pred), set(self.y[i])):
                num_correct += 1
        return num_correct / len(y_pred)

    def from_mask_optimized(self, mask):
        empty_object = type('', (), {})()
        empty_object.parent_mask = mask
        empty_object.get_single_labels = lambda: self.y[mask]
        return empty_object

    def from_mask(self, mask):
        subset = SimpleDataset(self.x[mask], self.y[mask], self.name)
        subset.parent_mask = mask

        if self.unique_labels_ is not None:
            subset.unique_labels_ = self.unique_labels_[mask]
        return subset

    def load_metadata_from_json(self, json_object):
        metadata = json_object['metadata']
        self.x_metadata = metadata['X_metadata']
        self.y_metadata = metadata['Y_metadata']

    def __len__(self):
        return len(self.x)

    def load_if_necessary(self):
        pass

    def set_treat_categorical_as_numeric(self):
        pass

def _dtcontrol_node_to_omdt_rec(node):
    if node.is_leaf():
        return TreeLeaf(node.actual_label)

    assert len(node.children) == 2

    left_child = _dtcontrol_node_to_omdt_rec(node.children[0])
    right_child = _dtcontrol_node_to_omdt_rec(node.children[1])
    return TreeNode(node.split.feature, node.split.threshold, left_child, right_child)


def _dtcontrol_tree_to_omdt(tree: DecisionTree):
    return Tree(_dtcontrol_node_to_omdt_rec(tree.root))

class DtControlSolver:
    def __init__(self, output_dir, verbose=False):
        self.output_dir = output_dir
        self.verbose = verbose

        # dtcontrol does not prove optimality
        self.optimal_ = False
        self.bound_ = 0
    
    def solve(self, mdp: MarkovDecisionProcess, gamma=0.99, max_iter=100000):
        V = mdp.value_iteration(gamma=gamma, max_iter=max_iter)

        state_action_values = np.sum(
            mdp.trans_probs * (mdp.rewards + gamma * V[np.newaxis, :, np.newaxis]), axis=1
        )

        # DtControl trains on a mapping from observations to actions.
        # While DtControl handles non-deterministic policies (multiple
        # possible actions per observation) this increases the size of
        # the learned trees because internally all combinations of
        # actions are turned into separate actions. Therefore we only
        # give one optimal action per observation.
        optimal_actions = np.argmax(state_action_values, axis=1)

        tree = DecisionTree([AxisAlignedSplittingStrategy()], Entropy(), 'CART')
        dataset = SimpleDataset(mdp.observations, optimal_actions)

        tree.fit(dataset)

        self.tree_policy_ = _dtcontrol_tree_to_omdt(tree)

    def act(self, obs):
        return self.tree_policy_.act(obs)

    def to_graphviz(
        self,
        feature_names,
        action_names,
        integer_features,
        colors=None,
        fontname="helvetica",
    ):
        return self.tree_policy_.to_graphviz(
            feature_names,
            action_names,
            integer_features,
            colors,
            fontname,
        )
