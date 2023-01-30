from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Tree:
    def __init__(self, root):
        self.root = root

    def prune(self):
        bounds = defaultdict(lambda: [float("-inf"), float("inf")])
        self.root = self.root.prune(bounds)

    def act(self, observation):
        return self.root.act(observation)

    def to_string(self, feature_names, action_names):
        return self.root.to_string(0, feature_names, action_names)

    def to_graphviz(
        self,
        feature_names,
        action_names,
        integer_features=None,
        colors=None,
        fontname="helvetica",
    ):
        # If no features are specified as integer then assume they are continuous.
        # this means that if you have integers and don't specify it splits will
        # be printed as <= 4.500 instead of <= 4
        if integer_features is None:
            integer_features = [False for _ in range(len(feature_names))]

        # If no colors are defined then create a default palette
        if colors is None:
            # Seaborn color blind palette
            palette = [
                "#0173b2",
                "#de8f05",
                "#029e73",
                "#d55e00",
                "#cc78bc",
                "#ca9161",
                "#fbafe4",
                "#949494",
                "#ece133",
                "#56b4e9",
            ]
            colors = []
            for i in range(len(action_names)):
                colors.append(palette[i % len(palette)])

        header = f"""digraph Tree {{
node [shape=box, style=\"filled, rounded\", color=\"gray\", fillcolor=\"white\" fontname=\"{fontname}\"] ;
edge [fontname=\"{fontname}\"] ;
"""

        body = self.root.to_graphviz(
            feature_names, action_names, integer_features, colors, 0
        )[0]

        footer = "}"

        return header + body.strip() + footer

    def count_nodes(self):
        return self.root.count_nodes()

    def count_depth(self):
        return self.root.count_depth()

    def __str__(self):
        return self.root.to_string(0, None, None)


class TreeNode:
    def __init__(self, feature, threshold, left_child, right_child):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child

    def act(self, observation):
        if observation[self.feature] <= self.threshold:
            return self.left_child.act(observation)
        else:
            return self.right_child.act(observation)

    def prune(self, bounds):
        old_bound = bounds[self.feature][1]
        bounds[self.feature][1] = self.threshold

        self.left_child = self.left_child.prune(bounds)

        bounds[self.feature][1] = old_bound

        old_bound = bounds[self.feature][0]
        bounds[self.feature][0] = self.threshold

        self.right_child = self.right_child.prune(bounds)

        bounds[self.feature][0] = old_bound

        if bounds[self.feature][0] > self.threshold:
            return self.right_child

        if bounds[self.feature][1] <= self.threshold:
            return self.left_child

        if isinstance(self.left_child, TreeLeaf) and isinstance(
            self.right_child, TreeLeaf
        ):
            if self.left_child.action == self.right_child.action:
                return self.left_child

        return self

    def to_string(self, depth, feature_names, action_names):
        left_string = self.left_child.to_string(depth + 1, feature_names, action_names)
        right_string = self.right_child.to_string(
            depth + 1, feature_names, action_names
        )
        padding = "    " * depth
        if feature_names is None:
            return f"{padding}if X[{self.feature}] <= {self.threshold}:\n{left_string}\n{padding}else:\n{right_string}"
        else:
            return f"{padding}if {feature_names[self.feature]} <= {self.threshold}:\n{left_string}\n{padding}else:\n{right_string}"

    def to_graphviz(
        self, feature_names, action_names, integer_features, colors, node_id
    ):
        left_id = node_id + 1
        left_dot, new_node_id = self.left_child.to_graphviz(
            feature_names, action_names, integer_features, colors, left_id
        )

        right_id = new_node_id + 1
        right_dot, new_node_id = self.right_child.to_graphviz(
            feature_names, action_names, integer_features, colors, right_id
        )

        if node_id == 0:
            edge_label_left = "yes"
            edge_label_right = "no"
        else:
            edge_label_left = ""
            edge_label_right = ""

        feature_name = feature_names[self.feature]

        if integer_features[self.feature]:
            split_condition = int(self.threshold)
        else:
            split_condition = f"{self.threshold:.3f}"

        predicate = f'{node_id} [label="if {feature_name} <= {split_condition}"] ;\n'
        yes = left_id
        no = right_id

        edge_left = (
            f'{node_id} -> {yes} [label="{edge_label_left}", fontcolor="gray"] ;\n'
        )
        edge_right = (
            f'{node_id} -> {no} [label="{edge_label_right}", fontcolor="gray"] ;\n'
        )

        return f"{predicate}{left_dot}{right_dot}{edge_left}{edge_right}", new_node_id

    def count_nodes(self):
        return 1 + self.left_child.count_nodes() + self.right_child.count_nodes()

    def count_depth(self):
        return 1 + max(self.left_child.count_depth(), self.right_child.count_depth())


class TreeLeaf:
    def __init__(self, action):
        self.action = action

    def act(self, _):
        return self.action

    def to_string(self, depth, _, action_names):
        padding = "    " * depth
        if action_names is None:
            return f"{padding}return '{self.action}'"
        else:
            return f"{padding}return '{action_names[self.action]}'"

    def to_graphviz(
        self, feature_names, action_names, integer_features, colors, node_id
    ):
        label = f"{action_names[self.action]}"
        color = colors[self.action]
        return (
            f'{node_id} [label="{label}", fillcolor="{color}", color="{color}", fontcolor=white] ;\n',
            node_id,
        )

    def prune(self, _):
        return self

    def count_nodes(self):
        return 0

    def count_depth(self):
        return 0


def sklearn_to_omdt_tree(tree: DecisionTreeClassifier):
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    value = tree.tree_.value

    def sklearn_to_omdt_tree_rec(node_i):
        if children_left[node_i] == children_right[node_i]:
            # If this is a leaf
            return TreeLeaf(np.argmax(value[node_i][0]))

        # If this is a node
        left_child = sklearn_to_omdt_tree_rec(children_left[node_i])
        right_child = sklearn_to_omdt_tree_rec(children_right[node_i])
        return TreeNode(
            feature[node_i],
            threshold[node_i],
            left_child,
            right_child,
        )

    return Tree(sklearn_to_omdt_tree_rec(0))
