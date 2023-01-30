import sys

import numpy as np

from dtcontrol.decision_tree.impurity.multi_label_impurity_measure import (
    MultiLabelImpurityMeasure,
)


class MultiLabelTwoingRule(MultiLabelImpurityMeasure):
    def calculate_impurity(self, dataset, split):
        if len(split.get_masks(dataset)) == 1:
            return sys.maxsize
        assert len(split.get_masks(dataset)) == 2
        y = dataset.get_single_labels()

        [left_mask, right_mask] = split.get_masks(dataset)
        left = y[left_mask]
        left_flat = left.flatten()
        left_flat = left_flat[left_flat != -1]
        right = y[right_mask]
        right_flat = right.flatten()
        right_flat = right_flat[right_flat != -1]
        if len(left) == 0 or len(right) == 0:
            return sys.maxsize
        num_labels = len(y)
        twoing_value = (len(left) / num_labels) * (len(right) / num_labels)

        left_index = -1
        left_counts = np.bincount(left_flat)[1:]
        r = np.where(left_counts == len(left))
        if len(r[0]) > 0:
            left_index = r[0][0] + 1

        right_index = -1
        right_counts = np.bincount(right_flat)[1:]
        r = np.where(right_counts == len(right))
        if len(r[0]) > 0:
            right_index = r[0][0] + 1

        s = 0
        y_flat = y.flatten()
        y_flat = y_flat[y_flat != -1]
        unique = np.unique(y_flat)
        for u in unique:
            num_left = len(left_flat[left_flat == u]) if left_index == -1 else (len(left) if u == left_index else 0)
            num_right = len(right_flat[right_flat == u]) if right_index == -1 else (len(right) if u == right_index else 0)
            s += abs(num_left / len(left) - num_right / len(right))
        twoing_value *= s * s
        if twoing_value == 0:
            return sys.maxsize
        return 1 / twoing_value
