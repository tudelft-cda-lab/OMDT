import sys

import numpy as np

from .determinizing_impurity_measure import DeterminizingImpurityMeasure


class TwoingRule(DeterminizingImpurityMeasure):
    def calculate_impurity(self, dataset, split):
        if len(split.get_masks(dataset)) == 1:
            return sys.maxsize
        assert len(split.get_masks(dataset)) == 2
        [left_mask, right_mask] = split.get_masks(dataset)
        left = self.determinizer.determinize(dataset.from_mask_optimized(left_mask))
        right = self.determinizer.determinize(dataset.from_mask_optimized(right_mask))
        if len(left) == 0 or len(right) == 0:
            return sys.maxsize
        num_labels = len(dataset)
        twoing_value = (len(left) / num_labels) * (len(right) / num_labels)
        s = 0
        unique = np.unique(np.append(left, right))
        for u in unique:
            num_left = len(left[left == u])
            num_right = len(right[right == u])
            s += abs(num_left / len(left) - num_right / len(right))
        twoing_value *= s * s
        if twoing_value == 0:
            return sys.maxsize
        return 1 / twoing_value

    def get_oc1_name(self):
        return 'twoing'
