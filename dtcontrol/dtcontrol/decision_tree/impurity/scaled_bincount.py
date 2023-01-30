import sys

import numpy as np

from dtcontrol.decision_tree.impurity.multi_label_impurity_measure import (
    MultiLabelImpurityMeasure,
)


class ScaledBincount(MultiLabelImpurityMeasure):
    def __init__(self, scaling_function):
        self.scaling_function = scaling_function

    def calculate_impurity(self, dataset, split):
        if len(split.get_masks(dataset)) == 1:
            return sys.maxsize
        y = dataset.get_single_labels()
        impurity = 0
        for mask in split.get_masks(dataset):
            subset = y[mask, :]
            if len(subset) == 0:
                return sys.maxsize
            impurity += (len(subset) / len(y)) * self.calculate_scaled_bincount(subset)
        return impurity

    def calculate_scaled_bincount(self, y):
        flattened_labels = y.flatten()
        flattened_labels = flattened_labels[flattened_labels != -1]  # -1 is only a filler
        label_counts = np.bincount(flattened_labels)[1:]
        if np.any(label_counts == len(y)):
            return 0
        label_counts = label_counts[label_counts != 0].astype('float')
        label_counts = label_counts / len(y)
        return len(label_counts) - sum(self.scaling_function(label_counts))
