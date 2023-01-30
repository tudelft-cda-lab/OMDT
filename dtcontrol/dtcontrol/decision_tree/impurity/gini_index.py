import sys

import numpy as np

from dtcontrol.decision_tree.impurity.determinizing_impurity_measure import (
    DeterminizingImpurityMeasure,
)


class GiniIndex(DeterminizingImpurityMeasure):
    def calculate_impurity(self, dataset, split):
        if len(split.get_masks(dataset)) == 1:
            return sys.maxsize
        impurity = 0
        for mask in split.get_masks(dataset):
            subset_labels = self.determinizer.determinize(dataset.from_mask_optimized(mask))
            if len(subset_labels) == 0:
                return sys.maxsize
            impurity += (len(subset_labels) / len(dataset)) * self.calculate_gini_index(subset_labels)
        return impurity

    @staticmethod
    def calculate_gini_index(y):
        num_labels = len(y)
        unique = np.unique(y)
        probabilities = [len(y[y == label]) / num_labels for label in unique]
        return 1 - sum(prob * prob for prob in probabilities)

    def get_oc1_name(self):
        return 'gini_index'
