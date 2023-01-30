import sys

import numpy as np

from dtcontrol.decision_tree.impurity.determinizing_impurity_measure import (
    DeterminizingImpurityMeasure,
)


class EntropyRatio(DeterminizingImpurityMeasure):
    def calculate_impurity(self, dataset, split):
        if any(np.all(mask == False) for mask in split.get_masks(dataset)) or \
                len(split.get_masks(dataset)) == 1:
            return sys.maxsize

        split_entropy = self.calculate_split_entropy(dataset, split)
        split_info = self.calculate_split_info(dataset, split)
        return split_entropy / split_info

    def calculate_split_entropy(self, dataset, split):
        entropy = 0
        for mask in split.get_masks(dataset):
            subset_labels = self.determinizer.determinize(dataset.from_mask_optimized(mask))
            entropy += (len(subset_labels) / len(dataset)) * EntropyRatio.calculate_entropy(subset_labels)
        assert entropy >= 0
        return entropy

    @staticmethod
    def calculate_entropy(y):
        num_labels = len(y)
        unique = np.unique(y)
        probabilities = [len(y[y == label]) / num_labels for label in unique]
        return sum(-prob * np.log2(prob) for prob in probabilities)

    def calculate_split_info(self, dataset, split):
        info = 0
        for mask in split.get_masks(dataset):
            subset_labels = self.determinizer.determinize(dataset.from_mask_optimized(mask))
            info -= (len(subset_labels) / len(dataset)) * np.log2((len(subset_labels) / len(dataset)))
        assert info > 0
        return info

    def get_oc1_name(self):
        return None
