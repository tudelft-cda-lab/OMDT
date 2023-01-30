import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dtcontrol.decision_tree.impurity.determinizing_impurity_measure import (
    DeterminizingImpurityMeasure,
)


class AUROC(DeterminizingImpurityMeasure):
    def calculate_impurity(self, dataset, split):
        if len(split.get_masks(dataset)) == 1:
            return sys.maxsize
        scores = []
        for mask in split.get_masks(dataset):
            subset_labels = self.determinizer.determinize(dataset.from_mask_optimized(mask))
            scores.append(self.calculate_auroc(dataset.x[mask], subset_labels))
        if any([s == 0 for s in scores]):
            return sys.maxsize
        return 1 / sum(scores)

    @staticmethod
    def calculate_auroc(x, y):
        if len(y) == 0:
            return 0
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            return 1  # perfect split
        label_to_auroc = {}
        for label in unique_labels:
            new_y = np.copy(y)
            label_mask = (new_y == label)
            new_y[label_mask] = 1
            new_y[~label_mask] = -1
            logreg = LogisticRegression(solver='lbfgs', penalty='none')
            logreg.fit(x, new_y)
            auroc = roc_auc_score(new_y, logreg.decision_function(x))
            label_to_auroc[label] = auroc
        weighted_avg = sum([auroc * (len(y[y == label]) / len(y)) for label, auroc in label_to_auroc.items()])
        return weighted_avg

    def get_oc1_name(self):
        return None
