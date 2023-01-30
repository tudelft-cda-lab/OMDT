from dtcontrol.decision_tree.impurity.multi_label_impurity_measure import (
    MultiLabelImpurityMeasure,
)
from dtcontrol.decision_tree.impurity.scaled_bincount import ScaledBincount


class MultiLabelGiniIndex(MultiLabelImpurityMeasure):

    def __init__(self):
        self.scaled_bincount = ScaledBincount(self.scaling_function)

    @staticmethod
    def scaling_function(x):
        return x**2

    def calculate_impurity(self, dataset, split):
        return self.scaled_bincount.calculate_impurity(dataset, split)
