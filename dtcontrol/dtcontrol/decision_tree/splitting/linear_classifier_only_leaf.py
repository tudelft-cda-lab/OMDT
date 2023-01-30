import numpy as np

from dtcontrol.decision_tree.determinization.label_powerset_determinizer import (
    LabelPowersetDeterminizer,
)
from dtcontrol.decision_tree.splitting.linear_classifier import LinearClassifierSplit
from dtcontrol.decision_tree.splitting.linear_split import LinearSplit
from dtcontrol.decision_tree.splitting.splitting_strategy import SplittingStrategy


class LinearClassifierOnlyLeafSplittingStrategy(SplittingStrategy):
    def __init__(self, classifier_class, determinizer=LabelPowersetDeterminizer(), **kwargs):
        super().__init__()
        self.determinizer = determinizer
        self.classifier_class = classifier_class
        self.kwargs = kwargs

    def find_split(self, dataset, impurity_measure, **kwargs):
        x_numeric = dataset.get_numeric_x()
        if x_numeric.shape[1] == 0:
            return None
        y = self.determinizer.determinize(dataset)
        if not self.is_binary(y):  # otherwise we certainly won't have pure leaves
            return None

        label = y[0]
        new_y = np.copy(y)
        label_mask = (new_y == label)
        new_y[label_mask] = 1
        new_y[~label_mask] = -1
        classifier = self.classifier_class(**self.kwargs)
        classifier.fit(x_numeric, new_y)

        if np.array_equal(classifier.predict(x_numeric), new_y):  # perfect split
            real_features = LinearSplit.map_numeric_coefficients_back(classifier.coef_[0], dataset)
            split = LinearClassifierSplit(classifier, real_features, dataset.numeric_columns)
            assert impurity_measure.calculate_impurity(dataset, split) == 0
            return split
        return None

    @staticmethod
    def is_binary(y):
        return len(np.unique(y)) == 2
