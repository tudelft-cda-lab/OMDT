import numpy as np

from dtcontrol.decision_tree.determinization.label_powerset_determinizer import (
    LabelPowersetDeterminizer,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_exceptions import (
    RicherDomainStrategyException,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_logger import (
    RicherDomainLogger,
)
from dtcontrol.decision_tree.splitting.linear_split import LinearSplit
from dtcontrol.decision_tree.splitting.splitting_strategy import SplittingStrategy


class LinearUnitsClassifier(SplittingStrategy):
    """
    Splitting Strategy is basically the same as in dtcontrol/decision_tree/splitting/linear_classifier.py, with the only difference,
    that the Linear expression will respect the units given in self.unit.

    self.unit is a List containing Strings with the specific units.
    Example:
        Units:
            ["meter", "kilogram" , "meter", "seconds"]
        (Caution: len(units) should be #columns_of_dataset)

    Procedure:
        0. Convert the dataset -> One for every unique unit
        1. Apply the basic linear classifier algorithm as described in linear_classifier.py
    """

    def __init__(self, classifier_class, units, debug=False, determinizer=LabelPowersetDeterminizer(), **kwargs):
        self.units = units
        self.determinizer = determinizer
        self.classifier_class = classifier_class
        self.kwargs = kwargs

        # logger
        self.logger = RicherDomainLogger("LinearUnitsClassifier_logger", debug)

    def find_split(self, dataset, impurity_measure, **kwargs):
        x_numeric = dataset.get_numeric_x()
        if x_numeric.shape[1] == 0:
            return None

        y = self.determinizer.determinize(dataset)
        splits = {}

        if x_numeric.shape[1] != len(self.units):
            self.logger.root_logger.critical(
                "Aborting: Invalid amount of given units. Please give one unit for every column. Given units: {}. Columns: {}.".format(
                    len(self.units), x_numeric.shape[1]))
            raise RicherDomainStrategyException("Aborting: Invalid amount of given units. Check logger or comments for more information.")

        self.logger.root_logger.info("Converting the dataset to every unique unit.")

        for unit in set(self.units):
            self.logger.root_logger.info("Starting to process linear expression for {}".format(unit))
            converted_x = np.copy(x_numeric)
            for index in range(len(self.units)):
                if self.units[index] != unit:
                    converted_x[:, index] = 0

            for label in np.unique(y):
                new_y = np.copy(y)
                label_mask = (new_y == label)
                new_y[label_mask] = 1
                new_y[~label_mask] = -1
                classifier = self.classifier_class(**self.kwargs)
                classifier.fit(converted_x, new_y)
                real_features = LinearSplit.map_numeric_coefficients_back(classifier.coef_[0], dataset)
                split = LinearClassifierSplit(classifier, real_features, dataset.numeric_columns, self.priority)
                splits[split] = impurity_measure.calculate_impurity(dataset, split)

            self.logger.root_logger.info("Finished linear expression for {}".format(unit))

        return min(splits.keys(), key=splits.get)


class LinearClassifierSplit(LinearSplit):
    def __init__(self, classifier, real_coefficients, numeric_columns, priority=1):
        super().__init__(classifier.coef_[0], classifier.intercept_[0], real_coefficients, numeric_columns)
        self.classifier = classifier
        self.numeric_columns = numeric_columns
        self.priority = priority

    def get_masks(self, dataset):
        mask = self.classifier.predict(dataset.get_numeric_x()) == -1
        return [mask, ~mask]

    def predict(self, features):
        return 0 if self.classifier.predict(features[:, self.numeric_columns])[0] == -1 else 1
