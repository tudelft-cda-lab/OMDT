import copy
import sys

import numpy as np

from .split import Split
from .splitting_strategy import SplittingStrategy


class CategoricalMultiSplittingStrategy(SplittingStrategy):
    def __init__(self, value_grouping=False, tolerance=1e-5):
        """
        Implements splitting on a single categorical feature, with possibly multiple branches.
        :param value_grouping: if True, tries to merge different branches using the attribute value grouping heuristic
        :param tolerance: the absolute increase in impurity measure a value grouping may produce in order to still be
        considered a better candidate than the original (non-grouped) split
        """
        super().__init__()
        self.value_grouping = value_grouping
        self.tolerance = tolerance

    def find_split(self, dataset, impurity_measure, **kwargs):
        x_categorical = dataset.get_categorical_x()
        splits = {}
        for feature in range(x_categorical.shape[1]):
            real_feature = dataset.map_categorical_feature_back(feature)
            split = CategoricalMultiSplit(real_feature)
            impurity = impurity_measure.calculate_impurity(dataset, split)
            if impurity == sys.maxsize:
                continue

            if self.value_grouping:
                value_groups, grouped_impurity = \
                    self.find_best_value_groups(dataset, impurity_measure, feature, impurity)
                grouped_split = CategoricalMultiSplit(real_feature, value_groups)
                splits[grouped_split] = grouped_impurity
            else:
                splits[split] = impurity

        if not splits:
            return None
        return min(splits.keys(), key=splits.get)

    def find_best_value_groups(self, dataset, impurity_measure, feature, initial_impurity):
        impurity = initial_impurity
        real_feature = dataset.map_categorical_feature_back(feature)

        values = sorted(set(dataset.get_categorical_x()[:, feature]))
        value_groups = [[v] for v in values]
        best_new_value_groups = value_groups
        best_new_impurity = impurity
        while best_new_impurity <= impurity or abs(best_new_impurity - impurity) <= self.tolerance:
            impurity = best_new_impurity
            value_groups = best_new_value_groups
            if len(value_groups) == 2:
                break
            best_new_impurity = sys.maxsize
            for i in range(len(value_groups)):
                for j in range(i + 1, len(value_groups)):
                    new_groups = copy.deepcopy(value_groups)
                    new_groups[i] += new_groups[j]
                    del new_groups[j]
                    new_split = CategoricalMultiSplit(real_feature, new_groups)
                    new_impurity = impurity_measure.calculate_impurity(dataset, new_split)
                    if new_impurity <= best_new_impurity:
                        best_new_impurity = new_impurity
                        best_new_value_groups = new_groups
        return value_groups, impurity


class CategoricalMultiSplit(Split):
    def __init__(self, feature, value_groups=None):
        super().__init__()
        self.feature = feature
        self.value_groups = value_groups
        if not self.value_groups:
            self.value_groups = []

    def predict(self, features):
        v = features[:, self.feature][0]
        for i in range(len(self.value_groups)):
            if v in self.value_groups[i]:
                return i
        assert False

    def get_masks(self, dataset):
        if not self.value_groups:
            self.value_groups = [[v] for v in sorted(set(dataset.x[:, self.feature]))]
        masks = []
        for group in self.value_groups:
            mask = np.zeros(len(dataset.x), dtype=bool)
            for v in group:
                mask |= dataset.x[:, self.feature] == v
            masks.append(mask)
        return masks

    def print_dot(self, variables=None, category_names=None):
        if variables:
            return variables[self.feature]
        return f'x[{self.feature}]'

    def print_c(self):
        return f'x[{self.feature}]'

    def print_vhdl(self):
        return f'x{self.feature}]'

    def to_json_dict(self, variables=None, **kwargs):
        return {
            "lhs":
                {"coeff": 1, "var": variables[self.feature] if variables else self.feature},
            "op": "multi",
            "rhs": ""}
