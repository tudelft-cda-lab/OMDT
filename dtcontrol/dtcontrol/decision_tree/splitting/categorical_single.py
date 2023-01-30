from .split import Split
from .splitting_strategy import SplittingStrategy


class CategoricalSingleSplittingStrategy(SplittingStrategy):
    def find_split(self, dataset, impurity_measure, **kwargs):
        x_categorical = dataset.get_categorical_x()
        splits = {}
        for feature in range(x_categorical.shape[1]):
            real_feature = dataset.map_categorical_feature_back(feature)
            for value in set(x_categorical[:, feature]):
                split = CategoricalSingleSplit(real_feature, value)
                splits[split] = impurity_measure.calculate_impurity(dataset, split)

        if not splits:
            return None
        return min(splits.keys(), key=splits.get)


class CategoricalSingleSplit(Split):
    """
    A split of the form feature == value.
    """

    def __init__(self, feature, value):
        super().__init__()
        self.feature = feature
        self.value = value

    def predict(self, features):
        v = features[:, self.feature][0]
        return 0 if v == self.value else 1

    def get_masks(self, dataset):
        mask = dataset.x[:, self.feature] == self.value
        return [mask, ~mask]

    def print_dot(self, variables=None, category_names=None):
        if variables:
            var = variables[self.feature]
        else:
            var = f'x[{self.feature}]'
        if category_names and self.feature in category_names:
            val = category_names[self.feature][self.value]
        else:
            val = self.value
        return f'{var} == {val}'

    def print_c(self):
        return f'x[{self.feature}] == {self.value}'

    def print_vhdl(self):
        return f'x{self.feature}] == {self.value}'

    def to_json_dict(self, variables=None, category_names=None):
        return {
            "lhs":
                {"coeff": 1, "var": variables[self.feature] if variables else self.feature},
            "op": "==",
            "rhs": category_names[self.feature][self.value] if category_names and self.feature in category_names else self.value
        }
