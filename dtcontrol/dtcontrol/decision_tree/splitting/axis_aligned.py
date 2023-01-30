from .split import Split
from .splitting_strategy import SplittingStrategy


class AxisAlignedSplittingStrategy(SplittingStrategy):
    def find_split(self, dataset, impurity_measure, **kwargs):
        x_numeric = dataset.get_numeric_x()
        splits = {}
        for feature in range(x_numeric.shape[1]):
            values = sorted(set(x_numeric[:, feature]))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                real_feature = dataset.map_numeric_feature_back(feature)
                split = AxisAlignedSplit(real_feature, threshold, self.priority)
                splits[split] = impurity_measure.calculate_impurity(dataset, split)
        if not splits:
            return None
        return min(splits.keys(), key=splits.get)


class AxisAlignedSplit(Split):
    """
    Represents an axis aligned split of the form x[i] <= b.
    """

    def __init__(self, feature, threshold, priority=1):
        super().__init__()
        self.feature = feature
        self.threshold = threshold
        self.priority = priority

    def get_masks(self, dataset):
        mask = dataset.x[:, self.feature] <= self.threshold
        return [mask, ~mask]

    def predict(self, features):
        return 0 if features[:, self.feature][0] <= self.threshold else 1

    def print_dot(self, variables=None, category_names=None):
        if variables:
            return f'{variables[self.feature]} <= {round(self.threshold, 6)}'
        return self.print_c()

    def print_c(self):
        return f'x[{self.feature}] <= {round(self.threshold, 6)}'

    def print_vhdl(self):
        return f'x{self.feature} <= {round(self.threshold, 6)}'

    def to_json_dict(self, rounded=False, variables=None, **kwargs):
        return {
            "lhs":
                {"coeff": 1, "var": variables[self.feature] if variables else self.feature},
            "op": "<=",
            "rhs": round(self.threshold, 6) if rounded else self.threshold}
