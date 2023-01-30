from abc import ABC

import numpy as np

from .split import Split


class LinearSplit(Split, ABC):
    """
    Represents a linear split of the form wTx + b <= 0.
    """

    def __init__(self, coefficients, intercept, real_coefficients, numeric_columns):
        """
        :param coefficients: the coefficients with respect to the numeric columns
        :param intercept: the intercept
        :param real_coefficients: the coefficients with respect to all columns
        :param numeric_columns: a list of integers defining the numeric columns
        """
        super().__init__()
        self.coefficients = coefficients
        self.intercept = intercept
        self.real_coefficients = real_coefficients
        self.numeric_columns = numeric_columns

    @staticmethod
    def map_numeric_coefficients_back(numeric_coefficients, dataset):
        dim = dataset.x.shape[1]
        new_coefficients = [0] * dim
        for i in range(len(numeric_coefficients)):
            new_coefficients[dataset.map_numeric_feature_back(i)] = numeric_coefficients[i]
        return np.array(new_coefficients)

    def get_masks(self, dataset):
        mask = np.dot(dataset.get_numeric_x(), self.coefficients) + self.intercept <= 0
        return [mask, ~mask]

    def predict(self, features):
        return 0 if np.dot(features[:, self.numeric_columns], self.coefficients) + self.intercept <= 0 else 1

    def print_dot(self, variables=None, category_names=None):
        return self.get_hyperplane_str(rounded=True, newlines=True, variables=variables)

    def print_c(self):
        return self.get_hyperplane_str()

    def print_vhdl(self):
        hyperplane = self.get_hyperplane_str()
        hyperplane.replace('[', '')
        hyperplane.replace(']', '')
        return hyperplane

    def get_hyperplane_str(self, rounded=False, newlines=False, variables=None):
        line = []
        for i in range(len(self.real_coefficients)):
            if self.real_coefficients[i] == 0:
                continue
            coefficient = round(self.real_coefficients[i], 6) if rounded else self.real_coefficients[i]
            variable = variables[i] if variables else f'x[{i}]'
            line.append(f"{coefficient}*{variable}")
        line.append(f"{round(self.intercept, 6) if rounded else self.intercept}")
        joiner = "\\n+" if newlines else "+"
        hyperplane = joiner.join(line) + " <= 0"
        return hyperplane.replace('+-', '-')

    def to_json_dict(self, rounded=False, variables=None, **kwargs):
        lhs = []
        for i in range(len(self.real_coefficients)):
            if self.real_coefficients[i] == 0:
                continue
            coefficient = round(self.real_coefficients[i], 6) if rounded else self.real_coefficients[i]
            lhs.append({"coeff": coefficient, "var": variables[i] if variables else i})
        lhs.append({"intercept": round(self.intercept, 6) if rounded else self.intercept})
        return {
            "lhs": lhs,
            "op": "<=",
            "rhs": 0}
