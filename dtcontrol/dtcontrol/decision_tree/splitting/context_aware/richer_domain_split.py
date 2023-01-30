import uuid
import warnings
from copy import deepcopy
from itertools import product

import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

from dtcontrol.decision_tree.splitting.context_aware.richer_domain_exceptions import (
    RicherDomainSplitException,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_logger import (
    RicherDomainLogger,
)
from dtcontrol.decision_tree.splitting.split import Split

# For tree edit distance functionality --> 6605ba8c0f9231784a0c45d682ddbcbfb4897b1c


class RicherDomainSplit(Split):
    """
    e.g.
    c_1 * x_1 - c_2 + x_2 - c_3  <= 0; x_2 in {1,2,3}; c_1 in (-inf, inf); c_2 in {1,2,3}; c_3 in {5, 10, 32, 40}

        column_interval     =       {x_1:(-Inf,Inf), x_2:{1,2,3}}                           --> Key: Sympy Symbol Value: Sympy Interval
        coef_interval       =       {c_1:(-Inf,Inf), c_2:{1,2,3}, c_3:{5,10,32,40}          --> Key: Sympy Symbol Value: Sympy Interval
        term                =       c_1 * x_1 - c_2 + x_2 - c_3                             --> sympy expression
        relation            =       '<='                                                    --> String

        Every symbol without a specific defined Interval will be assigned to the interval: (-Inf, Inf)

        coef_assignment     =       [(c_1,-8.23), (c_2,2), (c_3,40)]                  --> List containing substitution Tuples (Sympy Symbol, Value)
        It will be determined inside fit() and later used inside predict() (and get_masks())
        It describes a specific assignment of all variables to a value inside their interval in order to achieve the lowest impurity.
    """

    def __init__(self, column_interval, coef_interval, term, relation, debug=False, priority=1):
        self.priority = priority
        self.column_interval = column_interval
        self.coef_interval = coef_interval
        self.term = term
        self.relation = relation
        self.coef_assignment = None

        # Helper attributes used to speedup calculations inside fit()
        self.y = None
        self.coef_fit = None
        self.coefs_to_determine = None

        # Helper attributes used to speedup get_mask()
        self.get_mask_lookup = None

        # logger
        self.logger = RicherDomainLogger("RicherDomainSplit_logger", debug)

        # id
        self.id = uuid.uuid4()

    def __repr__(self):
        return "RicherDomainSplit: " + str(self.term) + " " + str(self.relation) + " 0"

    def helper_str(self):
        return str(self.term) + " " + str(self.relation) + " 0"

    def helper_equal(self, obj1):
        return isinstance(obj1, RicherDomainSplit) and obj1.column_interval == self.column_interval \
               and obj1.coef_interval == self.coef_interval \
               and obj1.term == self.term and obj1.relation == self.relation

    def get_fixed_coef_combinations(self):

        """
        Returns every combination of already fixed coefs:
        Example:

        Split: c_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3+c_4 <= 0;c_1 in {1,2,3}; c_2 in {-1,-3}

        -->         combinations = [[('c_1', 1), ('c_2', -3)], [('c_1', 1), ('c_2', -1)],
                                    [('c_1', 2), ('c_2', -3)], [('c_1', 2), ('c_2', -1)],
                                    [('c_1', 3), ('c_2', -3)], [('c_1', 3), ('c_2', -1)]]

        --> The other coefs (c_0, c_3, c_4) still have to be determined by fit (curve_fit)

        """
        fixed_coefs = {}
        # Checking if coef_interval is containing finite sets with fixed coefs
        for coef in self.coef_interval:
            if isinstance(self.coef_interval[coef], sp.FiniteSet):
                fixed_coefs[coef] = list(self.coef_interval[coef].args)

        # Creating all combinations
        if fixed_coefs:
            # unzipping
            coef, val = zip(*fixed_coefs.items())
            # calculation all combinations and zipping back together
            combinations = [list(zip(coef, nbr)) for nbr in product(*val)]
        else:
            combinations = [[]]

        return combinations

    def contains_unfixed_coefs(self):
        """

        Returns whether self contains unfixed coefs.
        Example:

        Split: c_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3+c_4 <= 0;c_1 in {1,2,3}; c_2 in {-1,-3}
            --> c_0, c_3, c_4 are unfixed --> True

        Split: c_1 + x_0 + c_2 <= 0;c_1 in {1,2,3}; c_2 in {-1,-3}
            --> no unfixed coefs --> False

        Intention of this function is to decide whether the fit function has to be applied or not.

        """
        # No coefs at all
        if not self.coef_interval:
            return False

        for coef in self.coef_interval:
            if not isinstance(self.coef_interval[coef], sp.FiniteSet):
                return True
        return False

    def fit(self, fixed_coefs, x, y, method="lm"):
        """
        determines the best values for every coefficient(key) inside coef_interval(dict), within the range of their interval(value)
        :param fixed_coefs: Substitution list of tuples containing already determined coef values [(c_1, 2.5), ... ]
        :param x: feature columns of a dataset
        :param y: labels of a dataset
        :param method: {‘lm’, ‘trf’, ‘dogbox’, 'optimized'} -> method used inside curve_fit()
        """
        self.logger.root_logger.info("Started fitting coef predicate: {}".format(str(self)))

        # Edge Case no coefs or no unfixed coefs used in the term
        if not self.contains_unfixed_coefs():
            self.logger.root_logger.info("Finished fitting. Predicate does not contain unfixed coefs.")
            return

        # Checking type & shape of arguments
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or x.shape[0] <= 0 or x.shape[0] != y.shape[0] or not isinstance(
                fixed_coefs, list):
            self.logger.root_logger.critical("Aborting: invalid structure of the arguments x, y.")
            raise RicherDomainSplitException("Aborting: invalid structure of arguments x, y. Check logger or comments for more information.")

        # Checking structure of fixed_coefs
        self.coefs_to_determine = sorted(self.coef_interval, key=lambda x: int(str(x).split("_")[1]))
        for (c_i, _) in fixed_coefs:
            if c_i not in self.coef_interval:
                # Checking if fixed_coefs are valid (every fixed coef must appear inside coef_interval)
                self.logger.root_logger.critical("Aborting: invalid fixed_coefs member found. (Does not appear inside coef_interval)")
                raise RicherDomainSplitException("Aborting: invalid fixed_coefs member found. Check logger or comments for more information.")
            else:
                # Calculate coefs to determine with curve_fit
                self.coefs_to_determine.remove(c_i)
        if not self.coefs_to_determine:
            self.coef_assignment = fixed_coefs
            return

        # Predicate was already fitted.
        if self.coef_assignment is not None:
            self.logger.root_logger.critical("Aborting: predicate was already fitted")
            raise RicherDomainSplitException("Aborting: predicate was already fitted. Check logger or comments for more information.")

        # Method checking
        if not (method == "optimized" or method == "lm" or method == "trf" or method == "dogbox"):
            self.logger.root_logger.critical("Aborting: invalid curve fitting method.")
            raise RicherDomainSplitException(
                "Aborting: invalid curve fitting method. Check logger or comments for more information.")

        if method == "optimized":
            if x.shape[0] < len(self.coefs_to_determine):
                method = 'trf'
            else:
                method = 'lm'

        term_copy = deepcopy(self.term)

        # Substitution of already fixed coefs in Term (important to improve performance)
        if fixed_coefs:
            self.term = self.term.subs(fixed_coefs)

        # initial guess is very important since otherwise, curve_fit doesn't know how many coefs to fit
        inital_guess = [1. for coef in self.coefs_to_determine]

        # Values that will be calculated later on
        self.y = None
        self.coef_fit = None

        # adapter function representing the term (for curve_fit usage)
        def adapter_function(x, *args):
            out = []
            subs_list = []

            for i in range(len(args)):
                subs_list.append((self.coefs_to_determine[i], args[i]))
            new_term = self.term.subs(subs_list)

            args = sorted(new_term.free_symbols, key=lambda x: int(str(x).split("_")[1]))
            func = sp.lambdify(args, new_term)
            used_args_index = [int(str(i).split("_")[1]) for i in args]
            data = x[:, used_args_index]

            for row in data:
                result = func(*row)
                out.append(result)

            self.y = out
            self.coef_fit = subs_list
            for index in range(len(out)):
                # Checking the offset
                if self.relation == "<=":
                    if not ((out[index] <= 0 and y[index] <= 0) or (out[index] > 0 and y[index] > 0)):
                        return np.array(out)
                elif self.relation == ">=":
                    if not ((out[index] >= 0 and y[index] >= 0) or (out[index] < 0 and y[index] < 0)):
                        return np.array(out)
                elif self.relation == ">":
                    if not ((out[index] > 0 and y[index] > 0) or (out[index] <= 0 and y[index] <= 0)):
                        return np.array(out)
                elif self.relation == "<":
                    if not ((out[index] < 0 and y[index] < 0) or (out[index] >= 0 and y[index] >= 0)):
                        return np.array(out)
                elif self.relation == "=":
                    if not ((out[index] == 0 and y[index] == 0) or (out[index] != 0 and y[index] != 0)):
                        return np.array(out)
                else:
                    self.logger.root_logger.critical("Aborting: invalid relation found.")
                    raise RicherDomainSplitException(
                        "Aborting: Split with invalid relation can not be fitted. Check logger or comments for more information.")

            # For optimization reasons, once the first solution was found (with right accuracy), the loop should end.
            raise Exception('ALREADY FOUND A FIT!')

        # Ignoring warnings, since for our purpose a failed fit can still be useful
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                calculated_coefs, cov = curve_fit(adapter_function, x, y, inital_guess, method=method)
            except Exception:
                # Even if the curve_fit fails, it may have still passed some useful information to self.y or self.coef_fit before stopping.
                pass

        # Calculations done --> Assigning calculated coefs
        if self.y is not None and self.coef_fit is not None:
            self.coef_fit.extend(fixed_coefs)
            self.coef_assignment = self.coef_fit
            self.term = term_copy
            self.logger.root_logger.info("Fitting done. Result: {}".format(str(self.coef_assignment)))
        else:
            self.logger.root_logger.info("No fit found for {}".format(str(self.coef_assignment)))

    def check_valid_column_reference(self, x):
        """
        :param x: the dataset to be split
        :return: boolean

        Checks whether used column reference index is existing or not.
            e.g.
            x_5 - c_0 >= 12; x_5 in {1,2,3}
            column_interval = {x_5:{1,2,3}}
            If the dataset got k columns with k > 5 --> True
            If the dataset got k columns with k <= 5 --> False
        """

        allowed_var_index = x.shape[1] - 1
        sorted_column_refs = sorted(set(self.column_interval), key=lambda x: int(str(x).split("_")[1]))
        highest_index = int(str(sorted_column_refs[-1]).split("x_")[1])
        return not highest_index > allowed_var_index

    def check_data_in_column_interval(self, x):
        """
        :param x: the dataset to be split
        :return: boolean

        Checks if the column intervals, contain all of the values inside a column.
            e.g.
            column_interval = {x_2:{1,3}} --> all values from the third column must be inside {1,3}
        """
        for column_reference in self.column_interval:
            interval = self.column_interval.get(column_reference)
            if interval != sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity):
                index = int(str(column_reference).split("x_")[1])
                column = x[:, index]
                for val in column:
                    if not interval.contains(val):
                        return False
        return True

    def check_offset(self, offset):
        """
        Checking result of a term.
        :param offset: value to be compared with.
        :return: boolean

            e.g.
            5       <=          0
            Offset  Relation    Term

            --> returns False
        """

        # Checking the offset
        if self.relation == "<=":
            check = offset <= 0
        elif self.relation == ">=":
            check = offset >= 0
        elif self.relation == ">":
            check = offset > 0
        elif self.relation == "<":
            check = offset < 0
        elif self.relation == "=":
            check = offset == 0
        else:
            self.logger.root_logger.critical("Aborting: invalid relation found from inside check_offset.")
            raise RicherDomainSplitException("Aborting: Invalid relation found. Check logger or comments for more information.")
        return check

    def predict(self, features):
        """
        Determines the child index of the split for one particular instance.
        :param features: the features of the instance
        :returns: the child index (0/1 for a binary split)
        """

        term = self.term.subs(self.coef_assignment) if self.coef_assignment is not None else self.term

        subs_list = []
        # Iterating over every possible value and creating a substitution list
        for i in range(len(features[0, :])):
            subs_list.append(("x_" + str(i), features[0, i]))

        result = term.subs(subs_list).evalf()
        return 0 if self.check_offset(result) else 1

    def get_masks(self, dataset):
        """
        Returns the masks specifying this split.
        :param dataset: the dataset to be split
        :return: a list of the masks corresponding to each subset after the split
        """

        mask = []
        if self.get_mask_lookup is not None:
            return self.get_mask_lookup
        elif self.y is not None:
            for result in self.y:
                mask.append(self.check_offset(result))
        else:
            term = self.term.subs(self.coef_assignment) if self.coef_assignment is not None else self.term

            args = sorted(term.free_symbols, key=lambda x: int(str(x).split("_")[1]))
            func = sp.lambdify(args, term)
            # Prepare dataset for required args
            data = dataset.get_numeric_x()
            used_args_index = [int(str(i).split("_")[1]) for i in args]
            data_filtered = data[:, used_args_index]

            for row in data_filtered:
                result = func(*row)
                mask.append(self.check_offset(result))

        mask = np.array(mask)
        self.get_mask_lookup = [mask, ~mask]
        return [mask, ~mask]

    def print_dot(self, variables=None, category_names=None):
        subs_list = self.coef_assignment if self.coef_assignment else []
        evaluated_predicate = str(self.term.subs(subs_list).evalf(5))

        sliced_term = []
        for i in range(0, len(evaluated_predicate), 20):
            sliced_term.append(evaluated_predicate[i:i + 20])

        for single_slice in sliced_term[1:-1]:
            if single_slice.__contains__(" + "):
                formated_slice = single_slice.replace(" + ", " + \\n", 1)
                sliced_term[sliced_term.index(single_slice)] = formated_slice
            elif single_slice.__contains__(" - "):
                formated_slice = single_slice.replace(" - ", " - \\n", 1)
                sliced_term[sliced_term.index(single_slice)] = formated_slice
            elif single_slice.__contains__(" * "):
                formated_slice = single_slice.replace(" * ", " * \\n", 1)
                sliced_term[sliced_term.index(single_slice)] = formated_slice

        out = "".join(single_slice for single_slice in sliced_term)
        return out + " \\n" + self.relation + " 0"

    def print_c(self):
        subs_list = self.coef_assignment if self.coef_assignment else []
        evaluated_predicate = self.term.subs(subs_list).evalf(5)
        return str(sp.ccode(evaluated_predicate)) + " " + self.relation + " 0"

    def print_vhdl(self):
        # TODO
        return self.print_dot()

    def to_json_dict(self, rounded=False, variables=None, **kwargs):
        subs_list = self.coef_assignment if self.coef_assignment else []
        evaluated_predicate = self.term.subs(subs_list).evalf(5)
        return {
            "lhs": evaluated_predicate,
            "op": self.relation,
            "rhs": 0}
