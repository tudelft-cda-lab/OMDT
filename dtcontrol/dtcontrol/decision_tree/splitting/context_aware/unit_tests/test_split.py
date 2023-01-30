import unittest
from copy import deepcopy

import numpy as np
import sympy as sp
from hypothesis import given

from dtcontrol.decision_tree.splitting.context_aware.richer_domain_exceptions import (
    RicherDomainSplitException,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_split import (
    RicherDomainSplit,
)


class TestSplitCurveFit(unittest.TestCase):
    """
    Test cases for fit() inside RicherDomainSplit Objects (richer_domain_split.py)
    """
    data_x_1 = np.array(
        [[1., 4.6, 1., 3.],
         [1., 4.6, 2., 3.],
         [2., 4., 3., 1.],
         [2., 4., 3., 2.],
         [1., 4., 4., 1.],
         [2., 4., 4., 2.],
         [2., 53., 2., 3.],
         [1., 228., 1., 5.],
         [2., 93., 1., 2.],
         [2., 59., 3., 2.]])
    data_y_1 = np.array([-1, -1, 1, 1, 1, 1, -1, -1, -1, 1])

    data_x_2 = np.array(
        [[2., 4., 3., 1.],
         [2., 4., 3., 2.],
         [1., 4., 4., 1.],
         [2., 4., 4., 2.],
         [2., 59., 3., 2.]])
    data_y_2 = np.array([1, 1, 1, 1, -1])

    data_x_3 = np.array(
        [[1., 4.6, 1., 3.],
         [1., 4.6, 2., 3.],
         [2., 53., 2., 3.],
         [1., 228., 1., 5.],
         [2., 93., 1., 2.]])
    data_y_3 = np.array([1, 1, -1, -1, -1])

    x_0, x_1, x_2, x_3, x_4, x_5, x_6, c_0, c_1, c_2, c_3, c_4, c_5 = sp.symbols('x_0 x_1 x_2 x_3 x_4 x_5 x_6 c_0 c_1 c_2 c_3 c_4 c_5')

    # Split 1
    column_interval1 = {x_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity), x_1: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                        x_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity), x_3: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    coef_interval1 = {c_4: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity), c_1: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                      c_3: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity), c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                      c_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    term1 = c_0 * x_0 + c_1 * x_1 + c_2 * x_2 + c_3 * x_3 + c_4
    relation1 = "<="
    split1 = RicherDomainSplit(column_interval1, coef_interval1, term1, relation1)

    # Split 2
    column_interval2 = {x_4: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    coef_interval2 = {c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    term2 = c_0 * x_4
    relation2 = "<="
    split2 = RicherDomainSplit(column_interval2, coef_interval2, term2, relation2)

    # Split 3
    column_interval3 = {x_4: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity), x_5: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    coef_interval3 = {c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    term3 = c_0 * x_4 * x_5
    relation3 = "<="
    split3 = RicherDomainSplit(column_interval3, coef_interval3, term3, relation3)

    # Split 4
    column_interval4 = {x_6: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    coef_interval4 = {c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    term4 = c_0 + x_6
    relation4 = "<="
    split4 = RicherDomainSplit(column_interval4, coef_interval4, term4, relation4)

    # Split 5
    column_interval5 = {x_0: sp.FiniteSet(1.0, 2.0)}
    coef_interval5 = {c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    term5 = c_0 - x_0
    relation5 = "<="
    split5 = RicherDomainSplit(column_interval5, coef_interval5, term5, relation5)

    # Split 6
    column_interval6 = {x_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity), x_1: sp.FiniteSet(6.0, 12.0)}
    coef_interval6 = {c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    term6 = c_0 * x_0 * x_1
    relation6 = "<="
    split6 = RicherDomainSplit(column_interval6, coef_interval6, term6, relation6)

    # Split 7
    column_interval7 = {x_0: sp.FiniteSet(3.0)}
    coef_interval7 = {c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
    term7 = c_0 - x_0
    relation7 = "<="
    split7 = RicherDomainSplit(column_interval7, coef_interval7, term7, relation7)

    def helper_fit(self, split, x, y):
        copy_split = deepcopy(split)
        copy_split.fit([], x, y)
        return copy_split.coef_assignment

    def test_fit_linear(self):
        c_0, c_1, c_2, c_3, c_4, c_5 = sp.symbols('c_0 c_1 c_2 c_3 c_4 c_5')

        coef_assignment_1 = [(c_0, 0.1720005117441602),
                             (c_1, 0.003330672196933837),
                             (c_2, 0.7146818484482542),
                             (c_3, -0.2652606736991465),
                             (c_4, -1.5064267251464263)]
        self.assertEqual(self.helper_fit(deepcopy(self.split1), self.data_x_1, self.data_y_1), coef_assignment_1)

        coef_assignment_2 = [(c_0, -1.1921147180515845e-07),
                             (c_1, -0.03636369539912798),
                             (c_2, -2.1791457527342573e-12),
                             (c_3, -2.1946888750790094e-12),
                             (c_4, 1.1454543134203623)]
        self.assertEqual(self.helper_fit(deepcopy(self.split1), self.data_x_2, self.data_y_2), coef_assignment_2)

        coef_assignment_3 = [(c_0, -1.6809487527583804),
                             (c_1, -0.006592031186399394),
                             (c_2, -2.225997164373439e-12),
                             (c_3, -0.2636783674879648),
                             (c_4, 3.502306420885194)]
        self.assertEqual(self.helper_fit(deepcopy(self.split1), self.data_x_3, self.data_y_3), coef_assignment_3)

    def test_fit_invalid_edge_cases(self):
        split = deepcopy(self.split1)
        # Invalid argument types
        with self.assertRaises(RicherDomainSplitException):
            split.fit([], [], [])
        with self.assertRaises(RicherDomainSplitException):
            split.fit([],
                      [[1., 4.6, 1., 3.],
                       [1., 4.6, 2., 3.],
                       [2., 4., 3., 1.],
                       [2., 4., 3., 2.],
                       [1., 4., 4., 1.],
                       [2., 4., 4., 2.],
                       [2., 53., 2., 3.],
                       [1., 228., 1., 5.],
                       [2., 93., 1., 2.],
                       [2., 59., 3., 2.]], np.array([1, 1, -1, -1, -1, -1, -1, -1, -1, -1]))
        with self.assertRaises(RicherDomainSplitException):
            split.fit([], np.array(
                [[1., 4.6, 1., 3.],
                 [1., 4.6, 2., 3.],
                 [2., 4., 3., 1.],
                 [2., 4., 3., 2.],
                 [1., 4., 4., 1.],
                 [2., 4., 4., 2.],
                 [2., 53., 2., 3.],
                 [1., 228., 1., 5.],
                 [2., 93., 1., 2.],
                 [2., 59., 3., 2.]]), [1, 1, -1, -1, -1, -1, -1, -1, -1, -1])
        with self.assertRaises(RicherDomainSplitException):
            # Invalid shapes (x rows > y column)
            split.fit([], np.array(
                [[1., 4.6, 1., 3.],
                 [1., 4.6, 2., 3.],
                 [2., 4., 3., 1.],
                 [2., 4., 3., 2.],
                 [1., 4., 4., 1.],
                 [2., 4., 4., 2.],
                 [2., 53., 2., 3.],
                 [1., 228., 1., 5.],
                 [2., 93., 1., 2.]]), np.array([1, 1, -1, -1, -1, -1, -1, -1, -1, -1]))
        with self.assertRaises(RicherDomainSplitException):
            # Invalid shapes (x rows < y column)
            split.fit([], np.array(
                [[1., 4.6, 1., 3.],
                 [1., 4.6, 2., 3.],
                 [2., 4., 3., 1.],
                 [2., 4., 3., 2.],
                 [1., 4., 4., 1.],
                 [2., 4., 4., 2.],
                 [2., 53., 2., 3.],
                 [1., 228., 1., 5.],
                 [2., 93., 1., 2.],
                 [2., 59., 3., 2.]]), np.array([1, 1, -1, -1, -1, -1, -1, -1, -1]))
        with self.assertRaises(RicherDomainSplitException):
            # Invalid shapes (x rows > y column)
            split.fit([], np.array(
                [[1., 4.6, 1., 3.],
                 [1., 4.6, 2., 3.],
                 [1., 4.6, 1., 5.],
                 [2., 4., 3., 1.],
                 [2., 4., 3., 2.],
                 [1., 4., 4., 1.],
                 [2., 4., 4., 2.],
                 [2., 53., 2., 3.],
                 [1., 228., 1., 5.],
                 [2., 93., 1., 2.],
                 [2., 59., 3., 2.]]), np.array([1, 1, -1, -1, -1, -1, -1, -1, -1, -1]))

        # all coefs already fixed
        c_0, c_1, c_2, c_3, c_4, c_5 = sp.symbols('c_0 c_1 c_2 c_3 c_4 c_5')
        self.assertIsNone(split.fit([(c_1, 1), (c_2, -3), (c_0, -3), (c_3, -3), (c_4, -3)], np.array(
            [[1., 4.6, 1., 3.],
             [1., 4.6, 2., 3.],
             [2., 4., 3., 1.],
             [2., 4., 3., 2.],
             [1., 4., 4., 1.],
             [2., 4., 4., 2.],
             [2., 53., 2., 3.],
             [1., 228., 1., 5.],
             [2., 93., 1., 2.],
             [2., 59., 3., 2.]]), np.array([1, 1, -1, -1, -1, -1, -1, -1, -1, -1])))

        with self.assertRaises(RicherDomainSplitException):
            # Too many fixed coefs
            split.fit([(c_1, 1), (c_2, -3), (c_0, -3), (c_3, -3), (c_4, -3), (c_5, 0)], np.array(
                [[1., 4.6, 1., 3.],
                 [1., 4.6, 2., 3.],
                 [2., 4., 3., 1.],
                 [2., 4., 3., 2.],
                 [1., 4., 4., 1.],
                 [2., 4., 4., 2.],
                 [2., 53., 2., 3.],
                 [1., 228., 1., 5.],
                 [2., 93., 1., 2.],
                 [2., 59., 3., 2.]]), np.array([1, 1, -1, -1, -1, -1, -1, -1, -1, -1]))


        # Invalid attribute configuration
        split.coef_interval = None
        self.assertIsNone(split.fit([], np.array(
            [[1., 4.6, 1., 3.],
             [1., 4.6, 2., 3.],
             [2., 4., 3., 1.],
             [2., 4., 3., 2.],
             [1., 4., 4., 1.],
             [2., 4., 4., 2.],
             [2., 53., 2., 3.],
             [1., 228., 1., 5.],
             [2., 93., 1., 2.],
             [2., 59., 3., 2.]]), np.array([1, 1, -1, -1, -1, -1, -1, -1, -1, -1])))

    def test_check_valid_column_reference(self):
        self.assertTrue(deepcopy(self.split1).check_valid_column_reference(self.data_x_1))
        self.assertTrue(deepcopy(self.split1).check_valid_column_reference(self.data_x_2))
        self.assertTrue(deepcopy(self.split1).check_valid_column_reference(self.data_x_3))

        self.assertFalse(deepcopy(self.split2).check_valid_column_reference(self.data_x_1))
        self.assertFalse(deepcopy(self.split2).check_valid_column_reference(self.data_x_2))
        self.assertFalse(deepcopy(self.split2).check_valid_column_reference(self.data_x_3))

        self.assertFalse(deepcopy(self.split3).check_valid_column_reference(self.data_x_1))
        self.assertFalse(deepcopy(self.split3).check_valid_column_reference(self.data_x_2))
        self.assertFalse(deepcopy(self.split3).check_valid_column_reference(self.data_x_3))

        self.assertFalse(deepcopy(self.split4).check_valid_column_reference(self.data_x_1))
        self.assertFalse(deepcopy(self.split4).check_valid_column_reference(self.data_x_2))
        self.assertFalse(deepcopy(self.split4).check_valid_column_reference(self.data_x_3))

        self.assertTrue(deepcopy(self.split5).check_valid_column_reference(self.data_x_1))
        self.assertTrue(deepcopy(self.split5).check_valid_column_reference(self.data_x_2))
        self.assertTrue(deepcopy(self.split5).check_valid_column_reference(self.data_x_3))

        self.assertTrue(deepcopy(self.split6).check_valid_column_reference(self.data_x_1))
        self.assertTrue(deepcopy(self.split6).check_valid_column_reference(self.data_x_2))
        self.assertTrue(deepcopy(self.split6).check_valid_column_reference(self.data_x_3))

        self.assertTrue(deepcopy(self.split7).check_valid_column_reference(self.data_x_1))
        self.assertTrue(deepcopy(self.split7).check_valid_column_reference(self.data_x_2))
        self.assertTrue(deepcopy(self.split7).check_valid_column_reference(self.data_x_3))

    def test_check_data_in_column_interval(self):
        self.assertTrue(deepcopy(self.split1).check_data_in_column_interval(self.data_x_1))
        self.assertTrue(deepcopy(self.split1).check_data_in_column_interval(self.data_x_2))
        self.assertTrue(deepcopy(self.split1).check_data_in_column_interval(self.data_x_3))

        self.assertTrue(deepcopy(self.split5).check_data_in_column_interval(self.data_x_1))
        self.assertTrue(deepcopy(self.split5).check_data_in_column_interval(self.data_x_2))
        self.assertTrue(deepcopy(self.split5).check_data_in_column_interval(self.data_x_3))

        self.assertFalse(deepcopy(self.split6).check_data_in_column_interval(self.data_x_1))
        self.assertFalse(deepcopy(self.split6).check_data_in_column_interval(self.data_x_2))
        self.assertFalse(deepcopy(self.split6).check_data_in_column_interval(self.data_x_3))

        self.assertFalse(deepcopy(self.split7).check_data_in_column_interval(self.data_x_1))
        self.assertFalse(deepcopy(self.split7).check_data_in_column_interval(self.data_x_2))
        self.assertFalse(deepcopy(self.split7).check_data_in_column_interval(self.data_x_3))

    def test_predict(self):
        x_0, x_1, x_2, x_3, x_4, x_5, x_6, c_0, c_1, c_2, c_3, c_4, c_5 = sp.symbols('x_0 x_1 x_2 x_3 x_4 x_5 x_6 c_0 c_1 c_2 c_3 c_4 c_5')

        split = deepcopy(self.split1)
        split.coef_assignment = [(c_4, -0.9697534633828704), (c_2, 0.595598997740995), (c_1, 0.000804917253418358),
                                 (c_0, -0.10072576315711212), (c_3, -0.22308508918656655)]

        self.assertEqual(split.predict(np.array([[1., 4.6, 1., 3.]])), 0)
        self.assertEqual(split.predict(np.array([[1., 4.6, 2., 3.]])), 0)
        self.assertEqual(split.predict(np.array([[2., 4., 3., 1.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 4., 3., 2.]])), 1)
        self.assertEqual(split.predict(np.array([[1., 4., 4., 1.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 4., 4., 2.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 53., 2., 3.]])), 0)
        self.assertEqual(split.predict(np.array([[1., 228., 1., 5.]])), 0)
        self.assertEqual(split.predict(np.array([[2., 93., 1., 2.]])), 0)
        self.assertEqual(split.predict(np.array([[2., 59., 3., 2.]])), 1)

        split.coef_assignment = [(c_4, 1.1454545454692278), (c_2, -2.1822543772032077e-12), (c_1, -0.03636363636362639),
                                 (c_0, -2.1651569426239803e-12), (c_3, -2.1969093211282598e-12)]

        self.assertEqual(split.predict(np.array([[2., 4., 3., 1.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 4., 3., 2.]])), 1)
        self.assertEqual(split.predict(np.array([[1., 4., 4., 1.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 4., 4., 2.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 59., 3., 2.]])), 0)

        split.coef_assignment = [(c_4, 1.1454545454692278), (c_1, -0.03636363636362639), (c_0, -2.1651569426239803e-12),
                                 (c_2, -2.1822543772032077e-12),
                                 (c_3, -2.1969093211282598e-12)]

        self.assertEqual(split.predict(np.array([[1., 4.6, 1., 3.]])), 1)
        self.assertEqual(split.predict(np.array([[1., 4.6, 2., 3.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 4., 3., 1.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 4., 3., 2.]])), 1)
        self.assertEqual(split.predict(np.array([[1., 4., 4., 1.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 4., 4., 2.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 53., 2., 3.]])), 0)
        self.assertEqual(split.predict(np.array([[1., 228., 1., 5.]])), 0)
        self.assertEqual(split.predict(np.array([[2., 93., 1., 2.]])), 0)
        self.assertEqual(split.predict(np.array([[2., 59., 3., 2.]])), 0)

        split.coef_assignment = [(c_0, -1.6809492419229106), (c_3, -0.26367831245735257), (c_2, -2.1596058275008545e-12),
                                 (c_1, -0.006591957811487801),
                                 (c_4, 3.5023071852310514)]

        self.assertEqual(split.predict(np.array([[1., 4.6, 1., 3.]])), 1)
        self.assertEqual(split.predict(np.array([[1., 4.6, 2., 3.]])), 1)
        self.assertEqual(split.predict(np.array([[2., 53., 2., 3.]])), 0)
        self.assertEqual(split.predict(np.array([[1., 228., 1., 5.]])), 0)
        self.assertEqual(split.predict(np.array([[2., 93., 1., 2.]])), 0)




if __name__ == '__main__':
    unittest.main()

print("The Critical Logger statements are supposed to appear on the console.")