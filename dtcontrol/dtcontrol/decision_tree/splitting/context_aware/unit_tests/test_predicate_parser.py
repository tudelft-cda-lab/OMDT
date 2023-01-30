import os
import unittest

import sympy as sp

from dtcontrol.decision_tree.splitting.context_aware.predicate_parser import (
    PredicateParser,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_exceptions import (
    RicherDomainPredicateParserException,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_split import (
    RicherDomainSplit,
)


class TestPredicateParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """

        FILE NAME                       |   BEHAVIOR
        --------------------------------|----------------------------------------
        test_file1.txt                  |   useful /correct predicates
        --------------------------------|----------------------------------------
        test_file2.txt                  |   predicates which do not use dataset
                                        |   -> No /wrong usage of variables
                                        |   e.g.
                                        |   12*pi >= {1}
                                        |   23 * 123 = {1,2,3}
        --------------------------------|----------------------------------------
        test_file3.txt                  |   predicates with general typos or use of unknown functions
                                        |   e.g.
                                        |   sqrrrt(2)*x_12 != {123}
                                        |   löög(10)*x_0 - x_1 < {13}
        --------------------------------|----------------------------------------
        test_file4.txt                  |   only invalid relations
                                        |   e.g.
                                        |   12*x_1*pi ? {1}
                                        |   2 / x_0 =9 {1,2}
                                        |   x_1 [1,9)
        --------------------------------|----------------------------------------

        """
        # Useful / correct predicates
        test_input_file1 = open("../input_data/test_file1.txt", "w+")
        test_input_file1.write(
            "c_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 <= c_4\nc_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 > c_4; c_0 in {0.96223}; c_1 in {-0.564809}; c_2 in {1.32869}; c_3 in {3.315577}; c_4 in {-1.088248}\nc_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 > c_4; c_0 in {0.96223}; c_1 in {-0.564809}; c_3 in {3.315577}; x_3 in [-1.088248,2)\nsqrt(x_0)*c_0 + log(x_1) -(c_0 / x_2) * c_1 + c_2 < c_3; c_3 in {0,1,2,3,4}\nsqrt(x_0) + c_2 < c_3; c_3 in {0,1,2,3,4}; x_0 in (12, 13); c_2 in {14,15,16}")
        test_input_file1.close()

        # No / wrong usage of variables
        test_input_file2 = open("../input_data/test_file2.txt", "w+")
        test_input_file2.write(
            "c_0*c_0+c_1*c_1+c_2*c_2+c_3*c_3 <= c_4; c_5 in {1}\nx_0 > c_0; c_1 in {1}\nc_0*c_0+c_1*c_1+c_2*c_2+c_3*c_3 <= c_4\nc_0*c_0+c_1*c_1+c_2*c_2+c_3*c_3 = c_4; c_0 in {0.96223}; c_1 in {-0.564809}; c_2 in {1.32869}; c_3 in {3.315577}; c_4 in {-1.088248}\nc_0*c_0+c_1*c_1+c_2*c_2+c_3*c_3 > c_4; c_0 in {0.96223}; c_1 in {-0.564809}; c_3 in {3.315577}; c_3 in [-1.088248,2)\nsqrt(c_0)*c_0 + log(c_1) -(c_0 / c_2) * c_1 + c_2 != c_3; c_3 in {0,1,2,3,4}\nsqrt(c_0) + c_2 != c_3; c_3 in {0,1,2,3,4}; c_0 in (12, 13); c_2 in {14,15,16}")
        test_input_file2.close()

        # General typos
        test_input_file3 = open("../input_data/test_file3.txt", "w+")
        test_input_file3.write(
            "cx_0+1c23_1*x_1+c_2*x_2+c_3*x_3 <= c_4\nc_0*adx_0+c_1*x_1+c_2*x_2+c_3*x_3 = c_4; c_0 in {0.96223}; c_1 in {-0.564809}; c_2 in 1.32869}; c_3 in {3.315577}; c_4 in {-1.088248}\nc_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 > c_4; c_0 in0.96223}; c_1 in {-0.564809}; c_3 in {3.315577}; x_3 in [-1.088248,2)\nsqrtx_0)*c_0 + log(x_1) -(c_0 / x_2) * c_1 + c_2 != c_3; c_3 in {0,1,2,3,4}\nsqrt(x_0) + c_2 != c_3; c_3 in {0,1,2,3,4}; x_0 (12, 13); c_2 in {14,15,16}")
        test_input_file3.close()

        # Invalid relations
        test_input_file4 = open("../input_data/test_file4.txt", "w+")
        test_input_file4.write(
            "c_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3  c_4\nc_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 y= c_4; c_0 in {0.96223}; c_1 in {-0.564809}; c_2 in {1.32869}; c_3 in {3.315577}; c_4 in {-1.088248}\nc_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 <> c_4; c_0 in {0.96223}; c_1 in {-0.564809}; c_3 in {3.315577}; x_3 in [-1.088248,2)\nsqrt(x_0)*c_0 + log(x_1) -(c_0 / x_2) * c_1 + c_2 !=! c_3; c_3 in {0,1,2,3,4}\nsqrt(x_0) + c_2 ? c_3; c_3 in {0,1,2,3,4}; x_0 in (12, 13); c_2 in {14,15,16}")
        test_input_file4.close()

    @classmethod
    def tearDownClass(cls):
        # Deleting test input files after usage
        for i in range(4):
            os.unlink(f"../input_data/test_file{i + 1}.txt")

    def test_useful_predicates(self):
        # USAGE OF FILE 1
        # Non existing input file
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.get_predicate(input_file_path="None")

        # Check if parsing of file 1 was successful
        parsed_predicate = PredicateParser.get_predicate(input_file_path="../input_data/test_file1.txt")

        # Checking right instance
        for obj in parsed_predicate:
            self.assertIsInstance(obj, RicherDomainSplit)

        x_0, x_1, x_2, x_3, x_4, x_5, c_0, c_1, c_2, c_3, c_4, c_5 = sp.symbols('x_0 x_1 x_2 x_3 x_4 x_5 c_0 c_1 c_2 c_3 c_4 c_5')

        # Checking predicate 1: c_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 <= c_4
        current_predicate = parsed_predicate[0]
        pred1_column_interval = {x_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_1: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_3: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
        self.assertEqual(current_predicate.column_interval, pred1_column_interval)

        pred1_coef_interval = {c_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                               c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                               c_1: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                               c_4: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                               c_3: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
        self.assertEqual(current_predicate.coef_interval, pred1_coef_interval)

        pred1_term = sp.sympify("c_0*x_0 + c_1*x_1 + c_2*x_2 + c_3*x_3 - c_4")
        self.assertEqual(current_predicate.term, pred1_term)
        self.assertEqual(current_predicate.relation, "<=")

        # Checking predicate 2: c_0 * x_0 + c_1 * x_1 + c_2 * x_2 + c_3 * x_3 = c_4;c_0 in {0.96223};c_1 in {-0.564809};c_2 in {1.32869};c_3 in {3.315577};c_4 in {-1.088248}
        current_predicate = parsed_predicate[1]
        pred2_column_interval = {x_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_1: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_3: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
        self.assertEqual(current_predicate.column_interval, pred2_column_interval)

        pred2_coef_interval = {c_4: sp.FiniteSet(-1.088248), c_0: sp.FiniteSet(0.96223), c_2: sp.FiniteSet(1.32869),
                               c_1: sp.FiniteSet(-0.564809), c_3: sp.FiniteSet(3.315577)}
        self.assertEqual(current_predicate.coef_interval, pred2_coef_interval)

        pred2_term = sp.sympify("c_0*x_0 + c_1*x_1 + c_2*x_2 + c_3*x_3 - c_4")
        self.assertEqual(current_predicate.term, pred2_term)
        self.assertEqual(current_predicate.relation, ">")

        # Checking predicate 3: c_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 > c_4; c_0 in {0.96223}; c_1 in {-0.564809}; c_3 in {3.315577}; x_3 in [-1.088248,2)
        current_predicate = parsed_predicate[2]
        pred3_column_interval = {x_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_1: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_3: sp.Interval.Ropen(sp.sympify(-1.088248).evalf(), sp.sympify(2).evalf())}
        self.assertEqual(current_predicate.column_interval, pred3_column_interval)

        pred3_coef_interval = {c_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity), c_3: sp.FiniteSet(3.315577),
                               c_4: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity), c_1: sp.FiniteSet(-0.564809),
                               c_0: sp.FiniteSet(0.96223)}
        self.assertEqual(current_predicate.coef_interval, pred3_coef_interval)
        pred3_term = sp.sympify("c_0*x_0 + c_1*x_1 + c_2*x_2 + c_3*x_3 - c_4")
        self.assertEqual(current_predicate.term, pred3_term)
        self.assertEqual(current_predicate.relation, ">")

        # Checking predicate 4: sqrt(x_0)*c_0 + log(x_1) -(c_0 / x_2) * c_1 + c_2 != c_3; c_3 in {0,1,2,3,4}
        current_predicate = parsed_predicate[3]
        pred4_column_interval = {x_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                                 x_1: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)}
        self.assertEqual(current_predicate.column_interval, pred4_column_interval)

        pred4_coef_interval = {c_2: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                               c_0: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                               c_1: sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity),
                               c_3: sp.FiniteSet(0, 1.0, 2.0, 3.0, 4.0)}
        self.assertEqual(current_predicate.coef_interval, pred4_coef_interval)
        pred4_term = sp.sympify("-c_0*c_1/x_2 + c_0*sqrt(x_0) + c_2 - c_3 + log(x_1)")
        self.assertEqual(current_predicate.term, pred4_term)
        self.assertEqual(current_predicate.relation, "<")

        # Checking predicate 5: sqrt(x_0) + c_2 != c_3; c_3 in {0,1,2,3,4}; x_0 in (12, 13); c_2 in {14,15,16}

        current_predicate = parsed_predicate[4]
        pred5_column_interval = {x_0: sp.Interval.open(12, 13)}
        self.assertEqual(current_predicate.column_interval, pred5_column_interval)

        pred5_coef_interval = {c_3: sp.FiniteSet(0, 1, 2, 3, 4), c_2: sp.FiniteSet(14, 15, 16)}
        self.assertEqual(current_predicate.coef_interval, pred5_coef_interval)
        pred5_term = sp.sympify("c_2 - c_3 + sqrt(x_0)")
        self.assertEqual(current_predicate.term, pred5_term)
        self.assertEqual(current_predicate.relation, "<")

    def test_predicates_without_column_ref(self):
        # USAGE OF FILE 2
        with self.assertRaises(RicherDomainPredicateParserException):
            parsed_predicate = PredicateParser.get_predicate(input_file_path="../input_data/test_file2.txt")

    def test_typo_predicates(self):
        # USAGE OF FILE 3
        with self.assertRaises(RicherDomainPredicateParserException):
            parsed_predicate = PredicateParser.get_predicate(input_file_path="../input_data/test_file3.txt")

    def test_invalid_relation_predicates(self):
        # USAGE OF FILE 4
        with self.assertRaises(RicherDomainPredicateParserException):
            parsed_predicate = PredicateParser.get_predicate(input_file_path="../input_data/test_file4.txt")

print("The Critical Logger statements are supposed to appear on the console.")