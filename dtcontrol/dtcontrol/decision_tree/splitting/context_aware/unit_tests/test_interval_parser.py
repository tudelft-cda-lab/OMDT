import unittest

import hypothesis.strategies as st
import sympy as sp
from hypothesis import given, settings

from dtcontrol.decision_tree.splitting.context_aware.predicate_parser import (
    PredicateParser,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_exceptions import (
    RicherDomainPredicateParserException,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_logger import (
    RicherDomainLogger,
)

# Setting up a logger instance for testing. --> Doesn't print out anything
test_logger = RicherDomainLogger("RicherDomainCliStrategy_logger", False)
test_logger.root_logger.disabled = True


class TestIntervalParser(unittest.TestCase):
    """
    Depends on edge case inside predicate parser if Empty set occurs :
    Currently:
    if (21,20) = Error:Return None      --> Error Interval = None
    """

    """
                    ------------------------            C A U T I O N           ------------------------
                    
    It is import to use str() function for comparing at some points bc in large number edge cases with -1,1238....1233e-44 the assertion
    will fail even though the results do not differ.
    (Problem is caused by the fact, that the user input from the predicate parser will (and always has to) be a string and strings will cut 
    the value while sympy rounds)
    
                    ------------------------            C A U T I O N           ------------------------
    """

    def open_intervals(self, x, y):
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval(f"({x},{x})", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval(f"({y},{y})", test_logger)

        if sp.sympify(str(x)).evalf() != sp.sympify(str(y)).evalf():
            self.assertEqual(PredicateParser.parse_user_interval(f"({str(min(x, y))},{str(max(x, y))})", test_logger),
                             sp.Interval.open(sp.sympify(str(min(x, y))).evalf(), sp.sympify(str(max(x, y))).evalf()))
            with self.assertRaises(RicherDomainPredicateParserException):
                PredicateParser.parse_user_interval(f"({str(max(x, y))},{str(min(x, y))})", test_logger)
        else:
            with self.assertRaises(RicherDomainPredicateParserException):
                PredicateParser.parse_user_interval(f"({str(x)},{str(y)})", test_logger)

    @settings(deadline=None)
    @given(x=st.integers(), y=st.integers())
    def test_open_interval_integer_hypothesis(self, x, y):
        self.open_intervals(x, y)

    @settings(deadline=None)
    @given(x=st.floats(allow_nan=False, allow_infinity=False), y=st.floats(allow_nan=False, allow_infinity=False))
    def test_open_interval_float_hypothesis(self, x, y):
        self.open_intervals(x, y)

    def test_valid_open_intervals(self):
        self.assertEqual(PredicateParser.parse_user_interval("(-Inf, Inf)", test_logger), sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("(-Inf, Inf) u (0,1)", test_logger), sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("(0,1) ∪ (12, 15)", test_logger), sp.Union(sp.Interval.open(0, sp.sympify(1).evalf()),
                                                                                           sp.Interval.open(sp.sympify(12).evalf(),
                                                                                                            sp.sympify(15).evalf())))
        self.assertEqual(PredicateParser.parse_user_interval("(-Inf, Inf) or (0,1)", test_logger), sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("(0,1) or (12, 15)", test_logger), sp.Union(sp.Interval.open(0, sp.sympify(1).evalf()),
                                                                                            sp.Interval.open(sp.sympify(12).evalf(),
                                                                                                             sp.sympify(15).evalf())))
        self.assertEqual(PredicateParser.parse_user_interval("(-0.00000000123, 1)", test_logger),
                         sp.Interval.open(sp.sympify(-0.00000000123).evalf(), sp.sympify(1).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("(-0.00000000123, 0.00000000124)", test_logger),
                         sp.Interval.open(sp.sympify(-0.00000000123).evalf(), sp.sympify(0.00000000124).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("(1231231231234124, 12312312312341249)", test_logger),
                         sp.Interval.open(sp.sympify(1231231231234124).evalf(), sp.sympify(12312312312341249).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("(-Inf, Inf) u (1231231231234124, 12312312312341249)", test_logger),
                         sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("(-Inf, Inf) u (-0.00000000123, 0.00000000124)", test_logger),
                         sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("(-0.0123003231312312123, 131213910)", test_logger),
                         sp.Interval.open(sp.sympify(-0.0123003231312312123).evalf(), sp.sympify(131213910).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("(-0.9123919380000123, 0.0000000000000000124)", test_logger),
                         sp.Interval.open(sp.sympify(-0.9123919380000123).evalf(), sp.sympify(0.0000000000000000124).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("(1231231231234124, 12312312312341249)", test_logger),
                         sp.Interval.open(sp.sympify(1231231231234124).evalf(), sp.sympify(12312312312341249).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval(
            "(-1230938391238129, -0.00000000000001323) u (-Inf, Inf) u (1231231231234124, 12312312312341249)", test_logger),
            sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval(
            "(12393839238293828, 9999999999999999999999) u (-Inf, Inf) u (-0.00000000123, 0.00000000124)", test_logger),
            sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("(-10, 100)", test_logger), sp.Interval.open(-10.0000, 100.000))
        self.assertEqual(PredicateParser.parse_user_interval("(20.231313, 123)", test_logger), sp.Interval.open(20.231313, 123.000))
        self.assertEqual(PredicateParser.parse_user_interval("(123, 200)", test_logger), sp.Interval.open(123.000, 200.000))
        self.assertEqual(PredicateParser.parse_user_interval("(0.000013, 0.0015)", test_logger), sp.Interval.open(0.0000130000, 0.00150000))
        self.assertEqual(PredicateParser.parse_user_interval("(-Inf, 0.000013)", test_logger), sp.Interval.open(sp.S.NegativeInfinity, 0.0000130000))
        self.assertEqual(PredicateParser.parse_user_interval("(90909, Inf)", test_logger), sp.Interval.open(90909, sp.S.Infinity))

    def test_invalid_open_intervals(self):
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(,20.231313, 123)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("s(0.000013, 0.0015d)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(x_1-Inf, 0.000013!)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-INF, inf#", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("((99090909, Inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(99090909, Inf))", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(1,1)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(1,x_0)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(1,)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval(")", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(((1,1)))", test_logger)

    # @given(text())
    # def test_random_input(self,s):
    #     self.assertEqual(PredicateParser.parse_user_interval(s), self.ErrorInterval)

    def closed_intervals(self, x, y):
        self.assertEqual(PredicateParser.parse_user_interval(f"[{str(x)},{str(x)}]", test_logger), sp.FiniteSet(sp.sympify(str(x)).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval(f"[{str(y)},{str(y)}]", test_logger), sp.FiniteSet(sp.sympify(str(y)).evalf()))
        if sp.sympify(str(x)).evalf() != sp.sympify(str(y)).evalf():
            self.assertEqual(PredicateParser.parse_user_interval(f"[{str(min(x, y))},{str(max(x, y))}]", test_logger),
                             sp.Interval(sp.sympify(str(min(x, y))).evalf(), sp.sympify(str(max(x, y))).evalf()))
            with self.assertRaises(RicherDomainPredicateParserException):
                PredicateParser.parse_user_interval(f"{str(max(x, y))},{str(min(x, y))}]", test_logger)
        else:
            self.assertEqual(PredicateParser.parse_user_interval(f"[{str(x)},{str(y)}]", test_logger), sp.FiniteSet(sp.sympify(str(y)).evalf()))

    @settings(deadline=None)
    @given(x=st.integers(), y=st.integers())
    def test_closed_interval_integer_hypothesis(self, x, y):
        self.closed_intervals(x, y)

    @settings(deadline=None)
    @given(x=st.floats(allow_nan=False, allow_infinity=False), y=st.floats(allow_nan=False, allow_infinity=False))
    def test_closed_interval_float_hypothesis(self, x, y):
        self.closed_intervals(x, y)

    @settings(deadline=None)
    @given(a=st.integers(), b=st.integers(), c=st.integers(), d=st.integers(), e=st.integers(), f=st.integers(), g=st.integers(),
           h=st.integers(), i=st.floats(allow_nan=False, allow_infinity=False), j=st.floats(allow_nan=False, allow_infinity=False),
           k=st.floats(allow_nan=False, allow_infinity=False), l=st.floats(allow_nan=False, allow_infinity=False))
    def test_finite_interval(self, a, b, c, d, e, f, g, h, i, j, k, l):
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{}", test_logger)
        self.assertEqual(PredicateParser.parse_user_interval(
            "{" + str(a) + "," + str(b) + "," + str(c) + "," + str(d) + "," + str(e) + "," + str(f) + "," + str(g) + "," + str(
                h) + "," + str(i) + "," + str(j) + "," + str(k) + "," + str(l) + "}", test_logger),
            sp.FiniteSet(sp.sympify(str(a)).evalf(), sp.sympify(str(b)).evalf(), sp.sympify(str(c)).evalf(),
                         sp.sympify(str(d)).evalf(), sp.sympify(str(e)).evalf(), sp.sympify(str(f)).evalf(),
                         sp.sympify(str(g)).evalf(), sp.sympify(str(h)).evalf(), sp.sympify(str(i)).evalf(),
                         sp.sympify(str(j)).evalf(), sp.sympify(str(k)).evalf(), sp.sympify(str(l)).evalf(), ))

    def test_valid_closed_intervals(self):
        self.assertEqual(PredicateParser.parse_user_interval("[-Inf, Inf]", test_logger), sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("[-Inf, Inf] u [0,1]", test_logger), sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("[0,1] ∪ [12, 15]", test_logger), sp.Union(sp.Interval(0, sp.sympify(1).evalf()),
                                                                                           sp.Interval(sp.sympify(12).evalf(),
                                                                                                       sp.sympify(15).evalf())))
        self.assertEqual(PredicateParser.parse_user_interval("[-0.00000000123, 1]", test_logger),
                         sp.Interval(sp.sympify(-0.00000000123).evalf(), sp.sympify(1).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("[-0.00000000123, 0.00000000124]", test_logger),
                         sp.Interval(sp.sympify(-0.00000000123).evalf(), sp.sympify(0.00000000124).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("[1231231231234124, 12312312312341249]", test_logger),
                         sp.Interval(sp.sympify(1231231231234124).evalf(), sp.sympify(12312312312341249).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("[-Inf, Inf] u [1231231231234124, 12312312312341249]", test_logger),
                         sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("[-Inf, Inf] u [-0.00000000123, 0.00000000124]", test_logger),
                         sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("[-Inf, Inf] or [1231231231234124, 12312312312341249]", test_logger),
                         sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("[-Inf, Inf] or [-0.00000000123, 0.00000000124]", test_logger),
                         sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("[-0.0123003231312312123, 131213910]", test_logger),
                         sp.Interval(sp.sympify(-0.0123003231312312123).evalf(), sp.sympify(131213910).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("[-0.9123919380000123, 0.0000000000000000124]", test_logger),
                         sp.Interval(sp.sympify(-0.9123919380000123).evalf(), sp.sympify(0.0000000000000000124).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("[1231231231234124, 12312312312341249]", test_logger),
                         sp.Interval(sp.sympify(1231231231234124).evalf(), sp.sympify(12312312312341249).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval(
            "[-1230938391238129, -0.00000000000001323] u [-Inf, Inf] u [1231231231234124, 12312312312341249]", test_logger),
            sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval(
            "[12393839238293828, 9999999999999999999999] u [-Inf, Inf] u [-0.00000000123, 0.00000000124]", test_logger),
            sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("[-10, 100]", test_logger), sp.Interval(-10.0000, 100.000))
        self.assertEqual(PredicateParser.parse_user_interval("[20.231313, 123]", test_logger), sp.Interval(20.231313, 123.000))
        self.assertEqual(PredicateParser.parse_user_interval("[123, 200]", test_logger), sp.Interval(123.000, 200.000))
        self.assertEqual(PredicateParser.parse_user_interval("[0.000013, 0.0015]", test_logger), sp.Interval(0.0000130000, 0.00150000))
        self.assertEqual(PredicateParser.parse_user_interval("[-Inf, 0.000013]", test_logger), sp.Interval(sp.S.NegativeInfinity, 0.0000130000))
        self.assertEqual(PredicateParser.parse_user_interval("[-INF, inf]", test_logger), sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("[90909, Inf]", test_logger), sp.Interval(90909, sp.S.Infinity))

    def test_invalid_closed_intervals(self):
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("-10, 100]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[,20.231313, 123]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[12as3,s 200]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("s[0.000013, 0.0015d]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[x_1-Inf, 0.000013!]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-INF, inf#", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[[99090909, Inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[99090909, Inf]]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[,-1]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[1,x_0]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[1,]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[[[1,1]]]", test_logger)

    def test_invalid_infinity_intervals(self):
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(Inf, Inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-Inf, -Inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(Inf, -Inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[Inf, Inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-Inf, -Inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[Inf, -Inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(Inf, Inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-Inf, -Inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(Inf, -Inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[Inf, Inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-Inf, -Inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[Inf, -Inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(INF, INF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-INF, -INF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(INF, -INF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[INF, INF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-INF, -INF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[INF, -INF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(INF, INF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-INF, -INF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(INF, -INF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[INF, INF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-INF, -INF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[INF, -INF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inf, inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-inf, -inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inf, -inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inf, inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-inf, -inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inf, -inf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inf, inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-inf, -inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inf, -inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inf, inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-inf, -inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inf, -inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(iNf, iNf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-iNf, -iNf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(iNf, -iNf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[iNf, iNf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-iNf, -iNf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[iNf, -iNf)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(iNf, iNf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-iNf, -iNf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(iNf, -iNf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[iNf, iNf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-iNf, -iNf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[iNf, -iNf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inF, inF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-inF, -inF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inF, -inF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inF, inF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-inF, -inF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inF, -inF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inF, inF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-inF, -inF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inF, -inF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inF, inF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-inF, -inF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inF, -inF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(iNF, iNF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-iNF, -iNF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(iNF, -iNF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[iNF, iNF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-iNF, -iNF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[iNF, -iNF)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(iNF, iNF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-iNF, -iNF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(iNF, -iNF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[iNF, iNF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-iNF, -iNF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[iNF, -iNF]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(oo, oo)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-oo, -oo)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(oo, -oo)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[oo, oo)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-oo, -oo)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[oo, -oo)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(oo, oo]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-oo, -oo]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(oo, -oo]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[oo, oo]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-oo, -oo]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[oo, -oo]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(INFINITY, INFINITY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-INFINITY, -INFINITY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(INFINITY, -INFINITY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[INFINITY, INFINITY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-INFINITY, -INFINITY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[INFINITY, -INFINITY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(INFINITY, INFINITY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-INFINITY, -INFINITY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(INFINITY, -INFINITY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[INFINITY, INFINITY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-INFINITY, -INFINITY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[INFINITY, -INFINITY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inFinItY, inFinItY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-inFinItY, -inFinItY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inFinItY, -inFinItY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inFinItY, inFinItY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-inFinItY, -inFinItY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inFinItY, -inFinItY)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inFinItY, inFinItY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-inFinItY, -inFinItY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(inFinItY, -inFinItY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inFinItY, inFinItY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[-inFinItY, -inFinItY]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[inFinItY, -inFinItY]", test_logger)

    def test_valid_mixed_intervals(self):
        self.assertEqual(PredicateParser.parse_user_interval("(0,1) ∪ [12, 15]", test_logger), sp.Union(sp.Interval.open(0, 1), sp.Interval(12, 15)))
        self.assertEqual(PredicateParser.parse_user_interval("[0,1)", test_logger), sp.Interval.Ropen(0, 1))
        self.assertEqual(PredicateParser.parse_user_interval("{1,2,3,4,5,6} ∪ {6,7,8}", test_logger), sp.FiniteSet(1, 2, 3, 4, 5, 6, 7, 8))
        self.assertEqual(PredicateParser.parse_user_interval("{1,2,3,4,5,6} ∪ {6,7,8} ∪ [12, 15]", test_logger),
                         sp.Union(sp.FiniteSet(1, 2, 3, 4, 5, 6, 7, 8), sp.Interval(12, 15)))
        self.assertEqual(PredicateParser.parse_user_interval("{1,2} ∪ (12, 15]", test_logger), sp.Union(sp.FiniteSet(1, 2), sp.Interval.Lopen(12, 15)))

    def test_valid_open_and_closed_intervals(self):
        self.assertEqual(PredicateParser.parse_user_interval("(-10, 100]", test_logger), sp.Interval.Lopen(-10, 100))
        self.assertEqual(PredicateParser.parse_user_interval("[20.231313, 123)", test_logger), sp.Interval.Ropen(20.231313, 123))
        self.assertEqual(PredicateParser.parse_user_interval("(123, 200]", test_logger), sp.Interval.Lopen(123, 200))
        self.assertEqual(PredicateParser.parse_user_interval("[0.000013, 0.0015)", test_logger), sp.Interval.Ropen(0.000013, 0.0015))
        self.assertEqual(PredicateParser.parse_user_interval("(-Inf, 0.000013]", test_logger), sp.Interval(sp.S.NegativeInfinity, 0.000013))
        self.assertEqual(PredicateParser.parse_user_interval("[-INF, inf)", test_logger), sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity))
        self.assertEqual(PredicateParser.parse_user_interval("(90909, Inf]", test_logger), sp.Interval.open(90909, sp.S.Infinity))

    def test_invalid_open_and_closed_intervals(self):
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(y-10, 100]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval(",20.231313, 123]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[12as3,s 200)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("s(0.000013, 0.0015d]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(x_1-Inf, 0.000013!]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(-INF, inf&]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[(99090909, Inf]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(99090909, Inf]]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(,-1]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(1,x_0]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(1,]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[[(1,1]])", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(1,1]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(1,daswad]", test_logger)

    def test_valid_finite_intervals(self):
        self.assertEqual(PredicateParser.parse_user_interval("{1233.123, sqrt(2), 123213}", test_logger),
                         sp.FiniteSet(sp.sympify("sqrt(2)").evalf(), 1233.123, 123213.0))
        self.assertEqual(PredicateParser.parse_user_interval("{log(12), sqrt(2), 123213}", test_logger),
                         sp.FiniteSet(sp.sympify("sqrt(2)").evalf(), sp.sympify("log(12)").evalf(),
                                      123213.0))
        self.assertEqual(PredicateParser.parse_user_interval("{1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8}", test_logger),
                         sp.FiniteSet(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        self.assertEqual(PredicateParser.parse_user_interval("{10,0.00002329, -1230123092029, 1230123}", test_logger),
                         sp.FiniteSet(10, 0.00002329, -1230123092029, 1230123))
        self.assertEqual(PredicateParser.parse_user_interval("{-123123123123.2323,-231,23,1,345343,1233,-231,-0.0000000323232023132}", test_logger),
                         sp.FiniteSet(-123123123123.2323, -231, 23, 1, 345343, 1233, -231, -0.0000000323232023132))
        self.assertEqual(PredicateParser.parse_user_interval("{12,312,3.123,123, 123,981,2918, 0}", test_logger),
                         sp.FiniteSet(12, 312, 3.123, 123, 123, 981, 2918, 0))
        self.assertEqual(PredicateParser.parse_user_interval("{63,5253.2131,23876,5432,3542,3123,123}", test_logger),
                         sp.FiniteSet(63, 5253.2131, 23876, 5432, 3542, 3123, 123))
        self.assertEqual(PredicateParser.parse_user_interval("{342,14.213,123}", test_logger), sp.FiniteSet(342, 14.213, 123))
        self.assertEqual(PredicateParser.parse_user_interval("{0.1231,231,23, 123,9812,918,636,6346,34,5643,44353,34534,5345,3453,115}", test_logger),
                         sp.FiniteSet(0.1231, 231, 23, 123, 9812, 918, 636, 6346, 34, 5643, 44353, 34534, 5345, 3453, 115))
        self.assertEqual(PredicateParser.parse_user_interval("{321,3123123.5435345}", test_logger), sp.FiniteSet(321, 3123123.5435345))
        self.assertEqual(PredicateParser.parse_user_interval("{41344.123123}", test_logger), sp.FiniteSet(41344.123123))
        self.assertEqual(PredicateParser.parse_user_interval("{0.0,0.03012,3102301}", test_logger), sp.FiniteSet(0.0, 0.03012, 3102301))
        self.assertEqual(PredicateParser.parse_user_interval("{123,19239,12,3}", test_logger), sp.FiniteSet(123, 19239, 12, 3))

    def test_invalid_finite_intervals(self):
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1233.123,, x_0*sqrt(2), 123213}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{log12), sqrt(2),!§ 123213}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1,{2},{3},4,5,6,x_0,8,9,1,2,3,4,5,6,7,8}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1x_0}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1,{2},{3},BAUM)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1,Inf,x_0", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("8{12}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{(,-}1", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1 2 3 }", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{0.1231,23x_1,23, 123,9812,918,636,6346,34,5643,44353,34534,5345,3453,115}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{0.1231,,,115}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{0.1asd231,we,,115}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("31,,,115}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{0.1", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1^^", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{Inf}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{Inf,1,2}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{-Inf}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{-1223,-Inf,1,2}", test_logger)

    def test_valid_union_intervals(self):
        self.assertEqual(PredicateParser.parse_user_interval(
            "{1233.123, sqrt(2), 123213} or (1,2) or (3,4) u [9,10) u (12,9999] OR (-1,0) Or {-1,-2,-3,-4}", test_logger),
            sp.Union(sp.FiniteSet(-4, -3, -2, 123213), sp.Interval.Ropen(-1, 0),
                     sp.Interval.open(1, 2),
                     sp.Interval.open(3, 4),
                     sp.Interval.Ropen(9, 10),
                     sp.Interval.Lopen(12, 9999)))
        self.assertEqual(PredicateParser.parse_user_interval("{12313} OR {1313,2323} u {log(12), sqrt(2), 123213}", test_logger),
                         sp.FiniteSet(sp.sympify("sqrt(2)").evalf(), sp.sympify("log(12)").evalf(), 1313, 2323, 12313.0, 123213.0))
        self.assertEqual(PredicateParser.parse_user_interval("{1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8} u {1,2,3,4,5}", test_logger),
                         sp.FiniteSet(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
        self.assertEqual(PredicateParser.parse_user_interval("[1,5] or (-1,2) Or {-10,-11,123.4}", test_logger),
                         sp.Union(sp.FiniteSet(-11.0, -10.0, 123.4), sp.Interval.Lopen(-1.00000000000000, 5.00000000000000)))
        self.assertEqual(PredicateParser.parse_user_interval("(1,9) or [-0.00000000123, 1]", test_logger),
                         sp.Interval.Ropen(-0.00000000123, sp.sympify(9).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("(9,321] u [-0.9123919380000123, 0.0000000000000000124]", test_logger),
                         sp.Union(sp.Interval(-0.9123919380000123, 0.0000000000000000124), sp.Interval.Lopen(9, 321)))
        self.assertEqual(PredicateParser.parse_user_interval("[12,14) u [-1230938391238129, -0.00000000000001323]", test_logger),
                         sp.Union(sp.Interval(sp.sympify(-1230938391238129).evalf(), -0.00000000000001323),
                                  sp.Interval.Ropen(sp.sympify(12).evalf(), sp.sympify(14).evalf())))
        self.assertEqual(PredicateParser.parse_user_interval("(1,21] u [-1230938391238129, -0.00000000000001323]", test_logger),
                         sp.Union(sp.Interval(sp.sympify(-1230938391238129).evalf(), -0.00000000000001323),
                                  sp.Interval.Lopen(sp.sympify(1).evalf(), sp.sympify(21).evalf())))
        self.assertEqual(PredicateParser.parse_user_interval("[0,3131313.123123) or [-0.00000000123, 1]", test_logger),
                         sp.Interval.Ropen(sp.sympify(-0.00000000123), 3131313.123123))
        self.assertEqual(PredicateParser.parse_user_interval("(212,13113) u [-0.9123919380000123, 0.0000000000000000124]", test_logger),
                         sp.Union(sp.Interval(-0.9123919380000123, sp.sympify(0.0000000000000000124).evalf()),
                                  sp.Interval.open(sp.sympify(212).evalf(), sp.sympify(13113).evalf())))
        self.assertEqual(PredicateParser.parse_user_interval("[0.00032,98989.12313] or (1,1.000000000000001)", test_logger),
                         sp.Interval(sp.sympify(0.00032).evalf(), sp.sympify(98989.12313).evalf()))
        self.assertEqual(PredicateParser.parse_user_interval("[123,1231313) or [-0.00000000123, 1]", test_logger),
                         sp.Union(sp.Interval(sp.sympify(-0.00000000123).evalf(), sp.sympify(1).evalf()),
                                  sp.Interval.Ropen(sp.sympify(123), sp.sympify(1231313).evalf())))

    def test_invalid_union_intervals(self):
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1233.123,, x_0*sqrt(2), 123213Or } or {123}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[123,1234] or {log12), sqrt(2),!§ 123213}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1,2} or {1,{2},{3},4,5,6,x_0,8,9,1,2,3,4,5,6,7,8}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(1,9) or {1x_0}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(8,9] or {1,{2},{3},BAUM)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[1 x_0 5] or -1,2) Or [-10,-11),123.4}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[12,14) u [-1230938391238129, -0.000000000000ad01323]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[123,1231313) or -0.00000000123, 1]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("[0.00032,98989.12313] and (1,1.000000000000001)", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("(212,13113) u [-0.9123919380000123, and , 0.0000000000000000124]", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1,(2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8} u {1,2,3,4,5}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1,2 3,4,5,6,7,8,9,1,2,3,4,5,6,7,8} u {1,2,3,4,5}", test_logger)
        with self.assertRaises(RicherDomainPredicateParserException):
            PredicateParser.parse_user_interval("{1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8 and  {1,2,3,4,5}", test_logger)


if __name__ == '__main__':
    unittest.main()