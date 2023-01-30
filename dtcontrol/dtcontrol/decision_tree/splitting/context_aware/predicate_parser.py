import re

import sympy as sp
from sympy.core.function import AppliedUndef

from dtcontrol.decision_tree.splitting.context_aware.richer_domain_exceptions import (
    RicherDomainPredicateParserException,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_logger import (
    RicherDomainLogger,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_split import (
    RicherDomainSplit,
)


class PredicateParser:

    @classmethod
    def parse_single_predicate(cls, single_predicate, logger, debug=False):
        """
        Function to parse a single predicate and return a RicherDomainSplitObject
        :param single_predicate: String containing the predicate
        :param logger_name: String containing the logger name
        :param debug: Bool for debug mode (see Logger)

        e.g.
        Input_line:  c_1 * x_1 - c_2 + x_2 - c_3  <= 0; x_2 in {1,2,3}; c_1 in (-inf, inf); c_2 in {1,2,3}; c_3 in {5, 10, 32, 40}
        Output: RicherDomainSplit Object with:

        column_interval     =       {x_1:(-Inf,Inf), x_2:{1,2,3}}                           --> Key: Sympy Symbol Value: Sympy Interval
        coef_interval       =       {c_1:(-Inf,Inf), c_2:{1,2,3}, c_3:{5,10,32,40}          --> Key: Sympy Symbol Value: Sympy Interval
        term                =       c_1 * x_1 - c_2 + x_2 - c_3                             --> sympy expression
        relation            =       '<='                                                    --> String

        Every symbol without a specific defined Interval will be assigned to the interval: (-Inf, Inf)

        EDGE CASE BEHAVIOR:
        Every column reference or coef without a specific defined Interval will be assigned this interval: (-Inf, Inf)
        Allowed interval types for columns: All (expect Empty Set)
        Allowed interval types for coef: Finite or (-Inf,Inf)

        In case the predicate obtained from the user has following structure:

        x_1     <=          5       (5 != 0)
        term    relation    bias    (with bias != 0)

        the whole predicate will be transferred to following structure:

        x_1 - 5         <=          0
        term - bias     relation    0


        ----------------    !!!!!!!!!!!!!!!!    C A U T I O N    !!!!!!!!!!!!!!!!   ----------------
        |                  COLUMN REFERENCING ONLY WITH VARIABLES OF STRUCTURE: x_i                |
        |                  COEFS ONLY WITH VARIABLES OF STRUCTURE: c_j                             |
        --------------------------------------------------------------------------------------------

        """
        logger.root_logger.info("Predicate to parse: " + str(single_predicate))

        # Currently supported types of relations
        supported_relation = ["<=", ">=", "<", ">", "="]

        for relation in supported_relation:
            if relation in single_predicate:
                # Deleting additional semi colon at the end
                if single_predicate[-1] == ";":
                    single_predicate = single_predicate[:-1]
                try:
                    # Cutting the input into separate strings. The first one should contain the term. The rest should be intervals
                    split_pred = single_predicate.split(";")
                    split_term = split_pred[0].split(relation)
                    term = sp.sympify(split_term[0] + "-(" + split_term[1] + ")")
                except Exception:
                    logger.root_logger.critical(
                        "Aborting: one predicate does not have a valid structure. Invalid predicate: {}. Please check for typos and read the comments inside predicate_parser.py. For more information take a look at the sympy library (https://docs.sympy.org/latest/tutorial/basic_operations.html#converting-strings-to-sympy-expressions).".format(
                            str(single_predicate)))
                    raise RicherDomainPredicateParserException()

                all_interval_defs = {}
                column_interval = {}
                coef_interval = {}

                # Parsing all additional given intervals and storing them inside --> all_interval_defs
                try:
                    for i in range(1, len(split_pred)):
                        split_coef_definition = split_pred[i].split("in", 1)
                        interval = cls.parse_user_interval(split_coef_definition[1], logger)
                        symbol = sp.sympify(split_coef_definition[0])
                        all_interval_defs[symbol] = interval
                except Exception:
                    logger.root_logger.critical(
                        "Aborting: one predicate does not have a valid structure. Invalid predicate: {}. Please check for typos and read the comments inside predicate_parser.py. For more information take a look at the sympy library (https://docs.sympy.org/latest/tutorial/basic_operations.html#converting-strings-to-sympy-expressions).".format(
                            str(single_predicate)))
                    raise RicherDomainPredicateParserException()

                """
                ----------------    !!!!!!!!!!!!!!!!    C A U T I O N    !!!!!!!!!!!!!!!!   ----------------
                |                  COLUMN REFERENCING ONLY WITH VARIABLES OF STRUCTURE: x_i                |
                |                  COEFS ONLY WITH VARIABLES OF STRUCTURE: c_j                             |
                --------------------------------------------------------------------------------------------
                """
                # Iterating over every symbol/variable and deciding whether it is a column reference or a coef
                infinity = sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)
                for var in term.free_symbols:
                    # Type: x_i -> column reference
                    if re.match(r"x_\d+", str(var)):
                        if not all_interval_defs.__contains__(var):
                            column_interval[var] = infinity
                        else:
                            column_interval[var] = all_interval_defs[var]
                            all_interval_defs.__delitem__(var)
                    # Type: c_i --> coef
                    elif re.match(r"c_\d+", str(var)):
                        if not all_interval_defs.__contains__(var):
                            coef_interval[var] = infinity
                        else:
                            # CHECKING: coefs are only allowed to have 2 kinds of intervals: FiniteSet or (-Inf,Inf)
                            check_interval = all_interval_defs[var]
                            all_interval_defs.__delitem__(var)
                            if isinstance(check_interval, sp.FiniteSet) or check_interval == infinity:
                                coef_interval[var] = check_interval
                            else:
                                # coef interval is not FiniteSet or (-Inf,Inf)
                                logger.root_logger.critical(
                                    "Aborting: invalid interval for a coefficient declared. Only finite or (-Inf,Inf) allowed. Coefficient: {} with invalid interval: {} from predicate: {}".format(
                                        str(var), str(check_interval), str(single_predicate)))
                                raise RicherDomainPredicateParserException()
                    else:
                        logger.root_logger.critical(
                            "Aborting: one symbol inside one predicate does not have a valid structure. Column refs only with x_i. Coefs only with c_i. Invalid symbol: '{}' inside predicate {}".format(
                                str(var), str(single_predicate)))
                        raise RicherDomainPredicateParserException()

                # Hidden edge case1: Undefined functions.
                # e.g. f(x) * c1 * x_3 >= 1 <--> f(x) is an undefined function.
                if term.atoms(AppliedUndef):
                    logger.root_logger.critical(
                        "Aborting: one predicate contains an undefined function. Undefined function: {}. Invalid predicate: {}".format(
                            str(term.atoms(AppliedUndef)), str(single_predicate)))
                    raise RicherDomainPredicateParserException()
                # Hidden edge case2: No symbols to reference columns used.
                # e.g. c_0 >= 1; c_0 in {5,7}
                elif not column_interval:
                    logger.root_logger.critical(
                        "Aborting: one predicate does not contain variables to reference columns. Invalid predicate: {}".format(
                            str(single_predicate)))
                    raise RicherDomainPredicateParserException()
                # Hidden edge case3: Term evaluates to zero.
                # e.g. 3-1.5*2 <= 0
                elif term == 0 or term.evalf() == 0:
                    logger.root_logger.critical(
                        "Aborting: one predicate does evaluate to zero. Invalid predicate: {}".format(str(single_predicate)))
                    raise RicherDomainPredicateParserException()
                # Hidden edge case4: Invalid states for important key variables reached.
                elif not split_pred or not term or not column_interval:
                    logger.root_logger.critical("Aborting: one predicate does not have a valid structure. Invalid predicate: {}".format(
                        str(single_predicate)))
                    raise RicherDomainPredicateParserException()

                # Checking if every interval-Definition, actually occurs in in the term.
                # e.g. x_0 <= c_0; c_5 in {1}  --> c_5 doesn't even occur in the term.
                for var in all_interval_defs:
                    # additional column restrictions are allowed
                    if re.match(r"x_\d+", str(var)):
                        column_interval[var] = all_interval_defs[var]
                    else:
                        logger.root_logger.critical(
                            "Aborting: invalid symbol in interval definition without symbol usage in the term found. Invalid symbol(s): {} inside predicate: {}".format(
                                str(all_interval_defs), str(single_predicate)))
                        raise RicherDomainPredicateParserException()

                return RicherDomainSplit(column_interval, coef_interval, term, relation, debug)

        logger.root_logger.critical(
            "Aborting: one predicate did not contain any relation. Invalid predicate: {}".format(str(single_predicate)))
        raise RicherDomainPredicateParserException()

    @classmethod
    def get_domain_knowledge(cls, debug=False,
                             input_file_path=r"dtcontrol/decision_tree/splitting/context_aware/input_data/input_domain_knowledge.txt"):

        """
        Function to parse domain knowledge obtained from user (stored in input_file_path)
        :param input_file_path: path with file containing user domain knowledge
        :param debug: Bool for debug mode (see Logger)
        :returns: Tuple. (Units, List of RicherDomainSplit Objects)
                        --> if input file does not contains units: (None, List of RicherDomainSplit Objects)

        Procedure:
            0. Checking whether input file path is correct/valid
            1. Processing the units from line 1
            2. Applying cls.parse_single_predicate() to every other line
            2. Returning List of all RicherDomainSplit Objects

        Structure of input file:
        # <UnitOfColumn_1> <UnitOfColumn_2> <UnitOfColumn_3> ... <UnitOfColumn_i>
        <DomainKnowledgeExpression_1>
        <DomainKnowledgeExpression_2>
        <DomainKnowledgeExpression_3>

        (<DomainKnowledgeExpression> contains a predicate, parsed by the function:  parse_single_predicate())
        """
        # Logger Init
        logger = RicherDomainLogger("GetDomainKnowledge_Logger", debug)
        logger.root_logger.info("Starting Domain Knowledge Parser.")

        # Opening and checking the input file
        try:
            with open(input_file_path, "r") as file:
                input_line = [predicate.rstrip() for predicate in file]
        except FileNotFoundError:
            logger.root_logger.info("Couldn't find input_domain_knowledge.txt file. Assuming there is no domain knowledge to start with.")
            return None, []

        # Edge Case user input == ""
        if not input_line:
            return None, []

        # output list containing all parsed predicates (+ Unit list if existing)
        output = []
        converted_units = None

        # checking whether first line contains units
        if input_line[0].startswith("#UNITS"):
            units = input_line[0].split(" ")[1:]
            converted_units = [str.lower(u) for u in units]
            input_line = input_line[1:]
            logger.root_logger.info("Units found: {}".format(converted_units))

        logger.root_logger.info(
            "Reading input file containing domain knowledge given by user. Found {} predicate(s).".format(len(input_line)))

        for single_predicate in input_line:
            logger.root_logger.info(
                "Processing user domain knowledge {} / {}.".format(input_line.index(single_predicate) + 1, len(input_line)))

            # Parse single predicate
            parsed_predicate = cls.parse_single_predicate(single_predicate, logger, debug=debug)

            logger.root_logger.info(
                "Parsed predicate {} / {} successfully. Result: {}".format(input_line.index(single_predicate) + 1,
                                                                           len(input_line), str(parsed_predicate)))
            output.append(parsed_predicate)

        logger.root_logger.info("Finished processing of user predicate. Shutting down Predicate Parser")
        return converted_units, output

    @classmethod
    def get_predicate(cls, debug=False, input_file_path=r"dtcontrol/decision_tree/splitting/context_aware/input_data/input_predicates.txt"):

        """
        Function to parse predicates obtained from user (stored in input_file_path)
        :param input_file_path: path with file containing user predicates (in every line one predicate)
        :param debug: Bool for debug mode (see Logger)
        :returns: List of RicherDomainSplit Objects

        Procedure:
            0. Checking whether input file path is correct/valid
            1. Applying cls.parse_single_predicate() to every line
            2. Returning List of all RicherDomainSplit Objects
        """
        # Logger Init
        logger = RicherDomainLogger("GetPredicate_Logger", debug)
        logger.root_logger.info("Starting Predciate Parser.")

        # Opening and checking the input file
        try:
            with open(input_file_path, "r") as file:
                input_line = [predicate.rstrip() for predicate in file]
        except FileNotFoundError:
            logger.root_logger.critical("Aborting: input file with user predicate(s) not found. Please check input file/path.")
            raise RicherDomainPredicateParserException()

        # Edge Case user input == ""
        if not input_line:
            logger.root_logger.critical("Aborting: input file with user predicates is empty. Please check file.")
            raise RicherDomainPredicateParserException()
        else:
            logger.root_logger.info(
                "Reading input file containing predicate(s) given by user. Found {} predicate(s).".format(len(input_line)))

        # output list containing all parsed predicates
        output = []
        for single_predicate in input_line:
            logger.root_logger.info("Processing user predicate {} / {}.".format(input_line.index(single_predicate) + 1, len(input_line)))

            # Parse single predicate
            parsed_predicate = cls.parse_single_predicate(single_predicate, logger, debug=debug)

            logger.root_logger.info(
                "Parsed predicate {} / {} successfully. Result: {}".format(input_line.index(single_predicate) + 1,
                                                                           len(input_line), str(parsed_predicate)))
            output.append(parsed_predicate)

        logger.root_logger.info("Finished processing of user predicate. Shutting down Predicate Parser")
        return output

    @classmethod
    def parse_user_interval(cls, user_input, logger):
        """
        Predicate Parser for the interval.
        :variable user_input: Interval as a string
        :returns: a sympy expression (to later use in self.interval of ContextAwareSplit objects)

        Option 1: user_input = (-oo, oo) = [-oo, oo]
        --> self.offset of ContextAwareSplit will be the value to achieve the 'best' impurity

        (with a,b ∊ R)
        Option 2: user_input is an interval
        Option 2.1: user_input = [a,b]
        --> Interval with closed boundary --> {x | a <= x <= b}
        Option 2.2: user_input = (a,b)
        --> Interval with open boundary --> {x | a < x < b}
        Option 2.3: user_input = (a.b]
        Option 2.4: user_input = [a,b)

        Option 3: user_input = {1,2,3,4,5}
        --> Finite set

        Option 4: user_input = [0,1) ∪ (8,9) ∪ [-oo, 1)
        --> Union of intervals


        Grammar G for an user given interval:

        G = (V, Σ, P, predicate)
        V = {predicate, combination, interval, real_interval, bracket_left, bracket_right, number, finite_interval, number_finit, num}
        Σ = { (, [, ), ], {, }, +oo, -oo, ,, ∪, -Inf, Inf, -INF, INF, -inf, inf, or, Or, OR, u}
        P:
        DEF                 -->     SET | SET ∪ SET
        SET                 -->     INFINITE_INTERVAL | FINITE_SET
        INFINITE_INTERVAL   -->     BRACKET_LEFT NUMBER_INFINITE , NUMBER_INFINITE BRACKET_RIGHT
        BRACKET_LEFT        -->     ( | [
        BRACKET_RIGHT       -->     ) | ]
        NUMBER_INFINITE     -->     x ∊ R | +oo | -oo
        FINITE_SET          -->     {NUMBER_FINITE NUM}
        NUMBER_FINITE       -->     x ∊ R
        NUM                 -->     epsilon | ,NUMBER_FINITE | ,NUMBER_FINITE NUM
        """

        logger.root_logger.info("User interval to process: {}".format(user_input))

        # simplest special case:
        if user_input == "$i":
            return sp.Interval(sp.S.NegativeInfinity, sp.S.Infinity)

        # super basic beginning and end char check of whole input
        if not user_input.strip():
            logger.root_logger.critical("Aborting: no interval found.")
            raise RicherDomainPredicateParserException()
        elif user_input.strip()[0] != '{' and user_input.strip()[0] != '(' and user_input.strip()[0] != '[':
            logger.root_logger.critical("Aborting: interval starts with an invalid char. Invalid interval: {}".format(user_input))
            raise RicherDomainPredicateParserException()
        elif user_input.strip()[-1] != '}' and user_input.strip()[-1] != ')' and user_input.strip()[
            -1] != ']':
            logger.root_logger.critical("Aborting: interval ends with an invalid char. Invalid interval: {}".format(user_input))
            raise RicherDomainPredicateParserException()

        user_input = user_input.lower()
        # Modify user_input and convert every union symbol/word into "∪" <-- ASCII Sign for Union not letter U
        user_input = user_input.replace("or", "∪")
        user_input = user_input.replace("u", "∪")

        # Modify user_input and convert every "Inf" to sympy supported symbol for infinity "oo"
        user_input = user_input.replace("inf", "oo")
        user_input = user_input.replace("infinity", "oo")

        # appending all intervals into this list and later union all of them
        interval_list = []

        user_input = user_input.split("∪")
        user_input = [x.strip() for x in user_input]

        # Parsing of every single interval
        for interval in user_input:
            """
            Basic idea: Evaluate/Parse every single predicate and later union them (if needed)
            Path is chosen based on first char of interval
            possible first char of an interval:
                --> {
                --> ( or [ (somehow belong to the same "family")

            """

            if interval[0] == "{":
                # finite intervals like {1,2,3}
                if interval[-1] == "}":
                    unchecked_members = interval[1:-1].split(",")
                    # Check each member whether they are valid
                    checked_members = []
                    for var in unchecked_members:
                        try:
                            tmp = sp.sympify(var).evalf()
                        except Exception:
                            logger.root_logger.critical("Aborting: invalid member found. Invalid interval: {}".format(user_input))
                            raise RicherDomainPredicateParserException()
                        if tmp == sp.nan:
                            logger.root_logger.critical("Aborting: invalid NaN member found. Invalid interval: {}".format(user_input))
                            raise RicherDomainPredicateParserException()
                        elif tmp == sp.S.Infinity or tmp == sp.S.NegativeInfinity:
                            logger.root_logger.critical("Aborting: infinity is an invalid member. Invalid interval: {}".format(user_input))
                            raise RicherDomainPredicateParserException()
                        elif isinstance(tmp, sp.Number):
                            checked_members.append(tmp)
                        else:
                            logger.root_logger.critical(
                                "Aborting: Invalid member found in finite interval. Invalid member: {} Invalid interval: {}".format(
                                    str(tmp), user_input))
                            raise RicherDomainPredicateParserException()
                    # Edge case: if no member is valid --> empty set will be returned.
                    out = sp.FiniteSet(*checked_members)
                    if out == sp.EmptySet:
                        logger.root_logger.critical("Aborting: Invalid empty interval found. Invalid interval: {}".format(user_input))
                        raise RicherDomainPredicateParserException()
                    else:
                        interval_list.append(out)
                else:
                    # Interval starts with { but does not end with }
                    logger.root_logger.critical("Aborting: Invalid char at end of interval found. Invalid interval: {}".format(user_input))
                    raise RicherDomainPredicateParserException()
            elif interval[0] == "(" or interval[0] == "[":
                # normal intervals of structure (1,2] etc

                # Checking of first char
                if interval[0] == "(":
                    left_open = True
                elif interval[0] == "[":
                    left_open = False
                else:
                    logger.root_logger.critical("Aborting: interval starts with an invalid char. Invalid interval: {}".format(user_input))
                    raise RicherDomainPredicateParserException()
                # Checking boundaries of interval
                tmp = interval[1:-1].split(",")
                if len(tmp) > 2:
                    logger.root_logger.critical("Aborting: too many numbers inside an interval. Invalid interval: {}".format(user_input))
                    raise RicherDomainPredicateParserException()
                try:
                    a = sp.sympify(tmp[0]).evalf()
                    b = sp.sympify(tmp[1]).evalf()
                    if a == sp.nan or b == sp.nan:
                        logger.root_logger.critical(
                            "Aborting: invalid NaN interval boundary found. Invalid interval: {}".format(user_input))
                        raise RicherDomainPredicateParserException()
                except Exception:
                    logger.root_logger.critical("Aborting: Invalid member found inside interval. Invalid interval: {}".format(user_input))
                    raise RicherDomainPredicateParserException()
                else:
                    if isinstance(a, sp.Number) and isinstance(b, sp.Number):
                        # Checking of last char
                        if interval[-1] == ")":
                            right_open = True

                        elif interval[-1] == "]":
                            right_open = False
                        else:
                            logger.root_logger.critical(
                                "Aborting: interval ends with an invalid char. Invalid interval: {}".format(user_input))
                            raise RicherDomainPredicateParserException()
                        out = sp.Interval(a, b, right_open=right_open, left_open=left_open)
                        if out == sp.EmptySet:
                            logger.root_logger.critical("Aborting: Invalid empty interval found. Invalid interval: {}".format(user_input))
                            raise RicherDomainPredicateParserException()
                        else:
                            interval_list.append(out)
                    else:
                        logger.root_logger.critical(
                            "Aborting: Invalid member found inside interval. Invalid interval: {}".format(user_input))
                        raise RicherDomainPredicateParserException()
            else:
                logger.root_logger.critical("Aborting: Invalid char found inside interval. Invalid interval: {}".format(user_input))
                raise RicherDomainPredicateParserException()

        # Union
        final_interval = interval_list[0]

        # union with all other intervals
        if len(interval_list) > 1:
            for item in interval_list:
                final_interval = sp.Union(final_interval, item)
        logger.root_logger.info("Finished processing user interval. Result: {}.".format(str(final_interval)))
        return final_interval

    @classmethod
    def parse_user_string(cls, user_input):
        """
        Method to parse a string obtained from the frontend and create the corresponding richer domain splits.
        """
        # Logger Init
        logger = RicherDomainLogger("GetPredicate_Logger", False)
        logger.root_logger.info("Starting Predciate Parser.")

        predicate_list = user_input.split("\n") if user_input else []

        # output list containing all parsed predicates
        output = []
        for single_predicate in predicate_list:
            logger.root_logger.info("Processing user predicate {} / {}.".format(predicate_list.index(single_predicate) + 1, len(predicate_list)))

            # Parse single predicate
            parsed_predicate = cls.parse_single_predicate(single_predicate, logger, debug=False)

            logger.root_logger.info(
                "Parsed predicate {} / {} successfully. Result: {}".format(predicate_list.index(single_predicate) + 1,
                                                                           len(predicate_list), str(parsed_predicate)))
            output.append(parsed_predicate)

        logger.root_logger.info("Finished processing of user predicate. Shutting down Predicate Parser")
        return output