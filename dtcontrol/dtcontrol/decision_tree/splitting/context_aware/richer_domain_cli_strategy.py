import json
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

from dtcontrol import util
from dtcontrol.decision_tree.decision_tree import Node
from dtcontrol.decision_tree.determinization.label_powerset_determinizer import (
    LabelPowersetDeterminizer,
)
from dtcontrol.decision_tree.determinization.max_freq_determinizer import (
    MaxFreqDeterminizer,
)
from dtcontrol.decision_tree.splitting.axis_aligned import (
    AxisAlignedSplit,
    AxisAlignedSplittingStrategy,
)
from dtcontrol.decision_tree.splitting.context_aware.context_aware_splitting_strategy import (
    ContextAwareSplittingStrategy,
)
from dtcontrol.decision_tree.splitting.context_aware.linear_units_classifier import (
    LinearUnitsClassifier,
)
from dtcontrol.decision_tree.splitting.context_aware.predicate_parser import (
    PredicateParser,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_exceptions import (
    RicherDomainPredicateParserException,
    RicherDomainStrategyException,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_logger import (
    RicherDomainLogger,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_split import (
    RicherDomainSplit,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_splitting_strategy import (
    RicherDomainSplittingStrategy,
)
from dtcontrol.decision_tree.splitting.linear_classifier import (
    LinearClassifierSplittingStrategy,
)
from dtcontrol.util import Caller, error_wrapper, interactive_queue, success_wrapper


class RicherDomainCliStrategy(ContextAwareSplittingStrategy):

    def __init__(self, domain_knowledge=None, determinizer=LabelPowersetDeterminizer(), debug=False):
        super().__init__()

        """
        Current path to 'domain knowledge file' = dtcontrol/decision_tree/splitting/context_aware/input_data/input_domain_knowledge.txt
        
        dataset_units:
            Storing the units of the current dataset 
                --> given in first line inside 'domain knowledge file': #UNITS ...
                --> Is optional. If the first line doesn't contain units, dataset_units will just remain None
                
        standard_predicates:
            List of RicherDomainSplit Objects given at startup inside 'domain knowledge file'
                
        """

        self.dataset_units, self.standard_predicates = PredicateParser.get_domain_knowledge(
            debug=debug) if domain_knowledge is None else domain_knowledge
        self.recently_added_predicates = []
        self.determinizer = determinizer
        self.root = None
        self.current_node = None

        # logger
        self.logger = RicherDomainLogger("RicherDomainCliStrategy_logger", debug)
        self.debug = debug

        """
        standard_predicates_imp and recently_added_predicates_imp have the same structure:
        --> List containing Tuple: [(Predicate, impurity)]
        
        Both contain the 'fitted' instances of their corresponding predicate collection
        
        standard_alt_predicates_imp:
            --> Contains all standard_predicates AND alternative_strategies predicates
        
        recently_added_predicates_imp:
            - used to store: predicates added by user via user_input_handler() --> '/add <Predicate>'
        """

        self.standard_alt_predicates_imp = []
        self.recently_added_predicates_imp = []

        # List containing alternative splitting strategies [axis, logreg, logreg_unit]
        self.alternative_strategies = self.setup_alternative_strategies()

        # Reference to richer domain splitting strategy, to get all possible splits
        self.richer_domain_strat = self.setup_richer_domain_strat()

        # Seen- Parent IDs
        self.known_parent_id = []

    def setup_alternative_strategies(self):
        """
        Function to setup the alternative splitting Strategies.
        --> Sets up Axis Aligned, Linear Classifier and if units are given Linear with respect of units.

        (Units can be given at the first line inside dtcontrol/decision_tree/splitting/context_aware/input_data/input_domain_knowledge.txt)
        """
        # Axis Aligned
        axis = AxisAlignedSplittingStrategy()
        axis.priority = 1

        # Linear
        logreg = LinearClassifierSplittingStrategy(LogisticRegression, solver='lbfgs', penalty='none')
        logreg.priority = 1

        # TODO: Enable this when we are more sure
        # Linear Units (Only if there are units given)
        # logreg_unit = LinearUnitsClassifier(LogisticRegression, self.dataset_units, solver='lbfgs', penalty='none')
        # logreg_unit.priority = 1

        # return [axis, logreg, logreg_unit] if self.dataset_units is not None else [axis, logreg]
        return [axis, logreg]

    def setup_richer_domain_strat(self):
        """
        Function to setup a richer domain splitting strategy instance.
        """

        tmp_richer_domain = RicherDomainSplittingStrategy(user_given_splits=None, debug=self.debug)
        tmp_richer_domain.priority = 1
        tmp_richer_domain.optimized_tree_check_version = False
        tmp_richer_domain.curve_fitting_method = "optimized"
        return tmp_richer_domain

    def get_all_richer_domain_splits(self, starting_predicates, dataset, impurity_measure):
        """
        Function to get all possible fitted instances of one predicate.
        :param starting_predicates: List of predicates to be processed.
        :param dataset: the subset of data at the current 'situation'
        :param impurity_measure: impurity measure to use.
        :returns: Dict containing all fitted instances. (key: split value: impurity)

        e.g:
        starting_predicate: c_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3 <= c_4

        returns: {  -0.70769*x_0 - 0.011934*x_1 - 0.38391*x_2 + 0.32438*x_3 + 1.222 <= 0  <= 0: 1.6,
                    -0.10073*x_0 + 0.00080492*x_1 + 0.5956*x_2 - 0.22309*x_3 - 0.96975 <= 0 <= 0: 1.3219280948873622,
                    0.6178*x_0 - 0.0033169*x_1 - 0.0099337*x_2 + 0.35789*x_3 - 2.4716 <= 0: inf,
                    0.19595*x_0 + 0.0047356*x_1 - 0.48916*x_2 - 0.57777*x_3 + 1.2301 <= 0 <= 0: 1.8529325012980808,
                    -0.27807*x_0 + 0.0071843*x_1 + 0.16831*x_2 + 0.16076*x_3 - 1.4741 <= 0 <= 0: 1.8529325012980808,
                    0.6178*x_0 - 0.0033169*x_1 - 0.0099337*x_2 + 0.35789*x_3 - 2.4716 <= 0: inf}

        """
        self.richer_domain_strat.first_run = True
        self.richer_domain_strat.user_given_splits = starting_predicates
        return self.richer_domain_strat.get_all_splits(dataset, impurity_measure)

    def print_dataset_specs(self, dataset, cli=True):
        """
        CAUTION!: It is not recommended to use this function alone. Use it via console_output()

        Function to print interesting specifications about the current dataset.
        :param cli: whether called from command-line interface
        :param dataset: the subset of data at the current split
        :returns: None. --> Console output
        """
        x_numeric = dataset.get_numeric_x()

        # Access metadata
        x_meta = dataset.x_metadata
        y_meta = dataset.y_metadata

        median = np.median(x_numeric, axis=0)

        ret = {}

        # FEATURE INFORMATION

        if x_meta.get('variables') is not None and x_meta.get('step_size') is not None:
            # Detailed meta data available
            table_feature = [["x_" + str(i), x_meta.get('variables')[i], str(np.min(x_numeric[:, i])),
                              str(np.max(x_numeric[:, i])),
                              str((np.min(x_numeric[:, i]) + np.max(x_numeric[:, i])) / 2),
                              str(median[i]),
                              str(x_meta.get('step_size')[i])] for i in range(x_numeric.shape[1])]

            header_feature = ["COLUMN", "NAME", "MIN", "MAX", "AVG", "MEDIAN", "STEP SIZE"]
            # Add Units if available
            if self.dataset_units is not None:
                for i in range(len(table_feature)):
                    table_feature[i].append(self.dataset_units[i])
                header_feature.append("UNIT")

            if cli:
                print("\n\t\t\t\t\t\t FEATURE INFORMATION\n" + tabulate(
                    table_feature,
                    header_feature,
                    tablefmt="psql"))
            else:
                ret.update({"feature_information": {"header": header_feature, "body": table_feature} })
        else:
            # Meta data not available
            table_feature = [["x_" + str(i), str(np.min(x_numeric[:, i])),
                              str(np.max(x_numeric[:, i])),
                              str((np.min(x_numeric[:, i]) + np.max(x_numeric[:, i])) / 2),
                              str(median[i])] for i in range(x_numeric.shape[1])]

            header_feature = ["COLUMN", "MIN", "MAX", "AVG", "MEDIAN"]
            # Add Units if available
            if self.dataset_units is not None:
                for i in range(len(table_feature)):
                    table_feature[i].append(self.dataset_units[i])
                header_feature.append("UNIT")

            if cli:
                print("\n\t\t\t FEATURE SPECIFICATION\n" + tabulate(
                    table_feature,
                    header_feature,
                    tablefmt="psql"))
            else:
                ret.update({"feature_specification":  {"header": header_feature, "body": table_feature}})

        # LABEL INFORMATION

        if y_meta.get('variables') is not None and \
                y_meta.get('min') is not None and \
                y_meta.get('max') is not None:
            # Detailed meta data available
            # TODO P: min and max should be first fixed in dataset loader, the issue is that currently ...
            # TODO P: ... we maintain a global min and max for all the datasets.
            table_label = [[y_meta.get('variables')[i], y_meta.get('min')[0], y_meta.get('max')[0],
                            y_meta.get('step_size')[i] if y_meta.get('step_size') is not None else "-"] for i in range(len(y_meta.get('variables')))]
            header_label = ["NAME", "MIN", "MAX", "STEP SIZE"]
            if cli:
                print("\n\t\t\t LABEL SPECIFICATION\n" + tabulate(
                    table_label,
                    header_label,
                    tablefmt="psql"))
            else:
                ret.update({"label_specification": {"header": header_label, "body": table_label }})
        else:
            # No meta data available
            if cli:
                print("\nNo detailed label information available.")
            else:
                ret.update({"label_specification": None})

        try:
            label_counts = MaxFreqDeterminizer.get_label_counts(dataset.get_single_labels())
            label_stats = []
            header_stats = [y_meta.get('variables')[i] for i in range(len(y_meta.get('variables')))] + ["frequency"]
            for (k, v) in enumerate(label_counts):
                if v > 0:
                    label_stats.append(list(map(str, list(map(dataset.index_label_to_actual, dataset.tuple_id_to_tuple[k])) + [v])))
            ret.update({"label_statistics": {"header": header_stats, "body": label_stats}})
        except Exception as e:
            ret.update({"label_statistics": None})

        return ret

    def print_standard_alt_predicates(self, cli=True):
        """
        CAUTION!: It is not recommended to use this function alone. Use it via console_output()

        Function to print standard_alt_predicates_imp.
        Uses only self.standard_alt_predicates_imp
        :returns: None -> Console Output
        """

        ret = {}
        if len(self.standard_alt_predicates_imp) > 0:
            table = [[self.standard_alt_predicates_imp.index(pred), round(pred[1], ndigits=3), pred[0].print_dot().replace("\\n", ""),
                      self.known_parent_id.index(pred[0].id) if isinstance(pred[0], RicherDomainSplit) else "Alternative Strategy"] for
                     pred in self.standard_alt_predicates_imp]
            header = ["INDEX", "IMPURITY", "EXPRESSION", "PARENT ID"]

            if cli:
                print("\n\t\t\t STANDARD AND ALTERNATIVE PREDICATES°\n" + tabulate(
                    table,
                    header,
                    tablefmt="psql") + "\n(°) Contains predicates obtained by user at startup, as well as one alternative Axis Aligned predicate and one or two Linear.")
            else:
                ret.update({"standard_alt_predicates": {"header": header, "body": table}})
        else:
            print("\nNo standard and alternative predicates.")

        return ret

    def print_recently_added_predicates(self, cli=True):
        """
        CAUTION!: It is not recommended to use this function alone. Use it via console_output()

        Function to print recently added predicates.
        Uses only self.recently_added_predicates_imp
        :returns: None -> Console Output
        """

        ret = {}

        if len(self.recently_added_predicates_imp) > 0:
            table = [[self.recently_added_predicates_imp.index(pred) + len(self.standard_alt_predicates_imp), round(pred[1], ndigits=3), pred[0].print_dot().replace("\\n", ""),
                      self.known_parent_id.index(pred[0].id)] for
                     pred in self.recently_added_predicates_imp]
            header = ["INDEX", "IMPURITY", "EXPRESSION", "PARENT ID"]
            if cli:
                print("\n\t\t\t RECENTLY ADDED PREDICATES\n" + tabulate(
                    table,
                    header,
                    tablefmt="psql"))
            else:
                ret.update({"recently_added_predicates": {"header": header, "body": [list(map(str, row)) for row in table]}})
        else:
            print("\nNo recently added predicates.")

        return ret

    def user_input_handler(self, dataset, impurity_measure):
        """
        Function to handle the user input via console.
        :param dataset: only used for console output (dataset infos) and to get all splits (via richer domain strat)
        :param impurity_measure: only used to get all splits (via richer domain strat)
        :returns: SplitObject

        """

        for input_line in sys.stdin:
            input_line = input_line.strip()
            if input_line == "/help":
                # display help window
                print("\n" + tabulate([["/help", "display help window"],
                                       ["/use <Index>", "select predicate at index to be returned. ('use and keep table')"],
                                       ["/use_empty <Index>",
                                        "select predicate at index to be returned. Works only on recently added table. ('use and empty table')"],
                                       ["/add <Expression>", "add an expression. (to 'recently added predicates' table)"],
                                       ["/add_standard <Expression>", "add an expression to standard and alternative predicates"],
                                       ["/del <PARENT ID>", "delete predicate with <PARENT ID>"],
                                       ["/del_all_recent", "clear recently_added_predicates list"],
                                       ["/del_all_standard", "clear standard and alternative predicates list"],
                                       ["/refresh", "refresh the console output"],
                                       ["/collection", "displays predicate collection"],
                                       ["/exit", "to exit"]],
                                      tablefmt="psql") + "\n")
            elif input_line == "/exit":
                # exit the program
                sys.exit(187)
            elif re.match("/use \d+", input_line):
                # select predicate at index to be returned. ('use and keep table')

                index = int(input_line.split("/use ")[1])
                # Index out of range check
                if index < 0 or index >= (len(self.standard_alt_predicates_imp) + len(self.recently_added_predicates_imp)):
                    print("Invalid index.")
                else:
                    users_choice = self.index_predicate_mapping(index)
                    if users_choice[1] < np.inf or self.user_double_check_inf(users_choice):
                        return users_choice[0]
                    else:
                        self.console_output(dataset)
            elif re.match("/use_empty \d+", input_line):
                # select predicate at index to be returned. Works only on recently added table. ('use and empty table')
                index = int(input_line.split("/use_empty ")[1])
                if index < len(self.standard_alt_predicates_imp):
                    print("Invalid index. /use_empty is only available on recently_added_predicates.")
                else:
                    users_choice = self.index_predicate_mapping(index)
                    if users_choice[1] < np.inf or self.user_double_check_inf(users_choice):
                        self.recently_added_predicates = []
                        self.recently_added_predicates_imp = []
                        return users_choice[0]
                    else:
                        self.console_output(dataset)
            elif input_line.startswith("/add "):
                # add an expression (to recently added predicates table)
                user_input = input_line.split("/add ")[1]
                try:
                    parsed_input = PredicateParser.parse_single_predicate(user_input, self.logger, self.debug)
                except RicherDomainPredicateParserException:
                    print("Invalid predicate entered. Please check logger or comments for more information.")
                else:
                    # Duplicate check
                    for pred in self.recently_added_predicates:
                        if pred.helper_equal(parsed_input):
                            print("ADDING FAILED: duplicate found.")
                            self.logger.root_logger.info("User tried to add a duplicate predicate to 'recently_added_predicates'.")
                            break
                    else:
                        try:
                            all_pred = self.get_all_richer_domain_splits([parsed_input], dataset, impurity_measure)
                        except RicherDomainStrategyException:
                            print("Invalid predicate parsed. Please check logger or comments for more information.")
                        else:
                            # store id
                            self.known_parent_id.append(parsed_input.id)
                            # add input to recently added predicates collection
                            self.recently_added_predicates.append(parsed_input)

                            # add all fitted instances to recently_added_predicates_imp
                            self.recently_added_predicates_imp.extend(list(all_pred.items()))
                            self.recently_added_predicates_imp.sort(key=lambda x: x[1])
                            # refresh console output
                            self.console_output(dataset)
            elif input_line.startswith("/add_standard "):
                # add an expression to standard and alternative predicates
                user_input = input_line.split("/add_standard ")[1]
                try:
                    parsed_input = PredicateParser.parse_single_predicate(user_input, self.logger, self.debug)
                except RicherDomainPredicateParserException:
                    print("Invalid predicate entered. Please check logger or comments for more information.")
                else:
                    # Duplicate check
                    for pred in self.standard_predicates:
                        if pred.helper_equal(parsed_input):
                            print("ADDING FAILED: duplicate found.")
                            self.logger.root_logger.info("User tried to add a duplicate predicate to 'standard_predicates'.")
                            break
                    else:
                        try:
                            all_pred = self.get_all_richer_domain_splits([parsed_input], dataset, impurity_measure)
                        except RicherDomainStrategyException:
                            print("Invalid predicate parsed. Please check logger or comments for more information.")
                        else:
                            # store id
                            self.known_parent_id.append(parsed_input.id)
                            # add input to standard predicates collection
                            self.standard_predicates.append(parsed_input)

                            # add all fitted instances to
                            self.standard_alt_predicates_imp.extend(list(all_pred.items()))
                            self.standard_alt_predicates_imp.sort(key=lambda x: x[1])
                            # refresh console output
                            self.console_output(dataset)
            elif input_line == "/del_all_recent":
                # clear recently_added_predicates list
                self.recently_added_predicates = []
                self.recently_added_predicates_imp = []
                self.console_output(dataset)
            elif input_line == "/del_all_standard":
                # clear standard and alternative predicates list
                self.standard_predicates = []
                new_standard_alt_predicates_imp = []
                for pred in self.standard_alt_predicates_imp:
                    if not isinstance(pred[0], RicherDomainSplit):
                        new_standard_alt_predicates_imp.append(pred)
                self.standard_alt_predicates_imp = new_standard_alt_predicates_imp
                self.console_output(dataset)
            elif re.match("/del \d+", input_line):
                # select predicate VIA ID to be deleted
                del_id = int(input_line.split("/del ")[1])
                if self.user_double_check_del():
                    if del_id < 0 or del_id >= len(self.known_parent_id) or len(self.known_parent_id) == 0:
                        print("Aborting: Invalid Id.")
                    else:
                        real_del_id = self.known_parent_id[del_id]
                        self.delete_parent_id(real_del_id)
                self.console_output(dataset)
            elif input_line == "/collection":
                # display the collections of predicates
                self.print_predicate_collections()
            elif input_line == "/refresh":
                # refresh the console output --> prints everything again
                self.console_output(dataset)
            else:
                print("Unknown command. Type '/help' for help.")

    def webui_input_handler(self, dataset, impurity_measure):
        """
        Function to handle interactions with the webui.
        :param dataset: only used for console output (dataset infos) and to get all splits (via richer domain strat)
        :param impurity_measure: only used to get all splits (via richer domain strat)
        :returns: SplitObject

        """

        while True:
            command = interactive_queue.get_from_front()
            print(command)

            if command["action"] == "use":
                # select predicate at index to be returned. ('use and keep table')

                index = int(command["body"])
                # Index out of range check
                if index < 0 or index >= (len(self.standard_alt_predicates_imp) + len(self.recently_added_predicates_imp)):
                    interactive_queue(error_wrapper("Invalid index."))
                else:
                    users_choice = self.index_predicate_mapping(index)
                    if users_choice[1] < np.inf:
                        interactive_queue.send_to_front(success_wrapper("use succeeded."))
                        return users_choice[0]
                    else:
                        interactive_queue.send_to_front(self.webui_output(dataset))
            elif command["action"] == "use_empty":
                # select predicate at index to be returned. Works only on recently added table. ('use and empty table')
                index = int(command["body"])
                if index < len(self.standard_alt_predicates_imp):
                    interactive_queue.send_to_front(error_wrapper("Invalid index. /use_empty is only available on recently_added_predicates."))
                else:
                    users_choice = self.index_predicate_mapping(index)
                    if users_choice[1] < np.inf:
                        self.recently_added_predicates = []
                        self.recently_added_predicates_imp = []
                        interactive_queue.send_to_front(success_wrapper("use_empty succeeded."))
                        return users_choice[0]
                    else:
                        interactive_queue.send_to_front(self.webui_output(dataset))
            elif command["action"] == "add":
                # add an expression (to recently added predicates table)
                user_input = command["body"]
                try:
                    parsed_input = PredicateParser.parse_single_predicate(user_input, self.logger, self.debug)
                except RicherDomainPredicateParserException:
                    interactive_queue.send_to_front(error_wrapper("Invalid predicate entered. Please check logger or comments for more information."))
                else:
                    # Duplicate check
                    for pred in self.recently_added_predicates:
                        if pred.helper_equal(parsed_input):
                            interactive_queue.send_to_front(error_wrapper("ADDING FAILED: duplicate found."))
                            self.logger.root_logger.info("User tried to add a duplicate predicate to 'recently_added_predicates'.")
                            break
                    else:
                        try:
                            all_pred = self.get_all_richer_domain_splits([parsed_input], dataset, impurity_measure)
                        except RicherDomainStrategyException:
                            interactive_queue.send_to_front(error_wrapper("Invalid predicate parsed. Please check logger or comments for more information."))
                        else:
                            # store id
                            self.known_parent_id.append(parsed_input.id)
                            # add input to recently added predicates collection
                            self.recently_added_predicates.append(parsed_input)

                            # add all fitted instances to recently_added_predicates_imp
                            self.recently_added_predicates_imp.extend(list(all_pred.items()))
                            self.recently_added_predicates_imp.sort(key=lambda x: x[1])
                            # refresh console output
                            interactive_queue.send_to_front(self.webui_output(dataset))
            elif command["action"] == "add_standard":
                # add an expression to standard and alternative predicates
                user_input = command["body"]
                try:
                    parsed_input = PredicateParser.parse_single_predicate(user_input, self.logger, self.debug)
                except RicherDomainPredicateParserException:
                    interactive_queue.send_to_front(error_wrapper("Invalid predicate entered. Please check logger or comments for more information."))
                else:
                    # Duplicate check
                    for pred in self.standard_predicates:
                        if pred.helper_equal(parsed_input):
                            interactive_queue.send_to_front(error_wrapper("ADDING FAILED: duplicate found."))
                            self.logger.root_logger.info("User tried to add a duplicate predicate to 'standard_predicates'.")
                            break
                    else:
                        try:
                            all_pred = self.get_all_richer_domain_splits([parsed_input], dataset, impurity_measure)
                        except RicherDomainStrategyException:
                            interactive_queue.send_to_front(error_wrapper("Invalid predicate parsed. Please check logger or comments for more information."))
                        else:
                            # store id
                            self.known_parent_id.append(parsed_input.id)
                            # add input to standard predicates collection
                            self.standard_predicates.append(parsed_input)

                            # add all fitted instances to
                            self.standard_alt_predicates_imp.extend(list(all_pred.items()))
                            self.standard_alt_predicates_imp.sort(key=lambda x: x[1])
                            # refresh console output
                            interactive_queue.send_to_front(self.webui_output(dataset))
            elif command["action"] == "del_all_recent":
                # clear recently_added_predicates list
                self.recently_added_predicates = []
                self.recently_added_predicates_imp = []
                interactive_queue.send_to_front(self.webui_output(dataset))
            elif command["action"] == "del_all_standard":
                # clear standard and alternative predicates list
                self.standard_predicates = []
                new_standard_alt_predicates_imp = []
                for pred in self.standard_alt_predicates_imp:
                    if not isinstance(pred[0], RicherDomainSplit):
                        new_standard_alt_predicates_imp.append(pred)
                self.standard_alt_predicates_imp = new_standard_alt_predicates_imp
                interactive_queue.send_to_front(self.webui_output(dataset))
            elif command["action"] == "del":
                # select predicate VIA ID to be deleted
                del_id = int(command["body"])
                if del_id < 0 or del_id >= len(self.known_parent_id) or len(self.known_parent_id) == 0:
                    interactive_queue.send_to_front(error_wrapper("Aborting: Invalid Id."))
                else:
                    real_del_id = self.known_parent_id[del_id]
                    self.delete_parent_id(real_del_id)
                interactive_queue.send_to_front(self.webui_output(dataset))
            elif command["action"] == "collection":
                # display the collections of predicates
                interactive_queue.send_to_front(self.print_predicate_collections(cli=False))
            elif command["action"] == "refresh":
                # refresh the console output --> prints everything again
                interactive_queue.send_to_front(self.webui_output(dataset))
            else:
                interactive_queue.send_to_front(error_wrapper("Unknown command."))

    def delete_parent_id(self, parent_id):
        # check inside standard_predicates
        new_standard_predicates = []
        for pred in self.standard_predicates:
            if not pred.id == parent_id:
                new_standard_predicates.append(pred)
        self.standard_predicates = new_standard_predicates
        # check inside standard_alt_predicates_imp
        new_standard_alt_predicates_imp = []
        for pred in self.standard_alt_predicates_imp:
            if not isinstance(pred[0], RicherDomainSplit) or not pred[0].id == parent_id:
                new_standard_alt_predicates_imp.append(pred)
        self.standard_alt_predicates_imp = new_standard_alt_predicates_imp
        # check inside recently_added_predicates
        new_recently_added_predicates = []
        for pred in self.recently_added_predicates:
            if not pred.id == parent_id:
                new_recently_added_predicates.append(pred)
        self.recently_added_predicates = new_recently_added_predicates
        # check inside recently_added_predictes imp
        new_recently_added_predicates_imp = []
        for pred in self.recently_added_predicates_imp:
            if not pred[0].id == parent_id:
                new_recently_added_predicates_imp.append(pred)
        self.recently_added_predicates_imp = new_recently_added_predicates_imp

    def print_predicate_collections(self, cli=True):
        """
        Function to give a visual representation of the predicates collection.
        """

        ret = {}

        header = ["ID", "TERM", "COLUMN INTERVAL", "COEF INTERVAL"]

        # Print standard and alternative collection
        table_standard = [[self.known_parent_id.index(pred.id), str(pred.term) + " " + pred.relation + " 0",
                           pred.column_interval, pred.coef_interval] for pred in
                          self.standard_predicates]
        if cli:
            print("\n\t\t\t\tSTANDARD PREDICATES COLLECTION\n" + tabulate(table_standard, header, tablefmt="psql") + "\n")
        else:
            ret.update({"standard_predicates_collection": {"header": header, "body": table_standard}})

        # Print standard and alternative collection
        table_recently_added = [[self.known_parent_id.index(pred.id), str(pred.term) + " " + pred.relation + " 0",
                                 str(pred.column_interval), str(pred.coef_interval)] for pred in
                                self.recently_added_predicates]
        # Print recently added collection
        if cli:
            print("\n\t\t\t\tRECENTLY ADDED PREDICATES COLLECTION\n" + tabulate(table_recently_added, header, tablefmt="psql") + "\n")
        else:
            ret.update({"recently_added_predicates_collection": {"header": header, "body": table_recently_added}})

        return ret

    def user_double_check_del(self):
        """
        Function to double check whether user really wants to delete a predicate.
        :returns: Bool
        """
        print("Are you sure, you want to delete that predicate? Remember that you will delete the predicate from the collection."
              "\nPlease enter y/n")
        for input_line in sys.stdin:
            input_line = input_line.strip()
            if input_line == "y" or input_line == "yes":
                return True
            elif input_line == "n" or input_line == "no":
                return False
            else:
                print("Unkown command. Please enter y/n")

    def user_double_check_inf(self, users_choice):
        """
        Function to double check whether user really wants to return a split with impurity inf
        :param users_choice: (Predicate, Impurity)
        :returns: Bool
        """
        user_split, imp = users_choice
        if imp < np.inf:
            return True
        else:
            # ARE YOU SURE - loop
            print("Are you sure, that you want to return a split with impurity of INFINITY ?\nPlease enter y/n")
            for input_line in sys.stdin:
                input_line = input_line.strip()
                if input_line == "y" or input_line == "yes":
                    return True
                elif input_line == "n" or input_line == "no":
                    return False
                else:
                    print("Unkown command. Please enter y/n")

    def process_standard_alt_predicates(self, dataset, impurity_measure):
        """
        Function to setup standard_alt_predicates_imp for future usage.
        :param dataset: the subset of data at the current 'situation'
        :param impurity_measure: impurity measure to use
        :returns: None. --> Sets up self.standard_alt_predicates_imp

        self.standard_alt_predicates_imp
        --> List containing Tuple: [(Predicate, impurity)]

        Contains the 'fitted' instances of its corresponding predicate collection --> (self.standard_predicates)
        AND alternative_strategies predicates

        Procedure:
            1. All predicates inside self.standard_predicates get fit and checked
            2. Add additional alternative splitting strategies:
                2.1 Axis Aligned
                2.2 Linear Basic
                2.3 If Units available: Linear with Unit respect
            3. Sort self.standard_alt_predicates_imp
        """
        for pred in self.standard_predicates:
            if not self.known_parent_id.__contains__(pred.id):
                self.known_parent_id.append(pred.id)

        all_predicates = self.get_all_richer_domain_splits(self.standard_predicates, dataset, impurity_measure)
        self.standard_alt_predicates_imp = list(all_predicates.items())

        # Add the split objects from self.alternative_strategies
        for strat in self.alternative_strategies:
            pred = strat.find_split(dataset, impurity_measure)
            imp = impurity_measure.calculate_impurity(dataset, pred)
            self.standard_alt_predicates_imp.append((pred, imp))

        self.standard_alt_predicates_imp.sort(key=lambda x: x[1])

    def process_recently_added_predicates(self, dataset, impurity_measure):
        """
        Function to setup recently_added_predicates_imp for future usage.
        :param dataset: the subset of data at the current 'situation'
        :param impurity_measure: impurity measure to use
        :returns: None. --> Sets up self.recently_added_predicates_imp

        self.recently_added_predicates_imp
        --> List containing Tuple: [(Predicate, impurity)]

        contains predicates added by user via user_input_handler() --> '/add <Predicate>'

        Procedure:
            1. All predicates inside self.recently_added_predicates get fit and checked
            2. Sort self.standard_alt_predicates_imp
        """
        for pred in self.recently_added_predicates:
            if not self.recently_added_predicates.__contains__(pred.id):
                self.known_parent_id.append(pred.id)

        all_predicates = self.get_all_richer_domain_splits(self.recently_added_predicates, dataset, impurity_measure)
        self.recently_added_predicates_imp = list(all_predicates.items())
        self.recently_added_predicates_imp.sort(key=lambda x: x[1])

    def find_split(self, dataset, impurity_measure, **kwargs):

        """
        :param **kwargs:
        :param dataset: the subset of data at the current split
        :param impurity_measure: impurity measure to use
        :returns: split object

        Procedure:
            1. Process Standard and alternative predicates
            2. Process recently added predicates
            3. Print console output
            4. Start user_input_handler()
                --> possibility for user to add/del predicates
            5. Returned split chosen by user via user_input_handler()

        """

        self.process_standard_alt_predicates(dataset, impurity_measure)
        self.process_recently_added_predicates(dataset, impurity_measure)

        if "caller" in kwargs and kwargs["caller"] == Caller.WEBUI:
            self.webui_output(dataset)

            # handle_user_input
            return_split = self.webui_input_handler(dataset, impurity_measure)
        else:
            self.console_output(dataset)

            # handle_user_input
            return_split = self.user_input_handler(dataset, impurity_measure)


        self.logger.root_logger.info("Returned split: {}".format(str(return_split)))
        return return_split

    def index_predicate_mapping(self, index):
        """
        Function to map an index to the corresponding predicate.
        Double checks if user want to return split with impurity of inf.
        :param index: Integer. Index of predicate as displayed in the console output.
        :returns: Split Object at index
        """
        if index < len(self.standard_alt_predicates_imp):
            return self.standard_alt_predicates_imp[index]
        else:
            return self.recently_added_predicates_imp[index - len(self.standard_alt_predicates_imp)]

    def console_output(self, dataset):
        """
        Function to print out the visual representation to the console.
        :param dataset: the subset of data at the current 'situation'
        :returns: None --> Console output
        """

        self.print_dataset_specs(dataset)
        self.print_standard_alt_predicates()
        self.print_recently_added_predicates()
        print("\nSTARTING INTERACTIVE SHELL. PLEASE ENTER YOUR COMMANDS. TYPE '/help' FOR HELP.\n")

    def webui_output(self, dataset):
        res = {}
        res.update(self.print_dataset_specs(dataset, cli=False))
        res.update(self.print_standard_alt_predicates(cli=False))
        res.update(self.print_recently_added_predicates(cli=False))
        res.update(self.print_predicate_collections(cli=False))
        return json.dumps({"type": "update", "body": res}, default=util.convert)
