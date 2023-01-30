from copy import deepcopy

import numpy as np

from dtcontrol.decision_tree.determinization.label_powerset_determinizer import (
    LabelPowersetDeterminizer,
)
from dtcontrol.decision_tree.splitting.context_aware.context_aware_splitting_strategy import (
    ContextAwareSplittingStrategy,
)
from dtcontrol.decision_tree.splitting.context_aware.predicate_parser import (
    PredicateParser,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_exceptions import (
    RicherDomainStrategyException,
)
from dtcontrol.decision_tree.splitting.context_aware.richer_domain_logger import (
    RicherDomainLogger,
)


class RicherDomainSplittingStrategy(ContextAwareSplittingStrategy):

    def __init__(self, user_given_splits="", determinizer=LabelPowersetDeterminizer(), debug=False):

        """
        :param user_given_splits: predicates/splits obtained by user to work with. Parsed by predicate_parser.py
        :param determinizer: determinizer
        """
        super().__init__()
        """
        parses a huge string of predicates, if this string is provided. Otherwise PredicateParser.get_predicate() will go into 
        dtcontrol/decision_tree/splitting/context_aware/input_data/input_predicates.txt and parse the predicates from there.
        """
        self.user_given_splits = PredicateParser.parse_user_string(user_given_splits) if user_given_splits != "" else PredicateParser.get_predicate()
        self.determinizer = determinizer
        self.first_run = True

        # helper attributes used to store the dt while it is being built
        # Will be set inside decision_tree.py and later used inside self.get_path_root_current()
        self.root = None
        self.current_node = None

        # Checks whether predicate without coefs was already used in current dt path. Can lead in small(!) dt to performance boost.
        self.optimized_tree_check_version = True

        """
        {‘lm’, ‘trf’, ‘dogbox’, 'optimized'}
        "The method ‘lm’ won’t work when the number of observations is less than the number of variables, use ‘trf’ or ‘dogbox’ in this 
        case." https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html (go to section: method)
        
        Default: 'lm' Levenberg-Marquardt
        'trf' Trust Region Reflective
        'dogbox' Dogleg algorithm
        'optimized' Uses 'lm' whenever possible. Else 'trf' 
        
        CAUTION RUNTIME:
        lm < optimized < trf < dogbox
        """
        self.curve_fitting_method = "lm"

        # logger
        self.logger = RicherDomainLogger("RicherDomainSplittingStrategy_logger", debug)

    def get_path_root_current(self, ancestor_range=0, observed_node=None, path=[]):

        """
        Standard depth first search.
        (Default: starts from self.root)
        !!!!!!!! This function is always searching for self.current_node !!!!!!!!!

        :param ancestor_range: number of closest ancestor nodes to return
        :param observed_node: current node being checked
        :param path: list containing ancestor path (from self.root to self.current_node)
        :returns: list of ancestor nodes from self.root to self.current_node (containing only the last ancestor_range ancestors)
        """

        # Default starting node
        if observed_node is None:
            observed_node = self.root

        path_copy = path.copy()
        path_copy.append(observed_node)
        if self.current_node in observed_node.children:
            for node in path:
                if node.split is None:
                    path_copy.remove(node)
            return path_copy[-ancestor_range:]
        elif not observed_node.children:
            return None
        else:
            for node in observed_node.children:
                result = self.get_path_root_current(ancestor_range=ancestor_range, observed_node=node, path=path_copy)
                if result:
                    return result
            return None

    def print_path_root_current(self, split_list):

        """
        Super simple debugging function to quickly visualize the ancestor path returned from self.get_path_root_current()
        :param split_list: ancestor path returned by get_path_root_current
        :returns: None (Just super simple visual representation for terminal)
        """

        print("\n----------------------------")
        if split_list:
            for i in split_list:
                print("Parent Split: ", i.split.print_dot())
                # self.get_tree_distance(i.split.predicate, None)

        else:
            print("No parent splits yet")
        print("----------------------------")

    def get_all_splits(self, dataset, impurity_measure):

        """
        :param dataset: the subset of data at the current split
        :param impurity_measure: the impurity measure to determine the quality of a potential split
        :returns: dict with all user given splits + impurity

        Procedure:
            0. predicates given by user (stored in: dtcontrol/decision_tree/splitting/context_aware/input_data/input_predicates.txt)
                get parsed by dtcontrol/decision_tree/splitting/context_aware/predicate_parser.py
            1. If first_run, some basic predicate checking will be done (valid variables and dataset combinations etc...)
            2. Iterating over every given predicate and check whether predicate contains coefs
                2.1 If predicate does not contain coefs:
                    - Compute impurity and store in dict
                2.2 If predicate does contain coefs:
                    - Iterate over all unique labels and fit those coefs to that specific label mask
            3. Return dict with key:split object and value:Impurity of the split
        """

        x_numeric = dataset.get_numeric_x()
        if x_numeric.shape[1] == 0:
            return

        y = self.determinizer.determinize(dataset)

        # Will be only executed once at startup.
        if self.first_run:
            # creating some additional logger information
            self.logger.root_logger.info(
                "Current dataset specification:\n\t- x_metadate: {}\n\t- y_metadata: {}\n\t- rows: {}\n\t- columns: {}".format(
                    dataset.x_metadata,
                    dataset.y_metadata,
                    x_numeric.shape[0],
                    x_numeric.shape[1]))
            # Checking whether used column references in user_given_splits are actually represented in the given dataset.
            for single_split in self.user_given_splits:
                if not single_split.check_valid_column_reference(x_numeric):
                    self.logger.root_logger.critical(
                        "Aborting: one predicate uses an invalid column reference. Invalid predicate {}".format(str(single_split)))
                    raise RicherDomainStrategyException(
                        "Aborting: one predicate uses an invalid column reference. Check logger or comments for more information.")
            self.first_run = False
        else:
            # creating some additional logger information
            self.logger.root_logger.info(
                "Calling RicherDomainSplittingStrategy with: \n- Current dataset size:\n\t- rows: {}\n\t- columns: {}".format(
                    x_numeric.shape[0],
                    x_numeric.shape[1]))

        predicate_list = []
        if self.optimized_tree_check_version:
            """
            Checking whether predicate without coefs was already used in current dt path. Can lead in small(!) dt to huge performance boost.
            """
            root_path = self.get_path_root_current()
            ancestor_splits = [node.split for node in root_path] if root_path is not None else []

            for i in self.user_given_splits:
                if i.coef_interval:
                    predicate_list.append(i)
                else:
                    mask = [i.helper_equal(predicate) for predicate in ancestor_splits]
                    if True not in mask:
                        predicate_list.append(i)
        else:
            predicate_list = self.user_given_splits

        """
        Iterating over every user given predicate/split and adjusting it to the current dataset,
        to achieve the lowest impurity possible.
        All adjusted predicate/split objects will be stored inside the dict 'splits' 
        Key: split object   Value:Impurity of the split
        """
        splits = {}
        term_collection = []
        # Similar approach as in linear_classifier.py
        for single_split in predicate_list:
            """
            Checking if every column reference only contains values of its Interval.
            e.g. 
            column_interval = {x_1:(-Inf,Inf), x_2:{1,2,3}}
            --> For column x_1 there are no restrictions
            --> Inside column x_2 the only allowed values are {1,2,3} 
            """
            self.logger.root_logger.info("Processing predicate {} / {}".format(predicate_list.index(single_split) + 1, len(predicate_list)))
            if single_split.check_data_in_column_interval(x_numeric):
                # Checking whether predicate has to be fitted to data or not.
                if single_split.contains_unfixed_coefs():
                    # Creating different copies of the predicate and fitting every copy to one single unique label.
                    for label in np.unique(y):
                        # Creating the label mask (see linear classifier)
                        new_y = np.copy(y)
                        label_mask = (new_y == label)
                        new_y[label_mask] = 0 if single_split.relation == "=" else 1
                        new_y[~label_mask] = -1

                        """
                        If there are already fixed coefs:
                        Applying fit function for every possible combination of already fixed coefs

                        e.g.
                        Split: c_0*x_0+c_1*x_1+c_2*x_2+c_3*x_3+c_4 <= 0;c_1 in {1,2,3}; c_2 in {-1,-3}

                        -->         combinations = [[('c_1', 1), ('c_2', -3)], [('c_1', 1), ('c_2', -1)],
                                                    [('c_1', 2), ('c_2', -3)], [('c_1', 2), ('c_2', -1)],
                                                    [('c_1', 3), ('c_2', -3)], [('c_1', 3), ('c_2', -1)]]

                        --> The other coefs (c_0, c_3, c_4) still have to be determined by fit (curve_fit)

                        """

                        combinations = single_split.get_fixed_coef_combinations()

                        # Creating and fitting predicate for every combination
                        for comb in combinations:
                            split_copy = deepcopy(single_split)
                            split_copy.fit(comb, x_numeric, new_y, method=self.curve_fitting_method)
                            split_copy.priority = self.priority

                            # Checking whether fitting was successful
                            if split_copy.coef_assignment is not None:
                                # Checking for duplicates
                                evaluated_term = split_copy.term.subs(split_copy.coef_assignment)
                                for t in term_collection:
                                    if evaluated_term.equals(t):
                                        break
                                else:
                                    term_collection.append(evaluated_term)
                                    splits[split_copy] = impurity_measure.calculate_impurity(dataset, split_copy)
                else:
                    # Predicate only contains fixed or no coefs
                    combinations = single_split.get_fixed_coef_combinations()
                    # Creating and fitting predicate for every combination
                    for comb in combinations:
                        split_copy = deepcopy(single_split)
                        split_copy.coef_assignment = comb
                        split_copy.priority = self.priority

                        # Checking for duplicates
                        evaluated_term = split_copy.term.subs(split_copy.coef_assignment)
                        for t in term_collection:
                            if evaluated_term.equals(t):
                                break
                        else:
                            term_collection.append(evaluated_term)
                            splits[split_copy] = impurity_measure.calculate_impurity(dataset, split_copy)
            self.logger.root_logger.info(
                "Finished processing predicate {} / {}".format(predicate_list.index(single_split) + 1, len(predicate_list)))

        # Returning dict containing all possible splits with their impurity
        return splits

    def find_split(self, dataset, impurity_measure, **kwargs):

        """
        :param **kwargs:
        :param dataset: the subset of data at the current split
        :param impurity_measure: the impurity measure to determine the quality of a potential split
        :returns: split object with lowest impurity

        Procedure:
            1. self.get_all_splits() computes all possible split given by user in
                dtcontrol/decision_tree/splitting/context_aware/input_data/input_predicates.txt
            2. Split with lowest impurity gets returned.
        """

        splits = self.get_all_splits(dataset, impurity_measure, **kwargs)

        # Returning split with lowest impurity
        output_split = min(splits.keys(), key=splits.get) if splits else None

        self.logger.root_logger.info("Returned split: {}".format(str(output_split)))
        return output_split
