from abc import ABC, abstractmethod

from ..splitting_strategy import SplittingStrategy


class ContextAwareSplittingStrategy(SplittingStrategy, ABC):
    """
    Represents a splitting strategy especially used inside richer_domain_splitting_strategy.py
    """

    def __init__(self):
        """
        self.root contains a reference to the root node of the current DT to access the DT while it is being build
        self.current_node contains a reference to the current_node which is beeing
        """
        super().__init__()
        self.root = None
        self.current_node = None

    @abstractmethod
    def find_split(self, dataset, impurity_measure, **kwargs):
        """
        :param **kwargs:
        :param dataset: the subset of data at the current split
        :param impurity_measure: the impurity measure to determine the quality of a potential split
        :returns: a split object
        """
        pass

    def set_root(self, root):
        self.root = root

    def set_current_node(self, current_node):
        self.current_node = current_node
