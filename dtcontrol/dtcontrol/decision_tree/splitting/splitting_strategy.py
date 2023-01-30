from abc import ABC, abstractmethod


class SplittingStrategy(ABC):
    def __init__(self):
        self.priority = 1

    @abstractmethod
    def find_split(self, dataset, impurity_measure, **kwargs):
        """
        :param dataset: the subset of data at the current split
        :param impurity_measure: the impurity measure to determine the quality of a potential split
        :returns: a split object
        """
        pass
