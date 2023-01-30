from abc import ABC, abstractmethod


class Determinizer(ABC):
    def __init__(self):
        self.pre_determinized_labels = None

    @abstractmethod
    def determinize(self, dataset):
        """
        :param dataset: the dataset to be determinized
        :return: the determinized labels
        """
        pass

    @abstractmethod
    def is_pre_split(self):
        """
        Returns true if this determinizer should use the pre-split optimization.
        """
        pass
