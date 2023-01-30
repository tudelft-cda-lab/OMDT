from abc import ABC, abstractmethod


class BenchmarkSuiteClassifier(ABC):
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    @abstractmethod
    def is_applicable(self, dataset):
        pass

    @abstractmethod
    def fit(self, dataset):
        """
        Trains the classifier on the given dataset.
        """
        pass

    @abstractmethod
    def predict(self, dataset, actual_values=True):
        """
        Classifies a dataset.
        :param dataset: the dataset to classify
        :param actual_values: if True, the actual float values are predicted. if False, the index labels are predicted.
        :return: a list of the predicted labels.
        """
        pass

    @abstractmethod
    def get_stats(self):
        """
        Returns a dictionary of statistics to be saved and displayed (e.g. the number of nodes in the tree).
        :return: the dictionary of statistics
        """
        pass

    @abstractmethod
    def save(self, file):
        """
        Saves the classifier to a file (for debugging purposes).
        """
        pass

    @abstractmethod
    def print_dot(self, x_metadata, y_metadata):
        """
        Prints the classifier in the dot (graphviz) format.
        :param x_metadata: metadata describing the columns of the data
        :param y_metadata: metadata describing the columns of the labels
        :return: the dot string
        """
        pass

    @abstractmethod
    def print_c(self):
        """
        Prints the classifier as nested if-else statements in the C syntax.
        :return: the C string
        """
        pass
