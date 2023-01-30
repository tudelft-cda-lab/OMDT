import numpy as np

from dtcontrol.decision_tree.determinization.determinizer import Determinizer


class MaxFreqDeterminizer(Determinizer):
    """
    This determinizer uses the maximum frequency determinization approach.
    """

    def __init__(self, pre_determinize=True):
        super().__init__()
        self.pre_determinize = pre_determinize

    def determinize(self, dataset):
        if self.is_pre_split() and self.pre_determinized_labels is not None:
            return self.pre_determinized_labels[dataset.parent_mask]
        return self.get_max_freq_labels(dataset.get_single_labels())

    @staticmethod
    def get_label_counts(labels):
        flattened_labels = labels.flatten()
        # remove -1 as we use it only as a filler
        flattened_labels = flattened_labels[flattened_labels != -1]
        label_counts = np.bincount(flattened_labels)
        return label_counts

    @staticmethod
    def get_max_freq_labels(labels):
        label_counts = MaxFreqDeterminizer.get_label_counts(labels)
        new_labels = []
        for i in range(len(labels)):
            current = labels[i]
            current = current[current != -1]
            max_label = max(list(current), key=lambda l: label_counts[l])
            assert max_label != -1
            new_labels.append(max_label)
        return np.array(new_labels)

    def is_pre_split(self):
        return self.pre_determinize
