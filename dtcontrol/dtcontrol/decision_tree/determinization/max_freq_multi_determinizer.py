from operator import itemgetter

import numpy as np

from dtcontrol.decision_tree.determinization.determinizer import Determinizer


class MaxFreqMultiDeterminizer(Determinizer):
    """
    WARNING: this determinizer is no longer supported!

    This determinizer uses the maximum frequency multi determinization approach.
    """

    def __init__(self, pre_determinize=True):
        super().__init__()
        self.pre_determinize = pre_determinize

    def determinize(self, dataset):
        """
        Given a y_train such as
        array([[[ 1,  2,  3],
                [ 1,  2,  1],
                [ 1,  2,  2],
                [ 3,  3, -1]],

               [[ 3,  4,  5],
                [ 3,  4,  4],
                [ 2,  6,  1],
                [ 3,  6, -1]]])

        gets determinized to

        array([[[ 1, -1, -1],
                [ 1, -1, -1],
                [ 1, -1, -1],
                [ 3, -1, -1]],

               [[ 3, -1, -1],
                [ 3, -1, -1],
                [ 2, -1, -1],
                [ 3, -1, -1]]])

        which is reduced to

        array([[[1],
                [1],
                [1],
                [3]],

               [[3],
                [3],
                [2],
                [3]]])
        """
        y = np.copy(dataset.y)
        determinized = False

        # list of tuples (ctrl_input_idx, input_encoding) which were already considered for keeping
        already_considered = set()

        while not determinized:
            ranks = self.get_ranks(y)

            # find the ctrl_idx and inp_enc which should be used in the next round of pruning
            # i.e. the first one from the ranking list whose input has not been already considered
            ctrl_idx = None
            inp_enc = None
            for (ctr, inp, _) in ranks:
                if (ctr, inp) not in already_considered:
                    already_considered.add((ctr, inp))
                    ctrl_idx = ctr
                    inp_enc = inp
                    break

            # Go through y[ctrl_idx] row by row
            # for each row, if it contains input_encoding, then change the remaining into -1
            # make the same -1 changes for rest of the control inputs
            for i in range(y.shape[1]):
                row = y[ctrl_idx, i]
                if inp_enc in row:
                    for j in range(y.shape[2]):
                        if row[j] != inp_enc:
                            y[:, i, j] = -1

            # check if all rows contain only one element
            determinized = True
            for ctrl_idx in range(y.shape[0]):
                for i in range(y.shape[1]):
                    row = y[ctrl_idx, i]
                    valid_row = row[row != -1]
                    determinized = determinized & (valid_row.size == 1)
        valid_y = np.array([np.array([yyy[yyy != -1] for yyy in yy]) for yy in y])
        zipped = np.stack(valid_y, axis=2)
        # [[[1,3], [1,3], [1,2], [3,3]]] -> [1,1,2,3] (tuple ids)
        return np.apply_along_axis(lambda x: self.dataset.get_tuple_to_tuple_id()[tuple(x)], axis=2,
                                   arr=zipped).flatten()

    def get_index_label(self, label):
        return self.dataset.map_tuple_id_back(label)

    def determinize_once_before_construction(self):
        return False

    def is_only_multioutput(self):
        return True

    def is_pre_split(self):
        return self.pre_determinize

    @staticmethod
    def get_ranks(y):
        """
        Generate a list of tuples (ctrl_idx, inp_enc, freq)
        where ctrl_idx is the control input index, inp_enc is the control input integer encoding and
        freq is the number of times the respective control input has occurred as the ctrl_idx'th component
        """
        ranks = []
        for ctrl_idx in range(y.shape[0]):
            flattened_control = y[ctrl_idx].flatten()
            flattened_control = flattened_control[flattened_control != -1]
            counter = list(zip(range(len(np.bincount(flattened_control))), np.bincount(flattened_control)))
            idx_input_count = [(ctrl_idx,) + l for l in counter]
            ranks.extend(idx_input_count)
        return sorted(ranks, key=itemgetter(2), reverse=True)
