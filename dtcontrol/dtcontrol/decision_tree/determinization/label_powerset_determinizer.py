from .determinizer import Determinizer


class LabelPowersetDeterminizer(Determinizer):
    """
    This determinizer doesn't actually do any determinization but simply uses the label powerset approach.
    """

    def determinize(self, dataset):
        if self.pre_determinized_labels is not None:
            return self.pre_determinized_labels[dataset.parent_mask]
        return dataset.get_unique_labels()

    def is_pre_split(self):
        return True
