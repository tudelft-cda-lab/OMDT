from abc import ABC

from .impurity_measure import ImpurityMeasure


class MultiLabelImpurityMeasure(ImpurityMeasure, ABC):

    def get_oc1_name(self):
        return None
