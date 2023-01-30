from abc import ABC

from ..determinization.label_powerset_determinizer import LabelPowersetDeterminizer
from .impurity_measure import ImpurityMeasure


class DeterminizingImpurityMeasure(ImpurityMeasure, ABC):

    def __init__(self, determinizer=LabelPowersetDeterminizer()):
        self.determinizer = determinizer
