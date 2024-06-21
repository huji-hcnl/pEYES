from typing import Sequence, Dict, Union, Optional

import numpy as np

from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum as EventLabel


from src.pEYES.sample_metrics.confusion_matrix import confusion_matrix
from src.pEYES.sample_metrics.transition_matrix import transition_matrix
from src.pEYES.sample_metrics.calculate_metrics import calculate


def accuracy(ground_truth: Sequence[EventLabel], prediction: Sequence[EventLabel],) -> float:
    return calculate(ground_truth, prediction, "accuracy")


def balanced_accuracy(ground_truth: Sequence[EventLabel], prediction: Sequence[EventLabel],) -> float:
    return calculate(ground_truth, prediction, "balanced_accuracy")


def cohen_kappa(ground_truth: Sequence[EventLabel], prediction: Sequence[EventLabel],) -> float:
    return calculate(ground_truth, prediction, "cohen's_kappa")


def mcc(ground_truth: Sequence[EventLabel], prediction: Sequence[EventLabel],) -> float:
    return calculate(ground_truth, prediction, "mcc")


def complement_nld(ground_truth: Sequence[EventLabel], prediction: Sequence[EventLabel],) -> float:
    return calculate(ground_truth, prediction, "1_nld")


def precision(
        ground_truth: Sequence[EventLabel],
        prediction: Sequence[EventLabel],
        pos_labels: Optional[Union[EventLabel, Sequence[EventLabel]]] = None,
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, "precision", pos_labels=pos_labels, average=average)


def recall(
        ground_truth: Sequence[EventLabel],
        prediction: Sequence[EventLabel],
        pos_labels: Optional[Union[EventLabel, Sequence[EventLabel]]] = None,
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, "recall", pos_labels=pos_labels, average=average)


def f1_score(
        ground_truth: Sequence[EventLabel],
        prediction: Sequence[EventLabel],
        pos_labels: Optional[Union[EventLabel, Sequence[EventLabel]]] = None,
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, "f1", pos_labels=pos_labels, average=average)

