from typing import Union, Optional

from peyes._utils import constants as cnst
from peyes._DataModels.EventLabelEnum import EventLabelSequenceType
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

from peyes.sample_metrics._counts_and_matrices import label_counts, transition_matrix, confusion_matrix
from peyes.sample_metrics._calculate_metrics import calculate


def accuracy(ground_truth: EventLabelSequenceType, prediction: EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, cnst.ACCURACY_STR)


def balanced_accuracy(ground_truth: EventLabelSequenceType, prediction: EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, cnst.BALANCED_ACCURACY_STR)


def cohen_kappa(ground_truth: EventLabelSequenceType, prediction: EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, cnst.COHENS_KAPPA_STR)


def mcc(ground_truth: EventLabelSequenceType, prediction: EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, cnst.MCC_STR)


def complement_nld(ground_truth: EventLabelSequenceType, prediction: EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, cnst.COMPLEMENT_NLD_STR)


def precision(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, cnst.PRECISION_STR, pos_labels=pos_labels, average=average)


def recall(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, cnst.RECALL_STR, pos_labels=pos_labels, average=average)


def f1_score(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, cnst.F1_STR, pos_labels=pos_labels, average=average)


def d_prime(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        pos_labels: Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType],
        correction: str = "loglinear",
) -> float:
    return calculate(ground_truth, prediction, cnst.D_PRIME_STR, pos_labels=pos_labels, correction=correction)


def criterion(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        pos_labels: Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType],
        correction: str = "loglinear",
) -> float:
    return calculate(ground_truth, prediction, cnst.CRITERION_STR, pos_labels=pos_labels, correction=correction)
