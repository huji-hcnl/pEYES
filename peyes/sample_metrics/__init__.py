from typing import Union, Optional

from peyes._utils import constants as _cnst
from peyes._DataModels.EventLabelEnum import EventLabelSequenceType as _EventLabelSequenceType
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType as _UnparsedEventLabelType
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelSequenceType as _UnparsedEventLabelSequenceType

from peyes.sample_metrics._counts_and_matrices import label_counts, transition_matrix, confusion_matrix
from peyes.sample_metrics._calculate_metrics import calculate


def accuracy(ground_truth: _EventLabelSequenceType, prediction: _EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, _cnst.ACCURACY_STR)


def balanced_accuracy(ground_truth: _EventLabelSequenceType, prediction: _EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, _cnst.BALANCED_ACCURACY_STR)


def cohen_kappa(ground_truth: _EventLabelSequenceType, prediction: _EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, _cnst.COHENS_KAPPA_STR)


def mcc(ground_truth: _EventLabelSequenceType, prediction: _EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, _cnst.MCC_STR)


def complement_nld(ground_truth: _EventLabelSequenceType, prediction: _EventLabelSequenceType,) -> float:
    return calculate(ground_truth, prediction, _cnst.COMPLEMENT_NLD_STR)


def precision(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Optional[Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, _cnst.PRECISION_STR, pos_labels=pos_labels, average=average)


def recall(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Optional[Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, _cnst.RECALL_STR, pos_labels=pos_labels, average=average)


def f1_score(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Optional[Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    return calculate(ground_truth, prediction, _cnst.F1_STR, pos_labels=pos_labels, average=average)


def d_prime(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType],
        correction: str = "loglinear",
) -> float:
    return calculate(ground_truth, prediction, _cnst.D_PRIME_STR, pos_labels=pos_labels, correction=correction)


def criterion(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType],
        correction: str = "loglinear",
) -> float:
    return calculate(ground_truth, prediction, _cnst.CRITERION_STR, pos_labels=pos_labels, correction=correction)
