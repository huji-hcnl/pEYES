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


def complement_nld(ground_truth: _EventLabelSequenceType, prediction: _EventLabelSequenceType) -> float:
    """
    Calculates the complement of normalized Levenshtein distance (1-NLD) between two sequences.
    Levenshtein distance is the number of edits required to transform one sequence into another, and its normalization
    is with respect to the size of the GT (equivalent to the WER measure used in speech recognition). This normalization
    may result in a value exceeding 1 if the predicted sequence is longer than the GT sequence, meaning the complement
    (1-NLD) may be negative.
    For more information on using 1-NLD in eye-tracking, see https://doi.org/10.3758/s13428-021-01763-7
    """
    return calculate(ground_truth, prediction, _cnst.COMPLEMENT_NLD_STR)


def precision(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Optional[Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    """
    Calculate the precision of the prediction sequence compared to the ground truth.
    :param ground_truth: the ground truth sequence of labels
    :param prediction: the predicted sequence of labels
    :param pos_labels: the positive label(s) to consider. Other labels (if exist) are considered negative.
    :param average: the averaging strategy for precision. default is "weighted"
    """
    return calculate(ground_truth, prediction, _cnst.PRECISION_STR, pos_labels=pos_labels, average=average)


def recall(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Optional[Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    """
    Calculate the recall of the prediction sequence compared to the ground truth.
    :param ground_truth: the ground truth sequence of labels
    :param prediction: the predicted sequence of labels
    :param pos_labels: the positive label(s) to consider. Other labels (if exist) are considered negative.
    :param average: the averaging strategy for recall. default is "weighted"
    """
    return calculate(ground_truth, prediction, _cnst.RECALL_STR, pos_labels=pos_labels, average=average)


def f1_score(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Optional[Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType]],
        average: str = "weighted",
) -> float:
    """
    Calculate the F1-score of the prediction sequence compared to the ground truth.
    :param ground_truth: the ground truth sequence of labels
    :param prediction: the predicted sequence of labels
    :param pos_labels: the positive label(s) to consider. Other labels (if exist) are considered negative.
    :param average: the averaging strategy for F1-score. default is "weighted"
    """
    return calculate(ground_truth, prediction, _cnst.F1_STR, pos_labels=pos_labels, average=average)


def d_prime(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType],
        correction: str = "loglinear",
) -> float:
    """
    Calculates the discriminability (d-prime) of the prediction sequence compared to the ground truth.
    :param ground_truth: the ground truth sequence of labels
    :param prediction: the predicted sequence of labels
    :param pos_labels: the positive label(s) to consider. Other labels (if exist) are considered negative.
    :param correction: the correction method for floor/ceiling effects on the hit-rate and/or false-alarm rate.
        default is "loglinear".
        See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
        See implementation details at https://lindeloev.net/calculating-d-in-python-and-php/.
    """
    return calculate(ground_truth, prediction, _cnst.D_PRIME_STR, pos_labels=pos_labels, correction=correction)


def criterion(
        ground_truth: _EventLabelSequenceType,
        prediction: _EventLabelSequenceType,
        pos_labels: Union[_UnparsedEventLabelType, _UnparsedEventLabelSequenceType],
        correction: str = "loglinear",
) -> float:
    """
    Calculates the criterion of the prediction sequence compared to the ground truth.
    :param ground_truth: the ground truth sequence of labels
    :param prediction: the predicted sequence of labels
    :param pos_labels: the positive label(s) to consider. Other labels (if exist) are considered negative.
    :param correction: the correction method for floor/ceiling effects on the hit-rate and/or false-alarm rate.
        default is "loglinear".
        See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
        See implementation details at https://lindeloev.net/calculating-d-in-python-and-php/.
    """
    return calculate(
        ground_truth, prediction, _cnst.CRITERION_STR, pos_labels=pos_labels, correction=correction
    )
