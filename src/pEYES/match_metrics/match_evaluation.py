from typing import Sequence, Union, Optional

import numpy as np

from src.pEYES._DataModels.Event import BaseEvent
from src.pEYES._DataModels.EventMatcher import OneToOneEventMatchesType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

from src.pEYES._utils.metric_utils import dprime


def precision_recall_f1(
        ground_truth: Sequence[BaseEvent],
        prediction: Sequence[BaseEvent],
        matches: OneToOneEventMatchesType,
        positive_labels: Union[EventLabelEnum, Sequence[EventLabelEnum]],
) -> (float, float, float):
    """
    Calculates the precision, recall, and F1-score for the given ground-truth and predicted events, where successfully
    matched events are considered true positives. The provided labels are considered as "positive" events.

    :param ground_truth: all ground-truth events
    :param prediction: all predicted events
    :param matches: the one-to-one matches between (subset of) ground-truth and (subset of) predicted events
    :param positive_labels: event-labels to consider as "positive" events
    :return: the precision, recall, and F1-score values
    """
    p, n, pp, tp = _extract_contingency_values(ground_truth, prediction, matches, positive_labels)
    recall = tp / p if p > 0 else np.nan
    precision = tp / pp if pp > 0 else np.nan
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else np.nan
    return precision, recall, f1


def d_prime(
        ground_truth: Sequence[BaseEvent],
        prediction: Sequence[BaseEvent],
        matches: OneToOneEventMatchesType,
        positive_labels: Union[EventLabelEnum, Sequence[EventLabelEnum]],
        correction: Optional[str] = "loglinear",
) -> float:
    """
    Calculates d-prime while optionally applying a correction for floor/ceiling effects on the hit-rate and/or
    false-alarm rate. See information on correction methods at https://stats.stackexchange.com/a/134802/288290.

    :param ground_truth: all ground-truth events
    :param prediction: all predicted events
    :param matches: the one-to-one matches between (subset of) ground-truth and (subset of) predicted events
    :param positive_labels: event-labels to consider as "positive" events
    :param correction: optional floor/ceiling correction method when calculating hit-rate and false-alarm rate
    :return: the d-prime value
    """
    p, n, pp, tp = _extract_contingency_values(ground_truth, prediction, matches, positive_labels)
    return dprime(p, n, pp, tp, correction)


def _extract_contingency_values(
        ground_truth: Sequence[BaseEvent],
        prediction: Sequence[BaseEvent],
        matches: OneToOneEventMatchesType,
        positive_labels: Union[EventLabelEnum, Sequence[EventLabelEnum]],
) -> (int, int, int, int):
    """
    Extracts contingency values, used to fill in the confusion matrix for the provided matches between ground-truth and
    predicted events, where the given labels are considered "positive" events.

    :return:
        p: int; number of positive GT events
        n: int; number of negative GT events
        pp: int; number of positive predicted events
        tp: int; number of true positive predictions
    """
    positive_labels = [positive_labels] if isinstance(positive_labels, EventLabelEnum) else positive_labels
    p = len([e for e in ground_truth if e.label in positive_labels])
    n = len(ground_truth) - p
    pp = len([e for e in prediction if e.label in positive_labels])
    tp = len([e for e in matches.values() if e.label in positive_labels])
    return p, n, pp, tp
