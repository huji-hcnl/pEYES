from typing import Optional, Union

import numpy as np

from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES._DataModels.EventMatcher import OneToOneEventMatchesType
from src.pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

from src.pEYES._utils.event_utils import parse_label
from src.pEYES._utils.metric_utils import dprime_and_criterion


def match_ratio(
        prediction: EventSequenceType,
        matches: OneToOneEventMatchesType,
        labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
) -> float:
    """
    Calculates the ratio of matched events to the total number of predicted events, optionally filtered by the given
    event labels. Returns NaN if there are no predicted events.

    :param prediction: all predicted events
    :param matches: the one-to-one matches between (subset of) ground-truth and (subset of) predicted events
    :param labels: optional event-labels to consider when calculating the ratio

    :return: the match ratio
    """
    if len(prediction) == 0:
        return np.nan
    if labels is None:
        labels = set(EventLabelEnum)
    elif isinstance(labels, UnparsedEventLabelType):
        labels = {parse_label(labels)}
    else:
        labels = set(parse_label(l) for l in labels)
    return sum(1 for e in matches.values() if e.label in labels) / len(prediction)


def precision_recall_f1(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        matches: OneToOneEventMatchesType,
        positive_label: UnparsedEventLabelType,
) -> (float, float, float):
    """
    Calculates the precision, recall, and F1-score for the given ground-truth and predicted events, where successfully
    matched events are considered true positives. The provided labels are considered as "positive" events.

    :param ground_truth: all ground-truth events
    :param prediction: all predicted events
    :param matches: the one-to-one matches between (subset of) ground-truth and (subset of) predicted events
    :param positive_label: event-label to consider as "positive" events
    :return: the precision, recall, and F1-score values
    """
    p, n, pp, tp = _extract_contingency_values(ground_truth, prediction, matches, positive_label)
    recall = tp / p if p > 0 else np.nan
    precision = tp / pp if pp > 0 else np.nan
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else np.nan
    return precision, recall, f1


def d_prime_and_criterion(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        matches: OneToOneEventMatchesType,
        positive_label: UnparsedEventLabelType,
        correction: Optional[str] = "loglinear",
) -> float:
    """
    Calculates d-prime and criterion while optionally applying a correction for floor/ceiling effects on the hit-rate
    and/or false-alarm rate. See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
    See implementation details at https://lindeloev.net/calculating-d-in-python-and-php/.

    :param ground_truth: all ground-truth events
    :param prediction: all predicted events
    :param matches: the one-to-one matches between (subset of) ground-truth and (subset of) predicted events
    :param positive_label: event-label to consider as "positive" events
    :param correction: optional floor/ceiling correction method when calculating hit-rate and false-alarm rate
    :return:
        - d_prime: float; the d-prime value
        - criterion: float; the criterion value
    """
    p, n, pp, tp = _extract_contingency_values(ground_truth, prediction, matches, positive_label)
    return dprime_and_criterion(p, n, pp, tp, correction)


def _extract_contingency_values(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        matches: OneToOneEventMatchesType,
        positive_label: UnparsedEventLabelType,
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
    positive_label = parse_label(positive_label)
    p = len([e for e in ground_truth if e.label == positive_label])
    n = len(ground_truth) - p
    pp = len([e for e in prediction if e.label == positive_label])
    tp = len([e for e in matches.values() if e.label == positive_label])
    return p, n, pp, tp
