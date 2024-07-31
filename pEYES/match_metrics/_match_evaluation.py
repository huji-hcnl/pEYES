from typing import Optional, Union

import numpy as np

from pEYES._DataModels.EventLabelEnum import EventLabelEnum
from pEYES._DataModels.Event import EventSequenceType
from pEYES._DataModels.EventMatcher import OneToOneEventMatchesType
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

from pEYES._utils.event_utils import parse_label
from pEYES._utils.metric_utils import dprime_and_criterion


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
    num_matched = sum(1 for e in matches.values() if e.label in labels)
    num_predicted = sum(1 for e in prediction if e.label in labels)
    return num_matched / num_predicted if num_predicted > 0 else np.nan


def precision_recall_f1(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        matches: OneToOneEventMatchesType,
        positive_label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]],
) -> (float, float, float):
    """
    Calculates the precision, recall, and F1-score for the given ground-truth and predicted events, where successfully
    matched events are considered true positives. The provided labels are considered as "positive" events.

    :param ground_truth: all ground-truth events
    :param prediction: all predicted events
    :param matches: the one-to-one matches between (subset of) ground-truth and (subset of) predicted events
    :param positive_label: event-label(s) to consider as "positive" events
    :return: the precision, recall, and F1-score values
    """
    p, n, pp, tp = _extract_contingency_values(ground_truth, prediction, matches, positive_label)
    recall = tp / p if p > 0 else np.nan
    precision = tp / pp if pp > 0 else np.nan
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else np.nan
    return precision, recall, f1


def false_alarm_rate(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        matches: OneToOneEventMatchesType,
        positive_label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]],
) -> float:
    """
    Calculates the false-alarm rate for the given ground-truth and predicted events, where successfully matched events
    are considered true positives. The provided labels are considered as "positive" events.
    Note: False-Alarm rate could exceed 1.0 if there are many false alarms (pp - tp) and few negative GT events (n).
    # TODO: Consider adding a "correction" parameter to handle this case.

    :param ground_truth: all ground-truth events
    :param prediction: all predicted events
    :param matches: the one-to-one matches between (subset of) ground-truth and (subset of) predicted events
    :param positive_label: event-label(s) to consider as "positive" events
    :return: the false-alarm rate value
    """
    p, n, pp, tp = _extract_contingency_values(ground_truth, prediction, matches, positive_label)
    if n <= 0:
        return np.nan
    if 0 <= pp - tp <= n:
        return (pp - tp) / n
    # FA rate exceed 1.0 due to over-predicted false-alarm events
    return np.nan


def d_prime_and_criterion(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        matches: OneToOneEventMatchesType,
        positive_label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]],
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
        positive_label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]],
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
    if isinstance(positive_label, UnparsedEventLabelType):
        positive_label = {positive_label}
    positive_label = set(parse_label(l) for l in positive_label)
    if positive_label == set(EventLabelEnum):
        raise ValueError("Cannot consider all event labels as `positive` events.")
    p = len([e for e in ground_truth if e.label in positive_label])
    n = len(ground_truth) - p
    pp = len([e for e in prediction if e.label in positive_label])
    tp = len([e for e in matches.values() if e.label in positive_label])
    return p, n, pp, tp
