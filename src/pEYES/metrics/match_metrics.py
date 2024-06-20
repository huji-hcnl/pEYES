from typing import Sequence, Dict, Union, Optional

import numpy as np
from tqdm import tqdm

from src.pEYES._DataModels.Event import BaseEvent
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._DataModels.EventMatcher import OneToOneEventMatchesType
from src.pEYES._utils.metric_utils import dprime


def matched_features(
        matches: OneToOneEventMatchesType, features: Union[str, Sequence[str]], verbose: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculates the difference in the specified features between matched ground-truth and predicted events.
    :param matches: the one-to-one matches between ground-truth and predicted events
    :param features: the feature(s) to calculate the difference for. Supported features are:
        - 'onset' or 'onset_difference': difference in onset time (ms)
        - 'offset' or 'offset_difference': difference in offset time (ms)
        - 'duration' or 'duration_difference': difference in duration (ms)
        - 'amplitude' or 'amplitude_difference': difference in amplitude (deg)
        - 'azimuth' or 'azimuth_difference': difference in azimuth (deg)
        - 'center_pixel_distance': distance between the center of the ground-truth and predicted events (pixels)
        - 'time_overlap': total overlap time between the ground-truth and predicted event (ms)
        - 'time_iou': intersection-over-union between the ground-truth and predicted event (unitless, [0, 1])
        - 'time_l2': L2 norm of the timing difference between the ground-truth and predicted event (ms)
    :param verbose: if True, display a progress bar while calculating the metrics
    :return: the calculated feature differences as a numpy array (for a single feature) or a dictionary of numpy arrays
        (for multiple features).
    :raises ValueError: if an unknown feature is provided
    """
    if isinstance(features, str):
        return _calculate_impl(matches, features)
    results = {}
    for metric in tqdm(features, desc="Calculating Features", disable=not verbose):
        results[metric] = _calculate_impl(matches, metric)
    return results


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


def _calculate_impl(matches: OneToOneEventMatchesType, feature: str,) -> np.ndarray:
    feature_name = feature.lower().strip().replace(" ", "_").replace("-", "_")
    feature_name = feature_name.removesuffix("_difference")
    if feature_name == "onset":
        return np.array([gt.start_time - pred.start_time for gt, pred in matches.items()])
    if feature_name == "offset":
        return np.array([gt.end_time - pred.end_time for gt, pred in matches.items()])
    if feature_name == "duration":
        return np.array([gt.duration - pred.duration for gt, pred in matches.items()])
    if feature_name == "amplitude":
        return np.array([gt.amplitude - pred.amplitude for gt, pred in matches.items()])
    if feature_name == "azimuth":
        return np.array([gt.azimuth - pred.azimuth for gt, pred in matches.items()])
    if feature_name == "center_pixel_distance":
        return np.array([gt.center_distance(pred) for gt, pred in matches.items()])
    if feature_name == "time_overlap":
        return np.array([gt.time_overlap(pred) for gt, pred in matches.items()])
    if feature_name == "time_iou":
        return np.array([gt.time_iou(pred) for gt, pred in matches.items()])
    if feature_name == "time_l2":
        return np.array([gt.time_l2(pred) for gt, pred in matches.items()])
    raise ValueError(f"Unknown feature: {feature}")


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
