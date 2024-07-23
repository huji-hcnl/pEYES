from typing import Union, Sequence

import numpy as np
import pandas as pd

import pEYES._utils.constants as cnst
from pEYES._DataModels.Event import EventSequenceType

from pEYES._base.postprocess_events import events_to_boolean_channel
from pEYES.channel_metrics._timing_differences import _timing_differences
from pEYES._utils.metric_utils import dprime_and_criterion


def onset_detection_metrics(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        threshold: Union[int, Sequence[int]],
        sampling_rate: float,
        min_num_samples=None,
        dprime_correction: str = "loglinear",
) -> pd.DataFrame:
    """
    1. Converts the event sequences to a boolean array indicating event onsets (MNE-style event channel).
    2. Matches between `True` indices (event onsets) of the two channels such that the difference between their indices
        is minimal.
    3. For each of threshold value, calculates:
        a) Contingency measures of the pairing (P, PP, TP, N)
        b) Signal Detection Theory (SDT) metrics (recall, precision, F1-score, false alarm rate, d-prime, criterion)

    :param ground_truth: array-like of Event objects
    :param prediction: array-like of Event objects
    :param threshold: int or array-like of int; threshold values to determine if a pairing is valid
    :param sampling_rate: sampling rate of the channels
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels
    :param dprime_correction: str; optional correction method for floor/ceiling effects on the hit-rate and/or false-alarm rate
    :return: DataFrame of contingency measures and SDT metrics
    """
    return _signal_detection_metrics(
        ground_truth, prediction, threshold, sampling_rate, "onset", min_num_samples, dprime_correction
    )


def offset_detection_metrics(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        threshold: Union[int, Sequence[int]],
        sampling_rate: float,
        min_num_samples=None,
        dprime_correction: str = "loglinear",
) -> pd.DataFrame:
    """
    1. Converts the event sequences to a boolean array indicating event offsets (MNE-style event channel).
    2. Matches between `True` indices (event offsets) of the two channels such that the difference between their indices
        is minimal.
    3. For each of threshold value, calculates:
        a) Contingency measures of the pairing (P, PP, TP, N)
        b) Signal Detection Theory (SDT) metrics (recall, precision, F1-score, false alarm rate, d-prime, criterion)

    :param ground_truth: array-like of Event objects
    :param prediction: array-like of Event objects
    :param threshold: int or array-like of int; threshold values to determine if a pairing is valid
    :param sampling_rate: sampling rate of the channels
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels
    :param dprime_correction: str; optional correction method for floor/ceiling effects on the hit-rate and/or false-alarm rate
    :return: DataFrame of contingency measures and SDT metrics
    """
    return _signal_detection_metrics(
        ground_truth, prediction, threshold, sampling_rate, "offset", min_num_samples, dprime_correction
    )


def _signal_detection_metrics(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        threshold: Union[int, Sequence[int]],
        sampling_rate: float,
        channel_type: str,
        min_num_samples=None,
        dprime_correction: str = "loglinear",
) -> pd.DataFrame:
    """
    Converts the events to boolean channels and matches between `True` indices of the two channels such that the
    difference between their indices is minimal. Then, for each threshold value, calculates the contingency measures
    of the pairing (P, PP, TP, N) and Signal Detection Theory (SDT) metrics (recall, precision, F1-score, false alarm
    rate, d-prime, criterion).
    Returns a DataFrame where the rows are the threshold values and the columns are the metrics.

    :param ground_truth: array-like of Event objects
    :param prediction: array-like of Event objects
    :param threshold: int or array-like of int; threshold values to determine if a pairing is valid
    :param sampling_rate: sampling rate of the channels
    :param channel_type: either 'onset' or 'offset'
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels
    :param dprime_correction: str; optional correction method for floor/ceiling effects on the hit-rate and/or false-alarm rate

    :return: DataFrame of contingency measures and SDT metrics
    """
    gt_channel = events_to_boolean_channel(ground_truth, channel_type, sampling_rate, min_num_samples)
    pred_channel = events_to_boolean_channel(prediction, channel_type, sampling_rate, min_num_samples)
    p, pp = gt_channel.sum(), pred_channel.sum()  # number of positive samples in GT and prediction
    all_matched_diffs = _timing_differences(ground_truth, prediction, sampling_rate, channel_type, min_num_samples)
    if isinstance(threshold, int):
        threshold = [threshold]
    results = {}
    for thresh in threshold:
        thresh_results = {"P": p, "PP": pp}
        tp = np.sum(np.abs(all_matched_diffs) <= thresh)
        thresh_results["TP"] = tp
        double_thresh = 2 * thresh + 1  # threshold can be before or after a particular sample
        n = (len(gt_channel) - double_thresh * p) / double_thresh  # number of negative "windows" in GT channel
        thresh_results["N"] = n

        recall = tp / p if p > 0 else np.nan    # true positive rate (TPR), sensitivity, hit-rate
        precision = tp / pp if pp > 0 else np.nan     # positive predictive value (PPV)
        if np.isfinite(precision + recall) and ((precision + recall) > 0):
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = np.nan
        false_alarm_rate = (pp - tp) / n if n > 0 else np.nan  # FPR, type I error, 1 - specificity
        d_prime, criterion = dprime_and_criterion(p, n, pp, tp, correction=dprime_correction)

        # update values:
        thresh_results[cnst.RECALL_STR] = recall
        thresh_results[cnst.PRECISION_STR] = precision
        thresh_results[cnst.F1_STR] = f1_score
        thresh_results[cnst.FALSE_ALARM_RATE_STR] = false_alarm_rate
        thresh_results[cnst.D_PRIME_STR] = d_prime
        thresh_results[cnst.CRITERION_STR] = criterion
        results[thresh] = thresh_results
    results = pd.DataFrame(results).T
    results.index.name = cnst.THRESHOLD_STR
    results.columns.name = cnst.METRIC_STR
    return results


