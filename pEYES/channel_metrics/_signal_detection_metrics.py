from typing import Union, Sequence

import numpy as np
import pandas as pd

import pEYES._utils.constants as cnst
from pEYES._DataModels.Event import EventSequenceType
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelSequenceType

from pEYES._base.create import create_boolean_channel
from pEYES.channel_metrics._timing_differences import timing_differences
from pEYES._utils.metric_utils import dprime_and_criterion


def onset_detection_metrics(
        ground_truth: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        prediction: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        threshold: Union[int, Sequence[int]],
        sampling_rate: float = None,
        min_num_samples=None,
        dprime_correction: str = "loglinear",
) -> pd.DataFrame:
    """
    1. Converts the label/event sequences to a boolean array indicating onsets (MNE-style event channel).
    2. Matches between `True` indices (event onsets) of the two channels such that the difference between their indices
        is minimal.
    3. For each of threshold value, calculates:
        a) Contingency measures of the pairing (P, PP, TP, N)
        b) Signal Detection Theory (SDT) metrics (recall, precision, F1-score, false alarm rate, d-prime, criterion)

    :param ground_truth: array-like of ground-truth labels or Event objects
    :param prediction: array-like of predicted labels or Event objects
    :param threshold: int or array-like of int; threshold values to determine if a pairing is valid
    :param sampling_rate: sampling rate of the channels; only used if `ground_truth` or `prediction` are Event objects
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels; only used if
        `ground_truth` or `prediction` are Event objects
    :param dprime_correction: str; optional correction method for floor/ceiling effects on the hit-rate and/or false-alarm rate
    :return: DataFrame of contingency measures and SDT metrics
    """
    return _signal_detection_metrics(
        ground_truth, prediction, threshold, "onset", sampling_rate, min_num_samples, dprime_correction
    )


def offset_detection_metrics(
        ground_truth: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        prediction: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        threshold: Union[int, Sequence[int]],
        sampling_rate: float = None,
        min_num_samples=None,
        dprime_correction: str = "loglinear",
) -> pd.DataFrame:
    """
    1. Converts the label/event sequences to a boolean array indicating offsets (MNE-style event channel).
    2. Matches between `True` indices (event offsets) of the two channels such that the difference between their indices
        is minimal.
    3. For each of threshold value, calculates:
        a) Contingency measures of the pairing (P, PP, TP, N)
        b) Signal Detection Theory (SDT) metrics (recall, precision, F1-score, false alarm rate, d-prime, criterion)

    :param ground_truth: array-like of ground-truth labels or Event objects
    :param prediction: array-like of predicted labels or Event objects
    :param threshold: int or array-like of int; threshold values to determine if a pairing is valid
    :param sampling_rate: sampling rate of the channels; only used if `ground_truth` or `prediction` are Event objects
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels; only used if
        `ground_truth` or `prediction` are Event objects
    :param dprime_correction: str; optional correction method for floor/ceiling effects on the hit-rate and/or false-alarm rate
    :return: DataFrame of contingency measures and SDT metrics
    """
    return _signal_detection_metrics(
        ground_truth, prediction, threshold, "offset", sampling_rate, min_num_samples, dprime_correction
    )


def _signal_detection_metrics(
        ground_truth: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        prediction: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        threshold: Union[int, Sequence[int]],
        channel_type: str,
        sampling_rate: float = None,
        min_num_samples=None,
        dprime_correction: str = "loglinear",
) -> pd.DataFrame:
    """
    Converts the labels/events to boolean channels and matches between `True` indices of the two channels such that the
    difference between their indices is minimal. Then, for each threshold value, calculates the contingency measures
    of the pairing (P, PP, TP, N) and Signal Detection Theory (SDT) metrics (recall, precision, F1-score, false alarm
    rate, d-prime, criterion).
    Returns a DataFrame where the rows are the threshold values and the columns are the metrics.

    :param ground_truth: array-like of ground-truth labels or Event objects
    :param prediction: array-like of predicted labels or Event objects
    :param threshold: int or array-like of int; threshold values to determine if a pairing is valid
    :param channel_type: either 'onset' or 'offset'
    :param sampling_rate: sampling rate of the channels; only used if `ground_truth` or `prediction` are Event objects
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels; only used if
        `ground_truth` or `prediction` are Event objects
    :param dprime_correction: str; optional correction method for floor/ceiling effects on the hit-rate and/or false-alarm rate

    :return: DataFrame of contingency measures and SDT metrics
    """
    gt_channel = create_boolean_channel(
        data=ground_truth, channel_type=channel_type, sampling_rate=sampling_rate, min_num_samples=min_num_samples
    )
    pred_channel = create_boolean_channel(
        data=prediction, channel_type=channel_type, sampling_rate=sampling_rate, min_num_samples=min_num_samples
    )
    p, pp = gt_channel.sum(), pred_channel.sum()  # number of positive samples in GT and prediction
    all_matched_diffs = timing_differences(
        ground_truth, prediction,
        max_diff=max(threshold), channel_type=channel_type, sampling_rate=sampling_rate, min_num_samples=min_num_samples
    )
    if isinstance(threshold, int):
        threshold = [threshold]
    results = {}
    for thresh in threshold:
        thresh_results = {"P": p, "PP": pp}
        tp = np.sum(np.abs(all_matched_diffs) <= thresh)
        thresh_results["TP"] = tp
        double_thresh = 2 * thresh + 1  # threshold can be before or after a particular sample
        n = max(0, (len(gt_channel) - double_thresh * p) / double_thresh)  # number of negative "windows" in GT channel
        thresh_results["N"] = n

        # true positive rate (TPR), sensitivity, hit-rate
        if p > 0 and 0 <= tp / p <= 1:
            recall = tp / p
        else:
            recall = np.nan
        # positive predictive value (PPV)
        if pp > 0 and 0 <= tp / pp <= 1:
            precision = tp / pp
        else:
            precision = np.nan
        # F1-score
        if np.isfinite(precision + recall) and ((precision + recall) > 0):
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = np.nan
        # FPR, type I error, 1 - specificity
        if n > 0 and 0 <= (pp - tp) / n <= 1:
            false_alarm_rate = (pp - tp) / n
        else:
            false_alarm_rate = np.nan
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


