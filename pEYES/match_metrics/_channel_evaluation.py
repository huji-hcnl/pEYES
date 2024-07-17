from typing import Union, Sequence

import numpy as np
import pandas as pd

from pEYES._DataModels.Event import EventSequenceType
from pEYES._utils.vector_utils import pair_boolean_arrays
from pEYES._utils.metric_utils import dprime_and_criterion

from src.pEYES import events_to_boolean_channels


def channel_timing_differences(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        sampling_rate: float,
        channel_type: str,
        min_num_samples=None,
) -> np.ndarray:
    """
    Converts each of the event sequences to a boolean channel, matches between `True` indices of the two channels such
    that the difference between their indices is minimal, and calculates the difference between the matched indices, in
    samples. Returns an array of timing differences between matched events.

    :param ground_truth: array-like of Event objects
    :param prediction: array-like of Event objects
    :param sampling_rate: sampling rate of the channels
    :param channel_type: either 'onset' or 'offset'
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels

    :return: array of timing differences between matched events (differences are in samples)
    """
    gt_chan, pred_chan = _to_channels(ground_truth, prediction, sampling_rate, channel_type, min_num_samples)
    matched_idxs = pair_boolean_arrays(gt_chan, pred_chan)
    diffs = np.diff(matched_idxs, axis=1).flatten()
    return diffs


def channel_signal_detection_measures(
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
    gt_chan, pred_chan = _to_channels(ground_truth, prediction, sampling_rate, channel_type, min_num_samples)
    p, pp = gt_chan.sum(), pred_chan.sum()  # number of positive samples in GT and prediction
    all_matched_diffs = channel_timing_differences(
        ground_truth, prediction, sampling_rate, channel_type, min_num_samples
    )
    if isinstance(threshold, int):
        threshold = [threshold]
    results = {}
    for thresh in threshold:
        thresh_results = {"P": p, "PP": pp}
        tp = np.sum(np.abs(all_matched_diffs) <= thresh)
        thresh_results["TP"] = tp
        double_thresh = 2 * thresh + 1  # threshold can be before or after a particular sample
        n = (len(gt_chan) - double_thresh * p) / double_thresh  # number of negative "windows" in GT channel
        thresh_results["N"] = n

        recall = tp / p if p > 0 else np.nan    # true positive rate (TPR), sensitivity, hit-rate
        precision = tp / pp if pp > 0 else np.nan     # positive predictive value (PPV)
        if np.isfinite(precision + recall) and ((precision + recall) > 0):
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = np.nan
        false_alarm_rate = (pp - tp) / n if n > 0 else np.nan  # false positive rate (FPR), type I error, 1 - specificity
        d_prime, criterion = dprime_and_criterion(tp, pp, p, n, correction=dprime_correction)

        # update values:
        thresh_results["recall"] = recall
        thresh_results["precision"] = precision
        thresh_results["f1_score"] = f1_score
        thresh_results["false_alarm_rate"] = false_alarm_rate
        thresh_results["d_prime"] = d_prime
        thresh_results["criterion"] = criterion
        results[thresh] = thresh_results
    results = pd.DataFrame(results).T
    results.index.name = "threshold"
    results.columns.name = "metric"
    return results


def _to_channels(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        sampling_rate: float,
        channel_type: str,
        min_num_samples=None,
) -> (np.ndarray, np.ndarray):
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling rate must be a positive finite number")
    gt_onset_channel, gt_offset_channel = events_to_boolean_channels(ground_truth, sampling_rate, min_num_samples)
    pred_onset_channel, pred_offset_channel = events_to_boolean_channels(prediction, sampling_rate, min_num_samples)
    if channel_type.lower() == "onset":
        return gt_onset_channel, pred_onset_channel
    if channel_type.lower() == "offset":
        return gt_offset_channel, pred_offset_channel
    raise ValueError("channel_type must be either 'onset' or 'offset'")
