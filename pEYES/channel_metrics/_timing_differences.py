from typing import Union

import numpy as np

from pEYES._DataModels.Event import EventSequenceType
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelSequenceType

from pEYES._base.create import create_boolean_channel
from pEYES._utils.vector_utils import pair_boolean_arrays


def onset_differences(
        ground_truth: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        prediction: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        sampling_rate: float = None,
        min_num_samples=None,
) -> np.ndarray:
    """
    1. Converts the label/event sequences to a boolean array indicating event onsets (MNE-style event channel).
    2. Matches between `True` indices (event onsets) of the two channels such that the difference between their indices
        is minimal.
    3. Calculates the difference between the matched indices, in samples.

    :param ground_truth: array-like of ground-truth labels or Event objects
    :param prediction: array-like of predicted labels or Event objects
    :param sampling_rate: sampling rate of the channels; only used if `ground_truth` or `prediction` are Event objects
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels; only used if
        `ground_truth` or `prediction` are Event objects
    :return: array of timing differences between matched onsets (differences are in sample units)
    """
    return timing_differences(ground_truth, prediction, "onset", sampling_rate, min_num_samples)


def offset_differences(
        ground_truth: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        prediction: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        sampling_rate: float = None,
        min_num_samples=None,
) -> np.ndarray:
    """
    1. Converts the label/event sequences to a boolean array indicating event offsets (MNE-style event channel).
    2. Matches between `True` indices (event offsets) of the two channels such that the difference between their indices
        is minimal.
    3. Calculates the difference between the matched indices, in samples.

    :param ground_truth: array-like of ground-truth labels or Event objects
    :param prediction: array-like of predicted labels or Event objects
    :param sampling_rate: sampling rate of the channels; only used if `ground_truth` or `prediction` are Event objects
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels; only used if
        `ground_truth` or `prediction` are Event objects
    :return: array of timing differences between matched offsets (differences are in sample units)
    """
    return timing_differences(ground_truth, prediction, "offset", sampling_rate, min_num_samples)


def timing_differences(
        ground_truth: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        prediction: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        channel_type: str,
        sampling_rate: float = None,
        min_num_samples=None,
) -> np.ndarray:
    """
    Converts each of the label/event sequences to a boolean channel, matches between `True` indices of the two channels such
    that the difference between their indices is minimal, and calculates the difference between the matched indices, in
    samples. Returns an array of timing differences between matched events.

    :param ground_truth: array-like of ground-truth labels or Event objects
    :param prediction: array-like of predicted labels or Event objects
    :param channel_type: either 'onset' or 'offset'
    :param sampling_rate: sampling rate of the channels; only used if `ground_truth` or `prediction` are Event objects
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels; only used if
        `ground_truth` or `prediction` are Event objects

    :return: array of timing differences between matched onsets/offsets (differences are in samples)
    """
    gt_channel = create_boolean_channel(
        data=ground_truth, channel_type=channel_type, sampling_rate=sampling_rate, min_num_samples=min_num_samples
    )
    pred_channel = create_boolean_channel(
        data=prediction, channel_type=channel_type, sampling_rate=sampling_rate, min_num_samples=min_num_samples
    )
    matched_idxs = pair_boolean_arrays(gt_channel, pred_channel)
    diffs = np.diff(matched_idxs, axis=1).flatten()
    return diffs
