import numpy as np

from pEYES._DataModels.Event import EventSequenceType
from pEYES._base.create import create_boolean_channel
from pEYES._utils.vector_utils import pair_boolean_arrays


def onset_differences(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        sampling_rate: float,
        min_num_samples=None,
) -> np.ndarray:
    """
    1. Converts the event sequences to a boolean array indicating event onsets (MNE-style event channel).
    2. Matches between `True` indices (event onsets) of the two channels such that the difference between their indices
        is minimal.
    3. Calculates the difference between the matched indices, in samples.

    :param ground_truth: array-like of Event objects
    :param prediction: array-like of Event objects
    :param sampling_rate: sampling rate of the channels
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels
    :return: array of timing differences between matched onsets (differences are in samples)
    """
    return _timing_differences(ground_truth, prediction, sampling_rate, "onset", min_num_samples)


def offset_differences(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        sampling_rate: float,
        min_num_samples=None,
) -> np.ndarray:
    """
    1. Converts the event sequences to a boolean array indicating event offsets (MNE-style event channel).
    2. Matches between `True` indices (event offsets) of the two channels such that the difference between their indices
        is minimal.
    3. Calculates the difference between the matched indices, in samples.

    :param ground_truth: array-like of Event objects
    :param prediction: array-like of Event objects
    :param sampling_rate: sampling rate of the channels
    :param min_num_samples: if not None, marks the minimal the number of samples in the channels
    :return: array of timing differences between matched offsets (differences are in samples)
    """
    return _timing_differences(ground_truth, prediction, sampling_rate, "offset", min_num_samples)


def _timing_differences(
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
    gt_channel = create_boolean_channel(
        events=ground_truth, channel_type=channel_type, sampling_rate=sampling_rate, min_num_samples=min_num_samples
    )
    pred_channel = create_boolean_channel(
        events=prediction, channel_type=channel_type, sampling_rate=sampling_rate, min_num_samples=min_num_samples
    )
    matched_idxs = pair_boolean_arrays(gt_channel, pred_channel)
    diffs = np.diff(matched_idxs, axis=1).flatten()
    return diffs
