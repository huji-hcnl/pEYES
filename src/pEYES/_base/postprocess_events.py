import numpy as np
import pandas as pd

from src.pEYES._utils import constants as cnst
from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum, EventLabelSequenceType


def summarize_events(
        events: EventSequenceType,
) -> pd.DataFrame:
    """ Converts the given events to a DataFrame, where each row is an event and columns are event features. """
    if len(events) == 0:
        return pd.DataFrame()
    summaries = [e.summary() for e in events]
    return pd.DataFrame(summaries)


def events_to_labels(events: EventSequenceType, sampling_rate: float, num_samples=None) -> EventLabelSequenceType:
    """
    Converts the given events to a sequence of labels, where each event is mapped to a sequence of labels with length
    matching the number of samples in the event's duration (rounded up to the nearest integer).
    Samples with no event are labeled as `EventLabelEnum.UNDEFINED`.

    :param events: array-like of Event objects
    :param sampling_rate: the sampling rate of the output labels
    :param num_samples: the number of samples in the output sequence. If None, the number of samples is determined by
        the total duration of the provided events.

    :return: sequence of labels
    """
    num_samples = __calc_num_samples(events, sampling_rate, num_samples)
    global_start_time = min(e.start_time for e in events)
    out = np.full(num_samples, EventLabelEnum.UNDEFINED)
    for e in events:
        corrected_start_time, corrected_end_time = e.start_time - global_start_time, e.end_time - global_start_time
        start_sample = int(np.round(corrected_start_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        end_sample = int(np.round(corrected_end_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        out[start_sample:end_sample] = e.label
    return out


def events_to_boolean_channels(
        events: EventSequenceType,
        sampling_rate: float,
        num_samples=None,
) -> (np.ndarray, np.ndarray):
    """
    Converts the given events to boolean arrays (MNE-style event channels), one indicating event onsets and the other
    indicating event offsets.

    :param events: array-like of Event objects
    :param sampling_rate: the sampling rate of the output arrays
    :param num_samples: the number of samples in the output sequence. If None, the number of samples is determined by
        the total duration of the provided events.

    :return: two arrays of booleans, one indicating event onsets and the other indicating offsets
    """
    num_samples = __calc_num_samples(events, sampling_rate, num_samples)
    global_start_time = min(e.start_time for e in events)
    is_onset, is_offset = np.zeros(num_samples, dtype=bool), np.zeros(num_samples, dtype=bool)
    for e in events:
        corrected_start_time, corrected_end_time = e.start_time - global_start_time, e.end_time - global_start_time
        start_sample = int(np.round(corrected_start_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        end_sample = int(np.round(corrected_end_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        # assert 0 <= start_sample < end_sample, f"start_sample={start_sample} is out of bounds"
        # assert start_sample < end_sample < num_samples, f"end_sample={end_sample} is out of bounds"
        is_onset[start_sample] = True
        is_offset[end_sample - 1] = True
    return is_onset, is_offset


def __calc_num_samples(events: EventSequenceType, sampling_rate: float, num_samples=None) -> int:
    min_start_time = min(e.start_time for e in events)
    max_end_time = max(e.end_time for e in events)
    duration = max_end_time - min_start_time
    min_num_samples = int(np.ceil(sampling_rate * duration / cnst.MILLISECONDS_PER_SECOND))
    if num_samples is not None and num_samples < min_num_samples:
        raise ValueError(
            f"The provided events last {min_num_samples} samples, " +
            f"which is longer than the given number of samples {num_samples}."
        )
    num_samples = min_num_samples if num_samples is None else num_samples
    return num_samples
