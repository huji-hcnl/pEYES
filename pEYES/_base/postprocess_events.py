import numpy as np
import pandas as pd

from pEYES._utils import constants as cnst
from pEYES._utils.event_utils import calculate_num_samples
from pEYES._DataModels.Event import EventSequenceType
from pEYES._DataModels.EventLabelEnum import EventLabelEnum, EventLabelSequenceType


def summarize_events(
        events: EventSequenceType,
) -> pd.DataFrame:
    """ Converts the given events to a DataFrame, where each row is an event and columns are event features. """
    if len(events) == 0:
        return pd.DataFrame()
    summaries = [e.summary() for e in events]
    return pd.DataFrame(summaries)


def events_to_labels(events: EventSequenceType, sampling_rate: float, min_num_samples=None) -> EventLabelSequenceType:
    """
    Converts the given events to a sequence of labels, where each event is mapped to a sequence of labels with length
    matching the number of samples in the event's duration (rounded up to the nearest integer).
    Samples with no event are labeled as `EventLabelEnum.UNDEFINED`.

    :param events: array-like of Event objects
    :param sampling_rate: the sampling rate of the output labels
    :param min_num_samples: the minimal number of samples in the output sequence. If None, the number of samples is
        determined by the total duration of the provided events.

    :return: sequence of labels
    """
    global_start_time = min(e.start_time for e in events)
    global_end_time = max(e.end_time for e in events)
    num_samples = calculate_num_samples(global_start_time, global_end_time, sampling_rate, min_num_samples)
    out = np.full(num_samples, EventLabelEnum.UNDEFINED)
    for e in events:
        corrected_start_time, corrected_end_time = e.start_time - global_start_time, e.end_time - global_start_time
        start_sample = int(np.round(corrected_start_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        end_sample = int(np.round(corrected_end_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        out[start_sample:end_sample] = e.label
    return out
