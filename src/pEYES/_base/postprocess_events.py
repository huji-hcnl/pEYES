from typing import Union, Sequence

import numpy as np
import pandas as pd

from src.pEYES._utils import constants as cnst
from src.pEYES._DataModels.Event import BaseEvent, EventSequenceType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum


def events_to_labels(
        events: Union[BaseEvent, EventSequenceType],
        sampling_rate: float,
        num_samples=None,
) -> Sequence[EventLabelEnum]:
    """
    Converts the given event(s) to a sequence of labels, where each event is mapped to a sequence of labels with length
    matching the number of samples in the event's duration (rounded up to the nearest integer).
    Samples with no event are labeled as `EventLabelEnum.UNDEFINED`.

    :param events: sequence of event(s)
    :param sampling_rate: the sampling rate of the output labels
    :param num_samples: the number of samples in the output sequence. If None, the number of samples is determined by
        the total duration of the provided events.
    :return: sequence of labels
    """
    if isinstance(events, BaseEvent):
        out = np.full(int(np.ceil(sampling_rate * events.duration / cnst.MILLISECONDS_PER_SECOND)), events.label)
        return out

    max_end_time = max(e.end_time for e in events)
    min_num_samples = int(np.ceil(sampling_rate * max_end_time / cnst.MILLISECONDS_PER_SECOND))
    if num_samples is not None and num_samples < min_num_samples:
        raise ValueError(
            f"The provided events last {min_num_samples} samples, " +
            f"which is longer than the given number of samples {num_samples}."
        )
    num_samples = min_num_samples if num_samples is None else num_samples
    out = np.full(num_samples, EventLabelEnum.UNDEFINED)
    for e in events:
        start_time, end_time = e.start_time, e.end_time
        start_sample = int(np.round(start_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        end_sample = int(np.round(end_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        out[start_sample:end_sample] = e.label
    return out


def summarize_events(
        events: Union[BaseEvent, EventSequenceType],
) -> pd.DataFrame:
    """ Converts the given event(s) to a DataFrame, where each row is an event and columns are event features. """
    if isinstance(events, BaseEvent):
        s = events.summary()
        return s.to_frame()
    if len(events) == 0:
        return pd.DataFrame()
    summaries = [e.summary() for e in events]
    return pd.DataFrame(summaries)
