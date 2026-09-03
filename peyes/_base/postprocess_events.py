from typing import Optional

import numpy as np
import pandas as pd

from peyes._utils import constants as cnst
from peyes._utils.event_utils import calculate_num_samples
from peyes._DataModels.Event import BaseEvent, EventSequenceType
from peyes._DataModels.EventLabelEnum import EventLabelEnum, EventLabelSequenceType


def summarize_events(
        events: EventSequenceType,
) -> pd.DataFrame:
    """
    Converts the given events to a DataFrame, where each row is an event and columns are event features.
    An empty input returns an empty frame carrying the full schema, not a column-less one.
    """
    if len(events) == 0:
        return pd.DataFrame(columns=BaseEvent.summary_columns())
    summaries = [e.summary() for e in events]
    return pd.DataFrame(summaries)


def events_to_labels(
        events: EventSequenceType, sampling_rate: float, min_num_samples=None, t_start: Optional[float] = None,
) -> EventLabelSequenceType:
    """
    Converts the given events to a sequence of labels, where each event is mapped to a sequence of labels with length
    matching the number of samples in the event's duration (rounded up to the nearest integer).
    Samples with no event are labeled as `EventLabelEnum.UNDEFINED`.

    :param events: array-like of Event objects
    :param sampling_rate: the sampling rate of the output labels
    :param min_num_samples: the minimal number of samples in the output sequence. If None, the number of samples is
        determined by the total duration of the provided events.
    :param t_start: the recording's true start time (same time units/origin as each event's `start_time`), used to
        anchor sample 0 of the output (C-27). If None (default), sample 0 anchors to the earliest event's own
        start time instead, same as before this parameter existed - a caller who needs the output aligned to the
        full recording's timeline (e.g. a leading gap before the first event) must pass it explicitly.

    :return: array of label values (integers matching `EventLabelEnum`), one per sample

    :raises ValueError: if `events` is empty, or if `t_start` is after the earliest event's start time (which
        would place that event's samples before index 0)
    """
    if len(events) == 0:
        raise ValueError("Cannot convert an empty event sequence to labels")
    earliest_start_time = min(e.start_time for e in events)
    if t_start is None:
        global_start_time = earliest_start_time
    elif t_start > earliest_start_time:
        raise ValueError(f"t_start ({t_start}) is after the earliest event's start time ({earliest_start_time})")
    else:
        global_start_time = t_start
    global_end_time = max(e.end_time for e in events)
    # +1 because `duration` is end_time - start_time, so an n-sample event spans (n-1) * dt
    # (see BaseEvent.duration); without it the output is one sample short of the input.
    num_samples = calculate_num_samples(global_start_time, global_end_time, sampling_rate, 1) + 1
    if min_num_samples is not None:
        num_samples = max(num_samples, min_num_samples)
    out = np.full(num_samples, EventLabelEnum.UNDEFINED, dtype=int)
    for e in events:
        corrected_start_time, corrected_end_time = e.start_time - global_start_time, e.end_time - global_start_time
        start_sample = int(np.round(corrected_start_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        end_sample = int(np.round(corrected_end_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        out[start_sample:end_sample + 1] = e.label      # end_sample is the event's last sample, inclusive
    return out
