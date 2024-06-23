from typing import Union, Sequence

import numpy as np
import pandas as pd

from src.pEYES import constants as cnst
from src.pEYES._DataModels.Event import BaseEvent, EventSequenceType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType

from src.pEYES._utils.event_utils import parse_label


def create_events(
        labels: Union[UnparsedEventLabelType, Sequence[UnparsedEventLabelType]],
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        pupil: np.ndarray,
        viewer_distance: float,
        pixel_size: float,
) -> Union[BaseEvent, EventSequenceType]:
    """
    Create gaze-events from the given data.
    If `labels` is a single label, creates a single event spanning the entire data.
    Otherwise, `labels` must have the same length as `t` and events will be created for each "chunk" of subsequent
    identical labels.

    :param labels: The event label(s)
    :param t: timestamps (ms)
    :param x: horizontal gaze position (pixels)
    :param y: vertical gaze position (pixels)
    :param pupil: pupil size (mm)
    :param viewer_distance: distance from the viewer's eyes to the screen (cm)
    :param pixel_size: size of a pixel on the screen (cm)
    :return: the generated event(s)
    """
    if isinstance(labels, UnparsedEventLabelType):
        label = parse_label(labels)
        return BaseEvent.make(label, t, x, y, pupil, viewer_distance, pixel_size)
    labels = np.vectorize(parse_label)(labels)
    return BaseEvent.make_multiple(labels, t, x, y, pupil, viewer_distance, pixel_size)


def events_to_labels(
        events: Union[BaseEvent, EventSequenceType],
        sampling_rate: float,
) -> Sequence[EventLabelEnum]:
    """
    Converts the given event(s) to a sequence of labels, where each event is mapped to a sequence of labels with length
    matching the number of samples in the event's duration (rounded up to the nearest integer).

    :param events: sequence of event(s)
    :param sampling_rate: the sampling rate of the output labels
    :return: sequence of labels
    """
    if isinstance(events, BaseEvent):
        out = np.full(int(np.ceil(sampling_rate * events.duration / cnst.MILLISECONDS_PER_SECOND)), events.label)
        return out
    max_end_time = max(e.end_time for e in events)
    out = np.full(int(np.ceil(sampling_rate * max_end_time / cnst.MILLISECONDS_PER_SECOND)), EventLabelEnum.UNDEFINED)
    for e in events:
        start_time, end_time = e.start_time, e.end_time
        start_sample = int(np.round(start_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        end_sample = int(np.round(end_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        out[start_sample:end_sample] = e.label
    return out


def events_to_table(
        events: Union[BaseEvent, EventSequenceType],
) -> pd.DataFrame:
    """
    Converts the given event(s) to a DataFrame, where each row represents an event and columns are event features.
    """
    if isinstance(events, BaseEvent):
        s = events.summary()
        return s.to_frame()
    if len(events) == 0:
        return pd.DataFrame()
    summaries = [e.summary() for e in events]
    return pd.DataFrame(summaries)
