from typing import Union, Sequence, List

import numpy as np

from src.pEYES import constants as cnst
from src.pEYES._DataModels.Event import BaseEvent as Event
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum as EventLabel
from src.pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType

from src.pEYES._utils.event_utils import parse_label


def from_labels(
        labels: Union[UnparsedEventLabelType, Sequence[UnparsedEventLabelType]],
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        pupil: np.ndarray,
        viewer_distance: float,
        pixel_size: float,
) -> Union[Event, List[Event]]:
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
        return Event.make(label, t, x, y, pupil, viewer_distance, pixel_size)
    labels = np.vectorize(parse_label)(labels)
    return Event.make_multiple(labels, t, x, y, pupil, viewer_distance, pixel_size)


def to_labels(
        events: Union[Event, Sequence[Event]],
        sampling_rate: float,
) -> Sequence[EventLabel]:
    if isinstance(events, Event):
        out = np.full(int(np.ceil(sampling_rate * events.duration / cnst.MILLISECONDS_PER_SECOND)), events.label)
        return out
    max_end_time = max(e.end_time for e in events)
    out = np.full(int(np.ceil(sampling_rate * max_end_time / cnst.MILLISECONDS_PER_SECOND)), EventLabel.UNDEFINED)
    for e in events:
        start_time, end_time = e.start_time, e.end_time
        start_sample = int(np.round(start_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        end_sample = int(np.round(end_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
        out[start_sample:end_sample] = e.label
    return out
