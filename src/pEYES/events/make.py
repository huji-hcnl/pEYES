from typing import Union, Sequence, List

import numpy as np

from src.pEYES._DataModels.Event import BaseEvent as Event
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum as EventLabel

from src.pEYES._utils.event_utils import UnparsedEventLabelType, parse_label


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
        total_duration = events.duration
        out = np.full(int(np.ceil(total_duration * sampling_rate)), events.label)
        return out
    total_duration = sum(e.duration for e in events)
    out = np.full(int(np.ceil(total_duration * sampling_rate)), EventLabel.UNDEFINED)
    start = 0
    for e in events:
        end = start + int(np.ceil(e.duration * sampling_rate))
        out[start:end] = e.label
        start = end
    return out
