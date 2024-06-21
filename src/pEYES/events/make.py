from typing import Union, Sequence, List

import numpy as np

from src.pEYES._utils.event_utils import UnparsedEventLabelType, parse_label
from src.pEYES._DataModels.Event import BaseEvent as Event


def make(
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
