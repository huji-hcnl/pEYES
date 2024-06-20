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
    TODO: Add documentation here
    :param labels:
    :param t:
    :param x:
    :param y:
    :param pupil:
    :param viewer_distance:
    :param pixel_size:
    :return:
    """
    if isinstance(labels, UnparsedEventLabelType):
        label = parse_label(labels)
        return Event.make(label, t, x, y, pupil, viewer_distance, pixel_size)
    labels = np.vectorize(parse_label)(labels)
    return Event.make_multiple(labels, t, x, y, pupil, viewer_distance, pixel_size)
