from typing import Union, List

import numpy as np

from src.pEYES._utils.event_utils import parse_label
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._DataModels.Event import BaseEvent


def create_single_event(
        label: Union[EventLabelEnum, BaseEvent, int, str, float],
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        pupil: np.ndarray,
        viewer_distance: float,
        pixel_size: float,
) -> BaseEvent:
    label = parse_label(label)
    return BaseEvent.make(label, t, x, y, pupil, viewer_distance, pixel_size)


def create_multiple_events(
        labels: np.ndarray,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        pupil: np.ndarray,
        viewer_distance: float,
        pixel_size: float,
) -> List[BaseEvent]:
    labels = np.vectorize(parse_label)(labels)
    return BaseEvent.make_multiple(labels, t, x, y, pupil, viewer_distance, pixel_size)
