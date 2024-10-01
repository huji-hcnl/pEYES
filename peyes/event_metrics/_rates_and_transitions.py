import numpy as np
import pandas as pd

from peyes._utils.constants import MILLISECONDS_PER_SECOND
from peyes._DataModels.Event import EventSequenceType
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType
from peyes._DataModels.EventLabelEnum import EventLabelEnum

from peyes._utils.event_utils import parse_label as _parse_label
from peyes._utils.metric_utils import transition_matrix as _transition_matrix


def event_rate(
        events: EventSequenceType,
        label: UnparsedEventLabelType,
) -> float:
    """
    Calculates the number of occurrences of the given event-label per second.
    :param events: sequence of gaze-events
    :param label: event-label to calculate the rate for
    :return: the rate (Hz)
    """
    last_offset = events[-1].end_time   # last event's end-time in ms
    label = _parse_label(label)
    label_events = [e for e in events if e.label == label]
    events_to_ms = len(label_events) / last_offset
    return events_to_ms * MILLISECONDS_PER_SECOND


def microsaccade_rate(events: EventSequenceType, max_amplitude: float = 1.0) -> float:
    """
    Calculate the rate of micro-saccades per second.
    :param events: sequence of gaze-events
    :param max_amplitude: the maximum amplitude of a micro-saccade (deg)
    :return: the rate of micro-saccades per second (Hz)
    """
    assert max_amplitude > 0, "Micro-saccade threshold must be positive"
    microsaccades = [e for e in events if e.amplitude <= max_amplitude and e.label == EventLabelEnum.SACCADE]
    return len(microsaccades) / events[-1].end_time * MILLISECONDS_PER_SECOND


def microsaccade_ratio(events: EventSequenceType, max_amplitude: float = 1.0, zero_division: float = np.nan) -> float:
    """
    Calculate the ratio of micro-saccades to all saccades.
    Returns `zero_division` if there are no saccades.

    :param events: sequence of gaze-events
    :param max_amplitude: the maximum amplitude of a micro-saccade (deg)
    :param zero_division: value to return if there are no saccades
    :return: the ratio of micro-saccades to all saccades
    """
    assert max_amplitude > 0, "Micro-saccade threshold must be positive"
    saccades = [e for e in events if e.label == EventLabelEnum.SACCADE]
    microsaccades = [e for e in saccades if e.amplitude <= max_amplitude]
    try:
        return len(microsaccades) / len(saccades)
    except ZeroDivisionError:
        return zero_division


def transition_matrix(
        seq: EventSequenceType,
        normalize_rows: bool = False
) -> pd.DataFrame:
    """
    Calculates the transition matrix from a sequence of event.
    If `normalize_rows` is True, the matrix will be normalized by the sum of each row, i.e. contains transition probabilities.
    Returns a DataFrame where rows indicate the origin event-label and columns indicate the destination event-label.
    """
    return _transition_matrix([e.label for e in seq], normalize_rows)
