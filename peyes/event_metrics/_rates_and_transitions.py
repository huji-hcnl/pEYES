import numpy as np
import pandas as pd

from peyes._utils.constants import MILLISECONDS_PER_SECOND
from peyes._DataModels.Event import EventSequenceType
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType
from peyes._DataModels.EventLabelEnum import EventLabelEnum

from peyes._utils.event_utils import parse_label as _parse_label
from peyes._utils.metric_utils import transition_matrix as _transition_matrix


def _recording_duration_ms(events: EventSequenceType) -> float:
    """
    Duration spanned by the given events, in milliseconds. Uses the span from the earliest onset to the latest
    offset rather than the last offset alone, so the result does not depend on where the recording's clock
    happens to start -- dataset timestamps and trial slices frequently do not start at zero.
    """
    if len(events) == 0:
        raise ValueError("Cannot compute a rate over an empty event sequence")
    return max(e.end_time for e in events) - min(e.start_time for e in events)


def event_rate(
        events: EventSequenceType,
        label: UnparsedEventLabelType,
) -> float:
    """
    Calculates the number of occurrences of the given event-label per second.
    :param events: sequence of gaze-events
    :param label: event-label to calculate the rate for
    :return: the rate (Hz), or NaN if the events span no time

    :raises ValueError: if `events` is empty
    """
    duration_ms = _recording_duration_ms(events)
    if duration_ms <= 0:
        return np.nan
    label = _parse_label(label)
    label_events = [e for e in events if e.label == label]
    return len(label_events) / duration_ms * MILLISECONDS_PER_SECOND


def microsaccade_rate(events: EventSequenceType, max_amplitude: float = 1.0) -> float:
    """
    Calculate the rate of micro-saccades per second.
    :param events: sequence of gaze-events
    :param max_amplitude: the maximum amplitude of a micro-saccade (deg)
    :return: the rate of micro-saccades per second (Hz)
    """
    if max_amplitude <= 0:
        raise ValueError("Micro-saccade threshold must be positive")
    duration_ms = _recording_duration_ms(events)
    if duration_ms <= 0:
        return np.nan
    microsaccades = [e for e in events if e.amplitude <= max_amplitude and e.label == EventLabelEnum.SACCADE]
    return len(microsaccades) / duration_ms * MILLISECONDS_PER_SECOND


def microsaccade_ratio(events: EventSequenceType, max_amplitude: float = 1.0, zero_division: float = np.nan) -> float:
    """
    Calculate the ratio of micro-saccades to all saccades.
    Returns `zero_division` if there are no saccades.

    :param events: sequence of gaze-events
    :param max_amplitude: the maximum amplitude of a micro-saccade (deg)
    :param zero_division: value to return if there are no saccades
    :return: the ratio of micro-saccades to all saccades
    """
    if max_amplitude <= 0:
        raise ValueError("Micro-saccade threshold must be positive")
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
    labels = list(EventLabelEnum)
    matrix = _transition_matrix([e.label for e in seq], normalize_rows)
    return matrix.reindex(index=labels, columns=labels, fill_value=0)
