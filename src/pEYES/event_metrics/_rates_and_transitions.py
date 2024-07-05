import numpy as np
import pandas as pd

from src.pEYES._utils.constants import MILLISECONDS_PER_SECOND
from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType

from src.pEYES._utils.event_utils import parse_label as _parse_label
from src.pEYES._utils.metric_utils import transition_matrix as _transition_matrix


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


def microsaccade_ratio(events: EventSequenceType, threshold: float = 1.0, zero_division: float = np.nan) -> float:
    """
    Calculate the ratio of micro-saccades to all saccades.
    Returns `zero_division` if there are no saccades.

    :param events: sequence of gaze-events
    :param threshold: the maximum amplitude of a micro-saccade (deg)
    :param zero_division: value to return if there are no saccades
    :return: the ratio of micro-saccades to all saccades
    """
    assert threshold > 0, "Micro-saccade threshold must be positive"
    saccades = [e for e in events if e.label == "Saccade"]
    microsaccades = [e for e in saccades if e.amplitude <= threshold]
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
