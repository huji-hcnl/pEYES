import numpy as np
import pandas as pd

from src.pEYES._DataModels.Event import EventSequenceType

from src.pEYES._utils.event_utils import count_labels as _count


def counts(events: EventSequenceType) -> pd.Series:
    """
    Count the number of occurrences of each event-label within the given events.
    Returns a pandas Series mapping each event-label to its count.
    """
    return _count(events)


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
