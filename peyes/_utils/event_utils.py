from typing import Optional

import numpy as np

import peyes._utils.constants as cnst
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._DataModels.Event import BaseEvent, EventSequenceType
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType


def calculate_sampling_rate(ms: np.ndarray) -> float:
    """
    Calculates the sampling rate of the given timestamps in Hz.
    :param ms: timestamps in milliseconds (floating-point, not integer)
    """
    if len(ms) < 2:
        raise ValueError("timestamps must be of length at least 2")
    sr = cnst.MILLISECONDS_PER_SECOND / np.median(np.diff(ms))
    if not np.isfinite(sr):
        raise RuntimeError("Error calculating sampling rate")
    return sr


def calculate_num_samples(
        start_time: float, end_time: float, sampling_rate: float, min_samples: int = 1
) -> int:
    """
    Calculates the number of samples between the given start and end times, given the sampling rate. The number of
    samples is rounded up to the nearest integer, but at least `min_samples` are returned.

    :param start_time: the start time in milliseconds
    :param end_time: the end time in milliseconds
    :param sampling_rate: the sampling rate in Hz
    :param min_samples: the minimum number of samples to return

    :return: the number of samples

    :raises ValueError: if the start or end times are invalid (not finite or start > end)
    :raises ValueError: if the sampling rate is invalid (not finite or non-positive)
    :raises RuntimeError: if an error occurs during the calculation
    """
    if not np.isfinite(start_time) or not np.isfinite(end_time) or start_time > end_time:
        raise ValueError("Invalid start or end time")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling rate must be a positive finite number")
    res = np.ceil((end_time - start_time) * sampling_rate / cnst.MILLISECONDS_PER_SECOND)
    if not np.isfinite(res) or res < 0:
        raise RuntimeError("Error calculating number of samples")
    return max(int(res), min_samples)


def parse_label(
        val: Optional[UnparsedEventLabelType],
        safe: bool = True
) -> EventLabelEnum:
    """
    Parses a gaze label from the original dataset's type to an EventLabelEnum
    :param val: the value to parse
    :param safe: if True, returns EventLabelEnum.UNDEFINED when the parsing fails; otherwise, raises an exception
    :return: the parsed gaze label
    """
    try:
        if isinstance(val, EventLabelEnum):
            return val
        if isinstance(val, BaseEvent):
            return val.label
        if isinstance(val, int):
            return EventLabelEnum(val)
        if isinstance(val, str):
            return EventLabelEnum[val.upper()]
        if isinstance(val, float):
            if not val.is_integer():
                raise ValueError(f"Invalid value: {val}")
            return EventLabelEnum(int(val))
        if np.isscalar(val) and isinstance(val, np.number):
            return parse_label(float(val), safe)
        raise TypeError(f"Incompatible type: {type(val)}")
    except Exception as err:
        if safe and (isinstance(err, ValueError) or isinstance(err, KeyError) or isinstance(err, TypeError)):
            return EventLabelEnum.UNDEFINED
        raise err


def microsaccade_ratio(events: EventSequenceType, amplitude_threshold) -> float:
    """
    Calculates the ratio of microsaccades to saccades. Returns NaN if there are no saccades.
    :param events: sequence of events
    :param amplitude_threshold: threshold for microsaccades (degrees)
    :return: the ratio of microsaccades to saccades
    """
    saccades = [e for e in events if e.label == EventLabelEnum.SACCADE]
    if len(saccades) == 0:
        return np.nan
    microsaccades = [e for e in saccades if e.amplitude < amplitude_threshold]
    return len(microsaccades) / len(saccades)
