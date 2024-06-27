from src.pEYES._DataModels.Event import EventSequenceType as _EventSequenceType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum as _EventLabelEnum

from src.pEYES.event_metrics.get_features import get_features as _features
from src.pEYES.event_metrics.counts_and_rates import event_rate as _event_rate

from src.pEYES.event_metrics.counts_and_rates import counts, microsaccade_ratio
from src.pEYES.event_metrics.transition_matrix import transition_matrix


def start_times(events: _EventSequenceType):
    return _features(events, "start_time", verbose=False)


def end_times(events: _EventSequenceType):
    return _features(events, "end_time", verbose=False)


def durations(events: _EventSequenceType):
    return _features(events, "duration", verbose=False)


def amplitudes(events: _EventSequenceType):
    return _features(events, "amplitude", verbose=False)


def azimuths(events: _EventSequenceType):
    return _features(events, "azimuth", verbose=False)


def center_pixels(events: _EventSequenceType):
    return _features(events, "center_pixel", verbose=False)


def saccade_rate(events: _EventSequenceType):
    return _event_rate(events, _EventLabelEnum.SACCADE)


def blink_rate(events: _EventSequenceType):
    return _event_rate(events, _EventLabelEnum.BLINK)
