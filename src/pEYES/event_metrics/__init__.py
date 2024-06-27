from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

from src.pEYES.event_metrics.counts_and_rates import counts, microsaccade_ratio
from src.pEYES.event_metrics.transition_matrix import transition_matrix
from src.pEYES.event_metrics.get_features import get_features as features
from src.pEYES.event_metrics.counts_and_rates import event_rate as _event_rate


def start_times(events: EventSequenceType):
    return features(events, "start_time", verbose=False)


def end_times(events: EventSequenceType):
    return features(events, "end_time", verbose=False)


def durations(events: EventSequenceType):
    return features(events, "duration", verbose=False)


def amplitudes(events: EventSequenceType):
    return features(events, "amplitude", verbose=False)


def azimuths(events: EventSequenceType):
    return features(events, "azimuth", verbose=False)


def center_pixels(events: EventSequenceType):
    return features(events, "center_pixel", verbose=False)


def saccade_rate(events: EventSequenceType):
    return _event_rate(events, EventLabelEnum.SACCADE)


def blink_rate(events: EventSequenceType):
    return _event_rate(events, EventLabelEnum.BLINK)
