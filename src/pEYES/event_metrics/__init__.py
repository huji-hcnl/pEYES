from src.pEYES._DataModels.Event import EventSequenceType as _EventSequenceType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum as _EventLabelEnum

from src.pEYES.event_metrics._get_features import get_features as _features
from src.pEYES.event_metrics._rates_and_transitions import event_rate as _event_rate
from src.pEYES.event_metrics._get_features import features_by_labels
from src.pEYES.event_metrics._rates_and_transitions import microsaccade_ratio, transition_matrix


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


def counts(events: _EventSequenceType):
    aggregated_features = features_by_labels(events)
    return aggregated_features["count"]


def saccade_rate(events: _EventSequenceType):
    return _event_rate(events, _EventLabelEnum.SACCADE)


def blink_rate(events: _EventSequenceType):
    return _event_rate(events, _EventLabelEnum.BLINK)
