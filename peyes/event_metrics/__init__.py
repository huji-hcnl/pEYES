import peyes._utils.constants as _cnst
from peyes._DataModels.Event import EventSequenceType as _EventSequenceType
from peyes._DataModels.EventLabelEnum import EventLabelEnum as _EventLabelEnum

from peyes.event_metrics._get_features import get_features as _features
from peyes.event_metrics._rates_and_transitions import event_rate as _event_rate
from peyes.event_metrics._get_features import features_by_labels
from peyes.event_metrics._rates_and_transitions import microsaccade_rate, microsaccade_ratio, transition_matrix


def start_times(events: _EventSequenceType):
    return _features(events, _cnst.START_TIME_STR, verbose=False)


def end_times(events: _EventSequenceType):
    return _features(events, _cnst.END_TIME_STR, verbose=False)


def durations(events: _EventSequenceType):
    return _features(events, _cnst.DURATION_STR, verbose=False)


def amplitudes(events: _EventSequenceType):
    return _features(events, _cnst.AMPLITUDE_STR, verbose=False)


def azimuths(events: _EventSequenceType):
    return _features(events, _cnst.AZIMUTH_STR, verbose=False)


def center_pixels(events: _EventSequenceType):
    return _features(events, _cnst.CENTER_PIXEL_STR, verbose=False)


def counts(events: _EventSequenceType):
    aggregated_features = features_by_labels(events)
    return aggregated_features[_cnst.COUNT_STR]


def saccade_rate(events: _EventSequenceType):
    return _event_rate(events, _EventLabelEnum.SACCADE)


def blink_rate(events: _EventSequenceType):
    return _event_rate(events, _EventLabelEnum.BLINK)
