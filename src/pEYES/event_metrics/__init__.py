from src.pEYES._DataModels.Event import EventSequenceType

from src.pEYES.event_metrics.counts import counts
from src.pEYES.event_metrics.micro_saccade_ratio import microsaccade_ratio
from src.pEYES.event_metrics.transition_matrix import transition_matrix
from src.pEYES.event_metrics.get_features import get_features as features


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
