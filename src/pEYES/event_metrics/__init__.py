from typing import Sequence

from src.pEYES._DataModels.Event import BaseEvent as Event

from src.pEYES.event_metrics.counts import counts
from src.pEYES.event_metrics.micro_saccade_ratio import microsaccade_ratio
from src.pEYES.event_metrics.transition_matrix import transition_matrix
from src.pEYES.event_metrics.get_features import get_features as features


def start_times(events: Sequence[Event]):
    return features(events, "start_time", verbose=False)


def end_times(events: Sequence[Event]):
    return features(events, "end_time", verbose=False)


def durations(events: Sequence[Event]):
    return features(events, "duration", verbose=False)


def amplitudes(events: Sequence[Event]):
    return features(events, "amplitude", verbose=False)


def azimuths(events: Sequence[Event]):
    return features(events, "azimuth", verbose=False)


def center_pixels(events: Sequence[Event]):
    return features(events, "center_pixel", verbose=False)
