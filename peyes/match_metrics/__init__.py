from peyes._DataModels.EventMatcher import OneToOneEventMatchesType as _OneToOneEventMatchesType

from peyes.match_metrics._get_features import get_features as features
from peyes.match_metrics._match_evaluation import (
    match_ratio, precision_recall_f1, false_alarm_rate, d_prime_and_criterion
)


def onset_difference(matches: _OneToOneEventMatchesType):
    return features(matches, 'onset', verbose=False)['onset']


def offset_difference(matches: _OneToOneEventMatchesType):
    return features(matches, 'offset', verbose=False)['offset']


def duration_difference(matches: _OneToOneEventMatchesType):
    return features(matches, 'duration', verbose=False)['duration']


def amplitude_difference(matches: _OneToOneEventMatchesType):
    return features(matches, 'amplitude', verbose=False)['amplitude']


def azimuth_difference(matches: _OneToOneEventMatchesType):
    return features(matches, 'azimuth', verbose=False)['azimuth']


def center_pixel_distance(matches: _OneToOneEventMatchesType):
    return features(matches, 'center_pixel_distance', verbose=False)['center_pixel_distance']


def time_overlap(matches: _OneToOneEventMatchesType):
    return features(matches, 'time_overlap', verbose=False)['time_overlap']


def time_iou(matches: _OneToOneEventMatchesType):
    return features(matches, 'time_iou', verbose=False)['time_iou']


def time_l2(matches: _OneToOneEventMatchesType):
    return features(matches, 'time_l2', verbose=False)['time_l2']
