import unittest

import numpy as np

import peyes
from peyes._DataModels.Event import BaseEvent
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes.event_metrics._get_features import get_features


def _event(start, n=5, label=EventLabelEnum.SACCADE):
    t = start + np.arange(n) * 2.0
    return BaseEvent.make(label, t=t, x=np.full(n, 1.0), y=np.full(n, 1.0),
                          pupil=np.ones(n), viewer_distance=60.0, pixel_size=0.0277)


class TestGetFeatures(unittest.TestCase):
    """ M-10: get_features() used to return a bare array for one feature and a dict for several. """

    EVENTS = [_event(0.0), _event(50.0)]

    def test_single_feature_still_returns_a_dict(self):
        result = get_features(self.EVENTS, "duration", verbose=False)
        self.assertIsInstance(result, dict)
        self.assertEqual({"duration"}, set(result.keys()))

    def test_multiple_features_return_a_dict(self):
        result = get_features(self.EVENTS, "duration", "amplitude", verbose=False)
        self.assertIsInstance(result, dict)
        self.assertEqual({"duration", "amplitude"}, set(result.keys()))

    def test_named_wrapper_still_returns_a_bare_array(self):
        """ Per-feature convenience functions (e.g. durations) must keep returning a plain array. """
        result = peyes.event_metrics.durations(self.EVENTS)
        self.assertIsInstance(result, np.ndarray)


if __name__ == "__main__":
    unittest.main()
