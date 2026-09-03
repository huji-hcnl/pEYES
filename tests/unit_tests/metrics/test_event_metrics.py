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

    def test_all_23_summary_columns_are_now_supported(self):
        """ M-14: get_features used to support only 6 of BaseEvent.summary_columns()'s 23 names. """
        for column in BaseEvent.summary_columns():
            with self.subTest(feature=column):
                result = get_features(self.EVENTS, column, verbose=False)[column]
                expected = np.array([e.summary()[column] for e in self.EVENTS])
                np.testing.assert_array_equal(result, expected)

    def test_previously_supported_names_still_match_direct_attribute_access(self):
        """ M-14: the original 6 (plus their aliases) must return byte-identical values after the rewrite. """
        direct_attrs = {
            "start_time": "start_time", "onset": "start_time", "end_time": "end_time", "offset": "end_time",
            "duration": "duration", "amplitude": "amplitude", "azimuth": "azimuth",
            "center_pixel": "center_pixel", "center": "center_pixel",
        }
        for name, attr in direct_attrs.items():
            with self.subTest(feature=name):
                result = get_features(self.EVENTS, name, verbose=False)[name]
                expected = np.array([getattr(e, attr) for e in self.EVENTS])
                np.testing.assert_array_equal(result, expected)

    def test_plural_fallback_still_works(self):
        """ M-15: a plural feature name falls back to the singular. """
        np.testing.assert_array_equal(
            get_features(self.EVENTS, "durations", verbose=False)["durations"],
            get_features(self.EVENTS, "duration", verbose=False)["duration"],
        )

    def test_unknown_feature_still_raises(self):
        self.assertRaises(ValueError, get_features, self.EVENTS, "not_a_real_feature", verbose=False)


if __name__ == "__main__":
    unittest.main()
