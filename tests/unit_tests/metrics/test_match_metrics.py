import unittest

import numpy as np

import peyes
from peyes._DataModels.Event import BaseEvent
from peyes._DataModels.EventLabelEnum import EventLabelEnum


def _event(start, n=5, label=EventLabelEnum.SACCADE):
    t = start + np.arange(n) * 2.0
    return BaseEvent.make(label, t=t, x=np.full(n, 1.0), y=np.full(n, 1.0),
                          pupil=np.ones(n), viewer_distance=60.0, pixel_size=0.0277)


class TestContingencyValues(unittest.TestCase):

    GT = [_event(0.0), _event(50.0, label=EventLabelEnum.FIXATION)]
    PRED = [_event(1.0), _event(51.0, label=EventLabelEnum.FIXATION)]

    def _matches(self):
        return peyes.match(self.GT, self.PRED, "onset")

    def test_positive_label_is_required(self):
        """ M-16: None is typed Optional but reached `for l in None` and raised TypeError. """
        with self.assertRaises(ValueError):
            peyes.match_metrics.precision_recall_f1(self.GT, self.PRED, self._matches(), None)

    def test_all_labels_positive_is_rejected(self):
        with self.assertRaises(ValueError):
            peyes.match_metrics.precision_recall_f1(
                self.GT, self.PRED, self._matches(), list(EventLabelEnum)
            )

    def test_true_positive_requires_both_sides_positive(self):
        """
        M-17: tp counted matched predictions only, so a negative-label GT event matched to a
        positive-label prediction counted as a hit. Reachable when cross-matching is enabled.
        """
        gt = [_event(0.0, label=EventLabelEnum.FIXATION)]
        pred = [_event(1.0, label=EventLabelEnum.SACCADE)]
        matches = peyes.match(gt, pred, "onset", allow_xmatch=True)
        self.assertEqual(1, len(matches), "cross-matching should pair these two events")
        precision, recall, _ = peyes.match_metrics.precision_recall_f1(gt, pred, matches, "saccade")
        # the single GT event is a fixation, so there is no saccade to recall and no true positive
        self.assertEqual(0.0, precision)
        self.assertTrue(np.isnan(recall))

    def test_matching_labels_still_count(self):
        precision, recall, f1 = peyes.match_metrics.precision_recall_f1(
            self.GT, self.PRED, self._matches(), "saccade"
        )
        self.assertEqual((1.0, 1.0, 1.0), (precision, recall, f1))


class TestMatchRatio(unittest.TestCase):

    GT = [_event(0.0), _event(50.0)]
    PRED = [_event(1.0), _event(51.0)]

    def test_one_to_one_matches(self):
        matches = peyes.match(self.GT, self.PRED, "onset")
        self.assertEqual(1.0, peyes.match_metrics.match_ratio(self.PRED, matches))

    def test_one_to_many_matches_are_rejected_clearly(self):
        """ M-18: `.label` on a list raised an opaque AttributeError. """
        matches = peyes.match(self.GT, self.PRED, "generic")
        with self.assertRaises(TypeError):
            peyes.match_metrics.match_ratio(self.PRED, matches)


class TestGetFeatures(unittest.TestCase):
    """ M-10: get_features() used to return a bare array for one feature and a dict for several. """

    GT = [_event(0.0), _event(50.0)]
    PRED = [_event(1.0), _event(51.0)]

    def _matches(self):
        return peyes.match(self.GT, self.PRED, "onset")

    def test_single_feature_still_returns_a_dict(self):
        result = peyes.match_metrics.features(self._matches(), "onset", verbose=False)
        self.assertIsInstance(result, dict)
        self.assertEqual({"onset"}, set(result.keys()))

    def test_multiple_features_return_a_dict(self):
        result = peyes.match_metrics.features(self._matches(), "onset", "offset", verbose=False)
        self.assertIsInstance(result, dict)
        self.assertEqual({"onset", "offset"}, set(result.keys()))

    def test_named_wrapper_still_returns_a_bare_array(self):
        """ Per-feature convenience functions (e.g. onset_difference) must keep returning a plain array. """
        result = peyes.match_metrics.onset_difference(self._matches())
        self.assertIsInstance(result, np.ndarray)
