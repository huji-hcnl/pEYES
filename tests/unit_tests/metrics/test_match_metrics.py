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
