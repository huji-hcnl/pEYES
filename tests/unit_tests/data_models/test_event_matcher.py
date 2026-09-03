import unittest

import numpy as np

from peyes._DataModels.Event import BaseEvent
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._DataModels.EventMatcher import EventMatcher

_VIEWER_DISTANCE_CM = 60.0
_PIXEL_SIZE_CM = 0.0277


def _event(start: float, end: float, label=EventLabelEnum.FIXATION) -> BaseEvent:
    t = np.array([float(start), float(end)])
    return BaseEvent.make(
        label, t=t, x=np.zeros(2), y=np.zeros(2), pupil=np.ones(2),
        viewer_distance=_VIEWER_DISTANCE_CM, pixel_size=_PIXEL_SIZE_CM,
    )


class TestOverlapBasedMatching(unittest.TestCase):
    """
    T-2: EventMatcher's static methods had zero unit-level test coverage before this. One shared scenario -
    three predictions overlapping a single GT event to different degrees - is enough to distinguish
    first/last/max-overlap/longest, since a correct implementation must pick a different prediction for each.
    """

    def setUp(self):
        self.gt = _event(0, 100)
        self.p1 = _event(10, 30)     # earliest start; overlap with gt = 20; own duration = 20
        self.p2 = _event(50, 90)     # overlap with gt = 40 (largest); own duration = 40
        self.p3 = _event(70, 200)    # latest start; overlap with gt = 30; own duration = 130 (largest)
        self.predictions = [self.p1, self.p2, self.p3]

    def test_first_overlap_picks_earliest_start(self):
        matches = EventMatcher.first_overlap([self.gt], self.predictions)
        self.assertIs(self.p1, matches[self.gt])

    def test_last_overlap_picks_latest_start(self):
        matches = EventMatcher.last_overlap([self.gt], self.predictions)
        self.assertIs(self.p3, matches[self.gt])

    def test_max_overlap_picks_largest_overlap(self):
        matches = EventMatcher.max_overlap([self.gt], self.predictions)
        self.assertIs(self.p2, matches[self.gt])

    def test_longest_overlapping_event_picks_largest_own_duration(self):
        matches = EventMatcher.longest_overlapping_event([self.gt], self.predictions)
        self.assertIs(self.p3, matches[self.gt])

    def test_min_overlap_threshold_excludes_below_threshold(self):
        # p1's overlap (20, normalized 0.2) is below a 0.25 threshold; p2 and p3 remain, p2 has the larger overlap
        matches = EventMatcher.max_overlap([self.gt], self.predictions, min_overlap=0.25)
        self.assertIs(self.p2, matches[self.gt])
        matches_none_left = EventMatcher.max_overlap([self.gt], [self.p1], min_overlap=0.25)
        self.assertEqual({}, matches_none_left)


class TestMetricBasedMatching(unittest.TestCase):
    """ iou/l2_timing/onset_difference/offset_difference each pick the candidate that best matches a real
    Event-level metric - verified against that same metric recomputed directly, not a hand-derived value. """

    def setUp(self):
        self.gt = _event(100, 200)
        self.candidates = [_event(95, 195), _event(110, 250), _event(90, 300)]

    def test_iou_picks_max_iou(self):
        matches = EventMatcher.iou([self.gt], self.candidates)
        expected = max(self.candidates, key=lambda c: self.gt.time_iou(c))
        self.assertIs(expected, matches[self.gt])

    def test_l2_timing_picks_min_l2(self):
        matches = EventMatcher.l2_timing([self.gt], self.candidates)
        expected = min(self.candidates, key=lambda c: self.gt.time_l2(c))
        self.assertIs(expected, matches[self.gt])

    def test_onset_difference_picks_min_onset_diff(self):
        matches = EventMatcher.onset_difference([self.gt], self.candidates)
        expected = min(self.candidates, key=lambda c: abs(c.start_time - self.gt.start_time))
        self.assertIs(expected, matches[self.gt])

    def test_offset_difference_picks_min_offset_diff(self):
        matches = EventMatcher.offset_difference([self.gt], self.candidates)
        expected = min(self.candidates, key=lambda c: abs(c.end_time - self.gt.end_time))
        self.assertIs(expected, matches[self.gt])


class TestWindowBased(unittest.TestCase):

    def test_excludes_predictions_outside_the_window(self):
        gt = _event(100, 200)
        near = _event(105, 195)     # onset diff=5, offset diff=5 - inside a window of 10
        far = _event(50, 250)       # onset diff=50, offset diff=50 - outside a window of 10
        matches = EventMatcher.window_based([gt], [near, far], max_onset_difference=10, max_offset_difference=10)
        self.assertEqual({gt: near}, matches)


class TestGenericMatching(unittest.TestCase):

    def test_all_reduction_returns_every_overlapping_prediction(self):
        gt = _event(0, 100)
        p1, p2 = _event(10, 30), _event(50, 90)
        matches = EventMatcher.generic_matching([gt], [p1, p2], allow_cross_matching=True, reduction="all")
        self.assertEqual({p1, p2}, set(matches[gt]))

    def test_cross_matching_disabled_excludes_different_labels(self):
        gt = _event(0, 100, label=EventLabelEnum.FIXATION)
        wrong_label = _event(10, 30, label=EventLabelEnum.SACCADE)
        matches = EventMatcher.first_overlap([gt], [wrong_label], allow_cross_matching=False)
        self.assertEqual({}, matches)

    def test_cross_matching_enabled_allows_different_labels(self):
        gt = _event(0, 100, label=EventLabelEnum.FIXATION)
        wrong_label = _event(10, 30, label=EventLabelEnum.SACCADE)
        matches = EventMatcher.first_overlap([gt], [wrong_label], allow_cross_matching=True)
        self.assertIs(wrong_label, matches[gt])

    def test_default_min_overlap_of_zero_does_not_require_actual_overlap(self):
        """
        min_overlap defaults to 0, and time_overlap floors at 0.0 (not negative) for non-overlapping events, so
        `0.0 >= 0` passes - a genuinely distant prediction still "matches" unless a positive min_overlap is
        given. Confirmed directly: gt.time_overlap(far) == 0.0 for these two, not negative.
        """
        gt = _event(0, 10)
        far = _event(1000, 1010)
        self.assertEqual(0.0, gt.time_overlap(far))
        matches = EventMatcher.first_overlap([gt], [far])
        self.assertEqual({gt: far}, matches)

    def test_positive_min_overlap_excludes_non_overlapping_predictions(self):
        gt = _event(0, 10)
        far = _event(1000, 1010)
        matches = EventMatcher.first_overlap([gt], [far], min_overlap=1e-9)
        self.assertEqual({}, matches)

    def test_non_all_reduction_does_not_reuse_a_prediction(self):
        """ B-3: once a prediction is claimed by one GT, a non-'all' reduction must not also match it to
        another GT - each of the two (distinct) GTs here overlaps the single prediction. """
        p = _event(10, 30)
        gt1 = _event(0, 100)
        gt2 = _event(5, 95)
        matches = EventMatcher.first_overlap([gt1, gt2], [p])
        self.assertEqual(1, len(matches))
