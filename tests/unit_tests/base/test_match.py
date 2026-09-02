import unittest

import numpy as np

import peyes
from peyes._DataModels.Event import BaseEvent
from peyes._DataModels.EventLabelEnum import EventLabelEnum


def _event(start, n=5, label=EventLabelEnum.FIXATION, sr=500.0):
    t = start + np.arange(n) * (1000.0 / sr)
    return BaseEvent.make(label, t=t, x=np.full(n, 100.0), y=np.full(n, 100.0),
                          pupil=np.ones(n), viewer_distance=60.0, pixel_size=0.0277)


class TestMatchAliases(unittest.TestCase):
    """ B-1: every alias the docstring advertises must reach the matcher it names. """

    GT = [_event(0.0), _event(50.0)]
    PRED = [_event(1.0), _event(52.0)]

    ONE_TO_ONE_ALIASES = [
        "first", "first overlap", "last", "last overlap", "max", "max overlap",
        "longest overlap", "iou", "intersection over union",
        "onset", "onset difference", "offset", "offset difference",
        "window", "window based", "l2",
    ]

    def test_documented_aliases_return_one_to_one_matches(self):
        for alias in self.ONE_TO_ONE_ALIASES:
            with self.subTest(match_by=alias):
                matches = peyes.match(self.GT, self.PRED, alias)
                self.assertTrue(matches, f"{alias!r} produced no matches")
                for value in matches.values():
                    self.assertIsInstance(
                        value, BaseEvent,
                        f"{alias!r} fell through to generic matching and returned {type(value).__name__}",
                    )

    def test_offset_difference_matches_offset(self):
        """ The specific typo: 'offset difference' used to fall through to generic_matching. """
        self.assertEqual(
            peyes.match(self.GT, self.PRED, "offset"),
            peyes.match(self.GT, self.PRED, "offset difference"),
        )

    def test_generic_fallback_still_returns_lists(self):
        matches = peyes.match(self.GT, self.PRED, "generic")
        self.assertTrue(all(isinstance(v, list) for v in matches.values()))


class TestMatchDefaults(unittest.TestCase):
    """ B-2: the wrappers overrode generic_matching's `inf` tolerances with 0, requiring exact equality. """

    GT = [_event(0.0), _event(50.0)]
    PRED = [_event(1.0), _event(52.0)]

    def test_tolerance_defaults_are_unbounded(self):
        for alias in ("onset", "offset", "l2", "window"):
            with self.subTest(match_by=alias):
                self.assertEqual(len(self.GT), len(peyes.match(self.GT, self.PRED, alias)))

    def test_explicit_tolerance_still_filters(self):
        self.assertEqual(0, len(peyes.match(self.GT, self.PRED, "onset", max_onset_difference=0)))
        self.assertEqual(2, len(peyes.match(self.GT, self.PRED, "onset", max_onset_difference=5)))
