import unittest

import numpy as np

import peyes
from peyes._DataModels.Event import BaseEvent
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._utils.metric_utils import _dprime_rates


def _event(start, n=5, label=EventLabelEnum.SACCADE, sr=500.0):
    t = start + np.arange(n) * (1000.0 / sr)
    return BaseEvent.make(label, t=t, x=np.full(n, 100.0), y=np.full(n, 100.0),
                          pupil=np.ones(n), viewer_distance=60.0, pixel_size=0.0277)


class TestEventRate(unittest.TestCase):
    """ M-2: the rate divided by the last event's absolute end time, not the recording duration. """

    @staticmethod
    def _recording(offset):
        return [_event(offset), _event(offset + 500.0), _event(offset + 1000.0, label=EventLabelEnum.FIXATION)]

    def test_rate_is_invariant_to_clock_offset(self):
        at_zero = peyes.event_metrics.saccade_rate(self._recording(0.0))
        at_100s = peyes.event_metrics.saccade_rate(self._recording(100_000.0))
        self.assertTrue(np.isclose(at_zero, at_100s),
                        f"rate changed with the clock origin: {at_zero} vs {at_100s}")

    def test_rate_value(self):
        events = self._recording(0.0)
        span_ms = max(e.end_time for e in events) - min(e.start_time for e in events)
        self.assertTrue(np.isclose(2 / span_ms * 1000.0, peyes.event_metrics.saccade_rate(events)))

    def test_empty_sequence_rejected(self):
        self.assertRaises(ValueError, peyes.event_metrics.saccade_rate, [])


class TestDprimeCorrectionSpelling(unittest.TestCase):
    """ M-5: the log-linear branch compared the raw string, not the normalised one. """

    ARGS = (10, 20, 15, 10)

    def test_equivalent_spellings_agree(self):
        expected = _dprime_rates(*self.ARGS, "loglinear")
        for spelling in ("log linear", "Log-Linear", "log_linear", "LOGLINEAR", "  loglinear  ", "ll", "hautus"):
            with self.subTest(correction=spelling):
                self.assertEqual(expected, _dprime_rates(*self.ARGS, spelling))

    def test_unknown_correction_still_rejected(self):
        self.assertRaises(ValueError, _dprime_rates, *self.ARGS, "nonsense")


class TestTransitionMatrixShape(unittest.TestCase):
    """ M-8: matrices only covered the labels present, so they could not be compared across sequences. """

    def test_sample_transition_matrix_covers_all_labels(self):
        matrix = peyes.sample_metrics.transition_matrix(
            [EventLabelEnum.FIXATION, EventLabelEnum.SACCADE, EventLabelEnum.FIXATION]
        )
        self.assertEqual((len(EventLabelEnum), len(EventLabelEnum)), matrix.shape)
        self.assertEqual(list(EventLabelEnum), list(matrix.index))

    def test_matrices_from_different_sequences_are_comparable(self):
        a = peyes.sample_metrics.transition_matrix([EventLabelEnum.FIXATION, EventLabelEnum.SACCADE])
        b = peyes.sample_metrics.transition_matrix([EventLabelEnum.PSO, EventLabelEnum.BLINK])
        self.assertEqual(a.shape, b.shape)
        self.assertEqual((len(EventLabelEnum), len(EventLabelEnum)), (a - b).shape)
