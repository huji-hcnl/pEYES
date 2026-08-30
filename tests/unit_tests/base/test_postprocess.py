import unittest

import numpy as np

import peyes
from peyes._DataModels.EventLabelEnum import EventLabelEnum


class TestEventsToLabels(unittest.TestCase):
    """
    C-2 / C-10: `duration` is end_time - start_time, so an n-sample event spans (n-1)*dt. The sample-count
    conversion has to add one in both directions or the round trip loses a sample per event.
    """

    SR = 500.0

    @staticmethod
    def _events(labels, sr):
        t = np.arange(len(labels)) * (1000.0 / sr)
        return peyes.create_events(
            list(labels), t=t, x=np.full(len(labels), 100.0), y=np.full(len(labels), 100.0),
            pupil=np.ones(len(labels)), viewer_distance=60.0, pixel_size=0.0277,
        )

    def _assert_round_trip(self, labels, sr):
        events = self._events(labels, sr)
        out = peyes.events_to_labels(events, sampling_rate=sr)
        self.assertEqual([int(l) for l in labels], [int(v) for v in out])

    def test_round_trip_is_lossless(self):
        self._assert_round_trip(
            [EventLabelEnum.FIXATION] * 3 + [EventLabelEnum.SACCADE] * 2, self.SR
        )

    def test_round_trip_with_several_events(self):
        self._assert_round_trip(
            [EventLabelEnum.FIXATION] * 4 + [EventLabelEnum.SACCADE] * 2
            + [EventLabelEnum.PSO] * 2 + [EventLabelEnum.FIXATION] * 5,
            self.SR,
        )

    def test_round_trip_at_other_sampling_rates(self):
        for sr in (100.0, 250.0, 1000.0):
            with self.subTest(sampling_rate=sr):
                self._assert_round_trip(
                    [EventLabelEnum.FIXATION] * 5 + [EventLabelEnum.SACCADE] * 3, sr
                )

    def test_output_length_matches_input(self):
        labels = [EventLabelEnum.FIXATION] * 7 + [EventLabelEnum.BLINK] * 4
        out = peyes.events_to_labels(self._events(labels, self.SR), sampling_rate=self.SR)
        self.assertEqual(len(labels), len(out))

    def test_min_num_samples_is_a_floor(self):
        labels = [EventLabelEnum.FIXATION] * 5
        events = self._events(labels, self.SR)
        self.assertEqual(5, len(peyes.events_to_labels(events, sampling_rate=self.SR, min_num_samples=3)))
        out = peyes.events_to_labels(events, sampling_rate=self.SR, min_num_samples=9)
        self.assertEqual(9, len(out))
        self.assertTrue(all(int(v) == EventLabelEnum.UNDEFINED for v in out[5:]))

    def test_empty_events_rejected(self):
        """ C-13: this used to raise an opaque ValueError from min() on an empty generator. """
        self.assertRaises(ValueError, peyes.events_to_labels, [], sampling_rate=self.SR)
