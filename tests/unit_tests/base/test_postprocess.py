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

    def test_t_start_none_matches_default_behavior(self):
        """ C-27: the default (no t_start) must stay anchored to the earliest event's own start time. """
        labels = [EventLabelEnum.FIXATION] * 5 + [EventLabelEnum.SACCADE] * 3
        events = self._events(labels, self.SR)
        out_default = peyes.events_to_labels(events, sampling_rate=self.SR)
        out_explicit_none = peyes.events_to_labels(events, sampling_rate=self.SR, t_start=None)
        np.testing.assert_array_equal(out_default, out_explicit_none)

    def test_t_start_before_earliest_event_pads_with_undefined(self):
        """ C-27: a leading gap before the first event must show up as leading UNDEFINED samples. """
        labels = [EventLabelEnum.FIXATION] * 5
        events = self._events(labels, self.SR)
        earliest_start_time = events[0].start_time
        lead_samples = 4
        t_start = earliest_start_time - lead_samples * (1000.0 / self.SR)
        out = peyes.events_to_labels(events, sampling_rate=self.SR, t_start=t_start)
        self.assertEqual(lead_samples + len(labels), len(out))
        self.assertTrue(all(int(v) == EventLabelEnum.UNDEFINED for v in out[:lead_samples]))
        self.assertEqual([int(l) for l in labels], [int(v) for v in out[lead_samples:]])

    def test_t_start_after_earliest_event_raises(self):
        """ C-27: a t_start later than the earliest event would place its samples before index 0. """
        labels = [EventLabelEnum.FIXATION] * 5
        events = self._events(labels, self.SR)
        earliest_start_time = events[0].start_time
        self.assertRaises(
            ValueError, peyes.events_to_labels, events, sampling_rate=self.SR,
            t_start=earliest_start_time + 1.0,
        )

    def test_t_end_none_matches_default_behavior(self):
        """ C-27: the default (no t_end) must stay anchored to the latest event's own end time. """
        labels = [EventLabelEnum.FIXATION] * 5 + [EventLabelEnum.SACCADE] * 3
        events = self._events(labels, self.SR)
        out_default = peyes.events_to_labels(events, sampling_rate=self.SR)
        out_explicit_none = peyes.events_to_labels(events, sampling_rate=self.SR, t_end=None)
        np.testing.assert_array_equal(out_default, out_explicit_none)

    def test_t_end_after_latest_event_pads_with_undefined(self):
        """ C-27: a trailing gap after the last event must show up as trailing UNDEFINED samples. """
        labels = [EventLabelEnum.FIXATION] * 5
        events = self._events(labels, self.SR)
        latest_end_time = events[-1].end_time
        trail_samples = 4
        t_end = latest_end_time + trail_samples * (1000.0 / self.SR)
        out = peyes.events_to_labels(events, sampling_rate=self.SR, t_end=t_end)
        self.assertEqual(len(labels) + trail_samples, len(out))
        self.assertEqual([int(l) for l in labels], [int(v) for v in out[:len(labels)]])
        self.assertTrue(all(int(v) == EventLabelEnum.UNDEFINED for v in out[len(labels):]))

    def test_t_end_before_latest_event_raises(self):
        """ C-27: a t_end earlier than the latest event would place its samples past the end of the array. """
        labels = [EventLabelEnum.FIXATION] * 5
        events = self._events(labels, self.SR)
        latest_end_time = events[-1].end_time
        self.assertRaises(
            ValueError, peyes.events_to_labels, events, sampling_rate=self.SR,
            t_end=latest_end_time - 1.0,
        )

    def test_t_start_and_t_end_together_pad_both_sides(self):
        """ C-27: t_start and t_end must compose - both a leading and a trailing gap padded at once. """
        labels = [EventLabelEnum.FIXATION] * 5
        events = self._events(labels, self.SR)
        dt = 1000.0 / self.SR
        lead_samples, trail_samples = 3, 2
        t_start = events[0].start_time - lead_samples * dt
        t_end = events[-1].end_time + trail_samples * dt
        out = peyes.events_to_labels(events, sampling_rate=self.SR, t_start=t_start, t_end=t_end)
        self.assertEqual(lead_samples + len(labels) + trail_samples, len(out))
        self.assertTrue(all(int(v) == EventLabelEnum.UNDEFINED for v in out[:lead_samples]))
        self.assertEqual(
            [int(l) for l in labels], [int(v) for v in out[lead_samples:lead_samples + len(labels)]],
        )
        self.assertTrue(all(int(v) == EventLabelEnum.UNDEFINED for v in out[lead_samples + len(labels):]))

    def test_narrower_window_than_the_events_raises_not_silently_truncates(self):
        """
        A caller-supplied [t_start, t_end] strictly inside the events' own [T0, T1] span would truncate real
        event data (some event's samples would land outside the output array) - must raise on either violated
        bound individually, not silently drop labels or return a subset.
        """
        labels = [EventLabelEnum.FIXATION] * 4 + [EventLabelEnum.SACCADE] * 4 + [EventLabelEnum.FIXATION] * 4
        events = self._events(labels, self.SR)
        earliest_start_time, latest_end_time = events[0].start_time, events[-1].end_time
        dt = 1000.0 / self.SR
        # only t_start narrows (t_end left at the true end)
        self.assertRaises(
            ValueError, peyes.events_to_labels, events, sampling_rate=self.SR,
            t_start=earliest_start_time + 2 * dt, t_end=latest_end_time,
        )
        # only t_end narrows (t_start left at the true start)
        self.assertRaises(
            ValueError, peyes.events_to_labels, events, sampling_rate=self.SR,
            t_start=earliest_start_time, t_end=latest_end_time - 2 * dt,
        )
        # both narrow at once
        self.assertRaises(
            ValueError, peyes.events_to_labels, events, sampling_rate=self.SR,
            t_start=earliest_start_time + 2 * dt, t_end=latest_end_time - 2 * dt,
        )


class TestSummarizeEvents(unittest.TestCase):

    @staticmethod
    def _event():
        t = np.arange(0.0, 30.0, 2.0)
        return peyes.create_events(
            "saccade", t=t, x=np.linspace(100.0, 400.0, len(t)), y=np.linspace(200.0, 260.0, len(t)),
            pupil=np.ones_like(t), viewer_distance=60.0, pixel_size=0.0277,
        )

    def test_empty_input_keeps_the_schema(self):
        """ Issue #25: an empty result used to be a (0, 0) frame with no columns. """
        empty = peyes.summarize_events([])
        self.assertEqual(0, len(empty))
        self.assertEqual(list(empty.columns), list(peyes.summarize_events([self._event()]).columns))

    def test_empty_results_can_be_concatenated(self):
        import pandas as pd
        empty = peyes.summarize_events([])
        self.assertIn("start_time", pd.concat([empty, empty]).columns)

    def test_empty_result_columns_are_selectable(self):
        empty = peyes.summarize_events([])
        self.assertEqual(0, len(empty["start_time"]))
