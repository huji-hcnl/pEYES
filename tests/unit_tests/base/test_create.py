import unittest

import numpy as np

import peyes
from peyes._DataModels.EventLabelEnum import EventLabelEnum


class TestBooleanChannel(unittest.TestCase):
    """ A 5-sample trace at 500 Hz: three fixation samples then two saccade samples. """

    SR = 500.0
    LABELS = [EventLabelEnum.FIXATION] * 3 + [EventLabelEnum.SACCADE] * 2

    def _events(self):
        t = np.arange(0.0, 10.0, 2.0)
        return peyes.create_events(
            self.LABELS, t=t, x=np.full(5, 100.0), y=np.full(5, 100.0), pupil=np.ones(5),
            viewer_distance=60.0, pixel_size=0.0277,
        )

    def test_onset_parity_between_labels_and_events(self):
        """ C-3: the two code paths must agree; the events path was off by one. """
        from_labels = peyes.create_boolean_channel("onset", self.LABELS)
        from_events = peyes.create_boolean_channel("onset", self._events(), sampling_rate=self.SR)
        self.assertTrue(np.array_equal(from_labels, from_events))
        self.assertTrue(np.array_equal(from_labels, np.array([1, 0, 0, 1, 0], dtype=bool)))

    def test_offset_parity_between_labels_and_events(self):
        """ C-3: offsets from events used to land one sample early. """
        from_labels = peyes.create_boolean_channel("offset", self.LABELS)
        from_events = peyes.create_boolean_channel("offset", self._events(), sampling_rate=self.SR)
        self.assertTrue(np.array_equal(from_labels, from_events))
        self.assertTrue(np.array_equal(from_labels, np.array([0, 0, 1, 0, 1], dtype=bool)))

    def test_min_num_samples_defaults_to_none(self):
        """ C-4: None is the documented default but raised TypeError. """
        channel = peyes.create_boolean_channel("onset", self._events(), sampling_rate=self.SR)
        self.assertEqual(5, len(channel))

    def test_min_num_samples_is_a_floor(self):
        events = self._events()
        self.assertEqual(5, len(peyes.create_boolean_channel("onset", events, sampling_rate=self.SR, min_num_samples=3)))
        self.assertEqual(12, len(peyes.create_boolean_channel("onset", events, sampling_rate=self.SR, min_num_samples=12)))

    def test_empty_data_with_min_num_samples(self):
        """ C-15: np.nanmin's second positional argument is `axis`, so this raised AxisError. """
        self.assertEqual(0, len(peyes.create_boolean_channel("onset", [])))
        self.assertEqual(100, len(peyes.create_boolean_channel("onset", [], min_num_samples=100)))

    def test_min_num_samples_too_small_is_widened(self):
        """
        C-16: sample indices were written without a bounds check. Treating `min_num_samples` as a floor
        rather than an exact length means an under-sized value is widened to fit instead of overflowing.
        """
        channel = peyes.create_boolean_channel("onset", self._events(), sampling_rate=self.SR, min_num_samples=2)
        self.assertEqual(5, len(channel))
        self.assertTrue(np.array_equal(channel, np.array([1, 0, 0, 1, 0], dtype=bool)))

    def test_events_require_a_sampling_rate(self):
        self.assertRaises(ValueError, peyes.create_boolean_channel, "onset", self._events())

    def test_invalid_channel_type(self):
        self.assertRaises(ValueError, peyes.create_boolean_channel, "sideways", self.LABELS)


class TestChannelSdtThreshold(unittest.TestCase):

    LABELS = [EventLabelEnum.FIXATION] * 3 + [EventLabelEnum.SACCADE] * 2

    def test_scalar_threshold(self):
        """ M-1: max(threshold) ran before the int-to-list normalisation. """
        result = peyes.channel_metrics.onset_detection_metrics(self.LABELS, self.LABELS, threshold=5)
        self.assertEqual(1, result.shape[0])

    def test_numpy_scalar_threshold(self):
        result = peyes.channel_metrics.onset_detection_metrics(self.LABELS, self.LABELS, threshold=np.int64(5))
        self.assertEqual(1, result.shape[0])

    def test_sequence_threshold(self):
        result = peyes.channel_metrics.onset_detection_metrics(self.LABELS, self.LABELS, threshold=np.arange(4))
        self.assertEqual(4, result.shape[0])

    def test_empty_threshold_is_rejected(self):
        self.assertRaises(
            ValueError, peyes.channel_metrics.onset_detection_metrics, self.LABELS, self.LABELS, threshold=[],
        )
