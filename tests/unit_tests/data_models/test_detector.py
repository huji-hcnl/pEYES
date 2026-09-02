import unittest

import numpy as np

from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._DataModels.Detector import (
    IVTDetector, IVVTDetector, IDTDetector, IDVTDetector, EngbertDetector, NHDetector, REMoDNaVDetector,
)
from peyes._utils.pixel_utils import visual_angle_to_pixels

_VIEWER_DISTANCE_CM = 60.0
_PIXEL_SIZE_CM = 0.03
_SAMPLING_RATE_HZ = 500.0
_DT_MS = 1000.0 / _SAMPLING_RATE_HZ


def _deg_to_px(deg: float) -> float:
    return visual_angle_to_pixels(deg, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)


def _make_trial(
        fixation_samples: int = 150, saccade_amplitude_deg: float = 10.0, num_saccades: int = 1,
        saccade_samples: int = 6,
):
    """
    Builds a synthetic trial: `fixation_samples` stationary samples, then (for each saccade) a ramp spread
    over `saccade_samples` samples covering `saccade_amplitude_deg` degrees total, then another
    `fixation_samples` stationary samples. Each ramp sample moves ~1.67deg in 2ms (~830deg/s) - unambiguously
    a saccade under every detector's default thresholds (all well under 1000deg/s) and unambiguously a large
    dispersion jump for IDT/IDVT - and, critically, sustained over enough samples to survive every detector's
    min_event_duration short-chunk filtering (a single-sample step is a 1-sample "event", shorter than any
    reasonable minimum, and gets silently reset to UNDEFINED before it can be asserted on).
    """
    n = fixation_samples * (num_saccades + 1) + num_saccades * saccade_samples
    t = np.arange(n) * _DT_MS
    x = np.full(n, 500.0)
    y = np.full(n, 500.0)
    jump_px = _deg_to_px(saccade_amplitude_deg)
    ramp = np.linspace(0, jump_px, saccade_samples + 1)[1:]
    idx = fixation_samples
    for _ in range(num_saccades):
        x[idx:idx + saccade_samples] += ramp
        x[idx + saccade_samples:] += jump_px
        idx += fixation_samples + saccade_samples
    return t, x, y


def _label_runs(labels):
    """Collapses a label sequence into (label, count) runs, e.g. [FIX,FIX,SAC,FIX] -> [(FIX,2),(SAC,1),(FIX,1)]."""
    runs = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1] = (label, runs[-1][1] + 1)
        else:
            runs.append((label, 1))
    return runs


class TestBaseDetectorPipeline(unittest.TestCase):
    """Shared BaseDetector.detect() pipeline behavior, exercised via IVT as the simplest concrete subclass."""

    def test_rejects_non_positive_min_event_duration(self):
        self.assertRaises(ValueError, IVTDetector, missing_value=np.nan, min_event_duration=0, pad_blinks_ms=0)
        self.assertRaises(ValueError, IVTDetector, missing_value=np.nan, min_event_duration=-1, pad_blinks_ms=0)

    def test_rejects_negative_pad_blinks_ms(self):
        self.assertRaises(ValueError, IVTDetector, missing_value=np.nan, min_event_duration=4, pad_blinks_ms=-1)

    def test_rejects_non_positive_viewer_distance_or_pixel_size(self):
        det = IVTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0)
        t, x, y = _make_trial()
        self.assertRaises(ValueError, det.detect, t, x, y, 0, _PIXEL_SIZE_CM)
        self.assertRaises(ValueError, det.detect, t, x, y, _VIEWER_DISTANCE_CM, 0)
        self.assertRaises(ValueError, det.detect, t, x, y, np.nan, _PIXEL_SIZE_CM)

    def test_blink_samples_are_labeled_blink_and_nanned_out(self):
        det = IVTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0)
        t, x, y = _make_trial()
        x[10:20] = np.nan
        y[10:20] = np.nan
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertTrue(all(l == EventLabelEnum.BLINK for l in labels[10:20]))

    def test_metadata_does_not_leak_across_calls_on_a_reused_instance(self):
        # D-20: a detector instance reused across trials must not carry stale per-call metadata forward.
        det = IVTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0)
        t1, x1, y1 = _make_trial(fixation_samples=50, num_saccades=0)
        _, metadata1 = det.detect(t1, x1, y1, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        t2, x2, y2 = _make_trial(fixation_samples=50, num_saccades=0)
        _, metadata2 = det.detect(t2, x2, y2, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertEqual(set(metadata1.keys()), set(metadata2.keys()))

    def test_warns_on_non_monotonic_t(self):
        det = IVTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0)
        t, x, y = _make_trial()
        t[5] = t[4]  # break monotonicity
        with self.assertWarns(UserWarning):
            det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)

    def test_short_chunks_are_dropped_or_merged(self):
        # A single-sample "fixation" between two saccade-adjacent samples is shorter than min_event_duration
        # and must not survive as its own labeled chunk.
        det = IVTDetector(missing_value=np.nan, min_event_duration=100, pad_blinks_ms=0)
        t, x, y = _make_trial(fixation_samples=50, num_saccades=2)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        runs = _label_runs(labels)
        min_samples = det.min_event_samples
        for label, count in runs:
            if label != EventLabelEnum.UNDEFINED:
                self.assertGreaterEqual(count, min_samples, f"{label} run of {count} samples is shorter than the {min_samples}-sample minimum")


class TestIVTDetector(unittest.TestCase):

    def test_classifies_fixation_then_saccade_then_fixation(self):
        det = IVTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, saccade_velocity_threshold=45)
        t, x, y = _make_trial(fixation_samples=50, saccade_amplitude_deg=10.0, num_saccades=1)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertIn(EventLabelEnum.FIXATION, labels)
        self.assertIn(EventLabelEnum.SACCADE, labels)
        # the ramp region (indices 50-55) must contain the fast classification
        self.assertIn(EventLabelEnum.SACCADE, labels[50:56])
        # index 0 is UNDEFINED by design (calculate_velocities has no prior sample to diff against there)
        self.assertEqual(labels[1], EventLabelEnum.FIXATION)
        self.assertEqual(labels[-1], EventLabelEnum.FIXATION)

    def test_rejects_non_positive_threshold(self):
        self.assertRaises(
            ValueError, IVTDetector, missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0,
            saccade_velocity_threshold=0,
        )


class TestIVVTDetector(unittest.TestCase):

    def test_classifies_intermediate_velocity_as_smooth_pursuit(self):
        det = IVVTDetector(
            missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0,
            saccade_velocity_threshold=100, smooth_pursuit_velocity_threshold=1,
        )
        # a shallow ramp (0.3deg over 6 samples at 500Hz) is ~25deg/s - comfortably between the 1deg/s SP
        # floor and the 100deg/s saccade ceiling, so it must land as smooth pursuit, not saccade.
        t, x, y = _make_trial(fixation_samples=50, saccade_amplitude_deg=0.3, num_saccades=1)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertIn(EventLabelEnum.SMOOTH_PURSUIT, labels[50:56])

    def test_classifies_fast_velocity_as_saccade(self):
        det = IVVTDetector(
            missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0,
            saccade_velocity_threshold=45, smooth_pursuit_velocity_threshold=1,
        )
        t, x, y = _make_trial(fixation_samples=50, saccade_amplitude_deg=10.0, num_saccades=1)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertIn(EventLabelEnum.SACCADE, labels[50:56])


class TestIDTDetector(unittest.TestCase):

    def test_stationary_window_is_fixation(self):
        det = IDTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, dispersion_threshold=2.7)
        t, x, y = _make_trial(fixation_samples=150, num_saccades=0)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertTrue(all(l == EventLabelEnum.FIXATION for l in labels))

    def test_large_jump_breaks_the_fixation_window(self):
        det = IDTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, dispersion_threshold=2.7)
        t, x, y = _make_trial(fixation_samples=150, saccade_amplitude_deg=10.0, num_saccades=1)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertIn(EventLabelEnum.SACCADE, labels)

    def test_rejects_non_positive_threshold(self):
        self.assertRaises(
            ValueError, IDTDetector, missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0,
            dispersion_threshold=0,
        )


class TestIDVTDetector(unittest.TestCase):

    def test_stationary_window_is_fixation(self):
        det = IDVTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, dispersion_threshold=2.7)
        t, x, y = _make_trial(fixation_samples=150, num_saccades=0)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertTrue(all(l == EventLabelEnum.FIXATION for l in labels))

    def test_large_jump_is_not_fixation(self):
        det = IDVTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, dispersion_threshold=2.7)
        t, x, y = _make_trial(fixation_samples=150, saccade_amplitude_deg=10.0, num_saccades=1)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertTrue(any(l != EventLabelEnum.FIXATION for l in labels[150:156]))


class TestEngbertDetector(unittest.TestCase):

    def test_classifies_fixation_then_saccade_then_fixation(self):
        det = EngbertDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, lambda_param=6)
        t, x, y = _make_trial(fixation_samples=100, saccade_amplitude_deg=10.0, num_saccades=1)
        labels, _ = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertIn(EventLabelEnum.FIXATION, labels)
        self.assertIn(EventLabelEnum.SACCADE, labels)
        self.assertIn(EventLabelEnum.SACCADE, labels[100:106])

    def test_rejects_non_positive_lambda(self):
        self.assertRaises(
            ValueError, EngbertDetector, missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, lambda_param=0,
        )


class TestNHDetector(unittest.TestCase):
    """
    NHDetector is the highest-complexity algorithm here (adaptive thresholding, Savitzky-Golay filtering) - these
    are broad sanity checks (runs cleanly, produces both label types on an unambiguous stimulus), not exact
    per-sample assertions. D-1 through D-9's specific edge-case findings are tracked separately in
    docs/CODE_REVIEW.md, not re-derived here.
    """

    def test_runs_without_error_and_produces_both_labels(self):
        det = NHDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0)
        t, x, y = _make_trial(fixation_samples=200, saccade_amplitude_deg=10.0, num_saccades=1)
        labels, metadata = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertEqual(len(labels), len(t))
        self.assertIn(EventLabelEnum.FIXATION, labels)
        self.assertIn(EventLabelEnum.SACCADE, labels)
        self.assertIn("sampling_rate", metadata)


class TestREMoDNaVDetector(unittest.TestCase):
    """Same disposition as TestNHDetector: broad sanity checks, not exact per-sample assertions."""

    def test_runs_without_error_and_produces_both_labels(self):
        det = REMoDNaVDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, show_warnings=False)
        t, x, y = _make_trial(fixation_samples=200, saccade_amplitude_deg=10.0, num_saccades=1)
        labels, metadata = det.detect(t, x, y, _VIEWER_DISTANCE_CM, _PIXEL_SIZE_CM)
        self.assertEqual(len(labels), len(t))
        self.assertIn(EventLabelEnum.FIXATION, labels)
        self.assertIn(EventLabelEnum.SACCADE, labels)


if __name__ == "__main__":
    unittest.main()
