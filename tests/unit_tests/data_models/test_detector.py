import unittest
import warnings
from unittest import mock

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

    def test_explicit_saccade_velocity_threshold_is_actually_applied(self):
        """
        D-12: the diamond-inheritance MRO used to land IDTDetector.__init__'s super() call on
        IVTDetector.__init__ WITH ITS OWN DEFAULT, silently discarding whatever saccade_velocity_threshold was
        passed here - only correct end-to-end because IDVTDetector.__init__ then overwrote it directly right
        after. This exercises exactly that path with a non-default value, not incidentally with the default.
        """
        det = IDVTDetector(
            missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0,
            dispersion_threshold=2.7, saccade_velocity_threshold=99.0,
        )
        self.assertEqual(99.0, det._saccade_velocity_threshold)
        self.assertNotEqual(IVTDetector._DEFAULT_SACCADE_VELOCITY_THRESHOLD, det._saccade_velocity_threshold)

    def test_all_three_thresholds_set_correctly_together(self):
        det = IDVTDetector(
            missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0,
            dispersion_threshold=1.5, window_duration=80.0, saccade_velocity_threshold=50.0,
        )
        self.assertEqual(1.5, det._dispersion_threshold)
        self.assertEqual(80.0, det._window_duration)
        self.assertEqual(50.0, det._saccade_velocity_threshold)


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

    def test_axial_velocities_matches_hand_computed_values(self):
        """ D-18: vectorized via sliding_window_view - pin down the exact formula on a hand-checked example. """
        arr = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0])  # window_size=4 -> half_ws=2
        velocities = EngbertDetector._axial_velocities_px(arr, sr=500.0, window_size=4)
        self.assertTrue(np.isnan(velocities[0]))
        self.assertTrue(np.isnan(velocities[1]))
        self.assertTrue(np.isnan(velocities[-1]))
        self.assertTrue(np.isnan(velocities[-2]))
        # idx=2: sum_before=arr[0:2]=0+1=1, sum_after=arr[3:5]=6+10=16, diff=15, v=15*500/4
        self.assertAlmostEqual(15 * 500.0 / 4, velocities[2])
        # idx=3: sum_before=arr[1:3]=1+3=4, sum_after=arr[4:6]=10+15=25, diff=21, v=21*500/4
        self.assertAlmostEqual(21 * 500.0 / 4, velocities[3])
        # idx=4: sum_before=arr[2:4]=3+6=9, sum_after=arr[5:7]=15+21=36, diff=27, v=27*500/4
        self.assertAlmostEqual(27 * 500.0 / 4, velocities[4])

    def test_axial_velocities_nan_stays_local_to_its_own_windows(self):
        """
        D-18: a NaN must only poison the windows whose sum_before/sum_after actually includes it, not every
        window from that point on (which a naive global-cumsum vectorization would do instead).
        """
        arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])  # window_size=2 -> half_ws=1
        velocities = EngbertDetector._axial_velocities_px(arr, sr=500.0, window_size=2)
        # idx=1's sum_after and idx=3's sum_before are single-element windows landing exactly on the NaN at
        # position 2; idx=2 itself and every idx>=4 draw from windows that never touch position 2.
        self.assertTrue(np.isnan(velocities[1]))
        self.assertFalse(np.isnan(velocities[2]))
        self.assertTrue(np.isnan(velocities[3]))
        self.assertFalse(np.isnan(velocities[4]))
        self.assertFalse(np.isnan(velocities[8]))

    def test_median_standard_deviation_clamps_a_negative_radicand(self):
        """
        D-17: `median(x)**2` can exceed `median(x**2)` by a hair due to floating-point cancellation (confirmed
        empirically: ~1 in 200k random arrays), even though the true radicand is mathematically >= 0 always -
        sqrt of that tiny negative silently produced NaN before. Forcing it via mock.patch since the real
        floating-point trigger is a fragile, platform-sensitive edge case unsuitable for a deterministic test.
        """
        with mock.patch("numpy.nanmedian", side_effect=[10.0, 99.0]):  # squared_median=100 > median_of_squares=99
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                sd = EngbertDetector._median_standard_deviation(np.array([1.0, 2.0, 3.0]))
        self.assertFalse(np.isnan(sd))
        self.assertEqual(sd, 1e-10)


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


class TestFrozenDefaultsFollowLiveConfig(unittest.TestCase):
    """
    D-13: IDTDetector/NHDetector/REMoDNaVDetector's config-derived defaults used to be bound as default values
    at class-definition time, so `set_event_configurations` silently had no effect on detectors constructed
    with defaults afterward - same mechanism as C-26 (Event.py), different file, not fixed by that patch.
    """

    def setUp(self):
        import copy
        import peyes._DataModels.config as cnfg
        self._cnfg = cnfg
        self._event_mapping = copy.deepcopy(cnfg.EVENT_MAPPING)

    def tearDown(self):
        self._cnfg.EVENT_MAPPING.clear()
        self._cnfg.EVENT_MAPPING.update(self._event_mapping)

    def test_idt_window_duration_follows_the_setter(self):
        import peyes
        peyes.set_event_configurations(EventLabelEnum.FIXATION, min_duration=12345)
        det = IDTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0)
        self.assertEqual(12345, det._window_duration)
        self.assertEqual(12345, IDTDetector.get_default_params()["window_duration"])

    def test_nh_duration_defaults_follow_the_setters(self):
        import peyes
        peyes.set_event_configurations(EventLabelEnum.SACCADE, min_duration=111)
        peyes.set_event_configurations(EventLabelEnum.FIXATION, min_duration=222)
        peyes.set_event_configurations(EventLabelEnum.PSO, max_duration=333)
        det = NHDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0)
        self.assertEqual(2 * 111, det._filter_duration)
        self.assertEqual(111, det._min_saccade_duration)
        self.assertEqual(222, det._min_fixation_duration)
        self.assertEqual(333, det._max_pso_duration)
        defaults = NHDetector.get_default_params()
        self.assertEqual(2 * 111, defaults["filter_duration_ms"])
        self.assertEqual(111, defaults["min_saccade_duration"])
        self.assertEqual(222, defaults["min_fixation_duration"])
        self.assertEqual(333, defaults["max_pso_duration"])

    def test_remodnav_duration_defaults_follow_the_setters(self):
        import peyes
        peyes.set_event_configurations(EventLabelEnum.SACCADE, min_duration=111)
        peyes.set_event_configurations(EventLabelEnum.SMOOTH_PURSUIT, min_duration=222)
        peyes.set_event_configurations(EventLabelEnum.FIXATION, min_duration=333)
        peyes.set_event_configurations(EventLabelEnum.BLINK, min_duration=444)
        peyes.set_event_configurations(EventLabelEnum.PSO, max_duration=555)
        det = REMoDNaVDetector(
            missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, show_warnings=False,
        )
        self.assertEqual(111, det._min_saccade_duration_ms)
        self.assertEqual(222, det._min_smooth_pursuit_duration_ms)
        self.assertEqual(333, det._min_fixation_duration_ms)
        self.assertEqual(444, det._min_blink_duration_ms)
        self.assertEqual(555, det._max_pso_duration_ms)
        defaults = REMoDNaVDetector.get_default_params()
        self.assertEqual(111, defaults["min_saccade_duration"])
        self.assertEqual(222, defaults["min_smooth_pursuit_duration"])
        self.assertEqual(333, defaults["min_fixation_duration"])
        self.assertEqual(444, defaults["min_blink_duration"])
        self.assertEqual(555, defaults["max_pso_duration"])

    def test_explicit_argument_still_overrides_the_live_default(self):
        """ The None-sentinel resolution must not swallow an explicitly-passed value. """
        import peyes
        peyes.set_event_configurations(EventLabelEnum.FIXATION, min_duration=12345)
        det = IDTDetector(missing_value=np.nan, min_event_duration=4, pad_blinks_ms=0, window_duration=99)
        self.assertEqual(99, det._window_duration)


if __name__ == "__main__":
    unittest.main()
