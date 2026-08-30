import unittest

import numpy as np

import peyes._utils.constants as cnst
from peyes._DataModels.Event import (
    BaseEvent, FixationEvent, SaccadeEvent, PSOEvent, SmoothPursuitEvent, BlinkEvent
)
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._utils.pixel_utils import pixels_to_visual_angle


class TestEvent(unittest.TestCase):
    _PS, _VD = 1, 1

    def test_init(self):
        t = np.arange(20)
        x = np.sin(np.pi * t / 10)
        y = np.cos(np.pi * t / 10)
        self.assertRaises(AssertionError, FixationEvent, t=t, x=x, y=y[:-1])
        self.assertRaises(AssertionError, FixationEvent, t=t, x=x, y=y, pixel_size=-self._PS)
        self.assertRaises(AssertionError, FixationEvent, t=t, x=x, y=y, viewer_distance=-self._VD)
        f1 = FixationEvent(t=t)
        f2 = FixationEvent(t=t, x=None, y=None)
        f3 = FixationEvent(t=t, x=x, y=y, pixel_size=self._PS, viewer_distance=self._VD)
        self.assertTrue(f1 == f2)
        self.assertFalse(f1 == f3)
        self.assertEqual(str(f1), "FIXATION(19.0ms)")

    def test_make(self):
        t = np.arange(10)
        self.assertIsNone(BaseEvent.make(EventLabelEnum.UNDEFINED, t=t))
        for label, cls in [
            (EventLabelEnum.FIXATION, FixationEvent), (EventLabelEnum.SACCADE, SaccadeEvent),
            (EventLabelEnum.PSO, PSOEvent), (EventLabelEnum.SMOOTH_PURSUIT, SmoothPursuitEvent),
            (EventLabelEnum.BLINK, BlinkEvent),
        ]:
            event = BaseEvent.make(label, t=t)
            self.assertIsInstance(event, cls)
            self.assertEqual(event.label, label)
            self.assertEqual(len(event), len(t))

    def test_make_multiple(self):
        t = np.arange(6)
        labels = np.array([
            EventLabelEnum.FIXATION, EventLabelEnum.FIXATION, EventLabelEnum.SACCADE,
            EventLabelEnum.SACCADE, EventLabelEnum.UNDEFINED, EventLabelEnum.BLINK,
        ])
        events = BaseEvent.make_multiple(labels, t=t)
        # UNDEFINED chunks are dropped, contiguous same-label samples become one event
        self.assertEqual([e.label for e in events],
                         [EventLabelEnum.FIXATION, EventLabelEnum.SACCADE, EventLabelEnum.BLINK])
        self.assertEqual([len(e) for e in events], [2, 2, 1])
        self.assertRaises(ValueError, BaseEvent.make_multiple, labels, t[:-1])

    def test_properties(self):
        fix_x, fix_y = np.full(21, 40), np.hstack([np.arange(50, 0, -5), np.arange(0, 51, 5)])
        t = np.arange(21)
        f = FixationEvent(t=t, x=fix_x, y=fix_y, pixel_size=self._PS, viewer_distance=self._VD)
        self.assertEqual(f.label, EventLabelEnum.FIXATION)
        self.assertEqual(f.duration, 20)
        self.assertEqual(f.center_pixel, (np.nanmean(fix_x), np.nanmean(fix_y)))
        self.assertEqual(f.pixel_std, (np.nanstd(fix_x), np.nanstd(fix_y)))
        self.assertEqual(f.distance, 0.0)
        self.assertEqual(f.amplitude, 0.0)
        self.assertEqual(f.azimuth, 0.0)
        self.assertEqual(f.cumulative_distance, 100)
        self.assertEqual(f.cumulative_amplitude, pixels_to_visual_angle(100, self._VD, self._PS))
        self.assertEqual(f.x_dispersion, 0.0)
        self.assertEqual(f.y_dispersion, pixels_to_visual_angle(50, self._VD, self._PS))
        self.assertEqual(f.dispersion, pixels_to_visual_angle(50, self._VD, self._PS))

    def test_velocity(self):
        t, x, y = np.arange(21), np.full(21, 40), np.hstack([np.arange(50, 0, -5), np.arange(0, 51, 5)])
        expected_px_vel = np.full_like(t, 5000, dtype=float)
        f = FixationEvent(t=t, x=x, y=y, pixel_size=self._PS, viewer_distance=self._VD)
        expected_px_vel[0] = np.nan  # first velocity is undefined
        self.assertTrue(np.allclose(f.velocities('px'), expected_px_vel, equal_nan=True))
        expected_deg_vel = np.vectorize(pixels_to_visual_angle)(expected_px_vel, self._VD, self._PS)
        self.assertTrue(np.allclose(f.velocities('deg'), expected_deg_vel, equal_nan=True))
        expected_rad_vel = np.vectorize(pixels_to_visual_angle)(expected_px_vel, self._VD, self._PS, use_radians=True)
        self.assertTrue(np.allclose(f.velocities('rad'), expected_rad_vel, equal_nan=True))
        self.assertRaises(ValueError, f.velocities, 'foobar')

    def test_overlaps(self):
        t, x, y = np.arange(21), np.full(21, 40), np.hstack([np.arange(50, 0, -5), np.arange(0, 51, 5)])
        f1 = FixationEvent(t=t, x=x, y=y, pixel_size=self._PS, viewer_distance=self._VD)
        f2 = FixationEvent(t=t+10, x=x, y=y, pixel_size=self._PS, viewer_distance=self._VD)
        self.assertFalse(f1 == f2)
        self.assertEqual(f1.time_overlap(f2, normalize=False), 10)
        self.assertEqual(f1.time_overlap(f2, normalize=True), 0.5)
        self.assertEqual(f1.time_iou(f2), 1/3)
        self.assertEqual(f1.time_l2(f2), 10 * np.sqrt(2))

    def test_duration_outliers(self):
        t = np.arange(21)
        s = SaccadeEvent(t=t)
        self.assertFalse(s.is_outlier)
        f = FixationEvent(t=t)
        self.assertTrue(f.is_outlier)
        self.assertEqual(f.get_outlier_reasons(), [cnst.MIN_DURATION_STR])


class TestEventSummary(unittest.TestCase):

    @staticmethod
    def _saccade():
        t = np.arange(0.0, 30.0, 2.0)
        return SaccadeEvent(
            t=t, x=np.linspace(100.0, 400.0, len(t)), y=np.linspace(200.0, 260.0, len(t)),
            pupil=np.ones_like(t), viewer_distance=60.0, pixel_size=0.0277,
        )

    def test_summary_reports_endpoints(self):
        """ Issue #24: center_pixel is the midpoint of a saccade; the endpoints were unrecoverable. """
        summary = self._saccade().summary()
        self.assertEqual(100.0, summary[cnst.START_X_STR])
        self.assertEqual(200.0, summary[cnst.START_Y_STR])
        self.assertEqual(400.0, summary[cnst.END_X_STR])
        self.assertEqual(260.0, summary[cnst.END_Y_STR])

    def test_summary_columns_matches_summary(self):
        self.assertEqual(BaseEvent.summary_columns(), list(self._saccade().summary().index))


class TestExtremumPixels(unittest.TestCase):
    """ C-8: argmin/argmax return the index of a NaN, so any missing sample poisoned these properties. """

    @staticmethod
    def _event(x, y):
        t = np.arange(len(x), dtype=float)
        return FixationEvent(t=t, x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float),
                             pupil=np.ones(len(x)), viewer_distance=60.0, pixel_size=0.0277)

    def test_ignores_nans(self):
        event = self._event([5.0, 1.0, np.nan, 9.0, 3.0], [2.0, 6.0, 7.0, 0.0, 8.0])
        self.assertEqual(1.0, event.left_pixel[0])
        self.assertEqual(9.0, event.right_pixel[0])
        self.assertEqual(0.0, event.top_pixel[1])
        self.assertEqual(8.0, event.bottom_pixel[1])

    def test_all_nan_returns_nan(self):
        event = self._event([np.nan] * 4, [np.nan] * 4)
        for pixel in (event.left_pixel, event.right_pixel, event.top_pixel, event.bottom_pixel):
            self.assertTrue(all(np.isnan(v) for v in pixel))

    def test_matches_naive_result_when_no_nans(self):
        x, y = [5.0, 1.0, 9.0, 3.0], [2.0, 6.0, 0.0, 8.0]
        event = self._event(x, y)
        self.assertEqual((x[int(np.argmin(x))], y[int(np.argmin(x))]), event.left_pixel)
        self.assertEqual((x[int(np.argmax(y))], y[int(np.argmax(y))]), event.bottom_pixel)
