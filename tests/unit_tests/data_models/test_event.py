import unittest

import numpy as np

import peyes._utils.constants as cnst
from peyes._DataModels.Event import FixationEvent, SaccadeEvent
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
        self.assertEqual(str(f1), "FIXATION(19.00ms)")

    def test_make(self):
        # TODO
        self.assertTrue(True)

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
        f.set_min_duration(10)
        self.assertFalse(f.is_outlier)
        s.set_max_duration(10)
        self.assertTrue(s.is_outlier)
        self.assertEqual(s.get_outlier_reasons(), [cnst.MAX_DURATION_STR])
        self.assertRaises(ValueError, s.set_max_duration, -1)
        self.assertRaises(ValueError, s.set_min_duration, 20)
