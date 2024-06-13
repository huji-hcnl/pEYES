import unittest

import numpy as np

from src.pEYES.helpers.pixel_utils import *
import src.pEYES.constants as cnst


class TestPixelUtils(unittest.TestCase):
    D = 1  # distance from screen to eye
    PS = 1  # pixel size

    def test_calculate_velocities(self):
        sqrt2 = np.sqrt(2)
        x_coords = np.arange(0, 5)
        y_coords1 = np.arange(0, 5)
        t_coords1 = np.arange(5)
        self.assertTrue(np.allclose(
            np.array([np.nan, sqrt2, sqrt2, sqrt2, sqrt2]) * cnst.MILLISECONDS_PER_SECOND,
            calculate_velocities(x_coords, y_coords1, t_coords1),
            equal_nan=True
        ))
        y_coords2 = np.zeros_like(x_coords)
        self.assertTrue(np.allclose(
            np.array([np.nan, 1, 1, 1, 1]) * cnst.MILLISECONDS_PER_SECOND,
            calculate_velocities(x_coords, y_coords2, t_coords1),
            equal_nan=True
        ))
        t_coords2 = t_coords1 * 2
        self.assertTrue(np.allclose(
            np.array([np.nan, 0.5, 0.5, 0.5, 0.5]) * cnst.MILLISECONDS_PER_SECOND,
            calculate_velocities(x_coords, y_coords2, t_coords2),
            equal_nan=True
        ))
        y_coords3 = y_coords1.copy().astype(float)
        y_coords3[2] = np.nan

        print(calculate_velocities(x_coords, y_coords3, t_coords1))

        self.assertTrue(np.allclose(
            np.array([np.nan, sqrt2, np.nan, np.nan, sqrt2]) * cnst.MILLISECONDS_PER_SECOND,
            calculate_velocities(x_coords, y_coords3, t_coords1),
            equal_nan=True
        ))
        t_coords3 = t_coords1[:-1].copy()
        self.assertRaises(AssertionError, calculate_velocities, x_coords, y_coords1, t_coords3)

    def test_pixels_to_visual_angle(self):
        self.assertEqual(0, pixels_to_visual_angle(num_px=0, d=self.D, pixel_size=self.PS))
        self.assertEqual(45, pixels_to_visual_angle(num_px=1, d=self.D, pixel_size=self.PS))
        self.assertEqual(np.pi / 4, pixels_to_visual_angle(num_px=1, d=self.D, pixel_size=self.PS, use_radians=True))
        self.assertTrue(np.isnan(pixels_to_visual_angle(num_px=np.inf, d=self.D, pixel_size=self.PS)))
        self.assertRaises(ValueError, pixels_to_visual_angle, num_px=-1, d=self.D, pixel_size=self.PS)
        self.assertRaises(ValueError, pixels_to_visual_angle, num_px=1, d=-1, pixel_size=self.PS)
        self.assertRaises(ValueError, pixels_to_visual_angle, num_px=1, d=self.D, pixel_size=-1)

    def test_calculate_azimuth(self):
        # angles are counter-clockwise from the positive x-axis, with y-axis pointing down
        self.assertEqual(0, calculate_azimuth(p1=(0, 0), p2=(0, 0), use_radians=False))
        self.assertEqual(45, calculate_azimuth(p1=(0, 0), p2=(1, -1), use_radians=False))
        self.assertEqual(315, calculate_azimuth(p1=(0, 0), p2=(1, 1), use_radians=False))
        self.assertEqual(90, calculate_azimuth(p1=(1, 1), p2=(1, -1), use_radians=False))
        self.assertEqual(180, calculate_azimuth(p1=(1, 1), p2=(1, -1), use_radians=False, zero_direction='S'))
        self.assertEqual(np.pi * 3 / 4, calculate_azimuth(p1=(1, 1), p2=(-2, -2), use_radians=True))
        self.assertEqual(np.pi, calculate_azimuth(p1=(1, 0), p2=(-1, 0), use_radians=True))
        self.assertEqual(np.pi * 5 / 4, calculate_azimuth(p1=(1, 0), p2=(0, 1), use_radians=True))
        self.assertEqual(np.pi * 3 / 4, calculate_azimuth(p1=(1, 0), p2=(0, 1), use_radians=True, zero_direction='N'))