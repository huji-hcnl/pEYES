import unittest

import src.pEYES._utils.constants as cnst
from src.pEYES._utils.pixel_utils import *


class TestPixelUtils(unittest.TestCase):
    D = 1  # distance from screen to eye
    PS = 1  # pixel size
    TOBII_WIDTH, TOBII_HEIGHT = 53.5, 30.0  # cm
    TOBII_RESOLUTION = (1920, 1080)         # pixels

    def test_cast_to_integers(self):
        # TODO
        pass

    def test_calculate_pixel_size(self):
        self.assertEqual(1, calculate_pixel_size(width=1, height=1, resolution=(1, 1)))
        self.assertEqual(0.5, calculate_pixel_size(width=1, height=1, resolution=(2, 2)))
        self.assertTrue(np.isclose(
            0.027844,
            calculate_pixel_size(width=self.TOBII_WIDTH, height=self.TOBII_HEIGHT, resolution=self.TOBII_RESOLUTION)
        ))

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
        self.assertEqual(0, pixels_to_visual_angle(num_px=0, d=1, pixel_size=1))
        self.assertEqual(90, pixels_to_visual_angle(num_px=2, d=1, pixel_size=1))
        self.assertEqual(np.pi / 2, pixels_to_visual_angle(num_px=2, d=1, pixel_size=1, use_radians=True))
        self.assertTrue(np.isclose(60, pixels_to_visual_angle(num_px=2*np.sqrt(1/3), d=1, pixel_size=1)))
        self.assertTrue(np.isclose(
            np.pi * 2/3, pixels_to_visual_angle(num_px=2*np.sqrt(3), d=1, pixel_size=1, use_radians=True))
        )
        self.assertTrue(np.isnan(pixels_to_visual_angle(num_px=np.inf, d=1, pixel_size=1)))
        self.assertRaises(ValueError, pixels_to_visual_angle, num_px=-1, d=1, pixel_size=1)
        self.assertRaises(ValueError, pixels_to_visual_angle, num_px=1, d=-1, pixel_size=1)
        self.assertRaises(ValueError, pixels_to_visual_angle, num_px=1, d=1, pixel_size=-1)

    def test_visual_angle_to_pixels(self):
        self.assertEqual(0, visual_angle_to_pixels(angle=0, d=1, pixel_size=1))
        self.assertTrue(np.isclose(2, visual_angle_to_pixels(angle=90, d=1, pixel_size=1)))
        self.assertTrue(np.isclose(2, visual_angle_to_pixels(angle=np.pi / 2, d=1, pixel_size=1, use_radians=True)))
        self.assertTrue(np.isclose(2*np.sqrt(1/3), visual_angle_to_pixels(angle=60, d=1, pixel_size=1)))
        self.assertTrue(np.isclose(
            -2*np.sqrt(3), visual_angle_to_pixels(
                angle=-np.pi * 2/3, d=1, pixel_size=1, use_radians=True, keep_sign=True
            ))
        )
        self.assertTrue(np.isnan(visual_angle_to_pixels(angle=np.inf, d=1, pixel_size=1)))
        self.assertRaises(ValueError, visual_angle_to_pixels, angle=1, d=-1, pixel_size=1)
        self.assertRaises(ValueError, visual_angle_to_pixels, angle=1, d=1, pixel_size=-1)

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
