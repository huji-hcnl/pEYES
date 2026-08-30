import os
import shutil
import tempfile
import unittest

import numpy as np

import peyes
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._utils.visualization_utils import create_image, get_label_colormap


class TestCreateImage(unittest.TestCase):

    def test_no_background_image(self):
        """ V-2: the documented default path is bg_image=None. """
        img = create_image((64, 48))
        self.assertEqual((48, 64, 4), img.shape)    # BGRA

    def test_rejects_out_of_range_alpha(self):
        self.assertRaises(ValueError, create_image, (64, 48), None, 1.5)
        self.assertRaises(ValueError, create_image, (64, 48), None, -0.1)

    def test_rejects_bad_resolution(self):
        self.assertRaises(ValueError, create_image, (0, 48))
        self.assertRaises(ValueError, create_image, (64,))


class TestVideo(unittest.TestCase):

    @staticmethod
    def _data(n=30, with_nans=True):
        t = np.arange(n) * 10.0
        x = np.linspace(100, 500, n)
        y = np.linspace(100, 300, n)
        if with_nans:
            x[5:8] = np.nan
            y[5:8] = np.nan
        labels = np.array([EventLabelEnum.FIXATION] * (n // 2) + [EventLabelEnum.SACCADE] * (n - n // 2))
        return t, x, y, labels

    def test_create_frames_tolerates_missing_samples(self):
        """ V-3: blinks and tracker loss put NaN in x/y; int(nan) used to raise. """
        _, x, y, labels = self._data()
        frames = peyes.visualization.create_frames(x=x, y=y, labels=labels, resolution=(64, 48))
        self.assertEqual(len(x), len(frames))
        self.assertEqual((48, 64, 4), frames[0].shape)

    def test_create_frames_uses_bgr_colours(self):
        """ V-8: the colormap is RGB but cv2 draws in BGR. """
        n = 5
        x, y = np.full(n, 32.0), np.full(n, 24.0)
        labels = np.array([EventLabelEnum.FIXATION] * n)
        frames = peyes.visualization.create_frames(
            x=x, y=y, labels=labels, resolution=(64, 48), gaze_radius=5,
        )
        r, g, b = get_label_colormap(None)[EventLabelEnum.FIXATION]
        centre = frames[0][24, 32]
        self.assertEqual([b, g, r], list(centre[:3]))

    def test_create_video_round_trip(self):
        """ V-1: create_video passed its arguments to create_frames positionally, misaligned. """
        t, x, y, labels = self._data()
        out_dir = tempfile.mkdtemp()
        try:
            path = peyes.visualization.create_video(
                t=t, x=x, y=y, labels=labels,
                output_path=os.path.join(out_dir, "clip"), resolution=(64, 48),
            )
            self.assertTrue(path.endswith(".mp4"))
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
