import unittest

import numpy as np

from src.pEYES._utils.event_utils import *
import src.pEYES.constants as cnst


class TestEventUtils(unittest.TestCase):

    def test_calculate_sampling_rate(self):
        t = np.arange(10)
        self.assertEqual(calculate_sampling_rate(t), cnst.MILLISECONDS_PER_SECOND)
        t = np.arange(0, 11, 2)
        self.assertEqual(calculate_sampling_rate(t), cnst.MILLISECONDS_PER_SECOND / 2)
        t = np.arange(10) * cnst.MILLISECONDS_PER_SECOND
        self.assertEqual(calculate_sampling_rate(t), 1)
        t = np.setdiff1d(np.arange(10), [2, 5, 8])
        self.assertEqual(calculate_sampling_rate(t), cnst.MILLISECONDS_PER_SECOND / 1.5)

    def test_parse_label(self):
        # TODO
        pass

    def test_count_labels(self):
        # TODO
        pass
