import unittest

import numpy as np

from src.pEYES._utils.event_utils import *
import src.pEYES.constants as cnst
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._DataModels.Event import BaseEvent


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
        self.assertEqual(parse_label(EventLabelEnum.FIXATION), EventLabelEnum.FIXATION)
        self.assertEqual(parse_label(BaseEvent.make(EventLabelEnum.FIXATION, np.arange(10))), EventLabelEnum.FIXATION)
        self.assertEqual(parse_label(1), EventLabelEnum.FIXATION)
        self.assertEqual(parse_label("fixation"), EventLabelEnum.FIXATION)
        self.assertEqual(parse_label(1.0), EventLabelEnum.FIXATION)
        self.assertRaises(ValueError, parse_label, 1.5, safe=False)
        self.assertEqual(parse_label(1.5, safe=True), EventLabelEnum.UNDEFINED)
        self.assertRaises(KeyError, parse_label, "foo", safe=False)
        self.assertEqual(parse_label("foo", safe=True), EventLabelEnum.UNDEFINED)
        self.assertRaises(TypeError, parse_label, None, safe=False)
        self.assertEqual(parse_label(None, safe=True), EventLabelEnum.UNDEFINED)

    def test_count_labels(self):
        exp = pd.Series({l: 0 for l in EventLabelEnum})
        self.assertTrue(count_labels(None).equals(exp))
        self.assertTrue(count_labels([]).equals(exp))
        exp[EventLabelEnum.FIXATION] = 3
        self.assertTrue(count_labels([EventLabelEnum.FIXATION] * 3).equals(exp))
        events = [
            BaseEvent.make(EventLabelEnum.FIXATION, np.arange(10)),
            BaseEvent.make(EventLabelEnum.SACCADE, np.arange(10)),
        ] * 2 + [
            BaseEvent.make(EventLabelEnum.FIXATION, np.arange(10)),
            BaseEvent.make(EventLabelEnum.SMOOTH_PURSUIT, np.arange(10)),
        ]
        exp[EventLabelEnum.SACCADE] = 2
        exp[EventLabelEnum.SMOOTH_PURSUIT] = 1
        self.assertTrue(count_labels(events).equals(exp))

