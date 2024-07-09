import unittest

from src.pEYES._utils.event_utils import *
from src.pEYES._utils.pixel_utils import visual_angle_to_pixels
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

    def test_microsaccade_ratio(self):
        viewer_distance, pixel_size = 60, 0.3
        events = []
        for i in range(5):
            last_x = visual_angle_to_pixels((i+1)/2, viewer_distance, pixel_size)
            sac = BaseEvent.make(
                EventLabelEnum.SACCADE,
                t=np.arange(10),
                x=np.linspace(0, last_x, 10),
                y=np.zeros(10),
                viewer_distance=viewer_distance,
                pixel_size=pixel_size,
            )
            self.assertTrue(np.isclose(sac.amplitude, (i+1)/2))
            events.append(sac)
        self.assertEqual(microsaccade_ratio(events, 1), 0.2)
        events[2] = events[0]
        self.assertEqual(microsaccade_ratio(events, 1), 0.4)
        self.assertEqual(microsaccade_ratio(events, 0.01), 0)
        self.assertTrue(np.isnan(microsaccade_ratio([], 1)))
