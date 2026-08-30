import unittest

from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._utils.visualization_utils import to_rgb, get_label_colormap


class TestVisualizationUtils(unittest.TestCase):

    def test_to_rgb_from_hex(self):
        self.assertEqual((255, 0, 0), to_rgb("#ff0000"))
        self.assertEqual((0, 17, 34), to_rgb("#001122"))
        self.assertEqual((255, 255, 255), to_rgb("#FFFFFF"))

    def test_to_rgb_from_tuple(self):
        self.assertEqual((1, 2, 3), to_rgb((1, 2, 3)))

    def test_to_rgb_rejects_bad_input(self):
        self.assertRaises(ValueError, to_rgb, "ff0000")      # no leading '#'
        self.assertRaises(ValueError, to_rgb, "#fff")        # too short
        self.assertRaises(ValueError, to_rgb, (1, 2))        # not 3 elements
        self.assertRaises(ValueError, to_rgb, 42)            # unsupported type

    def test_get_label_colormap_covers_every_label(self):
        colormap = get_label_colormap(None)
        for label in EventLabelEnum:
            self.assertIn(label, colormap)
            self.assertEqual(3, len(colormap[label]))

    def test_get_label_colormap_override(self):
        colormap = get_label_colormap({EventLabelEnum.FIXATION: "#010203"})
        self.assertEqual((1, 2, 3), colormap[EventLabelEnum.FIXATION])
        # untouched labels keep their default
        self.assertEqual(get_label_colormap(None)[EventLabelEnum.SACCADE],
                         colormap[EventLabelEnum.SACCADE])
