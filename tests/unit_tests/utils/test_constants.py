import unittest

import pandas as pd

import peyes._utils.constants as cnst


class TestStimulusTypeEnum(unittest.TestCase):
    """
    P-3: IMAGE_STR/VIDEO_STR/MOVING_DOT_STR became enum.StrEnum members instead of bare strings. Since a
    StrEnum member IS a real str instance equal to its value, every existing analysis/ comparison against these
    constants (==, .isin(), .str.lower(), dict/set membership) must keep working unchanged.
    """

    def test_members_are_real_strings(self):
        for member in (cnst.IMAGE_STR, cnst.VIDEO_STR, cnst.MOVING_DOT_STR):
            self.assertIsInstance(member, str)

    def test_equality_and_str_conversion_match_bare_strings(self):
        self.assertEqual("image", cnst.IMAGE_STR)
        self.assertEqual("image", str(cnst.IMAGE_STR))
        self.assertEqual("video", cnst.VIDEO_STR)
        self.assertEqual("moving_dot", cnst.MOVING_DOT_STR)

    def test_pandas_series_operations_treat_it_as_a_string(self):
        col = pd.Series([cnst.IMAGE_STR, cnst.VIDEO_STR, "IMAGE", "other"])
        pd.testing.assert_series_equal(
            col.str.lower().isin([cnst.IMAGE_STR]),
            pd.Series([True, False, True, False]),
        )

    def test_dict_and_set_membership_use_string_identity(self):
        self.assertEqual(1, {cnst.IMAGE_STR: 1}["image"])
        self.assertIn("video", {cnst.IMAGE_STR, cnst.VIDEO_STR})
