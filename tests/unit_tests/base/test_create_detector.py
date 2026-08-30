import unittest

import numpy as np

import peyes
from peyes._DataModels.Detector import (
    IVTDetector, IVVTDetector, IDTDetector, IDVTDetector, EngbertDetector, NHDetector, REMoDNaVDetector,
)

_BASE = dict(missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)


class TestCreateDetector(unittest.TestCase):

    EXPECTED = {
        "ivt": IVTDetector, "ivvt": IVVTDetector, "idt": IDTDetector, "idvt": IDVTDetector,
        "engbert": EngbertDetector, "nh": NHDetector, "remodnav": REMoDNaVDetector,
    }

    def test_every_algorithm_resolves(self):
        for name, cls in self.EXPECTED.items():
            with self.subTest(algorithm=name):
                self.assertIsInstance(peyes.create_detector(name, **_BASE), cls)

    def test_name_normalisation(self):
        """ C-19: 'IVT Detector' normalised to 'ivt ' and failed to match. """
        for spelling in ("ivt", "IVT", "I-VT", "  ivt  ", "IVTDetector", "IVT Detector", "ivt_detector"):
            with self.subTest(spelling=spelling):
                self.assertIsInstance(peyes.create_detector(spelling, **_BASE), IVTDetector)

    def test_unknown_algorithm(self):
        self.assertRaises(NotImplementedError, peyes.create_detector, "not-an-algorithm", **_BASE)

    def test_misspelled_keyword_is_rejected(self):
        """
        C-17: kwargs were filtered against get_default_params(), so a typo was silently dropped and the
        default used -- an easy way to run an experiment with the wrong parameters.
        """
        with self.assertRaises(TypeError):
            peyes.create_detector("ivt", saccade_velocity_treshold=45, **_BASE)

    def test_correct_keyword_is_applied(self):
        detector = peyes.create_detector("ivt", saccade_velocity_threshold=77, **_BASE)
        self.assertEqual(77, detector.saccade_velocity_threshold_deg)

    def test_defaults_are_applied_when_absent(self):
        detector = peyes.create_detector("ivt", **_BASE)
        self.assertEqual(
            IVTDetector.get_default_params()["saccade_velocity_threshold"],
            detector.saccade_velocity_threshold_deg,
        )

    def test_idvt_validates_its_saccade_threshold(self):
        """ D-12: IDVT assigned the threshold directly, bypassing IVTDetector's validation. """
        self.assertRaises(ValueError, peyes.create_detector, "idvt", saccade_velocity_threshold=-1, **_BASE)
        self.assertRaises(ValueError, peyes.create_detector, "idvt", saccade_velocity_threshold=0, **_BASE)
