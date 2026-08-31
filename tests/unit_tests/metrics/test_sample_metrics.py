import unittest

import peyes


class TestCalculate(unittest.TestCase):
    """ M-10: calculate() used to return a bare float for one metric and a dict for several. """

    GT = ["fixation", "fixation", "saccade", "saccade"]
    PRED = ["fixation", "saccade", "saccade", "saccade"]

    def test_single_metric_still_returns_a_dict(self):
        result = peyes.sample_metrics.calculate(self.GT, self.PRED, "accuracy")
        self.assertIsInstance(result, dict)
        self.assertEqual({"accuracy"}, set(result.keys()))

    def test_multiple_metrics_return_a_dict(self):
        result = peyes.sample_metrics.calculate(self.GT, self.PRED, "accuracy", "balanced_accuracy")
        self.assertIsInstance(result, dict)
        self.assertEqual({"accuracy", "balanced_accuracy"}, set(result.keys()))

    def test_named_wrapper_still_returns_a_bare_float(self):
        """ Per-metric convenience functions (e.g. accuracy) must keep returning a plain float. """
        result = peyes.sample_metrics.accuracy(self.GT, self.PRED)
        self.assertIsInstance(result, float)
        self.assertEqual(0.75, result)


if __name__ == "__main__":
    unittest.main()
