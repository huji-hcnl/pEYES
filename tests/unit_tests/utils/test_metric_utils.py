import unittest

import numpy as np
from scipy.stats import norm

from src.pEYES._utils.metric_utils import dprime, _dprime_rates


class TestMetricUtils(unittest.TestCase):

    def test_dprime(self):
        p = n = pp = 10
        tp = 5
        self.assertEqual(dprime(p, n, pp, tp, None), norm.ppf(tp/p) - norm.ppf((pp-tp)/n))
        self.assertEqual(dprime(p, n, pp, tp, "macmillan"), norm.ppf(tp/p) - norm.ppf((pp-tp)/n))
        self.assertEqual(dprime(p, n, pp, tp, "loglinear"), norm.ppf(tp/p) - norm.ppf((pp-tp)/n))
        self.assertEqual(dprime(p, n, pp, tp, "foo"), norm.ppf(tp/p) - norm.ppf((pp-tp)/n))
        tp = 10
        self.assertTrue(np.isinf(dprime(p, n, pp, tp, None)))
        self.assertEqual(dprime(p, n, pp, tp, "macmillan"), norm.ppf(1-0.5/p) - norm.ppf(0.5/n))
        hr_ll, far_ll = _dprime_rates(p, n, pp, tp, "loglinear")
        self.assertEqual(dprime(p, n, pp, tp, "loglinear"), norm.ppf(hr_ll) - norm.ppf(far_ll))
        self.assertRaises(ValueError, dprime, p, n, pp, tp, "foo")

    def test_dprime_rates(self):
        p, n, pp, tp = 10, 20, 15, 5
        # hr, far = dprime_rates(p, n, pp, tp, None)
        self.assertEqual((tp/p, (pp-tp)/n), _dprime_rates(p, n, pp, tp, None))
        self.assertEqual((tp/p, (pp-tp)/n), _dprime_rates(p, n, pp, tp, "macmillan"))
        self.assertEqual((tp/p, (pp-tp)/n), _dprime_rates(p, n, pp, tp, "loglinear"))
        self.assertEqual((tp/p, (pp-tp)/n), _dprime_rates(p, n, pp, tp, "foo"))
        tp = 10
        self.assertEqual((tp/p, (pp-tp)/n), _dprime_rates(p, n, pp, tp, None))
        self.assertEqual((1-0.5/p, (pp-tp)/n), _dprime_rates(p, n, pp, tp, "macmillan"))
        self.assertTrue(np.isclose(
            (tp + p / (p+n)) / (p + 2 * p / (p+n)), _dprime_rates(p, n, pp, tp, "loglinear")[0]
        ))
        self.assertTrue(np.isclose(
            (pp-tp + 1 - p / (p+n)) / (n + 2 * (1 - p / (p+n))), _dprime_rates(p, n, pp, tp, "loglinear")[1]
        ))
        self.assertRaises(ValueError, _dprime_rates, p, n, pp, tp, "foo")
        tp = 15
        self.assertRaises(AssertionError, _dprime_rates, p, n, pp, tp, None)
        p = tp = 0
        self.assertTrue(np.isnan(_dprime_rates(p, n, pp, tp, None)[0]))
