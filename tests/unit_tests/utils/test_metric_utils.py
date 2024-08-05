import unittest

import numpy as np
import Levenshtein
from scipy.stats import norm

from peyes._utils.metric_utils import transition_matrix, dprime_and_criterion, _dprime_rates
from peyes._utils.metric_utils import complement_normalized_levenshtein_distance as comp_nld


class TestMetricUtils(unittest.TestCase):

    def test_transition_matrix(self):
        seq = [0, 1, 2, 3, 2, 3, 2, 2]
        self.assertTrue(
            np.array_equal(transition_matrix(seq).values,
                           np.array([[1, 0, 0], [0, 1, 0], [0, 1, 2], [0, 2, 0]]))
        )
        self.assertTrue(
            np.array_equal(transition_matrix(seq, True).values,
                           np.array([[1, 0, 0], [0, 1, 0], [0, 1/3, 2/3], [0, 1, 0]]))
        )

    def test_complement_nld(self):
        gt = "kitten"
        pred = "sitting"
        self.assertEqual(1 - Levenshtein.distance(gt, pred) / max(len(gt), len(pred)), comp_nld(gt, pred))

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

    def test_dprime(self):
        p = n = pp = 10
        tp = 5
        exp = norm.ppf(tp / p) - norm.ppf((pp - tp) / n)
        self.assertEqual(dprime_and_criterion(p, n, pp, tp, None)[0], exp)
        self.assertEqual(dprime_and_criterion(p, n, pp, tp, "macmillan")[0], exp)
        self.assertEqual(dprime_and_criterion(p, n, pp, tp, "loglinear")[0], exp)
        self.assertEqual(dprime_and_criterion(p, n, pp, tp, "foo")[0], exp)
        tp = 10
        self.assertTrue(np.isinf(dprime_and_criterion(p, n, pp, tp, None)[0]))
        self.assertEqual(
            dprime_and_criterion(p, n, pp, tp, "macmillan")[0], norm.ppf(1 - 0.5 / p) - norm.ppf(0.5 / n)
        )
        hr_ll, far_ll = _dprime_rates(p, n, pp, tp, "loglinear")
        self.assertEqual(
            dprime_and_criterion(p, n, pp, tp, "loglinear")[0], norm.ppf(hr_ll) - norm.ppf(far_ll)
        )
        self.assertRaises(ValueError, dprime_and_criterion, p, n, pp, tp, "foo")

    def test_criterion(self):
        p = n = pp = 10
        tp = 5
        exp = -0.5 * (norm.ppf(tp / p) + norm.ppf((pp - tp) / n))
        self.assertEqual(dprime_and_criterion(p, n, pp, tp, None)[1], exp)
        self.assertEqual(dprime_and_criterion(p, n, pp, tp, "macmillan")[1], exp)
        self.assertEqual(dprime_and_criterion(p, n, pp, tp, "loglinear")[1], exp)
        self.assertEqual(dprime_and_criterion(p, n, pp, tp, "foo")[1], exp)
        tp = 10
        self.assertTrue(np.isnan(dprime_and_criterion(p, n, pp, tp, None)[1]))
        self.assertEqual(
            dprime_and_criterion(p, n, pp, tp, "macmillan")[1], -0.5 * (norm.ppf(1 - 0.5 / p) + norm.ppf(0.5 / n))
        )
        hr_ll, far_ll = _dprime_rates(p, n, pp, tp, "loglinear")
        self.assertEqual(
            dprime_and_criterion(p, n, pp, tp, "loglinear")[1], -0.5 * (norm.ppf(hr_ll) + norm.ppf(far_ll))
        )
        self.assertRaises(ValueError, dprime_and_criterion, p, n, pp, tp, "foo")

