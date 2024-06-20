from typing import Optional

import numpy as np
from scipy.stats import norm


def dprime(p: int, n: float, pp: int, tp: int, correction: Optional[str]) -> float:
    """
    Calculates d-prime while optionally applying a correction for floor/ceiling effects on the hit-rate and/or
    false-alarm rate. See information on correction methods at https://stats.stackexchange.com/a/134802/288290.

    :param p: int; number of positive GT events
    :param n: int; number of negative GT events
    :param pp: int; number of positive predicted events
    :param tp: int; number of true positive predictions
    :param correction: str; optional correction method for floor/ceiling effects
    :return: float; the d-prime value
    """
    hr, far = _dprime_rates(p, n, pp, tp, correction)
    return norm.ppf(hr) - norm.ppf(far)


def _dprime_rates(p: int, n: float, pp: int, tp: int, correction: Optional[str]) -> (float, float):
    """
    Calculates hit-rate and false-alarm rate for computing d-prime. Optionally applies a correction for floor/ceiling
    effects on the rates. See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
    Returns a tuple of (hit-rate, false-alarm rate).
    """
    fp = pp - tp
    assert 0 <= tp <= min(p, pp), f"True Positive count must be between 0 and min(p, pp) = {min(p, pp)}"
    assert 0 <= fp <= n, f"False Positive count must be between 0 and n = {n}"
    hit_rate = tp / p if p > 0 else np.nan
    false_alarm_rate = fp / n if n > 0 else np.nan
    if hit_rate != 0 and hit_rate != 1 and false_alarm_rate != 0 and false_alarm_rate != 1:
        # no correction needed
        return hit_rate, false_alarm_rate
    corr = (correction or "").lower().strip().replace(" ", "_").replace("-", "_")
    if corr is None or not corr:
        return hit_rate, false_alarm_rate
    if corr in {"mk", "m&k", "macmillan_kaplan", "macmillan"}:
        # apply Macmillan & Kaplan (1985) correction
        if hit_rate == 0:
            hit_rate = 0.5 / p
        if hit_rate == 1:
            hit_rate = 1 - 0.5 / p
        if false_alarm_rate == 0:
            false_alarm_rate = 0.5 / n
        if false_alarm_rate == 1:
            false_alarm_rate = 1 - 0.5 / n
        return hit_rate, false_alarm_rate
    if correction in {"ll", "loglinear", "log_linear", "hautus"}:
        # apply Hautus (1995) correction
        prevalence = p / (p + n)
        new_tp, new_fp = tp + prevalence, fp + 1 - prevalence
        new_p, new_n = p + 2 * prevalence, n + 2 * (1 - prevalence)
        hit_rate = new_tp / new_p if new_p > 0 else np.nan
        false_alarm_rate = new_fp / new_n if new_n > 0 else np.nan
        return hit_rate, false_alarm_rate
    raise ValueError(f"Invalid correction: {correction}")

