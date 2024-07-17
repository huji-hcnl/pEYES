import warnings
from itertools import islice
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import Levenshtein
from scipy.stats import norm


def transition_matrix(seq: Sequence, normalize_rows: bool = False) -> pd.DataFrame:
    """
    Calculates the transition matrix of a sequence of events. The matrix is a DataFrame where the rows represent the
    "from" event and the columns represent the "to" event. The values in the matrix are the counts of transitions
    between the events. If `normalize_rows` is True, the values in each row are normalized to sum to 1.
    See implementation details at https://stackoverflow.com/a/47298184/8543025.
    """
    def window(s, n=2):
        """ Sliding window width n from sequence s """
        it = iter(s)
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result
    pairs = pd.DataFrame(window(seq), columns=['From', 'To'])
    counts = pairs.groupby('From')['To'].value_counts().unstack(fill_value=0)
    if normalize_rows:
        counts = counts.div(counts.sum(axis=1), axis=0)
    return counts


def complement_normalized_levenshtein_distance(
        gt: Sequence, pred: Sequence,
) -> float:
    """ Calculates the complement of the normalized Levenshtein distance between two sequences. """
    d = Levenshtein.distance(gt, pred)
    normalized_d = d / max(len(gt), len(pred))
    return 1 - normalized_d


def dprime_and_criterion(p: int, n: float, pp: int, tp: int, correction: Optional[str]) -> (float, float):
    """
    Calculates d-prime and criterion while optionally applying a correction for floor/ceiling effects on the hit-rate
    and/or false-alarm rate. See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
    See implementation details at https://lindeloev.net/calculating-d-in-python-and-php/.

    :param p: int; number of positive GT events
    :param n: int; number of negative GT events
    :param pp: int; number of positive predicted events
    :param tp: int; number of true positive predictions
    :param correction: str; optional correction method for floor/ceiling effects
    :return:
        - d_prime: float; the d-prime value
        - criterion: float; the criterion value
    """
    hr, far = _dprime_rates(p, n, pp, tp, correction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_prime = norm.ppf(hr) - norm.ppf(far)
        criterion = -0.5 * (norm.ppf(hr) + norm.ppf(far))
        return d_prime, criterion


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
    if 0 < hit_rate < 1 and 0 < false_alarm_rate < 1:
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

