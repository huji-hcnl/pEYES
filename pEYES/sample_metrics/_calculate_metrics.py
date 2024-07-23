import warnings
from typing import Dict, Union, Optional

import numpy as np
from tqdm import tqdm
import sklearn.metrics as met

from pEYES._DataModels.EventLabelEnum import EventLabelEnum, EventLabelSequenceType
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import pEYES._utils.constants as cnst
from pEYES._utils.event_utils import parse_label
from pEYES._utils.metric_utils import complement_normalized_levenshtein_distance as _comp_nld
from pEYES._utils.metric_utils import dprime_and_criterion as _dprime_and_criterion


def calculate(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        *metrics: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        average: str = "weighted",
        correction: str = "loglinear",
        verbose: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Calculate the specified metrics between the ground truth and prediction sequences.
    :param ground_truth: sequence of ground truth labels
    :param prediction: sequence of predicted labels
    :param metrics: the metrics to calculate. Supported metrics are:
        - "accuracy"
        - "balanced_accuracy"
        - "recall"
        - "precision"
        - "f1"
        - "cohen's_kappa"
        - "d_prime" or "d'"
        - "criterion"
        - "mcc" or "mathew's_correlation"
        - "1_nld" or "complement_nld" - computed the complement to normalized Levenshtein distance
    :param pos_labels: the positive labels to consider for recall, precision, and f1-score
    :param average: the averaging strategy for recall, precision, and f1-score
    :param correction: the correction strategy for d-prime and criterion.
        See information on correction methods at https://stats.stackexchange.com/a/134802/288290.
        See implementation details at https://lindeloev.net/calculating-d-in-python-and-php/.
    :param verbose: if True, display a progress bar
    :return: the calculated metric(s) as a single float (if only one metric is specified) or a dictionary of metric
        names to values
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        results: Dict[str, float] = {}
        for metric in tqdm(metrics, desc="Calculating Metrics", disable=not verbose):
            results[metric] = _calculate_impl(ground_truth, prediction, metric, pos_labels, average, correction)
        if len(results) == 1:
            return next(iter(results.values()))
        return results


def _calculate_impl(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        metric: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        average: str = "weighted",
        correction: str = "loglinear",
) -> float:
    assert len(ground_truth) == len(prediction), "Ground truth and prediction must have the same length."
    pos_labels = pos_labels or set(EventLabelEnum)
    if isinstance(pos_labels, UnparsedEventLabelType):
        pos_labels = [parse_label(pos_labels)]
    else:
        pos_labels = list(set([parse_label(label) for label in pos_labels]))
    metric_lower = metric.lower().strip().replace(" ", "_").replace("-", "_").removesuffix("_score")
    average = average.lower().strip()
    if metric_lower == cnst.ACCURACY_STR:
        return met.accuracy_score(ground_truth, prediction)
    if metric_lower == cnst.BALANCED_ACCURACY_STR:
        return met.balanced_accuracy_score(ground_truth, prediction)
    if metric_lower == cnst.MCC_STR or metric_lower == "mathew's_correlation":
        return met.matthews_corrcoef(ground_truth, prediction)
    if metric_lower == cnst.COHENS_KAPPA_STR or metric_lower == "cohen_kappa":
        return met.cohen_kappa_score(ground_truth, prediction)
    if metric_lower == cnst.COMPLEMENT_NLD_STR or metric_lower == "1_nld":
        return _comp_nld(ground_truth, prediction)
    if metric_lower == cnst.RECALL_STR:
        return met.recall_score(ground_truth, prediction, labels=pos_labels, average=average, zero_division=np.nan)
    if metric_lower == cnst.PRECISION_STR:
        return met.precision_score(ground_truth, prediction, labels=pos_labels, average=average, zero_division=np.nan)
    if metric_lower == cnst.F1_STR:
        return met.f1_score(ground_truth, prediction, labels=pos_labels, average=average, zero_division=np.nan)
    if metric_lower.replace('_', '') in {"dprime", "d'", cnst.CRITERION_STR}:
        if pos_labels is None or pos_labels == 0 or set(pos_labels) == set([parse_label(l) for l in EventLabelEnum]):
            raise ValueError("Positive labels must be specified for d-prime and criterion calculations.")
        p = np.sum([1 for label in ground_truth if label in pos_labels])
        n = len(ground_truth) - p
        pp = np.sum([1 for label in prediction if label in pos_labels])
        tp = np.sum([1 for gt, pred in zip(ground_truth, prediction) if pred == gt and gt in pos_labels])
        dprime, crit = _dprime_and_criterion(p, n, pp, tp, correction)
        if metric == cnst.CRITERION_STR:
            return crit
        return dprime
    raise NotImplementedError(f"Unknown metric:\t{metric}")
