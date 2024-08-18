import warnings
from typing import Dict, Union, Optional

import numpy as np
from tqdm import tqdm
import sklearn.metrics as met

from peyes._DataModels.EventLabelEnum import EventLabelEnum, EventLabelSequenceType
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import peyes._utils.constants as cnst
from peyes._utils.event_utils import parse_label
from peyes._utils.metric_utils import complement_normalized_levenshtein_distance as _comp_nld
from peyes._utils.metric_utils import dprime_and_criterion as _dprime_and_criterion

_GLOBAL_METRICS = {
    cnst.ACCURACY_STR, cnst.BALANCED_ACCURACY_STR, cnst.COHENS_KAPPA_STR, cnst.MCC_STR, cnst.COMPLEMENT_NLD_STR,
    "mathew's_correlation", "cohen_kappa", "1_nld",
}
_SDT_METRICS = {
    cnst.RECALL_STR, cnst.PRECISION_STR, cnst.F1_STR, cnst.D_PRIME_STR, cnst.CRITERION_STR
}


def calculate(
        ground_truth: UnparsedEventLabelSequenceType,
        prediction: UnparsedEventLabelSequenceType,
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
        - "cohen's_kappa"
        - "mcc" or "mathew's_correlation"
        - "1_nld" or "complement_nld" - computed the complement to normalized Levenshtein distance
        - "recall"
        - "precision"
        - "f1"
        - "d_prime" or "d'"
        - "criterion"
    :param pos_labels: the positive labels to consider for recall, precision, f1-score, d-prime, and criterion
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
        assert len(ground_truth) == len(prediction), "Ground Truth and Prediction must have the same length."
        ground_truth = [parse_label(label) for label in ground_truth]
        prediction = [parse_label(label) for label in prediction]
        results: Dict[str, float] = {}
        for metric in tqdm(metrics, desc="Calculating Metrics", disable=not verbose):
            metric_lower = metric.lower().strip().replace(" ", "_").replace("-", "_").removesuffix("_score")
            if metric_lower in _GLOBAL_METRICS:
                results[metric] = _calculate_global_metrics(ground_truth, prediction, metric_lower)
            elif metric_lower in _SDT_METRICS:
                results[metric] = _calculate_sdt_metrics(
                    ground_truth, prediction, metric_lower, pos_labels, average, correction
                )
            else:
                raise NotImplementedError(f"Unknown metric:\t{metric}")
        if len(results) == 1:
            return next(iter(results.values()))
        return results


def _calculate_global_metrics(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        metric: str,
) -> float:
    if metric == cnst.ACCURACY_STR:
        return met.accuracy_score(ground_truth, prediction)
    if metric == cnst.BALANCED_ACCURACY_STR:
        return met.balanced_accuracy_score(ground_truth, prediction)
    if metric == cnst.MCC_STR or metric == "mathew's_correlation":
        return met.matthews_corrcoef(ground_truth, prediction)
    if metric == cnst.COHENS_KAPPA_STR or metric == "cohen_kappa":
        return met.cohen_kappa_score(ground_truth, prediction)
    if metric == cnst.COMPLEMENT_NLD_STR or metric == "1_nld":
        return _comp_nld(ground_truth, prediction)
    raise NotImplementedError(f"Unknown metric:\t{metric}")


def _calculate_sdt_metrics(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        metric: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        average: str = "weighted",
        correction: str = "loglinear",
) -> float:
    average = average.lower().strip()
    if pos_labels is None:
        pos_labels = [l for l in EventLabelEnum]
    elif isinstance(pos_labels, UnparsedEventLabelType):
        pos_labels = [parse_label(pos_labels)]
    else:
        pos_labels = [parse_label(l) for l in pos_labels]
    if metric == cnst.RECALL_STR:
        return met.recall_score(ground_truth, prediction, labels=pos_labels, average=average, zero_division=np.nan)
    if metric == cnst.PRECISION_STR:
        return met.precision_score(ground_truth, prediction, labels=pos_labels, average=average, zero_division=np.nan)
    if metric == cnst.F1_STR:
        return met.f1_score(ground_truth, prediction, labels=pos_labels, average=average, zero_division=np.nan)
    if metric in {"dprime", "d'", cnst.D_PRIME_STR, cnst.CRITERION_STR}:
        p = np.sum([1 for label in ground_truth if label in pos_labels])
        n = len(ground_truth) - p
        pp = np.sum([1 for label in prediction if label in pos_labels])
        tp = np.sum([1 for gt, pred in zip(ground_truth, prediction) if pred == gt and gt in pos_labels])
        dprime, crit = _dprime_and_criterion(p, n, pp, tp, correction)
        if metric == cnst.CRITERION_STR:
            return crit
        return dprime
    raise NotImplementedError(f"Unknown metric:\t{metric}")
