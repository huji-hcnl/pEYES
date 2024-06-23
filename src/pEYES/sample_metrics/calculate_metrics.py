from typing import Dict, Union, Optional

import numpy as np
from tqdm import tqdm
import sklearn.metrics as met

from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum, EventLabelSequenceType
from src.pEYES._utils.event_utils import parse_label
from src.pEYES._utils.metric_utils import complement_normalized_levenshtein_distance as _comp_nld

_parse_vectorized = np.vectorize(parse_label)


def calculate(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        *metrics: str,
        pos_labels: Optional[Union[EventLabelEnum, EventLabelSequenceType]] = None,
        average: str = "weighted",
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
        - "mcc" or "mathew's_correlation"
        - "1_nld" or "complement_nld" - computed the complement to normalized Levenshtein distance
    :param pos_labels: the positive labels to consider for recall, precision, and f1-score
    :param average: the averaging strategy for recall, precision, and f1-score
    :return: the calculated metric(s) as a single float (if only one metric is specified) or a dictionary of metric
        names to values
    """
    results: Dict[str, float] = {}
    for metric in tqdm(metrics, desc="Calculating Metrics"):
        results[metric] = _calculate_impl(
            ground_truth, prediction, metric, pos_labels, average
        )
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def _calculate_impl(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        metric: str,
        pos_labels: Optional[Union[EventLabelEnum, EventLabelSequenceType]] = None,
        average: str = "weighted",
        dprime_correction: Optional[str] = "loglinear"
) -> float:
    assert len(ground_truth) == len(prediction), "Ground truth and prediction must have the same length."
    if pos_labels is None:
        pos_labels = set(EventLabelEnum)
    elif isinstance(pos_labels, EventLabelEnum):
        pos_labels = {pos_labels}
    else:
        pos_labels = set(pos_labels)
    metric_lower = metric.lower().strip().replace(" ", "_").replace("-", "_").removesuffix("_score")
    average = average.lower().strip()
    if metric_lower == "accuracy":
        return met.accuracy_score(ground_truth, prediction)
    if metric_lower == "balanced_accuracy":
        return met.balanced_accuracy_score(ground_truth, prediction)
    if metric_lower == "mcc" or metric_lower == "mathew's_correlation":
        return met.matthews_corrcoef(ground_truth, prediction)
    if metric_lower == "cohen_kappa" or metric_lower == "cohen's_kappa":
        return met.cohen_kappa_score(ground_truth, prediction)
    if metric_lower == "1_nld" or metric_lower == "complement_nld":
        return _comp_nld(ground_truth, prediction)
    if metric_lower == "recall":
        return met.recall_score(ground_truth, prediction, labels=pos_labels, average=average)
    if metric_lower == "precision":
        return met.precision_score(ground_truth, prediction, labels=pos_labels, average=average)
    if metric_lower == "f1":
        return met.f1_score(ground_truth, prediction, labels=pos_labels, average=average)
    raise NotImplementedError(f"Unknown metric:\t{metric}")
