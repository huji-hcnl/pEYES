from typing import Sequence, Dict, Union, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.metrics as met


from src.pEYES._utils.event_utils import parse_label
from src.pEYES._utils.metric_utils import transition_matrix as _transition_matrix
from src.pEYES._utils.metric_utils import complement_normalized_levenshtein_distance as _comp_nld
from src.pEYES._utils.metric_utils import dprime as _dprime
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

_parse_vectorized = np.vectorize(parse_label)


def confusion_matrix(
        ground_truth: Sequence[EventLabelEnum],
        prediction: Sequence[EventLabelEnum],
        labels: Optional[Sequence[EventLabelEnum]] = None,
) -> np.ndarray:
    labels = set(EventLabelEnum) if labels is None else set(labels)
    return met.confusion_matrix(ground_truth, prediction, labels=labels)


def transition_matrix(
        seq: Sequence[EventLabelEnum],
        normalize_rows: bool = False
) -> pd.DataFrame:
    return _transition_matrix(seq, normalize_rows)


def calculate(
        ground_truth: Sequence[EventLabelEnum],
        prediction: Sequence[EventLabelEnum],
        metrics: Union[str, Sequence[str]],
        pos_labels: Optional[Union[EventLabelEnum, Sequence[EventLabelEnum]]] = None,
        average: str = "weighted",
        dprime_correction: Optional[str] = "loglinear"
) -> Union[float, Dict[str, float]]:
    if isinstance(metrics, str):
        return _calculate_impl(
            ground_truth, prediction, metrics, pos_labels, average, dprime_correction
        )
    results = {}
    for metric in tqdm(metrics, desc="Calculating Metrics"):
        results[metric] = _calculate_impl(
            ground_truth, prediction, metric, pos_labels, average, dprime_correction
        )
    return results


def _calculate_impl(
        ground_truth: Sequence[EventLabelEnum],
        prediction: Sequence[EventLabelEnum],
        metric: str,
        pos_labels: Optional[Union[EventLabelEnum, Sequence[EventLabelEnum]]] = None,
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
    if metric_lower == "1_nld" or metric_lower == "complement_nld":
        return _comp_nld(ground_truth, prediction)
    if metric_lower == "cohen_kappa" or metric_lower == "cohen's_kappa":
        return met.cohen_kappa_score(ground_truth, prediction, labels=pos_labels)
    if metric_lower == "recall":
        return met.recall_score(ground_truth, prediction, labels=pos_labels, average=average)
    if metric_lower == "precision":
        return met.precision_score(ground_truth, prediction, labels=pos_labels, average=average)
    if metric_lower == "f1":
        return met.f1_score(ground_truth, prediction, labels=pos_labels, average=average)
    if metric_lower == "d'" or metric_lower == "dprime" or metric_lower == "d_prime":
        p = len([l for l in ground_truth if l in pos_labels])
        n = len(ground_truth) - p
        pp = len([l for l in prediction if l in pos_labels])
        tp = len([1 for gt, pred in zip(ground_truth, prediction) if
                  gt == pred and gt in pos_labels and pred in pos_labels])
        return _dprime(p, n, pp, tp, dprime_correction)
    raise NotImplementedError(f"Unknown metric:\t{metric}")
