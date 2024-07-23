import os
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u

_SAMPLE_METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "recall", "precision", "f1", "cohen's_kappa", "mcc", "1_nld", "d_prime", "criterion",
]


def run_default(
        dataset_name: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
) -> pd.DataFrame:
    default_output_dir = u.get_default_output_dir(dataset_name)
    try:
        labels = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.LABELS_STR}.pkl"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find `{peyes.LABELS_STR}.pkl` in {default_output_dir}. Please preprocess the dataset first."
        )
    sample_metrics_dir = os.path.join(default_output_dir, f"{peyes.SAMPLE_STR}_{peyes.METRICS_STR}")
    os.makedirs(sample_metrics_dir, exist_ok=True)
    if isinstance(pos_labels, UnparsedEventLabelType):
        pos_labels = [pos_labels]
    fullpath = os.path.join(sample_metrics_dir, u.get_filename_for_labels(pos_labels, extension="pkl"))
    try:
        sample_metrics = pd.read_pickle(fullpath)
    except FileNotFoundError:
        metrics = _SAMPLE_METRIC_NAMES
        if not pos_labels:
            metrics = list(set(_SAMPLE_METRIC_NAMES) - {"d_prime", "criterion"})
        sample_metrics = calculate_sample_metrics(
            labels, u.DATASET_ANNOTATORS[dataset_name], pos_labels=pos_labels, metrics=metrics
        )
        sample_metrics.to_pickle(fullpath)
    return sample_metrics


def calculate_sample_metrics(
        labels: pd.DataFrame,
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        metrics: List[str] = None,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        average: str = "weighted",
        correction: str = "loglinear",
) -> pd.DataFrame:
    all_labelers = labels.columns.get_level_values(u.LABELER_STR).unique()
    pred_labelers = pred_labelers or all_labelers
    unknown_labelers = (set(pred_labelers) - set(all_labelers)) | (set(gt_labelers) - set(all_labelers))
    if unknown_labelers:
        raise ValueError(f"Unknown labelers: {unknown_labelers}")
    metrics = set(metrics or _SAMPLE_METRIC_NAMES)
    if not metrics <= set(_SAMPLE_METRIC_NAMES):
        raise ValueError(f"Unknown metrics: {metrics - set(_SAMPLE_METRIC_NAMES)}")
    if pos_labels is None:
        # cannot calculate d-prime and criterion without specifying a subset of positive labels
        metrics = metrics - {"d_prime", "criterion"}
    results = dict()
    for tr in tqdm(labels.columns.get_level_values(peyes.TRIAL_ID_STR).unique(), desc="Sample Metrics"):
        for gt_labeler in gt_labelers:
            gt_min_iteration = labels.xs((tr, gt_labeler), axis=1, level=[peyes.TRIAL_ID_STR, u.LABELER_STR]).columns.min()
            gt_labels = labels[tr, gt_labeler, gt_min_iteration].dropna().values.flatten()
            if gt_labels.size == 0:
                continue
            for pred_labeler in pred_labelers:
                pred_labels_all_iters = labels.xs((tr, gt_labeler), axis=1, level=[peyes.TRIAL_ID_STR, u.LABELER_STR])
                for pred_it in pred_labels_all_iters.columns.get_level_values(peyes.ITERATION_STR).unique():
                    if (pred_labeler == gt_labeler) and (pred_it == gt_min_iteration):
                        continue
                    pred_labels = pred_labels_all_iters[pred_it].dropna().values.flatten()
                    if pred_labels.size == 0:
                        continue
                    res = peyes.sample_metrics.calculate(
                        gt_labels, pred_labels, *metrics, pos_labels=pos_labels, average=average, correction=correction
                    )
                    results[(tr, gt_labeler, pred_labeler, pred_it)] = res
    results_df = pd.DataFrame(results).T
    results_df.index.names = [peyes.TRIAL_ID_STR, u.GT_STR, u.PRED_STR, peyes.ITERATION_STR]
    results_df = results_df[sorted(results_df.columns)]
    return results_df.T

