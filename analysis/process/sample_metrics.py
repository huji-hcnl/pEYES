import os
from typing import List, Optional, Union, Tuple

import pandas as pd
from tqdm import tqdm

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType
from pEYES.sample_metrics._calculate_metrics import _GLOBAL_METRICS, _SDT_METRICS

import analysis.utils as u
import analysis.process._helpers as h


def run_default(
        dataset_name: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    default_output_dir = h.get_default_output_dir(dataset_name)
    try:
        labels = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find `{peyes.constants.LABELS_STR}.pkl` in {default_output_dir}. Please preprocess the dataset first."
        )
    sample_metrics_dir = os.path.join(default_output_dir, f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}")
    os.makedirs(sample_metrics_dir, exist_ok=True)
    if pos_labels is None or len(pos_labels) == 0:
        pos_labels = None
    elif isinstance(pos_labels, UnparsedEventLabelType):
        pos_labels = [pos_labels]

    # calculate SDT metrics, regardless of whether pos_labels is None or not
    sdt_metrics_path = os.path.join(
        sample_metrics_dir, u.get_filename_for_labels(
            pos_labels, suffix=f"{u.SDT_STR}_{peyes.constants.METRICS_STR}", extension="pkl"
        )
    )
    try:
        sdt_metrics = pd.read_pickle(sdt_metrics_path)
    except FileNotFoundError:
        sdt_metric_names = _SDT_METRICS
        if not pos_labels:
            sdt_metric_names -= {peyes.constants.D_PRIME_STR, peyes.constants.CRITERION_STR}
        sdt_metrics = calculate_sdt_sample_metrics(
            labels, u.DATASET_ANNOTATORS[dataset_name], metrics=sdt_metric_names, pos_labels=pos_labels
        )
        sdt_metrics.to_pickle(sdt_metrics_path)
    if pos_labels is not None:
        return sdt_metrics

    # calculate global metrics only if pos_labels is None
    global_metrics_path = os.path.join(
        sample_metrics_dir, u.get_filename_for_labels(
            labels=pos_labels, suffix=f"{u.GLOBAL_STR}_{peyes.constants.METRICS_STR}", extension="pkl"
        )
    )
    try:
        global_metrics = pd.read_pickle(global_metrics_path)
    except FileNotFoundError:
        global_metrics = calculate_global_sample_metrics(
            labels, u.DATASET_ANNOTATORS[dataset_name], metrics=_GLOBAL_METRICS
        )
        global_metrics.to_pickle(global_metrics_path)
    return sdt_metrics, global_metrics


def calculate_global_sample_metrics(
        labels: pd.DataFrame,
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        metrics: List[str] = _GLOBAL_METRICS,
) -> pd.DataFrame:
    pred_labelers = h.check_labelers(labels, pred_labelers)
    metrics = set(m.lower().strip() for m in metrics) or set(_GLOBAL_METRICS)
    unknown_metrics = metrics - set(u.METRICS_CONFIG.keys())
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {unknown_metrics}")
    non_global_metrics = set(metrics) - set(_GLOBAL_METRICS)
    if non_global_metrics:
        raise ValueError(f"Metrics {non_global_metrics} are not \"global\" metrics.")
    metrics = sorted(metrics, key=lambda m: u.METRICS_CONFIG[m][1])
    return _calculate_impl(
        labels,
        gt_labelers,
        pred_labelers,
        metrics,
        f"Sample Metrics :: Global",
    )


def calculate_sdt_sample_metrics(
        labels: pd.DataFrame,
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        metrics: List[str] = _SDT_METRICS,
        pos_labels: Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType] = None,
        average: str = "weighted",
        correction: str = "loglinear",
) -> pd.DataFrame:
    pred_labelers = h.check_labelers(labels, pred_labelers)
    metrics = set(m.lower().strip() for m in metrics) or set(_SDT_METRICS)
    unknown_metrics = metrics - set(u.METRICS_CONFIG.keys())
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {unknown_metrics}")
    non_sdt_metrics = set(metrics) - set(_SDT_METRICS)
    if non_sdt_metrics:
        raise ValueError(f"Metrics {non_sdt_metrics} are not \"SDT\" metrics.")
    if pos_labels is None:
        # cannot calculate d-prime and criterion without specifying a subset of positive labels
        metrics = metrics - {peyes.constants.D_PRIME_STR, peyes.constants.CRITERION_STR}
    metrics = sorted(metrics, key=lambda m: u.METRICS_CONFIG[m][1])
    return _calculate_impl(
        labels,
        gt_labelers,
        pred_labelers,
        metrics,
        f"Sample Metrics :: SDT ({pos_labels})",
        pos_labels,
        average,
        correction
    )


def _calculate_impl(
        labels: pd.DataFrame,
        gt_labelers: List[str],
        pred_labelers: List[str],
        metrics: List[str],
        description: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        average: str = "weighted",
        correction: str = "loglinear",
) -> pd.DataFrame:
    results = dict()
    for tr in tqdm(labels.columns.get_level_values(peyes.constants.TRIAL_ID_STR).unique(), desc="Sample Metrics"):
        for gt_labeler in gt_labelers:
            gt_min_iteration = labels.xs(
                (tr, gt_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, peyes.constants.LABELER_STR]
            ).columns.min()
            gt_labels = labels[tr, gt_labeler, gt_min_iteration].dropna().values.flatten()
            if gt_labels.size == 0:
                continue
            for pred_labeler in pred_labelers:
                pred_labels_all_iters = labels.xs((tr, pred_labeler), axis=1,
                                                  level=[peyes.constants.TRIAL_ID_STR, peyes.constants.LABELER_STR])
                for pred_it in pred_labels_all_iters.columns.get_level_values(peyes.constants.ITERATION_STR).unique():
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
    results_df.index.names = [peyes.constants.TRIAL_ID_STR, u.GT_STR, u.PRED_STR, peyes.constants.ITERATION_STR]
    results_df = results_df[sorted(results_df.columns)]
    return results_df.T
