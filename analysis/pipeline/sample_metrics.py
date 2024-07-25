import os
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u


def run_default(
        dataset_name: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
) -> pd.DataFrame:
    default_output_dir = u.get_default_output_dir(dataset_name)
    try:
        labels = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find `{peyes.constants.LABELS_STR}.pkl` in {default_output_dir}. Please preprocess the dataset first."
        )
    sample_metrics_dir = os.path.join(default_output_dir, f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}")
    os.makedirs(sample_metrics_dir, exist_ok=True)
    if isinstance(pos_labels, UnparsedEventLabelType):
        pos_labels = [pos_labels]
    fullpath = os.path.join(sample_metrics_dir, u.get_filename_for_labels(pos_labels, extension="pkl"))
    try:
        sample_metrics = pd.read_pickle(fullpath)
    except FileNotFoundError:
        metrics = set(u.SAMPLE_METRICS.keys())
        if not pos_labels:
            metrics -= {"d_prime", "criterion"}
        sample_metrics = calculate_sample_metrics(
            labels, u.DATASET_ANNOTATORS[dataset_name], pos_labels=pos_labels, metrics=list(metrics)
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
    pred_labelers = u.check_labelers(labels, pred_labelers)
    metrics = set(metrics or u.SAMPLE_METRICS.keys())
    if not metrics <= set(u.SAMPLE_METRICS.keys()):
        raise ValueError(f"Unknown metrics: {metrics - set(u.SAMPLE_METRICS.keys())}")
    if pos_labels is None:
        # cannot calculate d-prime and criterion without specifying a subset of positive labels
        metrics = metrics - {"d_prime", "criterion"}
    results = dict()
    for tr in tqdm(labels.columns.get_level_values(peyes.constants.TRIAL_ID_STR).unique(), desc="Sample Metrics"):
        for gt_labeler in gt_labelers:
            gt_min_iteration = labels.xs(
                (tr, gt_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR]
            ).columns.min()
            gt_labels = labels[tr, gt_labeler, gt_min_iteration].dropna().values.flatten()
            if gt_labels.size == 0:
                continue
            for pred_labeler in pred_labelers:
                pred_labels_all_iters = labels.xs((tr, pred_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR])
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

