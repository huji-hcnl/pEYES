import os
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm

import src.pEYES as peyes
from src.pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u

SAMPLE_METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "recall", "precision", "f1", "cohen's_kappa", "mcc", "1_nld", "d_prime", "criterion",
]


def run_default(
        dataset_name: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
):
    default_output_dir = u.get_default_output_dir(dataset_name)
    try:
        labels = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.LABELS_STR}.pkl"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find `labels.pkl` in {default_output_dir}. Please preprocess the dataset first."
        )
    sample_metrics_dir = os.path.join(default_output_dir, peyes.SAMPLES_STR)
    os.makedirs(sample_metrics_dir, exist_ok=True)
    filename = "_".join([l.name.lower() for l in pos_labels]) + ".pkl" if pos_labels else "all_labels.pkl"
    fullpath = os.path.join(sample_metrics_dir, filename)
    try:
        sample_metrics = pd.read_pickle(fullpath)
    except FileNotFoundError:
        metrics = SAMPLE_METRIC_NAMES
        if not pos_labels:
            metrics = list(set(SAMPLE_METRIC_NAMES) - {"d_prime", "criterion"})
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
    metrics = metrics or SAMPLE_METRIC_NAMES
    if not set(metrics) <= set(SAMPLE_METRIC_NAMES):
        raise ValueError(f"Unknown metrics: {set(metrics) - set(SAMPLE_METRIC_NAMES)}")
    if (pos_labels is None) and (("d_prime" in metrics) or ("criterion" in metrics)):
        raise ValueError("Positive labels must be specified for d-prime and criterion metrics")
    results = dict()
    for tr in tqdm(labels.index.get_level_values(peyes.TRIAL_ID_STR).unique(), desc="Trials"):
        for gt_labeler in gt_labelers:
            gt_labels = labels.loc[tr, gt_labeler].dropna().values.flatten()
            if gt_labels.size == 0:
                continue
            for pred_labeler in pred_labelers:
                if pred_labeler == gt_labeler:
                    continue
                pred_labels_df = labels.loc[tr, pred_labeler]
                for it in pred_labels_df.columns.get_level_values(peyes.ITERATION_STR).unique():
                    pred_labels = pred_labels_df[it].dropna().values.flatten()
                    if pred_labels.size == 0:
                        continue
                    res = peyes.sample_metrics.calculate(
                        gt_labels, pred_labels, *metrics, pos_labels=pos_labels, average=average, correction=correction
                    )
                    results[(tr, gt_labeler, pred_labeler, it)] = res
    results_df = pd.DataFrame(results).T
    results_df.index.names = [peyes.TRIAL_ID_STR, "GT", "Pred", peyes.ITERATION_STR]
    results_df = results_df[sorted(results_df.columns)]
    return results_df

