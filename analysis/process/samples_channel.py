import os
from typing import List, Optional, Union, Sequence, Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType
from pEYES._DataModels.EventLabelEnum import EventLabelEnum

import analysis.utils as u
import analysis.process._helpers as h

_CHANNEL_TYPE_STR = "channel_type"


def run_default(
        dataset_name: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
):
    default_output_dir = h.get_default_output_dir(dataset_name)
    try:
        labels = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find `{peyes.constants.EVENTS_STR}.pkl` in {default_output_dir}. Please preprocess the dataset first."
        )

    channel_metrics_dir = os.path.join(default_output_dir, f"{peyes.constants.SAMPLES_STR}_{u.CHANNEL_STR}")
    os.makedirs(channel_metrics_dir, exist_ok=True)
    time_diffs_fullpath = os.path.join(
        channel_metrics_dir, u.get_filename_for_labels(pos_labels, suffix="timing_differences", extension="pkl")
    )
    try:
        time_diffs = pd.read_pickle(time_diffs_fullpath)
    except FileNotFoundError:
        time_diffs = timing_differences(labels, u.DATASET_ANNOTATORS[dataset_name], pos_labels=pos_labels)
        time_diffs.to_pickle(time_diffs_fullpath)

    sdt_metrics_fullpath = os.path.join(
        channel_metrics_dir, u.get_filename_for_labels(pos_labels, suffix="sdt_metrics", extension="pkl")
    )
    try:
        sdt_metrics = pd.read_pickle(sdt_metrics_fullpath)
    except FileNotFoundError:
        sdt_metrics = detection_metrics(labels, np.arange(20), u.DATASET_ANNOTATORS[dataset_name], pos_labels=pos_labels)
        sdt_metrics.to_pickle(sdt_metrics_fullpath)
    return time_diffs, sdt_metrics


def timing_differences(
        labels: pd.DataFrame,
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
) -> pd.DataFrame:
    results = _calculation_wrapper(
        labels,
        lambda gt_labels, pred_labels: peyes.channel_metrics.onset_differences(gt_labels, pred_labels),
        lambda gt_labels, pred_labels: peyes.channel_metrics.offset_differences(gt_labels, pred_labels),
        gt_labelers,
        pred_labelers,
        pos_labels,
        iteration_desc="Samples Channel :: Timing Differences",
    )
    results = pd.DataFrame.from_dict(results, orient="columns")
    results.columns.names = [peyes.constants.TRIAL_ID_STR, u.GT_STR, u.PRED_STR, peyes.constants.ITERATION_STR]
    results.index.names = [_CHANNEL_TYPE_STR]
    return results


def detection_metrics(
        labels: pd.DataFrame,
        threshold: Union[int, Sequence[int]],
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        dprime_correction: str = "loglinear",
) -> pd.DataFrame:
    onset_func = lambda gt_labels, pred_labels: peyes.channel_metrics.onset_detection_metrics(
        gt_labels, pred_labels, threshold=threshold, dprime_correction=dprime_correction
    )
    offset_func = lambda gt_labels, pred_labels: peyes.channel_metrics.offset_detection_metrics(
        gt_labels, pred_labels, threshold=threshold, dprime_correction=dprime_correction
    )
    results = _calculation_wrapper(
        labels,
        onset_func,
        offset_func,
        gt_labelers,
        pred_labelers,
        pos_labels,
        iteration_desc="Samples Channel :: SDT Metrics",
    )
    results = pd.concat({k: pd.concat(v, axis=0) for k, v in results.items()}, axis=1)
    results.columns.names = [
        peyes.constants.TRIAL_ID_STR, u.GT_STR, u.PRED_STR, peyes.constants.ITERATION_STR, peyes.constants.METRIC_STR
    ]
    results.index.names = [_CHANNEL_TYPE_STR, peyes.constants.THRESHOLD_STR]
    return results


def _calculation_wrapper(
        labels: pd.DataFrame,
        onset_func: Callable[[UnparsedEventLabelSequenceType, UnparsedEventLabelSequenceType], Any],
        offset_func: Callable[[UnparsedEventLabelSequenceType, UnparsedEventLabelSequenceType], Any],
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        iteration_desc: str = "Channel Metrics",
) -> dict:
    pred_labelers = h.check_labelers(labels, pred_labelers)
    if pos_labels is None:
        pos_labels = [l for l in EventLabelEnum]
    elif isinstance(pos_labels, UnparsedEventLabelType):
        pos_labels = [peyes.parse_label(pos_labels)]
    else:
        pos_labels = [peyes.parse_label(l) for l in pos_labels]

    results = dict()
    trials = labels.columns.get_level_values(level=peyes.constants.TRIAL_ID_STR).unique()
    for tr in tqdm(trials, desc=iteration_desc):
        for gt_lblr in gt_labelers:
            try:
                trial_gt_labels = labels.xs((tr, gt_lblr), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR])
            except KeyError:
                continue
            if trial_gt_labels.size == 0:
                continue
            gt_min_iteration = np.nanmin(trial_gt_labels.columns.get_level_values(peyes.constants.ITERATION_STR))
            gt_labels = labels[tr, gt_lblr, gt_min_iteration].dropna().values.flatten()
            gt_labels[~np.isin(gt_labels, pos_labels)] = EventLabelEnum.UNDEFINED
            for pred_lblr in pred_labelers:
                try:
                    pred_labels_all_iters = labels.xs(
                        (tr, pred_lblr), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR]
                    )
                except KeyError:
                    continue
                for pred_it in pred_labels_all_iters.columns.get_level_values(peyes.constants.ITERATION_STR).unique():
                    if (pred_lblr == gt_lblr) and (pred_it == gt_min_iteration):
                        continue
                    pred_labels = pred_labels_all_iters[pred_it].dropna().values.flatten()
                    pred_labels[~np.isin(pred_labels, pos_labels)] = EventLabelEnum.UNDEFINED
                    if pred_labels.size == 0:
                        continue
                    results[(tr, gt_lblr, pred_lblr, pred_it)] = {
                        "onset": onset_func(gt_labels, pred_labels),
                        "offset": offset_func(gt_labels, pred_labels)
                    }
    return results
