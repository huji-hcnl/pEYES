import os
from typing import List, Optional, Union, Sequence, Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType
from pEYES._DataModels.EventLabelEnum import EventLabelEnum
from pEYES._DataModels.Event import EventSequenceType
from pEYES._utils.event_utils import calculate_sampling_rate

import analysis.utils as u

_CHANNEL_TYPE_STR = "channel_type"


def run_default(
        dataset_name: str,
        dataset: pd.DataFrame,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
):
    default_output_dir = u.get_default_output_dir(dataset_name)
    try:
        events = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find `{peyes.constants.EVENTS_STR}.pkl` in {default_output_dir}. Please preprocess the dataset first."
        )
    channel_metrics_dir = os.path.join(default_output_dir, f"{u.CHANNEL_STR}_{peyes.constants.METRICS_STR}")
    os.makedirs(channel_metrics_dir, exist_ok=True)
    time_diffs_fullpath = os.path.join(
        channel_metrics_dir, u.get_filename_for_labels(pos_labels, suffix="timing_differences", extension="pkl")
    )
    try:
        time_diffs = pd.read_pickle(time_diffs_fullpath)
    except FileNotFoundError:
        time_diffs = timing_differences(dataset, events, u.DATASET_ANNOTATORS[dataset_name], pos_labels=pos_labels)
        time_diffs.to_pickle(time_diffs_fullpath)
    sdt_metrics_fullpath = os.path.join(
        channel_metrics_dir, u.get_filename_for_labels(pos_labels, suffix="sdt_metrics", extension="pkl")
    )
    try:
        sdt_metrics = pd.read_pickle(sdt_metrics_fullpath)
    except FileNotFoundError:
        sdt_metrics = detection_metrics(dataset, events, np.arange(10), u.DATASET_ANNOTATORS[dataset_name], pos_labels=pos_labels)
        sdt_metrics.to_pickle(sdt_metrics_fullpath)
    return time_diffs, sdt_metrics


def timing_differences(
        dataset: pd.DataFrame,
        events: pd.DataFrame,
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
) -> pd.DataFrame:
    results = _calculation_wrapper(
        dataset,
        events,
        peyes.channel_metrics.onset_differences,
        peyes.channel_metrics.offset_differences,
        gt_labelers,
        pred_labelers,
        pos_labels,
        iteration_desc="Channel :: Timing Differences",
    )
    results = pd.DataFrame.from_dict(results, orient="columns")
    results.columns.names = [peyes.constants.TRIAL_ID_STR, u.GT_STR, u.PRED_STR, peyes.constants.ITERATION_STR]
    results.index.names = [_CHANNEL_TYPE_STR]
    return results


def detection_metrics(
        dataset: pd.DataFrame,
        events: pd.DataFrame,
        threshold: Union[int, Sequence[int]],
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        dprime_correction: str = "loglinear",
) -> pd.DataFrame:
    onset_func = lambda gt, pred, sr, ns: peyes.channel_metrics.onset_detection_metrics(
        gt, pred, threshold, sr, ns, dprime_correction
    )
    offset_func = lambda gt, pred, sr, ns: peyes.channel_metrics.offset_detection_metrics(
        gt, pred, threshold, sr, ns, dprime_correction
    )
    results = _calculation_wrapper(
        dataset,
        events,
        onset_func,
        offset_func,
        gt_labelers,
        pred_labelers,
        pos_labels,
        iteration_desc="Channel :: SDT Metrics",
    )
    results = pd.concat({k: pd.concat(v, axis=0) for k, v in results.items()}, axis=1)
    results.columns.names = [
        peyes.constants.TRIAL_ID_STR, u.GT_STR, u.PRED_STR, peyes.constants.ITERATION_STR, peyes.constants.METRIC_STR
    ]
    results.index.names = [_CHANNEL_TYPE_STR, peyes.constants.THRESHOLD_STR]
    return results


def _calculation_wrapper(
        dataset: pd.DataFrame,
        events: pd.DataFrame,
        onset_func: Callable[[EventSequenceType, EventSequenceType, float, int], Any],
        offset_func: Callable[[EventSequenceType, EventSequenceType, float, int], Any],
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        iteration_desc: str = "Channel Metrics",
) -> dict:
    all_labelers = events.columns.get_level_values(u.LABELER_STR).unique()
    pred_labelers = pred_labelers or all_labelers
    unknown_labelers = (set(pred_labelers) - set(all_labelers)) | (set(gt_labelers) - set(all_labelers))
    if unknown_labelers:
        raise ValueError(f"Unknown labelers: {unknown_labelers}")

    if pos_labels is None:
        pos_labels = [l for l in EventLabelEnum]
    elif isinstance(pos_labels, UnparsedEventLabelType):
        pos_labels = [peyes.parse_label(pos_labels)]
    else:
        pos_labels = [peyes.parse_label(l) for l in pos_labels]

    results = dict()
    trials = events.columns.get_level_values(level=peyes.constants.TRIAL_ID_STR).unique()
    for tr in tqdm(trials, desc=iteration_desc):
        trial_timestamps = dataset[dataset[peyes.constants.TRIAL_ID_STR] == tr][peyes.T].values
        trial_num_samples = len(trial_timestamps)
        trial_sampling_rate = calculate_sampling_rate(trial_timestamps)
        for gt_lblr in gt_labelers:
            try:
                trial_gt_events = events.xs((tr, gt_lblr), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR])
            except KeyError:
                continue
            if trial_gt_events.size == 0:
                continue
            gt_min_iteration = np.nanmin(trial_gt_events.columns.get_level_values(peyes.constants.ITERATION_STR))
            gt_events = events[tr, gt_lblr, gt_min_iteration].dropna().values.flatten()
            gt_events = np.array([e for e in gt_events if e.label in pos_labels])
            if gt_events.size == 0:
                continue
            for pred_lblr in pred_labelers:
                try:
                    pred_events_all_iters = events.xs(
                        (tr, pred_lblr), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR]
                    )
                except KeyError:
                    continue
                for pred_it in pred_events_all_iters.columns.get_level_values(peyes.constants.ITERATION_STR).unique():
                    if (pred_lblr == gt_lblr) and (pred_it == gt_min_iteration):
                        continue
                    pred_events = pred_events_all_iters[pred_it].dropna().values.flatten()
                    pred_events = np.array([e for e in pred_events if e.label in pos_labels])
                    if pred_events.size == 0:
                        continue
                    results[(tr, gt_lblr, pred_lblr, pred_it)] = {
                        "onset": onset_func(gt_events, pred_events, trial_sampling_rate, trial_num_samples),
                        "offset": offset_func(gt_events, pred_events, trial_sampling_rate, trial_num_samples)
                    }
    return results
