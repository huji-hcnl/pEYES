import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.process._helpers as h

_MATCHED_FEATURE_NAMES = [
    f"{peyes.constants.ONSET_STR}_{peyes.constants.DIFFERENCE_STR}",
    f"{peyes.constants.OFFSET_STR}_{peyes.constants.DIFFERENCE_STR}",
    f"{peyes.constants.DURATION_STR}_{peyes.constants.DIFFERENCE_STR}",
    f"{peyes.constants.AMPLITUDE_STR}_{peyes.constants.DIFFERENCE_STR}",
    f"{peyes.constants.AZIMUTH_STR}_{peyes.constants.DIFFERENCE_STR}",
    f"center_{peyes.constants.PIXEL_STR}_{peyes.constants.DISTANCE_STR}",
    f"{peyes.constants.TIME_STR}_overlap",
    f"{peyes.constants.TIME_STR}_iou",
    f"{peyes.constants.TIME_STR}_l2",
]
_MATCH_SDT_METRICS = [
    u.MATCH_RATIO_STR,
    peyes.constants.PRECISION_STR, peyes.constants.RECALL_STR, peyes.constants.F1_STR,
    peyes.constants.D_PRIME_STR, peyes.constants.CRITERION_STR
]


def run_default(
        dataset_name: str,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
) -> (pd.DataFrame, pd.DataFrame):
    default_output_dir = h.get_default_output_dir(dataset_name)
    try:
        events = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find `{peyes.constants.EVENTS_STR}.pkl` in {default_output_dir}. Please preprocess the dataset first."
        )
    try:
        matches = pd.read_pickle(os.path.join(default_output_dir, f"{u.MATCHES_STR}.pkl"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Couldn't find `{u.MATCHES_STR}.pkl` in {default_output_dir}. Please preprocess the dataset first."
        )
    matches_metrics_dir = os.path.join(default_output_dir, f"{u.MATCHES_STR}_{peyes.constants.METRICS_STR}")
    os.makedirs(matches_metrics_dir, exist_ok=True)
    features_fullpath = os.path.join(
        matches_metrics_dir, u.get_filename_for_labels(labels=None, suffix="matched_features", extension="pkl")
    )
    try:
        matched_features = pd.read_pickle(features_fullpath)
    except FileNotFoundError:
        matched_features = calculate_matched_features(matches, features=_MATCHED_FEATURE_NAMES)
        matched_features.to_pickle(features_fullpath)
    sdt_fullpath = os.path.join(
        matches_metrics_dir, u.get_filename_for_labels(pos_labels, suffix="sdt_metrics", extension="pkl")
    )
    try:
        sdt_metrics = pd.read_pickle(sdt_fullpath)
    except FileNotFoundError:
        sdt_metrics = calculate_event_sdt_measures(events, matches, pos_labels)
        sdt_metrics.to_pickle(sdt_fullpath)
    return matched_features, sdt_metrics


def calculate_matched_features(
        matches: pd.DataFrame,
        features: List[str] = None,
):
    if features is None:
        features = _MATCHED_FEATURE_NAMES
    features = [f.lower().strip().replace(" ", "_").replace("-", "_") for f in features]
    if not set(features).issubset(_MATCHED_FEATURE_NAMES):
        raise ValueError(f"Unknown feature(s): {set(features) - set(_MATCHED_FEATURE_NAMES)}")
    results = dict()
    trials = matches.columns.get_level_values(level=peyes.constants.TRIAL_ID_STR).unique()
    gt_labelers = matches.columns.get_level_values(u.GT_STR).unique()
    pred_labelers = matches.columns.get_level_values(u.PRED_STR).unique()
    iterations = matches.columns.get_level_values(peyes.constants.ITERATION_STR).unique()
    matching_schemes = matches.index.get_level_values(u.MATCHING_SCHEME_STR).unique()
    for tr in tqdm(trials, desc="Matched Events :: Features"):
        for gt_labeler in gt_labelers:
            for pred_labeler in pred_labelers:
                for pred_it in iterations:
                    curr_results = dict()
                    for ms in matching_schemes:
                        try:
                            curr_matches = matches.loc[ms, (tr, gt_labeler, pred_labeler, pred_it)]
                        except KeyError:
                            continue
                        if len(curr_matches) == 0:
                            continue
                        for feat in features:
                            curr_results[(ms, feat)] = peyes.match_metrics.features(
                                curr_matches, feat, verbose=False
                            )
                    if curr_results:
                        results[(tr, gt_labeler, pred_labeler, pred_it)] = curr_results
    results = pd.DataFrame.from_dict(results, orient="columns")
    results.index.names = [u.MATCHING_SCHEME_STR, peyes.constants.FEATURE_STR]
    results.columns.names = matches.columns.names
    return results


def calculate_event_sdt_measures(
        events: pd.DataFrame,
        matches: pd.DataFrame,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
) -> pd.DataFrame:
    results = dict()
    trials = matches.columns.get_level_values(level=peyes.constants.TRIAL_ID_STR).unique()
    gt_labelers = matches.columns.get_level_values(u.GT_STR).unique()
    pred_labelers = matches.columns.get_level_values(u.PRED_STR).unique()
    iterations = matches.columns.get_level_values(peyes.constants.ITERATION_STR).unique()
    matching_schemes = matches.index.get_level_values(u.MATCHING_SCHEME_STR).unique()
    for tr in tqdm(trials, desc="Matched Events :: SDT Metrics"):
        for gt_labeler in gt_labelers:
            try:
                trial_gt_events = events.xs((tr, gt_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR])
            except KeyError:
                continue
            if trial_gt_events.size == 0:
                continue
            gt_min_iteration = np.nanmin(trial_gt_events.columns.get_level_values(peyes.constants.ITERATION_STR))
            gt_events = events[tr, gt_labeler, gt_min_iteration].dropna().values.flatten()
            if gt_events.size == 0:
                continue
            for pred_labeler in pred_labelers:
                try:
                    pred_events_all_iters = events.xs((tr, pred_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR])
                except KeyError:
                    continue
                for pred_it in iterations:
                    try:
                        pred_iter_events = pred_events_all_iters[pred_it].dropna().values.flatten()
                    except KeyError:
                        continue
                    if pred_iter_events.size == 0:
                        continue
                    curr_results = dict()
                    for ms in matching_schemes:
                        try:
                            curr_matches = matches.loc[ms, (tr, gt_labeler, pred_labeler, pred_it)]
                        except KeyError:
                            continue
                        if len(curr_matches) == 0:
                            continue
                        curr_results[(ms, u.MATCH_RATIO_STR)] = peyes.match_metrics.match_ratio(
                            pred_iter_events, curr_matches, labels=pos_labels
                        )
                        if pos_labels is not None:
                            prec, rec, f1 = peyes.match_metrics.precision_recall_f1(
                                gt_events, pred_iter_events, curr_matches, pos_labels
                            )
                            fa_rate = peyes.match_metrics.false_alarm_rate(
                                gt_events, pred_iter_events, curr_matches, pos_labels
                            )
                            d_prime, crit = peyes.match_metrics.d_prime_and_criterion(
                                gt_events, pred_iter_events, curr_matches, pos_labels
                            )
                            curr_results[(ms, peyes.constants.PRECISION_STR)] = prec
                            curr_results[(ms, peyes.constants.RECALL_STR)] = rec
                            curr_results[(ms, peyes.constants.F1_STR)] = f1
                            curr_results[(ms, peyes.constants.FALSE_ALARM_RATE_STR)] = fa_rate
                            curr_results[(ms, peyes.constants.D_PRIME_STR)] = d_prime
                            curr_results[(ms, peyes.constants.CRITERION_STR)] = crit
                    if curr_results:
                        results[(tr, gt_labeler, pred_labeler, pred_it)] = curr_results
    results = pd.DataFrame.from_dict(results, orient="columns")
    results.index.names = [u.MATCHING_SCHEME_STR, peyes.constants.METRIC_STR]
    results.columns.names = matches.columns.names
    return results
