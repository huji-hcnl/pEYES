import os
import time
import copy
import warnings
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import pEYES as peyes
from pEYES._DataModels.EventLabelEnum import EventLabelEnum
from pEYES._DataModels.Detector import BaseDetector
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u


DEFAULT_MATCHING_SCHEMES = {
    'onset': dict(max_onset_difference=15),
    'offset': dict(max_offset_difference=15),
    'window': dict(max_onset_difference=15, max_offset_difference=15),
    'l2': dict(max_l2=15),
    'iou': dict(min_iou=1/3),
    'max_overlap': dict(min_overlap=0.5),
}


def run_default(
        dataset_name: str, verbose: bool = True
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    start = time.time()
    dataset = load_dataset(dataset_name, verbose=True)
    default_output_dir = u.get_default_output_dir(dataset_name)
    try:
        labels = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
        metadata = pd.read_pickle(os.path.join(default_output_dir, f"{u.METADATA_STR}.pkl"))
        events = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    except FileNotFoundError:
        labels, metadata, events = detect_labels_and_events(
            dataset, u.DEFAULT_DETECTORS, u.DATASET_ANNOTATORS[dataset_name], verbose=verbose
        )
        if verbose:
            print(f"Saving labels & events to {default_output_dir}...")
        labels.to_pickle(os.path.join(default_output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
        metadata.to_pickle(os.path.join(default_output_dir, f"{u.METADATA_STR}.pkl"))
        events.to_pickle(os.path.join(default_output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    try:
        matches = pd.read_pickle(os.path.join(default_output_dir, f"{u.MATCHES_STR}.pkl"))
    except FileNotFoundError:
        matches = match_events(
            events, u.DATASET_ANNOTATORS[dataset_name], matching_schemes=None, allow_xmatch=False
        )
        if verbose:
            print(f"Saving matches to {default_output_dir}...")
        matches.to_pickle(os.path.join(default_output_dir, f"{u.MATCHES_STR}.pkl"))
    elapsed = time.time() - start
    if verbose:
        print(f"### PREPROCESS TIME:\t{elapsed:.2f} seconds ###")
    return dataset, labels, metadata, events, matches


def load_dataset(dataset_name: str, verbose: bool = True) -> pd.DataFrame:
    if dataset_name == "lund2013":
        dataset = peyes.datasets.lund2013(directory=u.DATASETS_DIR, save=True, verbose=verbose)
    elif dataset_name == "irf":
        dataset = peyes.datasets.irf(directory=u.DATASETS_DIR, save=True, verbose=verbose)
    elif dataset_name == "hfc":
        dataset = peyes.datasets.hfc(directory=u.DATASETS_DIR, save=True, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset


def detect_labels_and_events(
        dataset: pd.DataFrame,
        detectors: List[BaseDetector],
        annotators: List[str],
        num_iterations: int = 4,
        overwrite_label: Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType] = 2,
        verbose: bool = True,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if overwrite_label is None:
            overwrite_label = {l for l in EventLabelEnum}
        elif isinstance(overwrite_label, UnparsedEventLabelType):
            overwrite_label = {peyes.parse_label(overwrite_label)}
        else:
            overwrite_label = {peyes.parse_label(l) for l in overwrite_label}

        labels, metadata, events = dict(), dict(), dict()
        trials_prog_bar = tqdm(
            dataset[peyes.constants.TRIAL_ID_STR].unique(), desc="Detecting Labels & Events", leave=True, disable=False
        )
        for tr in trials_prog_bar:
            trial = dataset[dataset[peyes.constants.TRIAL_ID_STR] == tr]
            t = trial[peyes.constants.T].values
            x = trial[peyes.constants.X].values
            y = trial[peyes.constants.Y].values
            p = trial[peyes.constants.PUPIL].values
            vd = trial[peyes.constants.VIEWER_DISTANCE_STR].iloc[0]
            ps = trial[peyes.constants.PIXEL_SIZE_STR].iloc[0]
            for annot in tqdm(annotators, desc="\tAnnotators", leave=False, disable=not verbose, position=1):
                annotator_labels = trial[annot].values
                labels[(tr, annot, 1)] = np.array(annotator_labels)
                events[(tr, annot, 1)] = peyes.create_events(
                    labels=annotator_labels, t=t, x=x, y=y, pupil=p, viewer_distance=vd, pixel_size=ps
                )
            for det in tqdm(detectors, desc="\tDetectors", leave=False, disable=not verbose, position=1):
                det_name = det.name
                x_copy, y_copy, p_copy = copy.deepcopy(x), copy.deepcopy(y), copy.deepcopy(p)
                for it in trange(num_iterations, desc="\t\tIterations", leave=False, disable=not verbose, position=2):
                    it_labels, it_metadata = det.detect(t, x_copy, y_copy, viewer_distance_cm=vd, pixel_size_cm=ps)
                    labels[(tr, det_name, it+1)] = np.array(it_labels)
                    metadata[(tr, det_name, it+1)] = it_metadata
                    events[(tr, det_name, it+1)] = peyes.create_events(
                        labels=it_labels, t=t, x=x_copy, y=y_copy, pupil=p_copy, viewer_distance=vd, pixel_size=ps
                    )
                    to_overwrite = np.isin(it_labels, list(overwrite_label))
                    x_copy[to_overwrite] = np.nan
                    y_copy[to_overwrite] = np.nan
                    p_copy[to_overwrite] = np.nan
        labels = pd.concat([pd.Series(v, name=k) for k, v in labels.items()], axis=1)
        labels.index.name = peyes.constants.SAMPLE_STR
        metadata = pd.concat([pd.Series(v, name=k) for k, v in metadata.items()], axis=1)
        metadata.index.name = u.FIELD_NAME_STR
        events = pd.concat([pd.Series(v, name=k) for k, v in events.items()], axis=1)
        events.index.name = peyes.constants.EVENT_STR
        labels.columns.names = metadata.columns.names = events.columns.names = [
            peyes.constants.TRIAL_ID_STR, u.LABELER_STR, peyes.constants.ITERATION_STR
        ]
        return labels, metadata, events


def match_events(
        events: pd.DataFrame,
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        matching_schemes: Dict[str, Dict[str, Union[int, float]]] = None,
        allow_xmatch: bool = False,
):
    all_labelers = events.columns.get_level_values(u.LABELER_STR).unique()
    pred_labelers = pred_labelers or all_labelers
    unknown_labelers = (set(pred_labelers) - set(all_labelers)) | (set(gt_labelers) - set(all_labelers))
    if unknown_labelers:
        raise ValueError(f"Unknown labelers: {unknown_labelers}")
    matching_schemes = matching_schemes or DEFAULT_MATCHING_SCHEMES
    results = dict()
    for tr in tqdm(events.columns.get_level_values(level=peyes.constants.TRIAL_ID_STR).unique(), desc="Matching Events"):
        for gt_labeler in gt_labelers:
            trial_gt_events = events.xs((tr, gt_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR])
            if trial_gt_events.size == 0:
                continue
            gt_min_iteration = np.nanmin(trial_gt_events.columns.get_level_values(peyes.constants.ITERATION_STR))
            gt_events = events[tr, gt_labeler, gt_min_iteration].dropna().values.flatten()
            if gt_events.size == 0:
                continue
            for pred_labeler in pred_labelers:
                pred_events_all_iters = events.xs(
                    (tr, pred_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, u.LABELER_STR]
                )
                for pred_it in pred_events_all_iters.columns.get_level_values(peyes.constants.ITERATION_STR).unique():
                    if (pred_labeler == gt_labeler) and (pred_it == gt_min_iteration):
                        continue
                    pred_events = pred_events_all_iters[pred_it].dropna().values.flatten()
                    if pred_events.size == 0:
                        continue
                    matches = dict()
                    for match_by, match_params in matching_schemes.items():
                        matches[match_by] = peyes.match(
                            gt_events, pred_events, match_by, allow_xmatch=allow_xmatch, **match_params
                        )
                    results[(tr, gt_labeler, pred_labeler, pred_it)] = matches
    results = pd.DataFrame.from_dict(results, orient="columns")
    results.index.names = [u.MATCHING_SCHEME_STR]
    results.columns.names = [peyes.constants.TRIAL_ID_STR, u.GT_STR, u.PRED_STR, peyes.constants.ITERATION_STR]
    return results
