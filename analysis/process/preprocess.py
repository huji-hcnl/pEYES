import os
import time
import copy
import warnings
from typing import List, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import pEYES as peyes
from pEYES._DataModels.Detector import BaseDetector
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.process._helpers as h

_MATCH_BY_STR = "match_by"

DEFAULT_MATCHING_SCHEMES = {
    'iou': {_MATCH_BY_STR: 'iou', 'min_iou': 1/3},
    'max_overlap': {_MATCH_BY_STR: 'max_overlap', 'min_overlap': 0.5},
    'onset': {_MATCH_BY_STR: 'onset', 'max_onset_difference': 15},
    'offset': {_MATCH_BY_STR: 'offset', 'max_offset_difference': 15},
    'l2': {_MATCH_BY_STR: 'l2', 'max_l2': 15},
    # 'window': {_MATCH_BY_STR: 'window', 'max_onset_difference': 15, 'max_offset_difference': 15},
    **{f"window_{w}": {_MATCH_BY_STR: 'window', 'max_onset_difference': w, 'max_offset_difference': w} for w in np.arange(21)},
    # TODO: consider re-writing this to have `threshold` another argument
    # **{f"onset_{o}": {_MATCH_BY_STR: 'onset', 'max_onset_difference': o} for o in np.arange(21)},
    # **{f"iou_{iou:.1f}": {_MATCH_BY_STR: 'iou', 'min_iou': iou} for iou in np.arange(0.1, 1.01, 0.1)},
    # **{f"overlap_{ov:.1f}": {_MATCH_BY_STR: 'max_overlap', 'min_overlap': ov} for ov in np.arange(0.1, 1.01, 0.1)},
}

# peyes.match(gt_events, pred_events, match_by=match_by, allow_xmatch=allow_xmatch, **match_params)


def run_default(
        dataset_name: str, verbose: bool = True
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    start = time.time()
    dataset = u.load_dataset(dataset_name, verbose=True)
    default_output_dir = h.get_default_output_dir(dataset_name)
    try:
        labels = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
        metadata = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.METADATA_STR}.pkl"))
        events = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    except FileNotFoundError:
        default_detectors = [v[0] for v in u.LABELERS_CONFIG.values()]
        default_annotators = u.DATASET_ANNOTATORS[dataset_name]
        labels, metadata, events = detect_labels_and_events(
            dataset, default_detectors, default_annotators, verbose=verbose
        )
        if verbose:
            print(f"Saving labels & events to {default_output_dir}...")
        labels.to_pickle(os.path.join(default_output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
        metadata.to_pickle(os.path.join(default_output_dir, f"{peyes.constants.METADATA_STR}.pkl"))
        events.to_pickle(os.path.join(default_output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    try:
        matches = pd.read_pickle(os.path.join(default_output_dir, f"{peyes.constants.MATCHES_STR}.pkl"))
    except FileNotFoundError:
        matches = match_events(
            events, u.DATASET_ANNOTATORS[dataset_name], matching_schemes=None, allow_xmatch=False
        )
        if verbose:
            print(f"Saving matches to {default_output_dir}...")
        matches.to_pickle(os.path.join(default_output_dir, f"{peyes.constants.MATCHES_STR}.pkl"))
    elapsed = time.time() - start
    if verbose:
        print(f"### PREPROCESS TIME:\t{elapsed:.2f} seconds ###")
    return dataset, labels, metadata, events, matches


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
        if overwrite_label is None and num_iterations > 1:
            raise ValueError("Cannot have multiple iterations without specifying a label to overwrite.")
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
        metadata.index.name = peyes.constants.FIELD_NAME_STR
        events = pd.concat([pd.Series(v, name=k) for k, v in events.items()], axis=1)
        events.index.name = peyes.constants.EVENT_STR
        labels.columns.names = metadata.columns.names = events.columns.names = [
            peyes.constants.TRIAL_ID_STR, peyes.constants.LABELER_STR, peyes.constants.ITERATION_STR
        ]
        return labels, metadata, events


def match_events(
        events: pd.DataFrame,
        gt_labelers: List[str],
        pred_labelers: List[str] = None,
        matching_schemes: Dict[str, Dict[str, Union[str, int, float]]] = None,
        allow_xmatch: bool = False,
):
    """
    Matches between ground-truth and predicted events for each trial, labeler, and iteration.

    :param events: pd.DataFrame with MultiIndex columns (trial, labeler, iteration)
    :param gt_labelers: labelers to use as ground-truth
    :param pred_labelers: labelers to use as predictors
    :param matching_schemes: dictionary matching different scheme names to scheme parameters. Each set of parameters
        must include the key `match_by` which specifies the scheme to match by. If not provided, the scheme name is
        used as the `match_by` value.
    :param allow_xmatch: if True, allows cross-matching, i.e. matching between GT and Pred events of different types.

    :return: pd.DataFrame where each row is a matching-scheme, and columns are MultiIndex (trial, gt_labeler,
        pred_labeler, iteration). Cells contain dictionaries matching (a subset of) GT events to (a subset of) Pred events.
    """
    pred_labelers = h.check_labelers(events, pred_labelers)
    matching_schemes = matching_schemes or DEFAULT_MATCHING_SCHEMES
    results = dict()
    for tr in tqdm(events.columns.get_level_values(level=peyes.constants.TRIAL_ID_STR).unique(), desc="Matching Events"):
        for gt_labeler in gt_labelers:
            trial_gt_events = events.xs((tr, gt_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, peyes.constants.LABELER_STR])
            if trial_gt_events.size == 0:
                continue
            gt_min_iteration = np.nanmin(trial_gt_events.columns.get_level_values(peyes.constants.ITERATION_STR))
            gt_events = events[tr, gt_labeler, gt_min_iteration].dropna().values.flatten()
            if gt_events.size == 0:
                continue
            for pred_labeler in pred_labelers:
                pred_events_all_iters = events.xs(
                    (tr, pred_labeler), axis=1, level=[peyes.constants.TRIAL_ID_STR, peyes.constants.LABELER_STR]
                )
                for pred_it in pred_events_all_iters.columns.get_level_values(peyes.constants.ITERATION_STR).unique():
                    if (pred_labeler == gt_labeler) and (pred_it == gt_min_iteration):
                        continue
                    pred_events = pred_events_all_iters[pred_it].dropna().values.flatten()
                    if pred_events.size == 0:
                        continue
                    matches = dict()
                    for scheme_name, match_params in matching_schemes.items():
                        match_params[_MATCH_BY_STR] = match_params.get(_MATCH_BY_STR, scheme_name)
                        matches[scheme_name] = peyes.match(
                            gt_events, pred_events, allow_xmatch=allow_xmatch, **match_params
                        )
                    results[(tr, gt_labeler, pred_labeler, pred_it)] = matches
    results = pd.DataFrame.from_dict(results, orient="columns")
    results.index.names = [u.MATCHING_SCHEME_STR]
    results.columns.names = [peyes.constants.TRIAL_ID_STR, u.GT_STR, u.PRED_STR, peyes.constants.ITERATION_STR]
    return results
