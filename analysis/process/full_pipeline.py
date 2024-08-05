import os
import time
from typing import List, Union, Dict, Optional

import numpy as np
import pandas as pd

import pEYES as peyes
from pEYES._DataModels.Detector import BaseDetector
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.process.preprocess as preprocess
import analysis.process.sample_metrics as sample_metrics
import analysis.process.samples_channel as channel_metrics
import analysis.process.match_metrics as match_metrics


def run(
        output_dir: str,
        dataset_name: str,
        detectors: List[BaseDetector] = None,
        annotators: List[str] = None,
        num_iterations: int = 4,
        iterations_overwrite_label: Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType] = 2,
        matching_schemes: Dict[str, Dict[str, Union[str, int, float]]] = preprocess.DEFAULT_MATCHING_SCHEMES,
        allow_xmatch: bool = False,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        sample_sdt_average: str = "weighted",
        sample_dprime_correction: str = "loglinear",
        channel_dprime_correction: str = "loglinear",
        verbose: bool = True
):
    start = time.time()
    ## Load dataset ##
    dataset = u.load_dataset(dataset_name, verbose=verbose)
    output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    ## default detectors & annotators ##
    detectors = detectors or [v[0] for v in u.LABELERS_CONFIG.values() if v[0] is not None]
    annotators = annotators or u.DATASET_ANNOTATORS[dataset_name]

    ## labels, metadata, events, matches ##
    try:
        labels = pd.read_pickle(os.path.join(output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
        metadata = pd.read_pickle(os.path.join(output_dir, f"{peyes.constants.METADATA_STR}.pkl"))
        events = pd.read_pickle(os.path.join(output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    except FileNotFoundError:
        labels, metadata, events = preprocess.detect_labels_and_events(
            dataset, detectors, annotators, num_iterations, iterations_overwrite_label, verbose
        )
        labels.to_pickle(os.path.join(output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
        metadata.to_pickle(os.path.join(output_dir, f"{peyes.constants.METADATA_STR}.pkl"))
        events.to_pickle(os.path.join(output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    try:
        matches = pd.read_pickle(os.path.join(output_dir, f"{peyes.constants.MATCHES_STR}.pkl"))
    except FileNotFoundError:
        matches = preprocess.match_events(
            events, annotators, matching_schemes=matching_schemes, allow_xmatch=allow_xmatch
        )
        matches.to_pickle(os.path.join(output_dir, f"{peyes.constants.MATCHES_STR}.pkl"))

    ## Sample metrics ##
    sample_metrics_dir = os.path.join(output_dir, f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}")
    os.makedirs(sample_metrics_dir, exist_ok=True)

    # Sample SDT metrics
    sdt_sample_metrics_fullpath = os.path.join(
        sample_metrics_dir, u.get_filename_for_labels(
            pos_labels, suffix=f"{u.SDT_STR}_{peyes.constants.METRICS_STR}", extension="pkl"
        )
    )
    try:
        sdt_sample_metrics = pd.read_pickle(sdt_sample_metrics_fullpath)
    except FileNotFoundError:
        sdt_sample_metrics = sample_metrics.calculate_sdt_sample_metrics(
            labels, annotators, pos_labels=pos_labels, average=sample_sdt_average, correction=sample_dprime_correction
        )
        sdt_sample_metrics.to_pickle(sdt_sample_metrics_fullpath)

    # Sample Global metrics - only when pos_labels is None
    if pos_labels is None or (isinstance(pos_labels, list) and len(pos_labels) == 0):
        global_sample_metrics_fullpath = os.path.join(
            sample_metrics_dir, u.get_filename_for_labels(
                pos_labels, suffix=f"{u.GLOBAL_STR}_{peyes.constants.METRICS_STR}", extension="pkl"
            )
        )
        try:
            global_sample_metrics = pd.read_pickle(global_sample_metrics_fullpath)
        except FileNotFoundError:
            global_sample_metrics = sample_metrics.calculate_global_sample_metrics(labels, annotators)
            global_sample_metrics.to_pickle(global_sample_metrics_fullpath)
    else:
        global_sample_metrics = None

    ## Sample Channel metrics ##
    channel_metrics_dir = os.path.join(output_dir, peyes.constants.SAMPLES_CHANNEL_STR)
    os.makedirs(channel_metrics_dir, exist_ok=True)
    time_diffs_fullpath = os.path.join(
        channel_metrics_dir, u.get_filename_for_labels(pos_labels, suffix="timing_differences", extension="pkl")
    )
    try:
        time_diffs = pd.read_pickle(time_diffs_fullpath)
    except FileNotFoundError:
        time_diffs = channel_metrics.timing_differences(labels, annotators, pos_labels=pos_labels)
        time_diffs.to_pickle(time_diffs_fullpath)
    channel_sdt_metrics_fullpath = os.path.join(
        channel_metrics_dir, u.get_filename_for_labels(pos_labels, suffix="sdt_metrics", extension="pkl")
    )
    try:
        channel_sdt_metrics = pd.read_pickle(channel_sdt_metrics_fullpath)
    except FileNotFoundError:
        channel_sdt_metrics = channel_metrics.signal_detection_metrics(
            labels, np.arange(21), annotators, pos_labels=pos_labels, dprime_correction=channel_dprime_correction
        )
        channel_sdt_metrics.to_pickle(channel_sdt_metrics_fullpath)

    ## Match metrics ##
    match_metrics_dir = os.path.join(output_dir, f"{peyes.constants.MATCHES_STR}_{peyes.constants.METRICS_STR}")
    os.makedirs(match_metrics_dir, exist_ok=True)
    matched_features_fullpath = os.path.join(
        match_metrics_dir, u.get_filename_for_labels(labels=None, suffix="matched_features", extension="pkl")
    )
    try:
        matched_features = pd.read_pickle(matched_features_fullpath)
    except FileNotFoundError:
        matched_features = match_metrics.calculate_matched_features(matches)
        matched_features.to_pickle(matched_features_fullpath)
    matches_sdt_fullpath = os.path.join(
        match_metrics_dir, u.get_filename_for_labels(pos_labels, suffix="sdt_metrics", extension="pkl")
    )
    try:
        matches_sdt_metrics = pd.read_pickle(matches_sdt_fullpath)
    except FileNotFoundError:
        matches_sdt_metrics = match_metrics.calculate_event_sdt_measures(events, matches, pos_labels)
        matches_sdt_metrics.to_pickle(matches_sdt_fullpath)

    elapsed = time.time() - start
    if verbose:
        print(f"Finished in {elapsed:.2f}s")
    return (
        dataset,
        labels,
        metadata,
        events,
        matches,
        sdt_sample_metrics,
        global_sample_metrics,
        time_diffs,
        channel_sdt_metrics,
        matched_features,
        matches_sdt_metrics,
    )
