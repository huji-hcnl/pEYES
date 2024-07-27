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
        detectors: List[BaseDetector] = u.DEFAULT_DETECTORS,
        annotators: List[str] = None,
        num_iterations: int = 4,
        iterations_overwrite_label: Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType] = 2,
        matching_schemes: Dict[str, Dict[str, Union[int, float]]] = preprocess.DEFAULT_MATCHING_SCHEMES,
        allow_xmatch: bool = False,
        pos_labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        sample_metrics_average: str = "weighted",
        sample_dprime_correction: str = "loglinear",
        channel_dprime_correction: str = "loglinear",
        verbose: bool = True
):
    start = time.time()
    ## Load dataset ##
    dataset = u.load_dataset(dataset_name, verbose=verbose)
    output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    ## labels, metadata, events, matches ##
    annotators = annotators or u.DATASET_ANNOTATORS[dataset_name]
    try:
        labels = pd.read_pickle(os.path.join(output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
        metadata = pd.read_pickle(os.path.join(output_dir, f"{u.METADATA_STR}.pkl"))
        events = pd.read_pickle(os.path.join(output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    except FileNotFoundError:
        labels, metadata, events = preprocess.detect_labels_and_events(
            dataset, detectors, annotators, num_iterations, iterations_overwrite_label, verbose
        )
        labels.to_pickle(os.path.join(output_dir, f"{peyes.constants.LABELS_STR}.pkl"))
        metadata.to_pickle(os.path.join(output_dir, f"{u.METADATA_STR}.pkl"))
        events.to_pickle(os.path.join(output_dir, f"{peyes.constants.EVENTS_STR}.pkl"))
    try:
        matches = pd.read_pickle(os.path.join(output_dir, f"{u.MATCHES_STR}.pkl"))
    except FileNotFoundError:
        matches = preprocess.match_events(
            events, annotators, matching_schemes=matching_schemes, allow_xmatch=allow_xmatch
        )
        matches.to_pickle(os.path.join(output_dir, f"{u.MATCHES_STR}.pkl"))

    ## Sample metrics ##
    sample_metrics_dir = os.path.join(output_dir, f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}")
    os.makedirs(sample_metrics_dir, exist_ok=True)
    sample_metrics_fullpath = os.path.join(sample_metrics_dir, u.get_filename_for_labels(pos_labels, extension="pkl"))
    try:
        sample_mets = pd.read_pickle(sample_metrics_fullpath)
    except FileNotFoundError:
        sample_mets = sample_metrics.calculate_sample_metrics(
            labels,
            annotators,
            pos_labels=pos_labels,
            average=sample_metrics_average,
            correction=sample_dprime_correction,
        )
        sample_mets.to_pickle(sample_metrics_fullpath)

    ## Sample Channel metrics ##
    channel_metrics_dir = os.path.join(output_dir, f"{peyes.constants.SAMPLES_STR}_{u.CHANNEL_STR}")
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
        channel_sdt_metrics = channel_metrics.detection_metrics(
            labels,
            np.arange(20),
            annotators,
            pos_labels=pos_labels,
            dprime_correction=channel_dprime_correction,
        )
        channel_sdt_metrics.to_pickle(channel_sdt_metrics_fullpath)

    ## Match metrics ##
    match_metrics_dir = os.path.join(output_dir, f"{u.MATCHES_STR}_{peyes.constants.METRICS_STR}")
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
        sample_mets,
        time_diffs,
        channel_sdt_metrics,
        matched_features,
        matches_sdt_metrics,
    )
