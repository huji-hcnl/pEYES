"""
Shared pipeline logic for the dependency-drift regression test (`test_dependency_drift.py`) and the golden
fixture generator (`fixtures/generate_golden.py`). Both must call the exact same code so the comparison is
meaningful - this module is the single source of truth for that.

Mirrors `analysis/_article_results/lund2013/_helpers.py`'s real detector configs and
`analysis/process/full_pipeline.py`'s call graph (detect -> events -> match -> metrics), trimmed to a subset
of detectors/schemes that span the risk spectrum identified in Part B's Phase 1 audit: IVT (cheap baseline),
Engbert (common case), NH and REMoDNaV (scipy/external-package-heavy, highest drift risk), plus IDT/IDVT
(dispersion-threshold logic, the one detection strategy the other four don't exercise at all).
"""
from typing import Dict

import numpy as np
import pandas as pd

import peyes
from peyes._DataModels.Detector import BaseDetector
import analysis.process.preprocess as preprocess
import analysis.process.match_metrics as match_metrics

GT_LABELERS = ["RA", "MN"]

MATCHING_SCHEMES = {
    "iou": {"match_by": "iou", "min_iou": 1 / 3},
    "window_10": {"match_by": "window", "max_onset_difference": 10, "max_offset_difference": 10},
}

_DEFAULT_DETECTOR_PARAMS = dict(missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)


def build_detectors() -> Dict[str, BaseDetector]:
    detectors = {
        "ivt": peyes.create_detector(algorithm="ivt", saccade_velocity_threshold=45, **_DEFAULT_DETECTOR_PARAMS),
        "idt": peyes.create_detector(algorithm="idt", dispersion_threshold=2.7, **_DEFAULT_DETECTOR_PARAMS),
        "idvt": peyes.create_detector(algorithm="idvt", dispersion_threshold=2.7, **_DEFAULT_DETECTOR_PARAMS),
        "engbert": peyes.create_detector(algorithm="engbert", lambda_param=6, **_DEFAULT_DETECTOR_PARAMS),
        "nh": peyes.create_detector(algorithm="nh", **_DEFAULT_DETECTOR_PARAMS),
        "remodnav": peyes.create_detector(algorithm="remodnav", show_warnings=False, **_DEFAULT_DETECTOR_PARAMS),
    }
    for key, det in detectors.items():
        det.name = key
    return detectors


def run_slice(dataset: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Runs the real detect -> events -> match -> metrics pipeline (the same functions
    `analysis/process/full_pipeline.py` calls) on a small dataset slice, and returns the results in a
    directly-comparable (DataFrame-only) form.
    """
    detectors = build_detectors()
    labels, _metadata, events = preprocess.detect_labels_and_events(
        dataset, list(detectors.values()), GT_LABELERS, num_iterations=1, verbose=False
    )
    matches = preprocess.match_events(
        events, GT_LABELERS, matching_schemes=dict(MATCHING_SCHEMES), allow_xmatch=False
    )
    matched_features = match_metrics.calculate_matched_features(matches)
    sdt_measures = match_metrics.calculate_event_sdt_measures(events, matches, None)
    return {
        "labels": labels,
        "matched_features": matched_features,
        "sdt_measures": sdt_measures,
    }
