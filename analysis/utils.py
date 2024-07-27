import os
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
import plotly.express as px

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType


CWD = os.getcwd()
OUTPUT_DIR = os.path.join(CWD, "output")
DATASETS_DIR = os.path.join(CWD, "output", "datasets")

LABELER_STR = "labeler"
DEFAULT_STR = "default"
METADATA_STR = "metadata"
MATCHES_STR = "matches"
FIELD_NAME_STR = "field_name"
GT_STR, PRED_STR = "gt", "pred"
MATCHING_SCHEME_STR = "matching_scheme"
MATCH_RATIO_STR = "match_ratio"
CHANNEL_STR = "channel"
CHANNEL_TYPE_STR = f"{CHANNEL_STR}_{peyes.constants.TYPE_STR}"

DATASET_ANNOTATORS = {
    "lund2013": ["RA", "MN"],
    "irf": ['RZ'],
    "hfc": ['DN', 'IH', 'JB', 'JF', 'JV', 'KH', 'MN', 'MS', 'PZ', 'RA', 'RH', 'TC']
}
DETECTOR_NAMES = ["ivt", "ivvt", "idt", "engbert", "nh", "remodnav"]
DEFAULT_DETECTORS = [
    peyes.create_detector(det, missing_value=np.nan, min_event_duration=4, pad_blinks_time=0) for det in DETECTOR_NAMES
]

METRICS_CONFIG = {
    # key -> (name, order, value range)
    peyes.constants.ACCURACY_STR: ("Accuracy", 1, [0, 1]),
    peyes.constants.ONSET_STR: ("Onset", 1, None),
    MATCH_RATIO_STR: ("Match Ratio", 2, [0, 1]),
    peyes.constants.OFFSET_STR: ("Offset", 2, None),
    peyes.constants.BALANCED_ACCURACY_STR: ("Balanced Accuracy", 2, [0, 1]),
    peyes.constants.COHENS_KAPPA_STR: ("Cohen's Kappa", 3, [-1, 1]),
    peyes.constants.MCC_STR: ("MCC", 4, [-1, 1]),
    peyes.constants.RECALL_STR: ("Recall", 5, [0, 1]),
    peyes.constants.PRECISION_STR: ("Precision", 6, [0, 1]),
    peyes.constants.F1_STR: ("f1", 7, [0, 1]),
    peyes.constants.COMPLEMENT_NLD_STR: ("1 - NLD", 8, [0, 1]),
    peyes.constants.D_PRIME_STR: ("d'", 9, None),
    peyes.constants.CRITERION_STR: ("Criterion", 10, None),
}


DEFAULT_DISCRETE_COLORMAP = px.colors.qualitative.Dark24
DEFAULT_CONTINUOUS_COLORMAP = px.colors.sequential.Viridis

###########################################


def load_dataset(dataset_name: str, verbose: bool = True) -> pd.DataFrame:
    if dataset_name == "lund2013":
        dataset = peyes.datasets.lund2013(directory=DATASETS_DIR, save=True, verbose=verbose)
    elif dataset_name == "irf":
        dataset = peyes.datasets.irf(directory=DATASETS_DIR, save=True, verbose=verbose)
    elif dataset_name == "hfc":
        dataset = peyes.datasets.hfc(directory=DATASETS_DIR, save=True, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset


def get_filename_for_labels(
        labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        prefix: str = "",
        suffix: str = "",
        extension: str = "pkl"
) -> str:
    if prefix and not prefix.endswith("_"):
        prefix += "_"
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    if labels is None:
        return f"{prefix}all_labels{suffix}.{extension}"
    elif isinstance(labels, UnparsedEventLabelType):
        return f"{prefix}{peyes.parse_label(labels).name.lower()}{suffix}.{extension}"
    elif isinstance(labels, Sequence) and all(isinstance(l, UnparsedEventLabelType) for l in labels):
        return f"{prefix}{'_'.join([peyes.parse_label(l).name.lower() for l in labels])}{suffix}.{extension}"
    else:
        raise TypeError(f"Unknown pos_labels type: {type(labels)}")
