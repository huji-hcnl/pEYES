import os
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
import plotly.express as px

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

###########################

CWD = os.getcwd()
OUTPUT_DIR = os.path.join(CWD, peyes.constants.OUTPUT_STR)
DATASETS_DIR = os.path.join(CWD, peyes.constants.OUTPUT_STR, "datasets")

###########################

GT_STR, PRED_STR = "gt", "pred"
MATCHING_SCHEME_STR = "matching_scheme"

###########################

DEFAULT_DISCRETE_COLORMAP = px.colors.qualitative.Dark24
DEFAULT_CONTINUOUS_COLORMAP = px.colors.sequential.Viridis

###########################

DATASET_ANNOTATORS = {
    "lund2013": ["RA", "MN"],
    "irf": ['RZ'],
    "hfc": ['DN', 'IH', 'JB', 'JF', 'JV', 'KH', 'MN', 'MS', 'PZ', 'RA', 'RH', 'TC']
}

_default_detector_params = dict(missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)
DEFAULT_DETECTORS_CONFIG = {
    # detector name -> (detector object, order, color)
    "ivt": (peyes.create_detector("ivt", **_default_detector_params), 0, DEFAULT_DISCRETE_COLORMAP[0]),
    "ivvt": (peyes.create_detector("ivvt", **_default_detector_params), 1, DEFAULT_DISCRETE_COLORMAP[1]),
    "idt": (peyes.create_detector("idt", **_default_detector_params), 2, DEFAULT_DISCRETE_COLORMAP[2]),
    "engbert": (peyes.create_detector("engbert", **_default_detector_params), 3, DEFAULT_DISCRETE_COLORMAP[3]),
    "nh": (peyes.create_detector("nh", **_default_detector_params), 4, DEFAULT_DISCRETE_COLORMAP[4]),
    "remodnav": (peyes.create_detector("remodnav", **_default_detector_params), 5, DEFAULT_DISCRETE_COLORMAP[5]),
}

METRICS_CONFIG = {
    # metric -> (name, order, value range)

    peyes.constants.ACCURACY_STR: ("Accuracy", 1, [0, 1]),
    peyes.constants.ONSET_STR: ("Onset", 1, None),
    f"{peyes.constants.AMPLITUDE_STR}_{peyes.constants.DIFFERENCE_STR}": ("Amplitude Difference", 1, None),

    peyes.constants.BALANCED_ACCURACY_STR: ("Balanced Accuracy", 2, [0, 1]),
    peyes.constants.OFFSET_STR: ("Offset", 2, None),
    f"{peyes.constants.AZIMUTH_STR}_{peyes.constants.DIFFERENCE_STR}": ("Azimuth Difference", 2, None),
    peyes.constants.MATCH_RATIO_STR: ("Match Ratio", 2, [0, 1]),

    peyes.constants.COHENS_KAPPA_STR: ("Cohen's Kappa", 3, [-1, 1]),
    f"center_{peyes.constants.PIXEL_STR}_{peyes.constants.DISTANCE_STR}": ("Center Distance", 3, None),

    peyes.constants.MCC_STR: ("MCC", 4, [-1, 1]),
    f"{peyes.constants.ONSET_STR}_{peyes.constants.DIFFERENCE_STR}": ("Onset Difference", 4, [-15, 15]),

    peyes.constants.RECALL_STR: ("Recall", 5, [0, 1]),
    f"{peyes.constants.OFFSET_STR}_{peyes.constants.DIFFERENCE_STR}": ("Offset Difference", 5, [-15, 15]),

    peyes.constants.PRECISION_STR: ("Precision", 6, [0, 1]),
    f"{peyes.constants.DURATION_STR}_{peyes.constants.DIFFERENCE_STR}": ("Duration Difference", 6, [-30, 30]),

    peyes.constants.F1_STR: ("f1", 7, [0, 1]),
    f"{peyes.constants.TIME_STR}_overlap": ("Time Overlap", 7, [0, 1]),

    peyes.constants.COMPLEMENT_NLD_STR: ("1 - NLD", 8, [0, 1]),
    f"{peyes.constants.TIME_STR}_iou": ("Time IoU", 8, [0, 1]),

    peyes.constants.FALSE_ALARM_RATE_STR: ("F.A. Rate", 8, [0, 1]),
    f"{peyes.constants.TIME_STR}_l2": ("Time L2", 8, None),

    peyes.constants.D_PRIME_STR: ("d'", 9, None),

    peyes.constants.CRITERION_STR: ("Criterion", 10, None),
}

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
