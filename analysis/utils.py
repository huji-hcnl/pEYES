import os
from typing import Optional, Union, Sequence, List

import numpy as np
import pandas as pd
import plotly.express as px

import peyes
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

###########################

BASE_DIR = r'S:\Lab-Shared\Experiments\pEYES'   # lab-shared directory
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, peyes.constants.OUTPUT_STR)

###########################

GLOBAL_STR = "global"
SDT_STR = "sdt"
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


def get_trials_for_stimulus_type(
        dataset_name: str,
        stimulus_type: Union[str, List[str]],
) -> List[int]:
    if not stimulus_type:
        raise ValueError("Stimulus type is not specified")
    if isinstance(stimulus_type, str):
        stimulus_type = [stimulus_type]
    stimulus_type = list(set([stmtp.lower().strip() for stmtp in stimulus_type if isinstance(stmtp, str)]))
    dataset = load_dataset(dataset_name, verbose=False)
    is_stimulus_type = dataset[peyes.constants.STIMULUS_TYPE_STR].str.lower().isin(stimulus_type)
    trials = dataset.loc[is_stimulus_type, peyes.constants.TRIAL_ID_STR].unique()
    return trials


def get_labeler_index(labeler_name: str, detectors_names: Sequence[str]) -> int:
    all_labelers = [
        *detectors_names,
        *[annot for annotators in DATASET_ANNOTATORS.values() for annot in annotators],
    ]
    return all_labelers.index(labeler_name)


def get_labeler_color(detector_name: str, det_idx: int, colors: Optional[Sequence[str]] = None) -> str:
    colors = colors or DEFAULT_DISCRETE_COLORMAP
    c = colors.get(detector_name, colors[det_idx % len(colors)])
    return c
