import os
from typing import Optional, Union

import numpy as np

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
CHANNEL_STR = "channel"
CHANNEL_TYPE_STR = f"{CHANNEL_STR}_{peyes.TYPE_STR}"

DATASET_ANNOTATORS = {
    "lund2013": ["RA", "MN"],
    "irf": ['RZ'],
    "hfc": ['DN', 'IH', 'JB', 'JF', 'JV', 'KH', 'MN', 'MS', 'PZ', 'RA', 'RH', 'TC']
}
DETECTOR_NAMES = ["ivt", "ivvt", "idt", "engbert", "nh", "remodnav"]
DEFAULT_DETECTORS = [
    peyes.create_detector(det, missing_value=np.nan, min_event_duration=4, pad_blinks_time=0) for det in DETECTOR_NAMES
]

###########################################


def get_default_output_dir(dataset_name: str) -> str:
    res = os.path.join(OUTPUT_DIR, dataset_name, DEFAULT_STR)
    os.makedirs(res, exist_ok=True)
    return res


def get_filename_for_labels(
        labels: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        prefix: str = "",
        suffix: str = "",
        extension: str = "pkl"
) -> str:
    if labels is None:
        return f"{prefix}_all_labels_{suffix}.{extension}"
    elif isinstance(labels, UnparsedEventLabelType):
        return f"{prefix}_{peyes.parse_label(labels).name.lower()}_{suffix}.{extension}"
    elif isinstance(labels, UnparsedEventLabelSequenceType):
        return f"{prefix}_{'_'.join([peyes.parse_label(l).name.lower() for l in labels])}_{suffix}.{extension}"
    else:
        raise TypeError(f"Unknown pos_labels type: {type(labels)}")
