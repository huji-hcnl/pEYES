import os
from typing import List

import pandas as pd

import pEYES as peyes

import analysis.utils as u

_DEFAULT_STR = "default"


def get_default_output_dir(dataset_name: str) -> str:
    res = os.path.join(u.OUTPUT_DIR, dataset_name, _DEFAULT_STR)
    os.makedirs(res, exist_ok=True)
    return res


def check_labelers(data: pd.DataFrame, labelers: List[str] = None) -> List[str]:
    available_labelers = data.columns.get_level_values(peyes.constants.LABELER_STR).unique()
    labelers = set(labelers or available_labelers)
    unknown_labelers = labelers - set(available_labelers)
    if unknown_labelers:
        raise ValueError(f"Unknown labelers: {unknown_labelers}")
    return list(labelers)
