import os
from typing import List

import pandas as pd

import analysis.utils as u


def get_default_output_dir(dataset_name: str) -> str:
    res = os.path.join(u.OUTPUT_DIR, dataset_name, u.DEFAULT_STR)
    os.makedirs(res, exist_ok=True)
    return res


def check_labelers(data: pd.DataFrame, labelers: List[str] = None) -> List[str]:
    available_labelers = data.columns.get_level_values(u.LABELER_STR).unique()
    labelers = set(labelers or available_labelers)
    unknown_labelers = labelers - set(available_labelers)
    if unknown_labelers:
        raise ValueError(f"Unknown labelers: {unknown_labelers}")
    return list(labelers)
