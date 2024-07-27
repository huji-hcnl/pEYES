import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u


def get_data_impl(
        dataset_name: str,
        output_dir: str,
        data_dir_name: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        filename_prefix: str = "",
        filename_suffix: str = "",
        iteration: int = 1,
        stimulus_type: Optional[Union[str, List[str]]] = None,
        sub_index: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    data_dir_path = os.path.join(output_dir, dataset_name, data_dir_name)
    fullpath = os.path.join(data_dir_path, u.get_filename_for_labels(
        label, prefix=filename_prefix, suffix=filename_suffix, extension="pkl"
    ))
    try:
        data = pd.read_pickle(fullpath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find `{fullpath}`. Please preprocess the dataset first.")
    data = data.xs(iteration, level=peyes.constants.ITERATION_STR, axis=1)
    if stimulus_type:
        trial_ids = u.trial_ids_by_condition(dataset_name, key=peyes.constants.STIMULUS_TYPE_STR, values=stimulus_type)
        is_trial_ids = data.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(trial_ids)
        data = data.loc[:, is_trial_ids]
    if sub_index:
        data = data.loc[sub_index]
    return data
