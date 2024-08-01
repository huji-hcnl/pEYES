from typing import Optional, Union, Tuple, Sequence

import pandas as pd
import plotly.graph_objects as go

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.statistics._helpers as h

###################


def load(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        stimulus_type: Optional[Union[str, Sequence[str]]] = None,
        channel_type: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    return h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=peyes.constants.SAMPLES_CHANNEL_STR, label=label,
        filename_suffix="timing_differences", iteration=1, stimulus_type=stimulus_type,
        sub_index=channel_type
    )


def kruskal_wallis_dunns(
        data: pd.DataFrame,
        gt_cols: Union[str, Sequence[str]],
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return h.kruskal_wallis_dunns(data, gt_cols, multi_comp)


def distributions_figure(
        data: pd.DataFrame,
        gt1: str,
        gt2: str,
        title: str = "Samples Channel :: Difference Distributions",
        only_box: bool = False,
) -> go.Figure:
    return h.distributions_figure(data, gt1=gt1, gt2=gt2, title=title, only_box=only_box)
