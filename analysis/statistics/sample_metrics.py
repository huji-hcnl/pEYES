from typing import List, Optional, Union, Tuple, Sequence

import pandas as pd
import plotly.graph_objects as go

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.statistics._helpers as h

###################


def load(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        stimulus_type: Optional[Union[str, List[str]]] = None,
        metric: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    return h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}", label=label,
        iteration=1, stimulus_type=stimulus_type, sub_index=metric
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
        title: str = "Samples :: Metric Distributions",
        only_box: bool = False,
) -> go.Figure:
    return h.distributions_figure(data, gt1=gt1, gt2=gt2, title=title, only_box=only_box)
