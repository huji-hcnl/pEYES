from typing import List, Optional, Union, Tuple, Sequence

import pandas as pd
import plotly.graph_objects as go

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.statistics._helpers as h

###################


def load_sdt(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        stimulus_type: Optional[Union[str, List[str]]] = None,
        metric: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    return h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}",
        filename_suffix=f"{u.SDT_STR}_{peyes.constants.METRICS_STR}", label=label,
        iteration=1, stimulus_type=stimulus_type, sub_index=metric,
    )


def load_global_metrics(
        dataset_name: str,
        output_dir: str,
        stimulus_type: Optional[Union[str, List[str]]] = None,
        metric: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    return h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}",
        filename_suffix=f"{u.GLOBAL_STR}_{peyes.constants.METRICS_STR}", label=None,
        iteration=1, stimulus_type=stimulus_type, sub_index=metric,
    )


def kruskal_wallis_dunns(
        data: pd.DataFrame,
        gt_cols: Union[str, Sequence[str]],
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return h.kruskal_wallis_dunns(data, gt_cols, multi_comp)


def sdt_distributions_figure(
        data: pd.DataFrame,
        gt1: str,
        gt2: str,
        title: str = "",
        only_box: bool = False,
) -> go.Figure:
    title = title or "Samples :: SDT Metrics Distributions"
    return h.distributions_figure(data, gt1=gt1, gt2=gt2, title=title, only_box=only_box)


def global_metrics_distributions_figure(
        data: pd.DataFrame,
        gt1: str,
        gt2: str,
        title: str = "",
        only_box: bool = False,
) -> go.Figure:
    title = title or "Samples :: Global Metrics Distributions <br><sup>{peyes.constants.LABEL_STR.title()}:All</sup>"
    return h.distributions_figure(data, gt1=gt1, gt2=gt2, title=title, only_box=only_box)


# def load(
#         dataset_name: str,
#         output_dir: str,
#         label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
#         stimulus_type: Optional[Union[str, List[str]]] = None,
#         metric: Optional[Union[str, List[str]]] = None,
# ) -> pd.DataFrame:
#     return h.load_data(
#         dataset_name=dataset_name, output_dir=output_dir,
#         data_dir_name=f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}", label=label,
#         iteration=1, stimulus_type=stimulus_type, sub_index=metric
#     )


# def distributions_figure(
#         data: pd.DataFrame,
#         gt1: str,
#         gt2: str,
#         title: str = "Samples :: Metric Distributions",
#         only_box: bool = False,
# ) -> go.Figure:
#     return h.distributions_figure(data, gt1=gt1, gt2=gt2, title=title, only_box=only_box)
