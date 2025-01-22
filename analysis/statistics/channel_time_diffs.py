from typing import Optional, Union, Tuple, Sequence

import pandas as pd
import plotly.graph_objects as go

import peyes
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

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


def friedman_nemenyi(
        data: pd.DataFrame,
        gt_cols: Union[str, Sequence[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return h.friedman_nemenyi(data, gt_cols)


def post_hoc_table(
        ph_data: pd.DataFrame,
        metric: str,
        gt_cols: Union[str, Sequence[str]],
        alpha: float = 0.05,
        marginal_alpha: Optional[float] = 0.075,
) -> pd.DataFrame:
    if isinstance(gt_cols, str):
        gt_cols = [gt_cols]
    return h.create_post_hoc_table(ph_data, metric, *gt_cols, alpha=alpha, marginal_alpha=marginal_alpha)


def distributions_figure(
        data: pd.DataFrame,
        gt1: str,
        gt2: str,
        colors: u.COLORMAP_TYPE = None,
        title: str = "Samples Channel :: Difference Distributions",
        only_box: bool = False,
        show_other_gt: bool = False,
        share_x: bool = False,
        share_y: bool = False,
        subplots_vspace: float = 0.1,
        subplots_hspace: float = 0.1,
) -> go.Figure:
    fig = h.distributions_figure(
        data, gt1=gt1, gt2=gt2, colors=colors, title=title,
        only_box=only_box, show_other_gt=show_other_gt,
        share_x=share_x, share_y=share_y,
        subplots_vspace=subplots_vspace, subplots_hspace=subplots_hspace,
    )
    fig.update_yaxes(
        title=f"{peyes.constants.TIME_STR} {peyes.constants.DIFFERENCE_STR} ({peyes.constants.SAMPLES_STR})"
    )
    return fig
