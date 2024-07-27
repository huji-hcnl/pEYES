import os
from typing import Optional, Union, Tuple, Sequence

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.statistics._helpers as h

pio.renderers.default = "browser"

###################


def get_time_diffs(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        stimulus_type: Optional[Union[str, Sequence[str]]] = None,
        channel_type: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    return h.get_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{peyes.constants.SAMPLES_STR}_{u.CHANNEL_STR}", label=label,
        filename_suffix="timing_differences", stimulus_type=stimulus_type,
        iteration=1, sub_index=channel_type
    )


def get_sdt_metrics(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        stimulus_type: Optional[Union[str, Sequence[str]]] = None,
        channel_type: Optional[str] = None,
        threshold: Optional[Union[int, Sequence[int]]] = None,
        metrics: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Loads channel SDT metrics and re-arranges data for simpler analysis.
    Output DataFrame has the following MultiIndex:
    Index:
        level 0: Channel type (onset/offset)
        level 1: Metric (recall, precision, etc.)
        level 2: Threshold (0, 1, 2, ...)
    Columns:
        level 0: Trial ID (1, 2, ...)
        level 1: GT labeler (human annotators, e.g. RA, MN)
        level 2: Pred labeler (detection algorithms, e.g. EngbertDetector, etc.)
    """
    sdt_metrics = h.get_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{peyes.constants.SAMPLES_STR}_{u.CHANNEL_STR}",
        filename_suffix="sdt_metrics", label=label,
        stimulus_type=stimulus_type, sub_index=None, iteration=1,
    )
    sdt_metrics = sdt_metrics.stack(level=peyes.constants.METRIC_STR, future_stack=True)
    sdt_metrics = sdt_metrics.reorder_levels(
        [u.CHANNEL_TYPE_STR, peyes.constants.METRIC_STR, peyes.constants.THRESHOLD_STR], axis=0
    )
    return _extract_sdt_subframe(sdt_metrics, channel_type, threshold, metrics)


def sdt_single_threshold_analysis(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        threshold: int,
        gt_cols: Sequence[str],
        metrics: Union[str, Sequence[str]] = None,
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub_frame = _extract_sdt_subframe(sdt_metrics, channel_type, threshold, metrics)
    sub_frame = sub_frame.droplevel(  # remove single-value levels from index
        level=[u.CHANNEL_TYPE_STR, peyes.constants.THRESHOLD_STR], axis=0
    )
    statistics, pvalues, dunns, Ns = h.statistical_analysis(sub_frame, gt_cols=gt_cols, multi_comp=multi_comp)
    return statistics, pvalues, dunns, Ns


def sdt_single_threshold_figure(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        threshold: int,
        gt1: str,
        metrics: Union[str, Sequence[str]] = None,
        title: str = "",
        gt2: Optional[str] = None,
        only_box: bool = False,
) -> go.Figure:
    sub_frame = _extract_sdt_subframe(sdt_metrics, channel_type, threshold, metrics)
    sub_frame = sub_frame.droplevel(  # remove single-value levels from index
        level=[u.CHANNEL_TYPE_STR, peyes.constants.THRESHOLD_STR], axis=0
    )
    title = title if title else (
            f"SDT Metrics <br><sup>(Channel:{channel_type}\tMax Difference: {threshold} samples</sup>"
    )
    fig = h.distributions_figure(sub_frame, gt1=gt1, gt2=gt2, title=title, only_box=only_box)
    return fig


def _extract_sdt_subframe(
        sdt_metrics: pd.DataFrame,
        channel_type: Optional[str] = None,
        threshold: Optional[Union[int, Sequence[int]]] = None,
        metrics: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Given a DataFrame with SDT metrics that has multi-level index (channel_type, metric, threshold), extract a sub-frame
    based on the provided parameters. Returns a DataFrame with the same index structure as the input DataFrame.

    :param sdt_metrics: pd.DataFrame; SDT metrics
    :param channel_type: str; channel type (onset/offset)
    :param threshold: int or list of int; threshold value(s)
    :param metrics: str or list of str; metric(s) to extract

    :return: pd.DataFrame; sub-frame with SDT metrics
    """
    sub_frame = sdt_metrics
    if channel_type:
        channel_type = channel_type.lower().strip()
        sub_frame = sub_frame.xs(channel_type, level=u.CHANNEL_TYPE_STR, axis=0, drop_level=False)
    if threshold is None or (isinstance(threshold, Sequence) and len(threshold) == 0):
        threshold = sub_frame.index.get_level_values(peyes.constants.THRESHOLD_STR).unique().tolist()
    if isinstance(threshold, int):
        sub_frame = sub_frame.xs(key=threshold, level=peyes.constants.THRESHOLD_STR, axis=0, drop_level=False)
    else:
        assert len(threshold) > 0
        is_threshold = sub_frame.index.get_level_values(peyes.constants.THRESHOLD_STR).isin(threshold)
        sub_frame = sub_frame.loc[is_threshold]
    if metrics is None or len(metrics) == 0:
        metrics = sdt_metrics.index.get_level_values(peyes.constants.METRIC_STR).unique().tolist()
    elif isinstance(metrics, str):
        metrics = [metrics]
    metrics = [m.lower().strip() for m in metrics if m in u.METRICS_CONFIG.keys()]
    is_metrics = sub_frame.index.get_level_values(peyes.constants.METRIC_STR).isin(metrics)
    sub_frame = sub_frame.loc[is_metrics]
    return sub_frame.sort_index()


###################

DATASET_NAME = "lund2013"
GT1, GT2 = "RA", "MN"
MULTI_COMP = "fdr_bh"

##################
##  Time Diffs  ##

# time_diffs = get_time_diffs(
#     DATASET_NAME, os.path.join(u.OUTPUT_DIR, "default_values"), label=None, stimulus_type=peyes.constants.IMAGE_STR
# )
#
# statistics, pvalues, dunns, Ns = h.statistical_analysis(time_diffs, ["RA", "MN"], multi_comp=MULTI_COMP)
# time_diffs_fig = h.distributions_figure(time_diffs, GT1, gt2=GT2, title=f"Time Differences", only_box=False)
# time_diffs_fig.show()

###################
##  SDT Metrics  ##

sdt_metrics = get_sdt_metrics(
    dataset_name=DATASET_NAME,
    output_dir=os.path.join(u.OUTPUT_DIR, "default_values"),
    label=None,
    stimulus_type=peyes.constants.IMAGE_STR,
    channel_type=None
)

subframe = _extract_sdt_subframe(sdt_metrics, "onset", None, None).droplevel(u.CHANNEL_TYPE_STR, axis=0)
gt1_subframe = (subframe.loc[:, subframe.columns.get_level_values(u.GT_STR) == GT1]).droplevel(u.GT_STR, axis=1)
metrics = sorted([
    m for m in gt1_subframe.index.get_level_values(peyes.constants.METRIC_STR).unique() if m in u.METRICS_CONFIG.keys()
], key=lambda m: u.METRICS_CONFIG[m][1])

fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics, shared_xaxes=False, shared_yaxes=False)

for i, met in enumerate(metrics):
    r, c = i+1, 1
    met_frame = gt1_subframe.xs(met, level=peyes.constants.METRIC_STR, axis=0, drop_level=True)
    detectors = met_frame.columns.get_level_values(u.PRED_STR).unique()
    for j, det in enumerate(detectors):
        if det in {GT1, GT2}:
            continue
        met_det_frame = (met_frame.loc[:, met_frame.columns.get_level_values(u.PRED_STR) == det]).droplevel(u.PRED_STR, axis=1)
        thresholds = met_det_frame.index.get_level_values(peyes.constants.THRESHOLD_STR).unique()
        mean, std, n = met_det_frame.mean(axis=1), met_det_frame.std(axis=1), met_det_frame.count(axis=1)
        sem = std / np.sqrt(n)
        det_name = det.strip().removesuffix("Detector")
        det_color = u.DETECTORS_CONFIG[det_name.lower()][2]
        fig.add_trace(
            row=r, col=c, trace=go.Scatter(
                x=thresholds, y=met_det_frame.mean(axis=1),
                error_y=dict(type="data", array=sem),
                name=det_name, legendgroup=det_name,
                mode="lines+markers",
                marker=dict(size=5, color=det_color),
                line=dict(color=det_color),
                showlegend=i == 0,
            )
        )

fig.show()

del i, met, r, c, met_frame, detectors, j, det, met_det_frame, thresholds, mean, std, n, sem, det_name, det_color
# TODO: finish the above figure and write it as a function
