from typing import Optional, Union, Tuple, Sequence, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import peyes

import analysis.utils as u
import analysis.statistics._helpers as h
from peyes._utils.visualization_utils import make_empty_figure
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

###################


def load(
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
    sdt_metrics = h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=peyes.constants.SAMPLES_CHANNEL_STR, label=label,
        filename_suffix="sdt_metrics", iteration=1, stimulus_type=stimulus_type, sub_index=None
    )
    sdt_metrics = sdt_metrics.stack(level=peyes.constants.METRIC_STR, future_stack=True)
    sdt_metrics = sdt_metrics.reorder_levels(
        [peyes.constants.CHANNEL_TYPE_STR, peyes.constants.METRIC_STR, peyes.constants.THRESHOLD_STR], axis=0
    )
    return _extract_sdt_subframe(sdt_metrics, channel_type, threshold, metrics)


def wilcoxon(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        threshold: int,
        gt_cols: Union[str, Sequence[str]],
        metrics: Union[str, Sequence[str]] = None,
        alternative: str = "two-sided",
        method: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub_frame = _extract_sdt_subframe(sdt_metrics, channel_type, threshold, metrics)
    sub_frame = sub_frame.droplevel(  # remove single-value levels from index
        level=[peyes.constants.CHANNEL_TYPE_STR, peyes.constants.THRESHOLD_STR], axis=0
    )
    statistics, pvalues, Ns = h.wilcoxon(sub_frame, gt_cols=gt_cols, alternative=alternative, method=method)
    return statistics, pvalues, Ns



def kruskal_wallis_dunns(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        threshold: int,
        gt_cols: Union[str, Sequence[str]],
        metrics: Union[str, Sequence[str]] = None,
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub_frame = _extract_sdt_subframe(sdt_metrics, channel_type, threshold, metrics)
    sub_frame = sub_frame.droplevel(  # remove single-value levels from index
        level=[peyes.constants.CHANNEL_TYPE_STR, peyes.constants.THRESHOLD_STR], axis=0
    )
    statistics, pvalues, dunns, Ns = h.kruskal_wallis_dunns(sub_frame, gt_cols=gt_cols, multi_comp=multi_comp)
    return statistics, pvalues, dunns, Ns


def friedman_nemenyi(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        threshold: int,
        gt_cols: Union[str, Sequence[str]],
        metrics: Union[str, Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub_frame = _extract_sdt_subframe(sdt_metrics, channel_type, threshold, metrics)
    sub_frame = sub_frame.droplevel(  # remove single-value levels from index
        level=[peyes.constants.CHANNEL_TYPE_STR, peyes.constants.THRESHOLD_STR], axis=0
    )
    statistics, pvalues, nemenyi, Ns = h.friedman_nemenyi(sub_frame, gt_cols=gt_cols)
    return statistics, pvalues, nemenyi, Ns


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


def single_threshold_figure(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        threshold: int,
        gt1: str,
        metrics: Union[str, Sequence[str]] = None,
        colors: u.COLORMAP_TYPE = None,
        title: str = "",
        gt2: Optional[str] = None,
        only_box: bool = False,
        show_other_gt: bool = False,
        share_x: bool = False,
        share_y: bool = False,
) -> go.Figure:
    if metrics is None:
        metrics = [
            m for m in sdt_metrics.index.get_level_values(peyes.constants.METRIC_STR).unique()
            if m in u.METRICS_CONFIG.keys()
        ]
    sub_frame = _extract_sdt_subframe(sdt_metrics, channel_type, threshold, metrics)
    sub_frame = sub_frame.droplevel(  # remove single-value levels from index
        level=[peyes.constants.CHANNEL_TYPE_STR, peyes.constants.THRESHOLD_STR], axis=0
    )
    title = title if title else (
        "Samples Channel :: SDT Metrics <br>" +
        f"<sup>(Channel: {channel_type}  Max Difference: {threshold} samples)</sup>"
    )
    fig = h.distributions_figure(
        sub_frame, gt1=gt1, gt2=gt2, colors=colors, title=title,
        only_box=only_box, show_other_gt=show_other_gt, share_x=share_x, share_y=share_y,
    )
    return fig


def multi_threshold_figures(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        metrics: Union[str, Sequence[str]] = None,
        title: str = "",
        error_bars: Optional[str] = None,
        colors: u.COLORMAP_TYPE = None,
        show_other_gt: bool = True,
        show_err_bands: bool = False,
) -> Dict[str, go.Figure]:
    if metrics is None:
        metrics = [
            m for m in sdt_metrics.index.get_level_values(peyes.constants.METRIC_STR).unique()
            if m in u.METRICS_CONFIG.keys()
        ]
    subframe = _extract_sdt_subframe(sdt_metrics, channel_type, None, metrics)
    subframe = subframe.droplevel(peyes.constants.CHANNEL_TYPE_STR, axis=0)    # remove single-value levels from index
    gt_cols = subframe.columns.get_level_values(u.GT_STR).unique()
    assert 0 < len(gt_cols) <= 2, "Only 1 or 2 GT labelers are allowed for multi-threshold figures"
    figures = dict()
    for gt in gt_cols:
        # create figure for this GT labeler, with subplots for each metric
        gt_subframe = subframe.xs(gt, level=u.GT_STR, axis=1, drop_level=True)
        subframe_metrics = sorted(
            [m for m in gt_subframe.index.get_level_values(peyes.constants.METRIC_STR).unique()],
            key=lambda m: u.METRICS_CONFIG[m][1]
        )
        fig, nrows, ncols = make_empty_figure(
            subtitles=list(map(lambda met: u.METRICS_CONFIG[met][0] if met in u.METRICS_CONFIG else met, subframe_metrics)),
            sharex=False, sharey=False,
        )
        for i, met in enumerate(subframe_metrics):
            r, c = (i, 0) if ncols == 1 else divmod(i, ncols)
            met_frame = gt_subframe.xs(met, level=peyes.constants.METRIC_STR, axis=0, drop_level=True)
            detectors = u.sort_labelers(met_frame.columns.get_level_values(u.PRED_STR).unique())
            for j, det in enumerate(detectors):
                if det not in gt_cols:
                    # current detector is a prediction labeler (detection algorithm)
                    det_name = det.strip().removesuffix("Detector")
                    det_color = u.get_labeler_color(det_name, j, colors)
                    dash = None
                if det in gt_cols and show_other_gt:
                    # current detector is a GT labeler, and we want to refer to it as "Other GT"
                    det_name = "Other GT"
                    det_color = "#bab0ac"
                    dash = "dot"
                met_det_frame = met_frame.xs(det, level=u.PRED_STR, axis=1, drop_level=True)
                thresholds = met_det_frame.index.get_level_values(peyes.constants.THRESHOLD_STR).unique()
                mean = met_det_frame.mean(axis=1)
                errors = h.calc_error_bars(met_det_frame, error_bars)
                fig.add_trace(
                    row=r + 1, col=c + 1, trace=go.Scatter(
                        x=thresholds, y=mean, error_y=dict(type="data", array=errors),
                        name=det_name, legendgroup=det_name,
                        mode="lines+markers",
                        marker=dict(size=5, color=det_color),
                        line=dict(color=det_color, dash=dash),
                        showlegend=i == 0,
                    )
                )
                if show_err_bands and errors is not None and not errors.isna().all():
                    y_upper, y_lower = mean + errors, mean - errors
                    fig.add_trace(
                        row=r + 1, col=c + 1, trace=go.Scatter(
                            x=np.concatenate((thresholds, thresholds[::-1])),
                            y=np.concatenate((y_upper, y_lower[::-1])),
                            fill="toself", fillcolor=det_color, opacity=0.2,
                            line=dict(color=det_color, width=0),
                            name=det_name, legendgroup=det_name, showlegend=False, hoverinfo="skip",
                        )
                    )
            y_range = u.METRICS_CONFIG[met][2] if met in u.METRICS_CONFIG else None
            fig.update_yaxes(row=r + 1, col=c + 1, range=y_range)
        title = title if title else (
                "Samples Channel :: SDT Metrics <br>" + f"<sup>(GT: {gt}  Channel: {channel_type})</sup>"
        )
        fig.update_layout(title=title)
        figures[gt] = fig
    return figures


def multi_channel_figure(
        sdt_metrics: pd.DataFrame,
        metric: str,
        title: str = "",
        yaxis_title: str = "",
        error_bars: Optional[str] = None,
        colors: u.COLORMAP_TYPE = None,
        show_other_gt: bool = True,
        show_err_bands: bool = False,
) -> go.Figure:
    subframe = _extract_sdt_subframe(sdt_metrics, metrics=metric, channel_type=None, threshold=None)
    subframe = subframe.droplevel(peyes.constants.METRIC_STR, axis=0)  # remove single-value levels from index
    gt_cols = subframe.columns.get_level_values(u.GT_STR).unique()
    assert 0 < len(gt_cols) <= 2, "Only 1 or 2 GT labelers are allowed for multi-channel figure"
    channel_types = subframe.index.get_level_values(peyes.constants.CHANNEL_TYPE_STR).unique()
    fig = make_subplots(
        rows=len(channel_types), cols=len(gt_cols), shared_xaxes="all", shared_yaxes="all",
        vertical_spacing=0.025, horizontal_spacing=0.01,
        row_titles=list(map(lambda ch: ch.title(), channel_types)),
        column_titles=list(map(lambda gt: gt.upper(), gt_cols)),
    )
    for r, ch_type in enumerate(channel_types):
        for c, gt in enumerate(gt_cols):
            data = subframe.xs(ch_type, level=peyes.constants.CHANNEL_TYPE_STR, axis=0).xs(gt, level=u.GT_STR, axis=1)
            detectors = u.sort_labelers(data.columns.get_level_values(u.PRED_STR).unique())
            for k, det in enumerate(detectors):
                if det in gt_cols:
                    if show_other_gt:
                        # current detector is a GT labeler, and we want to refer to it as "Other GT"
                        det_name = "Other GT"
                        det_color = "#bab0ac"
                        dash = "dot"
                    else:
                        # current detector is a GT labeler, and we don't want to show it in the figure
                        continue
                else:
                    # current detector is a prediction labeler (detection algorithm)
                    det_name = det.strip().removesuffix("Detector")
                    det_color = u.get_labeler_color(det_name, k, colors)
                    dash = None
                det_data = data.xs(det, level=u.PRED_STR, axis=1)
                thresholds = det_data.index.get_level_values(peyes.constants.THRESHOLD_STR).unique()
                mean = det_data.mean(axis=1)
                errors = h.calc_error_bars(det_data, error_bars)
                fig.add_trace(
                    row=r + 1, col=c + 1, trace=go.Scatter(
                        x=thresholds, y=mean, error_y=dict(type="data", array=errors),
                        name=det_name, legendgroup=det_name,
                        mode="lines+markers",
                        marker=dict(size=5, color=det_color),
                        line=dict(color=det_color, dash=dash),
                        showlegend=(c == 0 and r == 0),
                    )
                )
                if show_err_bands and errors is not None and not errors.isna().all():
                    y_upper, y_lower = mean + errors, mean - errors
                    fig.add_trace(
                        row=r + 1, col=c + 1, trace=go.Scatter(
                            x=np.concatenate((thresholds, thresholds[::-1])),
                            y=np.concatenate((y_upper, y_lower[::-1])),
                            fill="toself", fillcolor=det_color, opacity=0.2,
                            line=dict(color=det_color, width=0),
                            name=det_name, legendgroup=det_name, showlegend=False, hoverinfo="skip",
                        )
                    )
            if c == 0:
                yaxis_title = yaxis_title if yaxis_title else metric.replace("_", " ").lower()
                fig.update_yaxes(title_text=yaxis_title, row=r + 1, col=1)
            if r == len(channel_types) - 1:
                fig.update_xaxes(title_text="Threshold (samples)", row=len(channel_types), col=c + 1)
    fig.update_layout(
        title=title if title else f"Samples Channel :: {metric.replace("_", " ").title()} for Increasing Thresholds",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0.25),
    )
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
        sub_frame = h.extract_subframe(
            sub_frame, level=peyes.constants.CHANNEL_TYPE_STR, value=channel_type, axis=0, drop_single_values=False
        )
    if threshold:
        sub_frame = h.extract_subframe(
            sub_frame, level=peyes.constants.THRESHOLD_STR, value=threshold, axis=0, drop_single_values=False
        )
    if metrics:
        sub_frame = h.extract_subframe(
            sub_frame, level=peyes.constants.METRIC_STR, value=metrics, axis=0, drop_single_values=False
        )
    return sub_frame
