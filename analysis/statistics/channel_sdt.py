from typing import Optional, Union, Tuple, Sequence, Dict

import numpy as np
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


def single_threshold_figure(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        threshold: int,
        gt1: str,
        metrics: Union[str, Sequence[str]] = None,
        title: str = "",
        gt2: Optional[str] = None,
        only_box: bool = False,
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
    fig = h.distributions_figure(sub_frame, gt1=gt1, gt2=gt2, title=title, only_box=only_box)
    return fig


def multi_threshold_figures(
        sdt_metrics: pd.DataFrame,
        channel_type: str,
        metrics: Union[str, Sequence[str]] = None,
        title: str = "",
        show_err_bands: bool = False
) -> Dict[str, go.Figure]:
    if metrics is None:
        metrics = [
            m for m in sdt_metrics.index.get_level_values(peyes.constants.METRIC_STR).unique()
            if m in u.METRICS_CONFIG.keys()
        ]
    subframe = _extract_sdt_subframe(sdt_metrics, channel_type, None, metrics)
    subframe = subframe.droplevel(peyes.constants.CHANNEL_TYPE_STR, axis=0)    # remove single-value levels from index
    gt_cols = subframe.columns.get_level_values(u.GT_STR).unique()
    figures = dict()
    for gt in gt_cols:
        # create figure for this GT labeler, with subplots for each metric
        gt_subframe = subframe.xs(gt, level=u.GT_STR, axis=1, drop_level=True)
        subframe_metrics = sorted(
            [m for m in gt_subframe.index.get_level_values(peyes.constants.METRIC_STR).unique()],
            key=lambda m: u.METRICS_CONFIG[m][1]
        )
        fig, nrows, ncols = h._make_empty_figure(subframe_metrics, sharex=False, sharey=False)
        for i, met in enumerate(subframe_metrics):
            r, c = (i, 0) if ncols == 1 else divmod(i, ncols)
            met_frame = gt_subframe.xs(met, level=peyes.constants.METRIC_STR, axis=0, drop_level=True)
            detectors = sorted(
                [d for d in met_frame.columns.get_level_values(u.PRED_STR).unique() if d not in gt_cols],
                key=lambda d: u.LABELERS_CONFIG[d.removesuffix("Detector").lower()][1]
            )
            for j, det in enumerate(detectors):
                met_det_frame = met_frame.xs(det, level=u.PRED_STR, axis=1, drop_level=True)
                thresholds = met_det_frame.index.get_level_values(peyes.constants.THRESHOLD_STR).unique()
                mean = met_det_frame.mean(axis=1)
                sem = met_det_frame.std(axis=1) / np.sqrt(met_det_frame.count(axis=1))
                det_name = det.strip().removesuffix("Detector")
                det_color = u.LABELERS_CONFIG[det_name.lower()][2]
                fig.add_trace(
                    row=r + 1, col=c + 1, trace=go.Scatter(
                        x=thresholds, y=mean, error_y=dict(type="data", array=sem),
                        name=det_name, legendgroup=det_name,
                        mode="lines+markers",
                        marker=dict(size=5, color=det_color),
                        line=dict(color=det_color),
                        showlegend=i == 0,
                    )
                )
                if show_err_bands:
                    y_upper, y_lower = mean + sem, mean - sem
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
