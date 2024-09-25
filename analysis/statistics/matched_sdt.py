from typing import Optional, Union, Sequence, Tuple, Dict

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
        matching_schemes: Optional[Union[str, Sequence[str]]] = None,
        metrics: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Loads matched-events' SDT metrics and extracts a sub-frame based on the provided parameters.
    Output DataFrame has the following MultiIndex:
    Index:
        level 0: Matching scheme (e.g. window_10, iou_0.5) - matching scheme & threshold
        level 1: Metric (dprime, recall, etc.)
    Columns:
        level 0: Trial ID (1, 2, ...)
        level 1: GT labeler (human annotators, e.g. RA, MN)
        level 2: Pred labeler (detection algorithms, e.g. EngbertDetector, etc.)
    """
    matches_sdt = h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{peyes.constants.MATCHES_STR}_{peyes.constants.METRICS_STR}", label=label,
        filename_suffix="sdt_metrics", iteration=1, stimulus_type=stimulus_type, sub_index=None
    )
    return _extract_sdt_subframe(matches_sdt, matching_schemes, metrics)


def kruskal_wallis_dunns(
        matches_sdt: pd.DataFrame,
        matching_scheme: str,
        gt_cols: Union[str, Sequence[str]],
        metrics: Union[str, Sequence[str]] = None,
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sub_frame = _extract_sdt_subframe(matches_sdt, matching_scheme, metrics)
    sub_frame = sub_frame.droplevel(level=[u.MATCHING_SCHEME_STR], axis=0)  # remove single-value levels from index
    statistics, pvalues, dunns, Ns = h.kruskal_wallis_dunns(sub_frame, gt_cols=gt_cols, multi_comp=multi_comp)
    return statistics, pvalues, dunns, Ns


def single_scheme_figure(
        matches_sdt: pd.DataFrame,
        matching_scheme: str,
        gt1: str,
        metrics: Union[str, Sequence[str]] = None,
        title: str = "",
        gt2: Optional[str] = None,
        only_box: bool = False,
) -> go.Figure:
    if metrics is None:
        metrics = [
            m for m in matches_sdt.index.get_level_values(peyes.constants.METRIC_STR).unique()
            if m in u.METRICS_CONFIG.keys()
        ]
    sub_frame = _extract_sdt_subframe(matches_sdt, matching_scheme, metrics)
    sub_frame = sub_frame.droplevel(level=[u.MATCHING_SCHEME_STR], axis=0)  # remove single-value levels from index
    if not title:
        split_scheme = matching_scheme.split("_")
        scheme_name, scheme_threshold = "_".join(split_scheme[:-1]), int(split_scheme[-1])
        threshold_str = f"{scheme_threshold:.2f}(a.u.)" if isinstance(scheme_threshold, float) else f"{scheme_threshold} (samples)"
        title = ("Matched Events :: SDT Metrics <br>" +
                 f"<sup>({u.MATCHING_SCHEME_STR.replace("_", " ").title()}: {scheme_name.replace("_", " ").title()}  " +
                 f"Threshold: {threshold_str}</sup>")
    fig = h.distributions_figure(sub_frame, gt1=gt1, gt2=gt2, title=title, only_box=only_box)
    return fig


def multi_threshold_figures(
        matches_sdt: pd.DataFrame,
        matching_scheme: str,
        metrics: Union[str, Sequence[str]] = None,
        title: str = "",
        show_other_gt: bool = True,
        show_err_bands: bool = False
) -> Dict[str, go.Figure]:
    all_schemes = sorted(
        [ms for ms in matches_sdt.index.get_level_values(u.MATCHING_SCHEME_STR).unique() if ms.startswith(matching_scheme)],
        key=lambda ms: int(ms.split("_")[-1])
    )
    if metrics is None:
        metrics = [
            m for m in matches_sdt.index.get_level_values(peyes.constants.METRIC_STR).unique()
            if m in u.METRICS_CONFIG.keys()
        ]
    subframe = _extract_sdt_subframe(matches_sdt, all_schemes, metrics)
    subframe.sort_index(
        level=subframe.index.names.index(u.MATCHING_SCHEME_STR),
        key=lambda x: [all_schemes.index(s) for s in x],
        axis=0, inplace=True,
    )
    gt_cols = subframe.columns.get_level_values(u.GT_STR).unique()
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
            detectors = sorted(
                [d for d in met_frame.columns.get_level_values(u.PRED_STR).unique()],
                key=lambda d: u.LABELERS_CONFIG[d.removesuffix("Detector").lower()][1]
            )
            for j, det in enumerate(detectors):
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
                    det_color = u.LABELERS_CONFIG[det_name.lower()][2]
                    dash = None
                met_det_frame = met_frame.xs(det, level=u.PRED_STR, axis=1, drop_level=True)
                # TODO: the following line is the only difference between this func & the similar one in channel_sdt.py,
                #  if we change how the thresholds are stored in the index, we could change this line
                thresholds = met_det_frame.index.to_series(name=peyes.constants.THRESHOLD_STR).apply(lambda ms: int(ms.split("_")[-1]))
                mean = met_det_frame.mean(axis=1)
                sem = met_det_frame.std(axis=1) / np.sqrt(met_det_frame.count(axis=1))
                fig.add_trace(
                    row=r + 1, col=c + 1, trace=go.Scatter(
                        x=thresholds, y=mean, error_y=dict(type="data", array=sem),
                        name=det_name, legendgroup=det_name,
                        mode="lines+markers",
                        marker=dict(size=5, color=det_color),
                        line=dict(color=det_color, dash=dash),
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
                "Matched Events :: SDT Metrics <br>" + f"<sup>(GT: {gt}  " +
                f"{u.MATCHING_SCHEME_STR.replace("_", " ").title()}: {matching_scheme.replace("_", " ").title()})</sup>"
        )
        fig.update_layout(title=title)
        figures[gt] = fig
    return figures


def multi_metric_figure(
        matches_sdt: pd.DataFrame,
        matching_scheme: str,
        metrics: Union[str, Sequence[str]] = None,
        title: str = "",
        show_other_gt: bool = True,
        show_err_bands: bool = False,
) -> go.Figure:
    all_schemes = sorted(
        [ms for ms in matches_sdt.index.get_level_values(u.MATCHING_SCHEME_STR).unique() if
         ms.startswith(matching_scheme)],
        key=lambda ms: int(ms.split("_")[-1])
    )
    if metrics is None:
        metrics = [
            m for m in matches_sdt.index.get_level_values(peyes.constants.METRIC_STR).unique()
            if m in u.METRICS_CONFIG.keys()
        ]
    subframe = _extract_sdt_subframe(matches_sdt, all_schemes, metrics)
    subframe.sort_index(
        level=subframe.index.names.index(u.MATCHING_SCHEME_STR),
        key=lambda x: [all_schemes.index(s) for s in x],
        axis=0, inplace=True,
    )
    gt_cols = subframe.columns.get_level_values(u.GT_STR).unique()
    assert 0 < len(gt_cols) <= 2, "Only 1 or 2 GT labelers are allowed for multi-channel figure"
    fig = make_subplots(
        rows=len(metrics), cols=len(gt_cols), shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.05, horizontal_spacing=0.01,
        row_titles=list(map(lambda met: met.replace("_", " ").title(), metrics)),
        column_titles=list(map(lambda gt: gt.upper(), gt_cols)),
    )
    for r, met in enumerate(metrics):
        for c, gt in enumerate(gt_cols):
            data = subframe.xs(
                met, level=peyes.constants.METRIC_STR, axis=0, drop_level=True).xs(
                gt, level=u.GT_STR, axis=1, drop_level=True
            )
            detectors = sorted(
                [d for d in data.columns.get_level_values(u.PRED_STR).unique()],
                key=lambda d: u.LABELERS_CONFIG[d.removesuffix("Detector").lower()][1]
            )
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
                    det_color = u.LABELERS_CONFIG[det_name.lower()][2]
                    dash = None

                det_data = data.xs(det, level=u.PRED_STR, axis=1)
                # TODO: the following line is the only difference between this func & the similar one in channel_sdt.py,
                #  if we change how the thresholds are stored in the index, we could change this line
                thresholds = det_data.index.to_series(name=peyes.constants.THRESHOLD_STR).apply(lambda ms: int(ms.split("_")[-1]))
                mean = det_data.mean(axis=1)
                sem = det_data.std(axis=1) / np.sqrt(det_data.count(axis=1))
                fig.add_trace(
                    go.Scatter(
                        x=thresholds, y=mean, error_y=dict(type="data", array=sem),
                        name=det_name, legendgroup=det_name,
                        mode="lines+markers",
                        marker=dict(size=5, color=det_color),
                        line=dict(color=det_color, dash=dash),
                        showlegend=r == 0 and c == 0,
                    ),
                    row=r + 1, col=c + 1
                )
                if show_err_bands:
                    y_upper, y_lower = mean + sem, mean - sem
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate((thresholds, thresholds[::-1])),
                            y=np.concatenate((y_upper, y_lower[::-1])),
                            fill="toself", fillcolor=det_color, opacity=0.2,
                            line=dict(color=det_color, width=0),
                            name=det_name, legendgroup=det_name,
                            showlegend=False, hoverinfo="skip",
                        ),
                        row=r + 1, col=c + 1
                    )
            # fig.update_yaxes(
            #     range=u.METRICS_CONFIG[met][2] if met in u.METRICS_CONFIG else None,
            #     row=r + 1, col=c + 1
            # )
            if r == len(metrics) - 1:
                fig.update_xaxes(title_text="Threshold (samples)", row=r + 1, col=c + 1)
            # if c == len(gt_cols) - 1:
            #     # TODO: update y-axis range, see https://stackoverflow.com/q/79023648/8543025
            #     y_range = u.METRICS_CONFIG[met][2] if met in u.METRICS_CONFIG else None
            #     fig.update_yaxes(range=y_range, row=r + 1, col=c + 1)
    fig.update_layout(
        title=title if title else (
                "Matched Events :: Metrics for Increasing Thresholds <br>" +
                f"<sup>({u.MATCHING_SCHEME_STR.replace('_', ' ').title()}: " +
                f"{matching_scheme.replace('_', ' ').title()})</sup>"
        ),
        legend=dict(orientation="h", yanchor="top", y=1.04, xanchor="left", x=0.3),
    )
    return fig


def _extract_sdt_subframe(
        matches_sdt: pd.DataFrame,
        matching_schemes: Optional[Union[str, Sequence[str]]] = None,
        metrics: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Given a DataFrame of matched-events' SDT metrics that has multi-level index (channel_type, metric, threshold),
    extract a sub-frame based on the provided parameters. Returns a DataFrame with the same index structure as the
    input DataFrame.

    :param matches_sdt: pd.DataFrame; SDT metrics
    :param matching_schemes: str or list of str; name of the matching scheme(s) to extract (e.g. `window_10`, `iou_0.5`)
    :param metrics: str or list of str; metric(s) to extract (e.g. `dprime`, `recall`)

    :return: pd.DataFrame; sub-frame with SDT metrics
    """
    sub_frame = matches_sdt
    if matching_schemes:
        sub_frame = h.extract_subframe(
            sub_frame, level=u.MATCHING_SCHEME_STR, value=matching_schemes, axis=0, drop_single_values=False
        )
    if metrics:
        sub_frame = h.extract_subframe(
            sub_frame, level=peyes.constants.METRIC_STR, value=metrics, axis=0, drop_single_values=False
        )
    return sub_frame
