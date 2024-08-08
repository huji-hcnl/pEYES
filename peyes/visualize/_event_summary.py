from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import peyes._utils.constants as cnst
import peyes._utils.visualization_utils as vis_utils
from peyes._utils.event_utils import parse_label
from peyes.event_metrics import features_by_labels
from peyes import summarize_events

from peyes._DataModels.Event import EventSequenceType
from peyes._DataModels.EventLabelEnum import EventLabelEnum

__INLIERS_STR, __OUTLIERS_STR = "inliers", "outliers"


def event_summary(events: EventSequenceType, **kwargs) -> go.Figure:
    """
    Creates a summary figure of the provided events. The figure contains four panels:
        1. A bar plot of the count of each event label.
        2. A violin plot showing the distribution of durations, per event label.
        3. A violin plot showing the distribution of amplitudes, per event label.
        4. A violin plot showing the distribution of peak velocities, per event label.
        5. A violin plot showing the distribution of areas of the ellipse bound each event, per event label.

    :param events: list (or other sequence) of events.
    :keyword label_colors: A dictionary mapping event labels to their respective colors. If a label is missing, the
        default color is used.
    :keyword show_outliers: whether to show outliers in the violin plots; default is False.
    :keyword outlier_opacity: if `show_outliers` is true, this controls the opacity of the outliers; default is 0.5.
    :keyword title: the title of the figure; default is "Event Summary".

    :return: the figure.
    """
    inlier_features = features_by_labels(list(filter(lambda e: not e.is_outlier, events)))
    outlier_features = features_by_labels(list(filter(lambda e: e.is_outlier, events)))
    assert inlier_features.index.equals(outlier_features.index)
    event_labels = [parse_label(l) for l in inlier_features.index]
    label_colors = vis_utils.get_label_colormap(kwargs.get("label_colors", None))
    show_outliers = kwargs.get("show_outliers", False)
    subplots = {
        cnst.COUNT_STR: '# instances',
        cnst.DURATION_STR: 'ms',
        cnst.AMPLITUDE_STR: 'deg',
        cnst.PEAK_VELOCITY_STR: 'deg/s',
        cnst.ELLIPSE_AREA_STR: 'deg^2',
    }
    fig = make_subplots(
        cols=1, rows=len(subplots), shared_xaxes=True,
        subplot_titles=list(map(lambda met: met.replace('_', ' ').title(), subplots.keys()))
    )
    for r, (metric, measure) in enumerate(subplots.items()):
        fig.update_yaxes(col=1, row=r + 1, title_text=measure)
        for evnt in event_labels:
            outlier_trace = None
            color = f"rgb{label_colors[evnt]}"
            if metric == cnst.COUNT_STR:
                inlier_trace = go.Bar(
                    x0=evnt.name,
                    y=[inlier_features.loc[evnt, metric]],
                    marker_color=color,
                    name=evnt.name, legendgroup=evnt.name, showlegend=True,
                )
                if show_outliers:
                    outlier_trace = go.Bar(
                        x0=evnt.name,
                        y=[outlier_features.loc[evnt, metric]],
                        marker_color=color, opacity=kwargs.get("outlier_opacity", 0.5),
                        name=f"{evnt.name} ({__OUTLIERS_STR})", legendgroup=evnt.name, showlegend=False,
                    )
            else:
                inlier_trace = go.Violin(
                    x0=evnt.name,
                    y=inlier_features.loc[evnt, metric],
                    fillcolor=color, spanmode='hard', showlegend=False,
                    side='positive' if show_outliers else None,
                    box_visible=True, meanline_visible=True, line_color='black',
                    name=evnt.name, legendgroup=evnt.name,
                )
                if show_outliers:
                    outlier_trace = go.Violin(
                        x0=evnt.name,
                        y=outlier_features.loc[evnt, metric],
                        fillcolor=color, spanmode='hard', showlegend=False, side='negative',
                        opacity=kwargs.get("outlier_opacity", 0.5),
                        box_visible=True, meanline_visible=True, line_color='black',
                        name=f"{evnt.name} ({__OUTLIERS_STR})", legendgroup=evnt.name,
                    )
            fig.add_trace(inlier_trace, col=1, row=r+1)
            if show_outliers and outlier_trace is not None:
                fig.add_trace(outlier_trace, col=1, row=r+1)
    fig.update_layout(
        title=kwargs.get("title", f"{cnst.EVENT_STR.title()} Summary"),
        barmode='stack',
        violinmode='overlay',
        violingap=0,
        violingroupgap=0.1,
    )
    return fig


def fixation_summary(fixations: EventSequenceType, **kwargs) -> go.Figure:
    """
    Creates a 2×2 figure with the following subplots:
        1. A violin plot showing the distribution of fixation durations (ms)
        2. A violin plot showing the distribution of fixation areas (deg^2)
        3. A violin plot showing the distribution of peak velocities (deg/s)
        4. A scatter plot showing the distribution of fixation centers (x-y coordinates)

    :param fixations: array-like of fixation events.

    :keyword show_outliers: whether to show outliers in the violin plots; default is False.
    :keyword inlier_color: the color of the inliers.
    :keyword outlier_color: the color of the outliers.
    :keyword outlier_opacity: the opacity of the outliers; default is 0.5.
    :keyword marker_size: the size of the markers in the scatter plot; default is 5.
    :keyword title: the title of the figure; default is "Fixation Summary".

    :return: the figure.
    """

    fixations = list(filter(lambda e: e.label == EventLabelEnum.FIXATION, fixations))   # filter out non-fixation events
    subplots = {
        cnst.DURATION_STR: 'ms', cnst.ELLIPSE_AREA_STR: 'deg^2',
        cnst.PEAK_VELOCITY_STR: 'deg/s', cnst.CENTER_PIXEL_STR: 'x coordinate',
    }
    return _create_single_event_figure(fixations, subplots, **kwargs)


def saccade_summary(saccades: EventSequenceType, **kwargs) -> go.Figure:
    """
    Creates a 2×2 figure with the following subplots:
        1. A violin plot showing the distribution of saccade durations (ms)
        2. A violin plot showing the distribution of saccade amplitudes (deg)
        3. A violin plot showing the distribution of peak velocities (deg/s)
        4. A polar histogram showing the distribution of saccade azimuths (deg)

    :param saccades: array-like of saccade events.

    :keyword show_outliers: whether to show outliers in the violin plots; default is False.
    :keyword inlier_color: the color of the inliers.
    :keyword outlier_color: the color of the outliers.
    :keyword outlier_opacity: the opacity of the outliers; default is 0.5.
    :keyword azimuth_nbins: the number of bins in the azimuth histogram; default is 16.
    :keyword title: the title of the figure; default is "Saccade Summary".

    :return: the figure.
    """
    saccades = list(filter(lambda e: e.label == EventLabelEnum.SACCADE, saccades))   # filter out non-saccade events
    subplots = {
        cnst.DURATION_STR: 'ms', cnst.AMPLITUDE_STR: 'deg',
        cnst.PEAK_VELOCITY_STR: 'deg/s', cnst.AZIMUTH_STR: 'deg',
    }
    return _create_single_event_figure(saccades, subplots, **kwargs)


def _create_single_event_figure(
        events: EventSequenceType,
        subplots: Dict[str, str],
        **kwargs,
) -> go.Figure:
    """
    Creates a figure with multiple subplots, each showing a different metric of the provided events.

    :param events: array-like of events. All events should be of the same type
    :param subplots: a dictionary mapping metric names to their respective measures (e.g. `peak velocity` -> `deg/s`)

    :keyword num_cols: the number of columns in the figure; default is 2.
        Number of rows is calculated as np.ceil(len(subplots) / num_cols).
    :keyword inlier_color: the color of the inliers; default is the color of the event label.
    :keyword show_outliers: whether to show outliers in the violin plots; default is False.
    :keyword outlier_color: the color of the outliers; default is the color of the event label.
    :keyword outlier_opacity: the opacity of the outliers; default is 0.5.
    :keyword marker_size: the size of the markers in scatter plots; default is 5.
    :keyword azimuth_nbins: the number of bins in the azimuth histogram; default is 16.
    :keyword title: the title of the figure; default is "{Event-Label} Summary" (e.g. "Saccade Summary").

    :return: the figure.
    """
    event_label = events[0].label
    assert all(e.label == event_label for e in events), "All events must have the same label."
    summary_df = summarize_events(events)
    inlier_df, outlier_df = summary_df[~summary_df[cnst.IS_OUTLIER_STR]], summary_df[summary_df[cnst.IS_OUTLIER_STR]]

    # create figure:
    num_cols = kwargs.get("num_cols", 2)
    num_rows = np.ceil(len(subplots) / num_cols).astype(int).item()
    num_subfigs = num_rows * num_cols
    subplot_titles = list(map(lambda met: met.replace('_', ' ').title(), subplots.keys())) + [None] * (num_subfigs - len(subplots))
    specs = [{}] * len(subplots) + [None] * (num_subfigs - len(subplots))
    if cnst.AZIMUTH_STR in subplots.keys():
        specs[list(subplots.keys()).index(cnst.AZIMUTH_STR)] = {'type': 'polar'}
    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        subplot_titles=subplot_titles,
        specs=np.array(specs).reshape(num_rows, num_cols).tolist(),
    )

    # populate subplots with traces:
    for i, (metric, measure) in enumerate(subplots.items()):
        r, c = divmod(i, num_cols)
        fig.add_trace(row=r + 1, col=c + 1, trace=__create_single_event_trace(
            data=inlier_df[metric],
            metric=metric,
            is_inlier=True,
            color=kwargs.get(
                "inlier_color", f"rgb{vis_utils.get_label_colormap(None)[event_label]}"
            ),
            marker_size=kwargs.get("marker_size", 5),
            nbins_polar=kwargs.get(f"{cnst.AZIMUTH_STR}_nbins", 16),
            show_legend=i == 1,  # show legend on top-right figure only
        ))
        if kwargs.get("show_outliers", False):
            fig.add_trace(row=r + 1, col=c + 1, trace=__create_single_event_trace(
                data=outlier_df[metric],
                metric=metric,
                is_inlier=False,
                color=kwargs.get(
                    "outlier_color", f"rgb{vis_utils.get_label_colormap(None)[event_label]}"
                ),
                outlier_opacity=kwargs.get("outlier_opacity", 0.5),
                marker_size=kwargs.get("marker_size", 5),
                nbins_polar=kwargs.get(f"{cnst.AZIMUTH_STR}_nbins", 16),
                show_legend=i == 1,  # show legend on top-right figure only
            ))
        fig.update_xaxes(title_text=measure, row=r + 1, col=c + 1)
        if metric == cnst.CENTER_PIXEL_STR:
            fig.update_yaxes(title_text="y coordinate", row=r + 1, col=c + 1)

    # adjust layout:
    fig.update_layout(
        title=kwargs.get("title", f"{event_label.name.title()} Summary"),
        violinmode='overlay',
        violingap=0,
        violingroupgap=0,
    )
    return fig


def __create_single_event_trace(
        data: pd.Series,
        metric: str,
        is_inlier: bool,
        color: str,
        marker_size: int,
        nbins_polar: int,
        outlier_opacity: float = 1,
        show_legend: bool = False,
):
    name = __INLIERS_STR if is_inlier else __OUTLIERS_STR
    opacity = 1 if is_inlier else outlier_opacity
    if metric == cnst.CENTER_PIXEL_STR:
        px = pd.DataFrame(data.to_list(), columns=[cnst.X, cnst.Y])  # data is a 2-tuple of x-y coordinates
        return go.Scatter(
            x=px[cnst.X], y=px[cnst.Y], mode="markers", name=name, legendgroup=name,
            marker=dict(size=marker_size, color=color, opacity=opacity), showlegend=show_legend,
        )
    if metric == cnst.AZIMUTH_STR:
        half_bin = 360 / nbins_polar / 2
        corrected_azimuths = (data.values + half_bin) % 360
        edges = np.linspace(0, 360, nbins_polar + 1, endpoint=True)
        counts, _ = np.histogram(corrected_azimuths, bins=edges)
        return go.Barpolar(
            r=counts, name=name, legendgroup=name, showlegend=show_legend,
            opacity=opacity, marker=dict(color=color),
        )
    if metric in {
        cnst.DURATION_STR, cnst.AMPLITUDE_STR, cnst.PEAK_VELOCITY_STR, cnst.ELLIPSE_AREA_STR
    }:
        violin_side = 'positive' if is_inlier else 'negative'
        return go.Violin(
            x=data, side=violin_side, showlegend=show_legend,
            name=name, legendgroup=name, scalegroup=metric, fillcolor=color, opacity=opacity,
            box_visible=True, meanline_visible=True, line_color='black',
            spanmode='hard', orientation='h', points=False,
        )
    raise ValueError(f"Invalid metric: {metric}")
