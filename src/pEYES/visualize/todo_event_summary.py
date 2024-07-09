import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.pEYES._utils.constants as cnst
import src.pEYES._utils.visualization_utils as vis_utils
from src.pEYES._utils.event_utils import parse_label
from src.pEYES.event_metrics import features_by_labels
from src.pEYES._base.postprocess_events import summarize_events

from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

__INLIERS_STR, __OUTLIERS_STR = "inliers", "outliers"


def event_summary(
        events: EventSequenceType,
        **kwargs,
) -> go.Figure:
    """
    Creates a summary figure of the provided events. The figure contains four panels:
        1. A bar plot of the count of each event label.
        2. A violin plot showing the distribution of durations per event label.
        3. A violin plot showing the distribution of amplitudes per event label.
        4. A violin plot showing the distribution of peak velocities per event label.

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
    }
    fig = make_subplots(
        cols=1, rows=len(subplots), shared_xaxes=True,
        subplot_titles=list(map(lambda met: met.replace('_', ' ').title(), subplots.keys()))
    )
    for r, (metric, measure) in enumerate(subplots.items()):
        for evnt in event_labels:
            color = f"rgb{label_colors[evnt]}"
            if metric == cnst.COUNT_STR:
                fig.add_trace(
                    col=1, row=r+1,
                    trace=go.Bar(
                        x0=evnt.name,
                        y=[inlier_features.loc[evnt, metric]],
                        marker_color=color,
                        name=evnt.name, legendgroup=evnt.name, showlegend=True,
                    )
                )
                if show_outliers:
                    fig.add_trace(
                        col=1, row=r+1,
                        trace=go.Bar(
                            x0=evnt.name,
                            y=[outlier_features.loc[evnt, metric]],
                            marker_color=color, opacity=kwargs.get("outlier_opacity", 0.5),
                            name=f"{evnt.name} ({__OUTLIERS_STR})", legendgroup=evnt.name, showlegend=False,
                        )
                    )
            else:
                fig.add_trace(
                    col=1, row=r+1,
                    trace=go.Violin(
                        x0=evnt.name,
                        y=inlier_features.loc[evnt, metric],
                        fillcolor=color, spanmode='hard', showlegend=False,
                        side='positive' if show_outliers else None,
                        box_visible=True, meanline_visible=True, line_color='black',
                        name=evnt.name, legendgroup=evnt.name,
                    )
                )
                if show_outliers:
                    fig.add_trace(
                        col=1, row=r+1,
                        trace=go.Violin(
                            x0=evnt.name,
                            y=outlier_features.loc[evnt, metric],
                            fillcolor=color, spanmode='hard', showlegend=False, side='negative',
                            opacity=kwargs.get("outlier_opacity", 0.5),
                            box_visible=True, meanline_visible=True, line_color='black',
                            name=f"{evnt.name} ({__OUTLIERS_STR})", legendgroup=evnt.name,
                        )
                    )
        fig.update_yaxes(col=1, row=r+1, title_text=measure)
    fig.update_layout(
        title=kwargs.get("title", f"{cnst.EVENT_STR.title()} Summary"),
        barmode='stack',
        violinmode='overlay',
        violingap=0,
        violingroupgap=0.1,
    )
    return fig


def fixation_summary(
        fixations: EventSequenceType,
        **kwargs,
) -> go.Figure:
    summary_df = summarize_events(list(filter(lambda e: e.label == EventLabelEnum.FIXATION, fixations)))
    inlier_df, outlier_df = summary_df[~summary_df[cnst.IS_OUTLIER_STR]], summary_df[summary_df[cnst.IS_OUTLIER_STR]]
    inlier_color = kwargs.get("inlier_color", f"rgb{vis_utils.get_label_colormap(None)[EventLabelEnum.FIXATION]}")
    show_outliers = kwargs.get("show_outliers", False)
    outlier_color = kwargs.get("outlier_color", f"rgb{vis_utils.get_label_colormap(None)[EventLabelEnum.FIXATION]}")
    opacity = kwargs.get("opacity", 0.5)
    subplots = {
        cnst.DURATION_STR: 'ms',
        cnst.CENTER_PIXEL_STR: 'x coordinate',
        cnst.PIXEL_STD_STR: 'px',
        cnst.PEAK_VELOCITY_STR: 'deg/s',
    }
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(map(lambda met: met.replace('_', ' ').title(), subplots.keys()))
    )
    for i, (metric, measure) in enumerate(subplots.items()):
        r, c = divmod(i, 2)
        if metric == cnst.CENTER_PIXEL_STR:
            px = pd.DataFrame(inlier_df[cnst.CENTER_PIXEL_STR].to_list(), columns=[cnst.X, cnst.Y])
            inlier_trace = go.Scatter(
                x=px[cnst.X], y=px[cnst.Y], mode="markers", name=None, showlegend=False,
                marker=dict(size=kwargs.get("marker_size", 5), color=inlier_color),
            )
            fig.update_yaxes(title_text="y coordinate", row=r + 1, col=c + 1)
            if show_outliers:
                px = pd.DataFrame(outlier_df[cnst.CENTER_PIXEL_STR].to_list(), columns=[cnst.X, cnst.Y])
                outlier_trace = go.Scatter(
                    x=px[cnst.X], y=px[cnst.Y], mode="markers", name=None, showlegend=False,
                    marker=dict(size=kwargs.get("marker_size", 5), color=outlier_color, opacity=opacity),
                )
        else:
            inlier_trace = go.Violin(
                x=inlier_df[metric], spanmode='hard', side='positive', showlegend=False,
                name=metric, legendgroup=metric, scalegroup=metric,
                fillcolor=inlier_color, box_visible=True, meanline_visible=True, line_color='black',
            )
            if show_outliers:
                outlier_trace = go.Violin(
                    x=outlier_df[metric], spanmode='hard', side='positive', showlegend=False,
                    name=metric, legendgroup=metric, scalegroup=metric,
                    fillcolor=outlier_color, opacity=opacity,
                    box_visible=True, meanline_visible=True, line_color='black',
                )
        fig.add_trace(inlier_trace, row=r+1, col=c+1)
        if show_outliers:
            fig.add_trace(outlier_trace, row=r+1, col=c+1)
        fig.update_xaxes(title_text=measure, row=r+1, col=c+1)
    fig.update_layout(
        title=kwargs.get("title", f"{EventLabelEnum.FIXATION.name.title()} Summary"),
        violinmode='overlay',
        violingap=0,
        violingroupgap=0,
    )
    return fig


# TODO: saccade summary figure - a multi-panel figure with the following panels:
    # saccade durations histogram / violin
    # saccade amplitudes histogram / violin
    # saccade azimuth histogram (radial histogram)
    # saccade main sequence (peak velocity vs. amplitude)

# TODO: fixation summary figure - a multi-panel figure with the following panels:
    # fixation durations histogram / violin
    # fixation center locations (heatmap or just scatter plot)
    # fixation dispersion (pixel std) histogram / violin
