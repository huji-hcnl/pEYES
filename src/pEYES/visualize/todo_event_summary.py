import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pEYES._DataModels.Event import EventSequenceType

from src.pEYES._utils.event_utils import count_labels

__INLIERS_STR, __OUTLIERS_STR = "Inliers", "Outliers"
__COUNTS_STR = "Counts"
__DURATIONS_STR = "Durations"
__AMPLITUDES_STR = "Amplitudes"
__AMPLITUDE_VS_DURATION_STR = f"{__AMPLITUDES_STR} vs. {__DURATIONS_STR}"


def _aggregate_events(events: EventSequenceType) -> pd.DataFrame:
    # TODO
    return None


def event_summary(
        events: EventSequenceType,
        title: str = "Event Summary",
        inlier_color: str = '#0000ff',
        outlier_color: str = '#ff0000',
        opacity: float = 0.6,
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[__COUNTS_STR, __DURATIONS_STR, __AMPLITUDES_STR, __AMPLITUDE_VS_DURATION_STR],
    )
    inliers, outliers = [e for e in events if not e.is_outlier], [e for e in events if e.is_outlier]

    # counts sub figure
    inlier_count, outlier_count = count_labels(inliers), count_labels(outliers)
    fig.add_trace(
        row=1, col=1,
        trace=go.Bar(
            name=__INLIERS_STR, x=list(inlier_count.keys()), y=list(inlier_count.values()), marker_color=inlier_color,
        )
    )
    fig.add_trace(
        row=1, col=1,
        trace=go.Bar(name=__OUTLIERS_STR, x=list(outlier_count.keys()), y=list(outlier_count.values())),
    )

    # durations sub figure
    inlier_durations, outlier_durations = [e.duration for e in inliers], [e.duration for e in outliers]
    fig.add_trace(
        row=1, col=2,
        trace=go.Violin(
            y=inlier_durations,
            name=__INLIERS_STR, legendgroup=__INLIERS_STR, scalegroup=__INLIERS_STR, side='positive',
            box_visible=True, meanline_visible=True, line_color='black',
            fillcolor='#0000ff', opacity=opacity,
        ),
    )
    fig.add_trace(
        row=1, col=2,
        trace=go.Violin(
            y=outlier_durations,
            name=__OUTLIERS_STR, legendgroup=__OUTLIERS_STR, scalegroup=__OUTLIERS_STR, side='negative',
            box_visible=True, meanline_visible=True, line_color='black',
            fillcolor='#ff0000', opacity=opacity,
        ),
    )

    fig.update_layout(
        barmode='stack',
        violinmode='overlay',
        violingap=0,
        violingroupgap=0,
    )

    return None



# TODO: all-event summary figure - a multi-panel figure with the following panels:
    # event counts (bar plot)
    # event durations (histogram / box plot / violin plot)
    # event amplitudes (histogram / box plot / violin plot)
    # event amplitude vs duration (scatter plot with colors based on event labels)

# TODO: saccade summary figure - a multi-panel figure with the following panels:
    # saccade durations histogram / violin
    # saccade amplitudes histogram / violin
    # saccade azimuth histogram (radial histogram)
    # saccade main sequence (peak velocity vs. amplitude)

# TODO: fixation summary figure - a multi-panel figure with the following panels:
    # fixation durations histogram / violin
    # fixation center locations (heatmap or just scatter plot)
    # fixation dispersion (pixel std) histogram / violin
