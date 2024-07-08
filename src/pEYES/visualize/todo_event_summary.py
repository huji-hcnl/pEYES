import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.pEYES._utils.constants as cnst
from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES.event_metrics import features_by_labels

__INLIERS_STR, __OUTLIERS_STR = "Inliers", "Outliers"
__AMPLITUDE_VS_DURATION_STR = f"{cnst.AMPLITUDE_STR} vs. {cnst.DURATION_STR}"


def event_summary(
        events: EventSequenceType,
        title: str = "Event Summary",
        inlier_color: str = '#0000ff',
        outlier_color: str = '#ff0000',
        opacity: float = 0.6,
        show_outliers: bool = True,
) -> go.Figure:
    subplots = {
        cnst.COUNT_STR: '#',
        cnst.DURATION_STR: 'ms',
        cnst.AMPLITUDE_STR: 'deg',
    }
    fig = make_subplots(
        cols=1, rows=len(subplots), subplot_titles=list(map(lambda met: met.title(), subplots.keys()))
    )
    inlier_features = features_by_labels(list(filter(lambda e: not e.is_outlier, events)))
    outlier_features = features_by_labels(list(filter(lambda e: e.is_outlier, events)))
    assert inlier_features.index.equals(outlier_features.index)
    event_labels = inlier_features.index.tolist()

    for r, (metric, measure) in enumerate(subplots.items()):
        if r == 0:
            # sub-figure: counts
            fig.add_trace(
                col=1, row=r+1,
                trace=go.Bar(
                    name=__INLIERS_STR, legendgroup=__INLIERS_STR,
                    x=event_labels, y=inlier_features[metric].values,
                    marker_color=inlier_color, opacity=opacity,
                    showlegend=True,

                )
            )
            if show_outliers:
                fig.add_trace(
                    col=1, row=r+1,
                    trace=go.Bar(
                        name=__OUTLIERS_STR, legendgroup=__OUTLIERS_STR,
                        x=event_labels, y=outlier_features[metric].values,
                        marker_color=outlier_color, opacity=opacity,
                        showlegend=True,
                    )
                )
        else:
            # sub-figure: durations and amplitudes
            for evnt in event_labels:
                fig.add_trace(
                    row=r+1, col=1,
                    trace=go.Violin(
                        name=evnt, legendgroup=evnt, scalegroup=evnt, side='positive', x0=evnt,
                        y=inlier_features.loc[evnt, metric], fillcolor=inlier_color, opacity=opacity,
                        box_visible=True, meanline_visible=True, line_color='black',
                        showlegend=False,
                    )
                )
                if show_outliers:
                    fig.add_trace(
                        row=r+1, col=1,
                        trace=go.Violin(
                            name=evnt, legendgroup=evnt, scalegroup=evnt, side='negative', x0=evnt,
                            y=outlier_features.loc[evnt, metric], fillcolor=outlier_color, opacity=opacity,
                            box_visible=True, meanline_visible=True, line_color='black',
                            showlegend=False,
                        )
                    )
        fig.update_yaxes(col=1, row=r+1, title_text=measure)
        fig.update_xaxes(
            col=1, row=r + 1, title_text='Event Labels',
            type='category', categoryorder='array', categoryarray=event_labels
        )
    fig.update_layout(
        title=title,
        barmode='stack',
        violinmode='overlay',
        violingap=0,
        violingroupgap=0,
    )
    return fig



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
