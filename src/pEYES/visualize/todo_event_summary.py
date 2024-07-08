import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.pEYES._utils.constants as cnst
import src.pEYES._utils.visualization_utils as vis_utils
from src.pEYES._utils.event_utils import parse_label
from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES.event_metrics import features_by_labels

__INLIERS_STR, __OUTLIERS_STR = "inliers", "outliers"


def event_summary(
        events: EventSequenceType,
        title: str = "Event Summary",
        label_colors: vis_utils.LabelColormapType = None,
        outlier_opacity: float = 1/3,
        show_outliers: bool = True,
) -> go.Figure:
    inlier_features = features_by_labels(list(filter(lambda e: not e.is_outlier, events)))
    outlier_features = features_by_labels(list(filter(lambda e: e.is_outlier, events)))
    assert inlier_features.index.equals(outlier_features.index)
    event_labels = [parse_label(l) for l in inlier_features.index]
    label_colors = vis_utils.get_label_colormap(label_colors)

    subplots = {
        cnst.COUNT_STR: '# instances',
        cnst.DURATION_STR: 'ms',
        cnst.AMPLITUDE_STR: 'deg',
    }
    fig = make_subplots(
        cols=1, rows=len(subplots), shared_xaxes=True,
        subplot_titles=list(map(lambda met: met.title(), subplots.keys()))
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
                            marker_color=color, opacity=outlier_opacity,
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
                            fillcolor=color, spanmode='hard', showlegend=False, opacity=outlier_opacity,
                            side='negative',
                            box_visible=True, meanline_visible=True, line_color='black',
                            name=f"{evnt.name} ({__OUTLIERS_STR})", legendgroup=evnt.name,
                        )
                    )
        fig.update_yaxes(col=1, row=r+1, title_text=measure)
    fig.update_layout(
        title=title,
        barmode='stack',
        violinmode='overlay',
        violingap=0,
        violingroupgap=0.1,
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
