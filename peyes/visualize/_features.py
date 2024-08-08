import warnings
from typing import Union, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import peyes
import peyes._utils.constants as cnst
import peyes._utils.visualization_utils as vis_utils
from peyes._DataModels.Event import EventSequenceType
from peyes._DataModels.EventLabelEnum import EventLabelEnum

from peyes import summarize_events


def feature_comparison(
        features: Union[str, Sequence[str]],
        *event_sequences: EventSequenceType,
        **kwargs
) -> go.Figure:
    """
    Creates a Ridge Plot comparing the distribution of features across multiple event sequences.
    Each feature is shown in a separate subplot.

    :param features: name(s) of the feature(s) to compare.
    :param event_sequences: array-like of Event objects

    :keyword include_outliers: bool; whether to include outliers in the plot (default is False)
    :keyword labels: array-like of str; labels for each event sequence (default is 1, 2, ...)
    :keyword colors: dict; color map for each label (default is the colormap from `utils.visualization_utils.py`)
    :keyword opacity: float; opacity of the violins (default is 0.75)
    :keyword line_width: float; width of the violin lines (default is 2)
    :keyword show_box: bool; whether to show the box plot (default is True)
    :keyword title: str; title for the plot (default is "Feature Comparison")

    :return: the plotly figure
    """
    include_outliers = kwargs.get("include_outliers", False)
    labels = kwargs.get("labels", list(range(len(event_sequences))))
    colors = vis_utils.get_label_colormap(kwargs.get("colors", None))
    if isinstance(features, str):
        features = [features]
    fig, nrows, ncols = vis_utils.make_empty_figure(
        list(map(lambda feat: feat.strip().replace("_", " ").title(), features)), sharex=False, sharey=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (seq_name, ev_seq) in enumerate(zip(labels, event_sequences)):
            summary_df = summarize_events(ev_seq)
            if not include_outliers:
                summary_df = summary_df[~summary_df["is_outlier"]]
            for j, feat in enumerate(features):
                color = (
                        colors.get(seq_name, None) or colors.get(seq_name.strip().lower(), None) or
                        colors.get(seq_name.strip().upper(), None) or colors[i]
                )
                color = f"rgb{color}"
                r, c = (j, 0) if ncols == 1 else divmod(j, ncols)
                if feat.lower().strip() == peyes.constants.COUNT_STR:
                    num_events = summary_df.shape[0]
                    fig.add_trace(
                        row=r+1, col=c+1, trace=go.Bar(
                            x=[num_events], y=[seq_name], orientation='h',
                            marker=dict(color=color), showlegend=j == 0,
                        )
                    )
                else:
                    fig.add_trace(
                        row=r + 1, col=c + 1, trace=go.Violin(
                            x=summary_df[feat].dropna().values,
                            name=seq_name, legendgroup=seq_name, scalegroup=feat,
                            line_color=color, opacity=kwargs.get("opacity", 0.75), width=kwargs.get("line_width", 2),
                            box_visible=kwargs.get("show_box", True), points=False,
                            orientation='h', spanmode='hard', side='positive',
                            showlegend=j == 0,
                        )
                    )
        fig.update_layout(
            title=kwargs.get("title", "Feature Comparison"),
        )
    return fig


def main_sequence(
        saccades: EventSequenceType,
        y_feature: str = "duration",
        include_outliers: bool = True,
) -> (go.Figure, pd.DataFrame):
    """
    Creates a saccades "Main Sequence" plot, showing the relationship between saccade amplitude and either duration or
    peak velocity.

    :param saccades: array-like of saccade Event objects
    :param y_feature: str; the feature to plot on the y-axis. Must be either "duration" or "peak_velocity"
    :param include_outliers: bool; whether to include outliers in the plot (default is True)

    :return:
        - fig: go.Figure; the figure object
        - stat_results: pd.DataFrame; statistics of the inlier (and outlier) trendline(s)
    """
    saccades = [s for s in saccades if s.label == EventLabelEnum.SACCADE]
    if y_feature.lower() == "duration":
        params = dict(
            y_feature="duration", title="Main Sequence: Duration vs. Amplitude",
            y_feature_units="ms", trendline_log_x=False,
        )
    elif y_feature.lower() == "peak_velocity":
        params = dict(
            y_feature="peak_velocity", title="Main Sequence: Peak Velocity vs. Amplitude",
            y_feature_units="deg/s", trendline_log_x=True,
        )
    else:
        raise KeyError(f"Invalid `y_feature` argument: {y_feature}. Must be `duration` or `peak_velocity`.")
    fig, stat_results = feature_relationship(
        events=saccades, x_feature="amplitude", x_feature_units="deg",
        include_outliers=include_outliers, trendline="trace", marginal_x='box',
        marginal_y='box', **params
    )
    return fig, stat_results


def feature_relationship(
        events: EventSequenceType, x_feature: str, y_feature: str, title: str = None, **kwargs
) -> (go.Figure, pd.DataFrame):
    """
    Creates a scatter plot of the two features of the provided events, with optional trendline(s).

    :param events: array-like of Event objects
    :param x_feature: str; name of the feature to plot on the x-axis
    :param y_feature: str; name of the feature to plot on the y-axis
    :param title: str; title of the plot. if None, the default title is "`y_feature` vs. `x_feature`"

    :keyword include_outliers: bool; whether to include outliers in the plot (default is True)
    :keyword log_x: bool; whether to use a log scale on the x-axis (default is False)
    :keyword log_y: bool; whether to use a log scale on the y-axis (default is False)
    :keyword marginal_x: str: None, 'rug', 'box', 'violin', 'histogram'; type of marginal plot for the x-axis (default is "box")
    :keyword marginal_y: str: None, 'rug', 'box', 'violin', 'histogram'; type of marginal plot for the y-axis (default is "box")
    :keyword trendline: str: None, 'trace', 'overall'; type of trendline to add to the plot (default is None)
        if None - no trendline is added
        if 'trace' - a trendline is added for each trace
        if 'overall' - a single trendline is added for all traces
    :keyword trendline_color: str; color of the overall trendline (default is 'black'). has no effect if trendline is not 'overall'
    :keyword trendline_log_x: bool; whether to use a log scale on the x-axis of the trendline (default is False)
    :keyword trendline_log_y: bool; whether to use a log scale on the y-axis of the trendline (default is False)
    :keyword x_feature_units: str; units of the x-axis feature (default is "")
    :keyword y_feature_units: str; units of the y-axis feature (default is "")

    :return:
        - fig: go.Figure; the figure object
        - stat_results: pd.DataFrame; statistics of the trendline(s) (if any)
    """
    summary_df = summarize_events(events)
    if x_feature not in summary_df.columns:
        raise KeyError(f"Feature '{x_feature}' not found in the summary DataFrame.")
    if y_feature not in summary_df.columns:
        raise KeyError(f"Feature '{y_feature}' not found in the summary DataFrame.")
    if not kwargs.get("include_outliers", True):
        summary_df = summary_df[~summary_df["is_outlier"]]
    summary_df[cnst.LABEL_STR] = summary_df[cnst.LABEL_STR].map(
        lambda l: EventLabelEnum(l).name.replace("_", " ").title()
    )
    trendline = kwargs.get("trendline", None)
    if trendline is None:
        trendline_options = dict()
    elif trendline == "trace":
        trendline_options = dict(
            trendline="ols", trendline_scope="trace", trendline_options=dict(
                log_x=kwargs.get("trendline_log_x", False), log_y=kwargs.get("trendline_log_y", False),
            ))
    elif trendline == "overall":
        trendline_options = dict(
            trendline="ols", trendline_scope="overall", trendline_color_override=kwargs.get("trendline_color", 'black'),
            trendline_options=dict(
                log_x=kwargs.get("trendline_log_x", False), log_y=kwargs.get("trendline_log_y", False),
            ))
    else:
        raise ValueError(f"Invalid `trendline` argument: {trendline}")
    marg_x, marg_y = kwargs.get("marginal_x", "box"), kwargs.get("marginal_y", "box")
    fig = px.scatter(
        summary_df, x=x_feature, y=y_feature,
        color=cnst.LABEL_STR, symbol="is_outlier", symbol_map={True: "x", False: "circle"},
        # TODO: use color_discrete_map argument to set default event colors
        log_x=kwargs.get("log_x", False), marginal_x=marg_x, log_y=kwargs.get("log_y", False), marginal_y=marg_y,
        **trendline_options
    )
    stat_results = px.get_trendline_results(fig)

    # remove trendlines from marginal subplots (see https://stackoverflow.com/q/78746094/8543025):
    fig.update_traces(visible=False, selector=dict(name='Overall Trendline'))
    fig.update_traces(visible=True, showlegend=True, selector=dict(name='Overall Trendline', xaxis='x'))

    # update titles
    x_feature_title = x_feature.replace("_", " ").title()
    y_feature_title = y_feature.replace("_", " ").title()
    x_feature_units, y_feature_units = kwargs.get("x_feature_units", ""), kwargs.get("y_feature_units", "")
    fig.update_layout(
        title=title or f"{y_feature_title} vs. {x_feature_title}",
        xaxis_title=f"{x_feature_title} ({x_feature_units})" if x_feature_units else x_feature_title,
        yaxis_title=f"{y_feature_title} ({y_feature_units})" if y_feature_units else y_feature_title,
    )
    return fig, stat_results
