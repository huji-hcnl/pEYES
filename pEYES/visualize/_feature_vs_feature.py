import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import pEYES._utils.constants as cnst
from pEYES._DataModels.Event import EventSequenceType
from pEYES._DataModels.EventLabelEnum import EventLabelEnum

from pEYES import summarize_events


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
    fig, stat_results = feature_vs_feature(
        events=saccades, x_feature="amplitude", x_feature_units="deg", include_outliers=include_outliers,
        trendline="trace", marginal_x='box', marginal_y='box', **params
    )
    return fig, stat_results


def feature_vs_feature(
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
    if trendline == "overall" and ((marg_x is not None) or (marg_y is not None)):
        # Remove the trendlines from the marginal subplots
        # TODO: see https://stackoverflow.com/q/78746094/8543025
        num_traces_to_remove = (marg_x is not None) + (marg_y is not None)
        fig.data = fig.data[:-num_traces_to_remove]
        fig.data[-1].showlegend = True

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
