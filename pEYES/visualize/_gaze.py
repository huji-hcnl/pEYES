from typing import Tuple
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from plotly.subplots import make_subplots

import pEYES._utils.constants as cnst
import pEYES._utils.visualization_utils as vis_utils
from pEYES._utils.vector_utils import is_one_dimensional, normalize
from pEYES._utils.pixel_utils import cast_to_integers

# TODO: create figure of y-coordinates over x-coordinates with color changing according to event labels


def gaze_trajectory(
        x: np.ndarray,
        y: np.ndarray,
        resolution: Tuple[int, int],
        t: np.ndarray = None,
        title: str = "Gaze Visualization",
        **kwargs,
) -> go.Figure:
    """
    Plots the gaze trajectory, optionally with color changing through time.

    :param x: the x-coordinates of the gaze.
    :param y: the y-coordinates of the gaze.
    :param resolution: the screen resolution in pixels (width, height).
    :param t: optional; time axis (will be mapped to varying color).
    :param title: the title of the figure.

    :keyword bg_image: the background image to overlay on, defaults to None.
    :keyword bg_image_format: the color format of the background image (if provided), defaults to "BGR".
    :keyword bg_color: the background color if no image is provided, defaults to white.
    :keyword marker_size: the size of the markers, defaults to 5.
    :keyword opacity: the opacity of the markers, defaults to 1 if no background image is provided, else 0.5.
    :keyword colorscale: the colorscale to use for displaying time (if `t` is provided), defaults to "Jet".
    :keyword color: the color of the markers if `t` is not provided, defaults to black.

    :return: the figure.
    """
    x, y, t, _ = __verify_arrays(x=x, y=y, t=t, v=None)
    bg_image = kwargs.get("bg_image", None)
    bg = vis_utils.create_image(
        resolution,
        bg_image=bg_image,
        color_format=kwargs.get("bg_image_format", "BGR"),
        bg_color=kwargs.get("bg_color", "#ffffff")  # default background color is white
    )
    fig = go.Figure(
        data=go.Image(z=bg),
        layout=dict(width=resolution[0], height=resolution[1], margin=dict(l=0, r=0, b=0, t=0)),
    )
    marker_size = kwargs.get("marker_size", 5)
    opacity = kwargs.get("opacity", 1 if bg_image is None else 0.5)
    if t is None:
        color = kwargs.get("color", "#000000")
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", marker=dict(color=color, size=marker_size, opacity=opacity))
        )
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", marker=dict(
                color=t,
                size=marker_size,
                opacity=opacity,
                colorscale=kwargs.get("colorscale", "Jet"),
                colorbar=dict(title=cnst.TIME_STR),
            )
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False),
        yaxis=dict(visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False),
    )
    return fig


def gaze_heatmap(
        x: np.ndarray,
        y: np.ndarray,
        resolution: Tuple[int, int],
        **kwargs,
) -> go.Figure:
    """
    Creates a heatmap of the given x and y coordinates, overlayed on a background image or a white background.

    :param x: 1D array of x coordinates
    :param y: 1D array of y coordinates
    :param resolution: tuple of (width, height) in pixels

    :keyword bg_image: the background image to overlay on, defaults to None.
    :keyword bg_image_format: the color format of the background image (if provided), defaults to "BGR".
    :keyword bg_color: the background color if no image is provided, defaults to white.
    :keyword sigma: standard deviation of the Gaussian filter that smooths the heatmap, defaults to 10.0
    :keyword colorscale: name of the color scale to use. Must be one of the named color scales in plotly.express.colors
    :keyword opacity: opacity of the heatmap (0-1). Default is 0.5

    :return: the figure.
    """
    x, y, _, _ = __verify_arrays(x=x, y=y)
    bg = vis_utils.create_image(
        resolution,
        bg_image=kwargs.get("bg_image", None),
        color_format=kwargs.get("bg_image_format", "BGR"),
        bg_color=kwargs.get("bg_color", "#ffffff")  # default background color is white
    )
    fig = go.Figure(
        data=go.Image(z=bg),
        layout=dict(width=resolution[0], height=resolution[1], margin=dict(l=0, r=0, b=0, t=0)),
    )
    counts = __pixel_counts(x, y, resolution)
    normalized_counts = normalize(gaussian_filter(counts, kwargs.get("sigma", 10.0)))
    fig.add_trace(go.Heatmap(
        z=normalized_counts,
        colorscale=kwargs.get("colorscale", "Jet"),
        opacity=kwargs.get("opacity", 0.5),
        showscale=False,
    ))
    fig.update_xaxes(
        range=[0, resolution[0]], visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False
    )
    fig.update_yaxes(
        range=[0, resolution[1]], visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False
    )
    return fig


def gaze_over_time(
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        v: np.ndarray = None,
        vert_lines: np.ndarray = None,
        title: str = "Gaze Coordinates Over Time",
        **kwargs,
) -> go.Figure:
    """
    Plots the x- and y-coordinates of the gaze over time, optionally with velocity as a secondary y-axis and/or vertical
    lines at specified time points.

    :param x: gaze x-coordinates.
    :param y: gaze y-coordinates.
    :param t: the time axis.
    :param v: gaze velocity, optional.
    :param vert_lines: time points to add vertical lines at, optional.
    :param title: the title of the figure.

    :keyword marker_size: the size of the markers, defaults to 5.
    :keyword x_color: the color of the x-coordinates, defaults to red.
    :keyword y_color: the color of the y-coordinates, defaults to blue.
    :keyword v_color: the color of the velocity, defaults to light gray.
    :keyword v_measure: the measure of the velocity, defaults to "deg/s".
    :keyword vert_line_colors: the colors of the vertical lines, should either be a single color or a list of colors the
        same length as `vert_lines`. Defaults to black.
    :keyword vert_line_width: the width of the vertical lines, defaults to 1.

    :return: the figure.
    """
    x, y, t, v = __verify_arrays(x=x, y=y, t=t, v=v)
    marker_size = kwargs.get("marker_size", 5)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=t, y=x, mode="markers", name="x", marker=dict(color=kwargs.get("x_color", "#ff0000"), size=marker_size)
        ), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=y, mode="markers", name="y", marker=dict(color=kwargs.get("y_color", "#0000ff"), size=marker_size)
        ), secondary_y=False,
    )

    if v is not None:
        v_measure = kwargs.get("v_measure", "deg/s")
        y_axis2_title = f"{cnst.VELOCITY_STR} ({v_measure})"
        fig.add_trace(
            go.Scatter(
                x=t, y=v, mode="markers", name="v", marker=dict(color=kwargs.get("v_color", "#dddddd"), size=marker_size)
            ), secondary_y=True,
        )
    else:
        y_axis2_title = None

    if vert_lines is not None:
        vert_line_color = kwargs.get("vert_line_color", ["#000000"] * len(vert_lines))
        vert_line_color = [vert_line_color] * len(vert_lines) if isinstance(vert_line_color, str) else vert_line_color
        assert len(vert_lines) == len(vert_line_color), "Length mismatch: `vert_lines` and `vert_line_color`"
        vert_line_width = kwargs.get("vert_line_width", 1)
        for v, c in zip(vert_lines, vert_line_color):
            fig.add_vline(x=v, line=dict(color=c, width=vert_line_width, dash="dash"))

    fig.update_layout(
        title=title,
        xaxis_title=f"{cnst.TIME_STR} ({cnst.SAMPLE_STR})",
        yaxis_title=f"{cnst.PIXEL_STR} coordinates",
        yaxis2_title=y_axis2_title,
    )
    return fig


def __verify_arrays(x: np.ndarray, y: np.ndarray, t: np.ndarray = None, v: np.ndarray = None):
    """  Verifies all input arrays are one dimensional and have the same length.  """
    if not is_one_dimensional(x):
        raise ValueError("`x` must be one-dimensional")
    if not is_one_dimensional(y):
        raise ValueError("`y` must be one-dimensional")
    x, y = x.reshape(-1), y.reshape(-1)
    if len(x) != len(y):
        raise ValueError("`x` and `y` must have the same length")
    if t is not None:
        if not is_one_dimensional(t):
            raise ValueError("`t` must be one-dimensional")
        t = t.reshape(-1)
        if len(x) != len(t):
            raise ValueError("`x` and `t` must have the same length")
    if v is not None:
        if not is_one_dimensional(v):
            raise ValueError("`v` must be one-dimensional")
        v = v.reshape(-1)
        if len(x) != len(v):
            raise ValueError("`x` and `v` must have the same length")
    return x, y, t, v


def __pixel_counts(
        x: np.ndarray,
        y: np.ndarray,
        resolution: Tuple[int, int],
) -> np.ndarray:
    w, h = resolution
    counts = np.zeros((h, w), dtype=int)
    x_int, y_int = cast_to_integers(x, y)
    counter = Counter(zip(y_int, x_int))
    for (y_, x_), count in counter.items():
        counts[y_, x_] = count
    return counts
