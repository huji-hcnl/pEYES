from typing import Tuple, Union
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from plotly.subplots import make_subplots

import peyes._utils.constants as cnst
import peyes._utils.visualization_utils as vis_utils
from peyes._utils.vector_utils import is_one_dimensional
from peyes._utils.pixel_utils import cast_to_integers

# TODO: create figure of y-coordinates over x-coordinates with color changing according to event labels


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
    :keyword bg_alpha: the alpha value of the background image (range [0, 1]), defaults to 1.
    :keyword sigma: standard deviation of the Gaussian filter that smooths the heatmap, defaults to 10.0
    :keyword scale: controls how much of the smoothed density map is masked out as "low value" before
        display, relative to its own default of sigma^2 (V-13 fix - a uniform scale factor on the raw pixel
        counts has no effect by itself, since the result is always min-max normalized to [0, 1] regardless;
        scale instead adjusts the post-normalization masking *percentile*, proportionally to how far it is
        from sigma^2 - a percentile rather than a value-ratio specifically so this stays meaningful even when
        the un-scaled threshold would be exactly 0, which is common whenever most of the canvas has no gaze
        samples nearby). A larger scale reveals more of the map (lower masking percentile, clipped to
        [1, 99]); a smaller one shows only the most concentrated regions. Passing exactly sigma^2 (or leaving
        scale unset) reproduces the 50th-percentile (median) threshold every existing caller already sees,
        unchanged.
    :keyword colorscale: name of the color scale to use. Must be one of the named color scales in plotly.express.colors
    :keyword opacity: opacity of the heatmap (0-1). Default is 0.5

    :return: the figure.
    """
    x, y = __verify_same_length(x, y)
    bg = vis_utils.create_image(
        resolution,
        image=kwargs.get("bg_image", None),
        alpha=kwargs.get("bg_alpha", 1),
        color_format=kwargs.get("bg_image_format", "BGR"),
        default_color=kwargs.get("bg_color", "#ffffff"),
    )
    fig = go.Figure(
        data=go.Image(z=bg),
        layout=dict(width=resolution[0], height=resolution[1], margin=dict(l=0, r=0, b=0, t=0)),
    )
    sig = kwargs.get("sigma", 10.0)
    default_scale = sig ** 2
    scale = kwargs.get("scale", default_scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    # __pixel_counts returns an integer array; gaussian_filter preserves input dtype, so without this cast the
    # "smoothed" density silently truncates to integers (often all-zero for sparse input relative to sigma) -
    # found while verifying the scale fix below, not a change it depends on, but adjacent enough to fix here
    # rather than leave a truncated-to-int heatmap next to newly-meaningful threshold arithmetic.
    counts = __pixel_counts(x, y, resolution).astype(float)
    filtered_counts = gaussian_filter(counts, sigma=sig)
    counts_min, counts_max = np.nanmin(filtered_counts), np.nanmax(filtered_counts)
    if counts_max > counts_min:
        heatmap = (filtered_counts - counts_min) / (counts_max - counts_min)
    else:
        # flat input (e.g. all-zero counts): every cell is masked out below anyway, skip the 0/0 division (V-11)
        heatmap = np.full_like(filtered_counts, np.nan)
    # remove low values; `scale` (V-13) shifts this threshold's percentile relative to its sigma^2 default -
    # see docstring. mask_percentile == 50 (median) exactly when scale == default_scale, matching every
    # existing caller's behavior unchanged.
    mask_percentile = np.clip(50 * (default_scale / scale), 1, 99)
    threshold = np.nanpercentile(heatmap, mask_percentile)
    heatmap[(~np.isfinite(heatmap)) | (heatmap <= threshold)] = np.nan
    fig.add_trace(go.Heatmap(
        z=heatmap,
        colorscale=kwargs.get("colorscale", "Jet"),
        opacity=kwargs.get("opacity", 0.5),
        showscale=False,
    ))
    fig.update_xaxes(
        visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False
    )
    fig.update_yaxes(
        visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False
    )
    fig.update_layout(
        paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)',  # transparent background
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
    :param v: gaze velocity, optional. If provided, will be plotted on a secondary y-axis. Note the units of the
        velocity are px/s by default, see `v_measure`.
    :param vert_lines: time points to add vertical lines at, optional.
    :param title: the title of the figure.

    :keyword mode: 'lines'/'markers'/'lines+markers', defaults to 'lines'.
    :keyword line_width: width of the plotted lines, if `mode` contains 'lines'. defaults to 2.
    :keyword marker_size: the size of the markers, if `mode` contains 'markers'. defaults to 4.
    :keyword x_color: the color of the x-coordinates, defaults to red.
    :keyword y_color: the color of the y-coordinates, defaults to blue.
    :keyword v_color: the color of the velocity, defaults to light gray.
    :keyword v_measure: the measure of the velocity, defaults to "px/s".
    :keyword vert_line_colors: the colors of the vertical lines, should either be a single color or a list of colors the
        same length as `vert_lines`. Defaults to black.
    :keyword vert_line_width: the width of the vertical lines, defaults to 1.

    :return: the figure.
    """
    if v is not None:
        x, y, t, v = __verify_same_length(x, y, t, v)
    else:
        x, y, t = __verify_same_length(x, y, t)
    mode = kwargs.get('mode', 'lines')
    line_width = kwargs.get('line_width', 2)
    marker_size = kwargs.get("marker_size", 4)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=t, y=x, name="x",
            mode=mode,
            line=dict(color=kwargs.get("x_color", "#ff0000"), width=line_width),
            marker=dict(color=kwargs.get("x_color", "#ff0000"), size=marker_size),
        ), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=y, name="y",
            mode=mode,
            line=dict(color=kwargs.get("y_color", "#0000ff"), width=line_width),
            marker=dict(color=kwargs.get("y_color", "#0000ff"), size=marker_size),
        ), secondary_y=False,
    )

    if v is not None:
        v_measure = kwargs.get("v_measure", "px/s")
        y_axis2_title = f"{cnst.VELOCITY_STR} ({v_measure})"
        fig.add_trace(
            go.Scatter(
                x=t, y=v, name="v",
                mode=mode,
                line=dict(color=kwargs.get("v_color", "#bbbbbb"), width=line_width),
                marker=dict(color=kwargs.get("v_color", "#bbbbbb"), size=marker_size),
            ), secondary_y=True,
        )

        # align y-axis scales on (0_1, 0_2) and (max-val_1, max-val_2)
        v_max = np.nanmax(v)
        xy_max = max(np.nanmax(x), np.nanmax(y))
        scaleratio = xy_max / v_max if v_max else 1  # avoid a division by zero for constant/zero velocity (V-11)
        fig.update_layout(yaxis2=dict(scaleanchor='y', scaleratio=scaleratio, constraintoward='bottom'))

    else:
        y_axis2_title = None

    if vert_lines is not None:
        vert_line_colors = kwargs.get("vert_line_colors", ["#000000"] * len(vert_lines))
        vert_line_colors = [vert_line_colors] * len(vert_lines) if isinstance(vert_line_colors, str) else vert_line_colors
        if len(vert_lines) != len(vert_line_colors):
            raise ValueError("Length mismatch: `vert_lines` and `vert_line_colors`")
        vert_line_width = kwargs.get("vert_line_width", 1)
        for vline, c in zip(vert_lines, vert_line_colors):
            fig.add_vline(x=vline, line=dict(color=c, width=vert_line_width, dash="dash"))

    fig.update_layout(
        title=title,
        xaxis_title=f"{cnst.TIME_STR} (ms)",
        yaxis_title=f"{cnst.PIXEL_STR} coordinates",
        yaxis2_title=y_axis2_title,
    )
    return fig


def gaze_trajectory(
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        resolution: Tuple[int, int],
        title: str = "Gaze Visualization",
        **kwargs,
) -> go.Figure:
    return _visualize_gaze_trajectory(
        x=x,
        y=y,
        resolution=resolution,
        title=title,
        bg_image=kwargs.get("bg_image", None),
        bg_image_format=kwargs.get("bg_image_format", "BGR"),
        bg_alpha=kwargs.get("bg_alpha", 1),
        bg_color=kwargs.get("bg_color", "#ffffff"),
        marker_color=t,
        marker_size=kwargs.get("marker_size", 5),
        marker_alpha=kwargs.get("marker_alpha", 1),
        marker_colorscale=kwargs.get("colorscale", "Jet"),
        marker_colorbar_title=cnst.TIME_STR.title(),
    )



def _visualize_gaze_trajectory(
        x: np.ndarray,
        y: np.ndarray,
        resolution: Tuple[int, int],
        title: str = None,
        bg_image: np.ndarray = None,
        bg_image_format: str = "BGR",
        bg_alpha: float = 1,
        bg_color: str = "#ffffff",
        marker_color: Union[str, np.ndarray] = "#000000",
        marker_size: Union[int, np.ndarray] = 5,
        marker_alpha: Union[float, np.ndarray] = 1,
        marker_colorscale: str = "Jet",
        marker_colorbar_title: str = None,
) -> go.Figure:
    if isinstance(marker_color, str):
        # np.full_like(x, marker_color) tries to cast the color string into x's own (numeric) dtype and
        # raises; build a plain object array of the color string instead (V-9).
        marker_color = np.full(len(x), marker_color, dtype=object)
    if isinstance(marker_size, int) or isinstance(marker_size, float):
        marker_size = np.full_like(x, marker_size)
    if isinstance(marker_alpha, int) or isinstance(marker_alpha, float):
        marker_alpha = np.full_like(x, marker_alpha)
    x, y, marker_color, marker_size, marker_alpha = __verify_markers(x, y, marker_color, marker_size, marker_alpha)
    bg = vis_utils.create_image(
        resolution, image=bg_image, color_format=bg_image_format, alpha=bg_alpha, default_color=bg_color
    )
    fig = go.Figure(
        data=go.Image(z=bg, colormodel='rgba256'),
        layout=dict(width=resolution[0], height=resolution[1], margin=dict(l=0, r=0, b=0, t=0)),
    )
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", marker=dict(
            color=marker_color,
            size=marker_size,
            opacity=marker_alpha,
            colorscale=marker_colorscale,
            colorbar=dict(title=marker_colorbar_title),
            line=dict(width=0),
        )
    ))
    fig.update_layout(
        title=title, title_y=0.98, title_yanchor='top',
        xaxis=dict(visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False),
        yaxis=dict(visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False),
        paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)',  # transparent background
    )
    return fig


def __verify_markers(
        x: np.ndarray, y: np.ndarray, color: np.ndarray, size: np.ndarray, alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Verifies all marker arrays have valid values and the same length as the x and y arrays. """
    if not all(size >= 0):
        raise ValueError("`size` must be non-negative")
    if not all((0 <= alpha) & (alpha <= 1)):
        raise ValueError("`alpha` must be in the range [0, 1]")
    x, y, color, size, alpha = __verify_same_length(x, y, color, size, alpha)
    return x, y, color, size, alpha


def __verify_same_length(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """  Verifies all input arrays are one dimensional and have the same length.  """
    if not all(is_one_dimensional(arr) for arr in arrays):
        raise ValueError("All arrays must be one-dimensional")
    if not all(len(arr) == len(arrays[0]) for arr in arrays):
        raise ValueError("All arrays must have the same length")
    return tuple(arr.reshape(-1) for arr in arrays)


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
        if not np.isfinite(x_) or not np.isfinite(y_):
            continue
        row, col = int(y_), int(x_)
        if not (0 <= row < h and 0 <= col < w):
            continue    # off-screen sample: drop it rather than let a negative index wrap
        counts[row, col] = count
    return counts
