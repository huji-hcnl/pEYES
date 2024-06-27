from typing import Tuple
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import plotly.express.colors as px_colors

from src.pEYES._utils.vector_utils import is_one_dimensional, normalize
from src.pEYES._utils.pixel_utils import cast_to_integers
import src.pEYES._utils.visualization_utils as vis_utils


def heatmap(
        x: np.ndarray,
        y: np.ndarray,
        resolution: Tuple[int, int],
        bg_image: np.ndarray = None,
        bg_image_format: str = "BGR",
        sigma: float = 10.0,
        color_scale: str = "jet",
        opacity: float = 0.5,
) -> go.Figure:
    """
    Creates a heatmap of the given x and y coordinates, overlayed on a background image or a white background.

    :param x: 1D array of x coordinates
    :param y: 1D array of y coordinates
    :param resolution: tuple of (width, height) in pixels
    :param bg_image: (optional) background image as a numpy array
    :param bg_image_format: color format of the background image (RGB/GRAY/BGR). Default is BGR
    :param sigma: standard deviation of the Gaussian filter
    :param color_scale: name of the color scale to use. Must be one of the named color scales in plotly.express.colors
    :param opacity: opacity of the heatmap (0-1). Default is 0.5

    :return: plotly figure of the heatmap
    """
    if not is_one_dimensional(x) or not is_one_dimensional(y):
        raise ValueError("x and y must be 1D arrays")
    if not len(x) == len(y):
        raise ValueError("x and y must be of the same length")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite number")
    if not resolution or len(resolution) != 2 or resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("resolution must be a tuple of two positive integers")
    if not color_scale or color_scale.lower() not in px_colors.named_colorscales():
        raise ValueError("Invalid color scale")
    if not 0 <= opacity <= 1:
        raise ValueError("Opacity must be between 0 and 1")
    bg = vis_utils.create_image(resolution, bg_image, bg_image_format, (255, 255, 255))   # default background color is white
    fig = go.Figure(
        data=go.Image(z=bg),
        layout=dict(width=resolution[0], height=resolution[1], margin=dict(l=0, r=0, b=0, t=0)),
    )
    counts = __pixel_counts(x, y, resolution)
    normalized_counts = normalize(gaussian_filter(counts, sigma))
    colorscale = px_colors.get_colorscale(color_scale)
    fig.add_trace(go.Heatmap(z=normalized_counts, colorscale=colorscale, opacity=opacity, showscale=False))
    fig.update_xaxes(
        range=[0, resolution[0]], visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False
    )
    fig.update_yaxes(
        range=[0, resolution[1]], visible=False, showticklabels=False, showgrid=False, showline=False, zeroline=False
    )
    return fig


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
