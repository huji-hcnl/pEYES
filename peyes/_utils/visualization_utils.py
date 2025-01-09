import os
from typing import Union, Tuple, Dict, Sequence

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from peyes._DataModels.EventLabelEnum import EventLabelEnum as _EventLabelEnum
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType as _UnparsedEventLabelType

ColorType = Union[str, Tuple[int, int, int]]
LabelColormapType = Dict[_UnparsedEventLabelType, ColorType]

_DISCRETE_COLORMAP = px.colors.qualitative.Dark24
_DEFAULT_COLORMAP = {
    _EventLabelEnum.UNDEFINED: "#dddddd",
    _EventLabelEnum.FIXATION: "#1f78b4",
    _EventLabelEnum.SACCADE: "#33a02c",
    _EventLabelEnum.PSO: "#b2df8a",
    _EventLabelEnum.SMOOTH_PURSUIT: "#fb9a99",
    _EventLabelEnum.BLINK: "#222222",
    "ra": _DISCRETE_COLORMAP[0],
    "rz": _DISCRETE_COLORMAP[0],
    'ih': _DISCRETE_COLORMAP[0],
    "mn": _DISCRETE_COLORMAP[1],
    'dn': _DISCRETE_COLORMAP[1],
    "ivt": _DISCRETE_COLORMAP[2],
    "ivtdetector": _DISCRETE_COLORMAP[2],
    "ivvt": _DISCRETE_COLORMAP[3],
    "ivvtdetector": _DISCRETE_COLORMAP[3],
    "idt": _DISCRETE_COLORMAP[4],
    "idtdetector": _DISCRETE_COLORMAP[4],
    "idvt": _DISCRETE_COLORMAP[5],
    "idvtdetector": _DISCRETE_COLORMAP[5],
    "engbert": _DISCRETE_COLORMAP[6],
    "engbertdetector": _DISCRETE_COLORMAP[6],
    "nh": _DISCRETE_COLORMAP[7],
    "nhdetector": _DISCRETE_COLORMAP[7],
    "remodnav": _DISCRETE_COLORMAP[8],
    "remodnavdetector": _DISCRETE_COLORMAP[8],
}


def save_figure(
        fig: go.Figure, fig_name: str, output_dir: str,
        as_json: bool = False, as_html: bool = False, as_png: bool = False, as_eps: bool = False
):
    os.makedirs(output_dir, exist_ok=True)
    if as_json:
        fig.write_json(os.path.join(output_dir, f"{fig_name}.json"))
    if as_html:
        fig.write_html(os.path.join(output_dir, f"{fig_name}.html"))
    if as_png:
        fig.write_image(os.path.join(output_dir, f"{fig_name}.png"))
    if as_eps:
        fig.write_image(os.path.join(output_dir, f"{fig_name}.eps"))


def create_image(
        resolution: Tuple[int, int],
        image: np.ndarray = None,
        alpha: float = 1,
        color_format: str = "BGR",
        default_color: ColorType = (0, 0, 0),
):
    """
    Creates an image in BGR format with the specified resolution. If a background image is not provided, an image with
    the specified background color will be created.

    :param resolution: tuple of (width, height) in pixels
    :param image: background image (numpy array)
    :param alpha: alpha (opacity) value of the background image, range [0, 1]. Only used if input image doesn't have
    an alpha channel (i.e. `color_format` is not RGBA or BGRA). Default is 1 (opaque).
    :param color_format: color format of the background image (RGB/GRAY/BGR). Default is BGR.
    :param default_color: background color (RGB tuple or hex string). Default is black.

    :return: numpy array of the image
    """
    if not resolution or len(resolution) != 2 or resolution[0] <= 0 or resolution[1] <= 0:
        raise ValueError("resolution must be a tuple of two positive integers")
    if image.ndim != 4 and (alpha < 0 or alpha > 1):
        raise ValueError("bg_alpha must be in the range [0, 1]")

    # If bg_image is not provided or invalid, create an image with the specified RGB background color
    if image is None or image.size == 0 or image.ndim not in (2, 3, 4):
        default_color = to_rgb(default_color)
        image = np.full((resolution[1], resolution[0], 3), default_color, dtype=np.uint8)
        color_format = "RGB"

    # Convert the background image to BGRA format
    if color_format.upper() == "BGRA":
        bg = image
    elif color_format.upper() == "BGR":
        bg = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        bg[:, :, 3] = round(alpha * 255)
    elif color_format.upper() == "RGBA":
        bg = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    elif color_format.upper() == "RGB":
        bg = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
        bg[:, :, 3] = round(alpha * 255)
    elif color_format.upper() == "GRAY" or color_format.upper() == "GREY":
        bg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        bg[:, :, 3] = round(alpha * 255)
    else:
        raise ValueError(f"Invalid color format: {color_format}")
    return cv2.resize(bg, resolution)


def get_label_colormap(
        label_colors: LabelColormapType = None
):
    """
    Returns a dictionary mapping event labels to RGB colors.
    If a custom mapping is provided, it will override the default colors for the labels that are present, and use the
    default colors for the rest.
    """
    default_colors_rgb = {k: to_rgb(v) for k, v in _DEFAULT_COLORMAP.items()}
    if label_colors is None:
        return default_colors_rgb
    event_colors_rgb = {k: to_rgb(v) if isinstance(v, str) else v for k, v in label_colors.items()}
    return {**default_colors_rgb, **event_colors_rgb}


def to_rgb(color: ColorType) -> Tuple[int, int, int]:
    """
    Converts a hex color code to an RGB tuple.
    :param color: color code in hex string (e.g.: '#FF0000') or RGB tuple (e.g.: (255, 0, 0))
    :return: RGB tuple (R, G, B)
    """
    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError("RGB tuple must have 3 elements.")
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        return rgb
    if isinstance(color, str):
        if not color.startswith("#") or len(color) != 7:
            raise ValueError("Hex color code must be in the format '#RRGGBB'.")
        color = color.removeprefix("#")
        rgb = tuple(int(color[i: i + 2], 16) for i in (0, 2, 4))
        return rgb
    raise ValueError("Invalid color format. Must be hex string or RGB tuple.")


def make_empty_figure(
        subtitles: Union[str, Sequence[str]],
        sharex=False,
        sharey=False,
        subplots_vspace: float = 0.1,
        subplots_hspace: float = 0.1,
) -> Tuple[go.Figure, int, int]:
    if isinstance(subtitles, str):
        subtitles = [subtitles]
    ncols = 1 if len(subtitles) <= 3 else 2 if len(subtitles) <= 8 else 3
    nrows = len(subtitles) if len(subtitles) <= 3 else sum(divmod(len(subtitles), ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols,
        shared_xaxes=sharex, shared_yaxes=sharey,
        vertical_spacing=subplots_vspace, horizontal_spacing=subplots_hspace,
        subplot_titles=subtitles,
    )
    return fig, nrows, ncols
