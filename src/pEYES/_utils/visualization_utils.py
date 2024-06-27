from typing import Union, Tuple, Dict

from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum as _EventLabelEnum
from src.pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType as _UnparsedEventLabelType

ColorType = Union[str, Tuple[int, int, int]]
LabelColormapType = Dict[_UnparsedEventLabelType, ColorType]

_DEFAULT_LABEL_COLORMAP = {
    _EventLabelEnum.UNDEFINED: "#dddddd",
    _EventLabelEnum.FIXATION: "#1f78b4",
    _EventLabelEnum.SACCADE: "#33a02c",
    _EventLabelEnum.PSO: "#b2df8a",
    _EventLabelEnum.SMOOTH_PURSUIT: "#fb9a99",
    _EventLabelEnum.BLINK: "#222222",
}


def get_label_colormap(
        label_colors: LabelColormapType = None
):
    """
    Returns a dictionary mapping event labels to RGB colors.
    If a custom mapping is provided, it will override the default colors for the labels that are present, and use the
    default colors for the rest.
    """
    default_colors_rgb = {k: to_rgb(v) for k, v in _DEFAULT_LABEL_COLORMAP.items()}
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
