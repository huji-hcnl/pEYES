import numpy as np

import peyes._utils.constants as cnst
import peyes._DataModels.config as cnfg
from peyes._utils.pixel_utils import calculate_pixel_size as calc_ps
from peyes._utils.event_utils import parse_label
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType


def set_viewer_distance(distance_cm: float) -> None:
    """ Sets the default viewer distance (in centimeters). """
    if distance_cm <= 0:
        raise ValueError("Viewer distance must be positive")
    cnfg.VIEWER_DISTANCE = distance_cm


def set_screen_monitor(
        width_cm: float = None, height_cm: float = None, width_px: int = None, height_px: int = None,
) -> None:
    """
    Sets the default screen monitor configuration: width and height in centimeters, and resolution in pixels.
    Then, calculates the default pixel size based on the provided dimensions and resolution.
    """
    if width_cm is not None:
        if width_cm <= 0:
            raise ValueError("Screen width (cm) must be positive")
        cnfg.SCREEN_MONITOR[cnst.WIDTH_STR] = width_cm
    if height_cm is not None:
        if height_cm <= 0:
            raise ValueError("Screen height (cm) must be positive")
        cnfg.SCREEN_MONITOR[cnst.HEIGHT_STR] = height_cm
    if width_px is not None or height_px is not None:
        width_px = width_px if width_px is not None else cnfg.SCREEN_MONITOR[cnst.RESOLUTION_STR][0]
        height_px = height_px if height_px is not None else cnfg.SCREEN_MONITOR[cnst.RESOLUTION_STR][1]
        if width_px <= 0 or height_px <= 0:
            raise ValueError("Screen resolution (px) must be positive")
        cnfg.SCREEN_MONITOR[cnst.RESOLUTION_STR] = (width_px, height_px)
    cnfg.SCREEN_MONITOR[cnst.PIXEL_SIZE_STR] = calc_ps(
        cnfg.SCREEN_MONITOR[cnst.WIDTH_STR],
        cnfg.SCREEN_MONITOR[cnst.HEIGHT_STR],
        cnfg.SCREEN_MONITOR[cnst.RESOLUTION_STR],
    )


def set_event_configurations(
        event_type: UnparsedEventLabelType,
        min_duration: float = None,
        max_duration: float = None,
        hex_color: str = None,
        min_velocity: float = None,
        max_velocity: float = None,
        min_acceleration: float = None,
        max_acceleration: float = None,
) -> None:
    """
    Sets the event-type's default configuration:
    - minimum/maximum duration (in milliseconds)
    - minimum/maximum peak velocity (in degrees/second), used for outlier detection
    - minimum/maximum peak acceleration (in degrees/second^2), used for outlier detection
    - plotting color (in hex format)

    Pass `np.nan` for a velocity/acceleration threshold to explicitly disable that specific outlier check.
    """
    if min_duration is not None:
        _set_min_duration(event_type, min_duration)
    if max_duration is not None:
        _set_max_duration(event_type, max_duration)
    if hex_color is not None:
        set_event_color(event_type, hex_color)
    if min_velocity is not None:
        _set_outlier_threshold(event_type, cnst.MIN_VELOCITY_STR, min_velocity)
    if max_velocity is not None:
        _set_outlier_threshold(event_type, cnst.MAX_VELOCITY_STR, max_velocity)
    if min_acceleration is not None:
        _set_outlier_threshold(event_type, cnst.MIN_ACCELERATION_STR, min_acceleration)
    if max_acceleration is not None:
        _set_outlier_threshold(event_type, cnst.MAX_ACCELERATION_STR, max_acceleration)


def _set_outlier_threshold(event_type: UnparsedEventLabelType, key: str, value: float) -> None:
    """ Sets a min/max velocity or acceleration outlier threshold for the specified event type. `np.nan` disables
    that specific check (a missing/None/NaN threshold is a no-op in `BaseEvent.get_outlier_reasons`). """
    if not np.isnan(value) and value < 0:
        raise ValueError(f"`{key}` must be non-negative, or NaN to disable this check")
    label = parse_label(event_type, safe=False)
    cnfg.EVENT_MAPPING[label][key] = value


def _set_min_duration(event_type: UnparsedEventLabelType, duration: float) -> None:
    """ Sets the minimum duration (in milliseconds) for the specified event type. """
    if duration <= 0:
        raise ValueError("Minimum duration must be positive")
    label = parse_label(event_type, safe=False)
    cnfg.EVENT_MAPPING[label][cnst.MIN_DURATION_STR] = duration


def _set_max_duration(event_type: UnparsedEventLabelType, duration: float) -> None:
    """ Sets the maximum duration (in milliseconds) for the specified event type. """
    if duration <= 0:
        raise ValueError("Maximum duration must be positive")
    label = parse_label(event_type, safe=False)
    cnfg.EVENT_MAPPING[label][cnst.MAX_DURATION_STR] = duration


def set_event_color(event_type: UnparsedEventLabelType, hex_color: str) -> None:
    """ Sets the color (in hex format) for the specified event type. """
    assertion_message = "Color must be a valid hex string."
    if len(hex_color) == 7:
        if hex_color[0] != "#" or not all(c in "0123456789abcdefABCDEF" for c in hex_color[1:]):
            raise ValueError(assertion_message)
    elif len(hex_color) == 6:
        if not all(c in "0123456789abcdefABCDEF" for c in hex_color):
            raise ValueError(assertion_message)
        hex_color = "#" + hex_color
    else:
        raise ValueError(assertion_message)
    label = parse_label(event_type, safe=False)
    cnfg.EVENT_MAPPING[label][cnst.COLOR_STR] = hex_color

