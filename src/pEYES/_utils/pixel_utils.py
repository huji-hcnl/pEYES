from typing import Tuple

import numpy as np

import src.pEYES.constants as cnst


def calculate_pixel_size(width: float, height: float, resolution: Tuple[int, int]) -> float:
    """ Calculates the approximate length of one pixel in centimeters (assuming square pixels): cm/px """
    diagonal_length = np.sqrt(np.power(width, 2) + np.power(height, 2))  # size of diagonal in centimeters
    diagonal_pixels = np.sqrt(np.power(resolution[0], 2) + np.power(resolution[1], 2))  # size of diagonal in pixels
    return diagonal_length / diagonal_pixels


def calculate_velocities(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """
    Calculates the velocity between subsequent pixels in the given x and y coordinates, in pixels per second.
    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param ts: 1D array of timestamps (in milliseconds)
    :return: velocity (pixel / second) between subsequent pixels
    """
    assert len(xs) == len(ys) == len(ts), "`xs`, `ys` and `ts` arrays must be of the same length"
    time_diff = np.diff(ts) / cnst.MILLISECONDS_PER_SECOND  # convert from milliseconds to seconds
    px_distance = np.sqrt(np.power(np.diff(xs), 2) + np.power(np.diff(ys), 2))
    velocities = px_distance / time_diff
    velocities = np.concatenate(([np.nan], velocities))  # first velocity is NaN
    return velocities


def pixels_to_visual_angle(num_px: float, d: float, pixel_size: float, use_radians=False) -> float:
    """
    Calculates the visual angle that corresponds to `num_px` pixels, given that the viewer is sitting at a distance of
    `d` centimeters from the screen, and that the size of each pixel is `pixel_size` centimeters.

    See details on calculations in Kaiser, Peter K. "Calculation of Visual Angle". The Joy of Visual Perception: A Web Book:
        http://www.yorku.ca/eye/visangle.htm

    :param num_px: the number of pixels.
    :param d: the distance (in cm) from the screen.
    :param pixel_size: the size (of the diagonal) of a pixel (in cm).
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.

    :return: the visual angle (in degrees) that corresponds to the given number of pixels.
        If `num_px` is not finite, returns np.nan.

    :raises ValueError: if any of the arguments is negative.
    """
    if not np.isfinite([num_px, d, pixel_size]).all():
        return np.nan
    if (np.array([num_px, d, pixel_size]) < 0).any():
        raise ValueError("arguments `num_px`, `d` and `pixel_size` must be non-negative numbers")
    full_edge_cm = num_px * pixel_size
    half_angle = np.arctan(full_edge_cm / (2 * d))
    angle = 2 * half_angle
    if use_radians:
        return angle
    return np.rad2deg(angle)


def visual_angle_to_pixels(
        angle: float, d: float, pixel_size: float, use_radians: bool = False, keep_sign: bool = False
) -> float:
    """
    Calculates the number of pixels that are equivalent to a visual angle `angle`, given that the viewer is sitting at
    a distance of `d` centimeters from the screen, and that the size of each pixel is `pixel_size` centimeters.
    See details on calculations in Kaiser, Peter K. "Calculation of Visual Angle". The Joy of Visual Perception: A Web
        Book: http://www.yorku.ca/eye/visangle.htm
    :param angle: the visual angle (degrees/radians).
    :param d: the distance of the viewer from the screen (cm).
    :param pixel_size: the size of each pixel (cm).
    :param use_radians: if True, `angle` is in radians; otherwise, it is in degrees.
    :param keep_sign: if True, returns a negative number if `angle` is negative. Otherwise, returns the absolute value.
    :return: the number of pixels that correspond to the given visual angle. If `angle` is not finite, returns np.nan.
    """
    if not np.isfinite([angle, d, pixel_size]).all():
        return np.nan
    if (np.array([d, pixel_size]) <= 0).any():
        raise ValueError("Arguments `d` and `pixel_size` must be positive numbers")
    if angle == 0:
        return 0
    abs_angle = abs(angle) if use_radians else np.deg2rad(abs(angle))
    half_edge_cm = d * np.tan(abs_angle / 2)        # half the edge size in cm
    edge_pixels = 2 * half_edge_cm / pixel_size     # full edge size in pixels
    edge_pixels = np.sign(angle) * edge_pixels if keep_sign else edge_pixels
    return edge_pixels


def calculate_azimuth(
        p1: Tuple[float, float], p2: Tuple[float, float], zero_direction: str = 'E', use_radians: bool = False
) -> float:
    """
    Calculates the counter-clockwise angle between the line starting from p1 and ending at p2, and the line starting
    from p1 and pointing in the direction of `zero_direction`.

    :param p1: the (x,y) coordinates of the starting point of the line
    :param p2: the (x,y) coordinates of the ending point of the line
    :param zero_direction: the direction of the zero angle. Must be one of 'E', 'W', 'S', 'N' (case insensitive).
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.

    :return: the angle between the two lines, in range [0, 2*pi) or [0, 360), or np.nan if either p1 or p2 is invalid.

    :raises ValueError: if zero_direction is not one of 'E', 'W', 'S', 'N' (case insensitive).
    """
    # exit early for invalid pixels:
    if not (np.isfinite(p1[0]) and np.isfinite(p1[1]) and np.isfinite(p2[0]) and np.isfinite(p2[1])):
        return np.nan
    # verify 0-direction
    zero_direction = zero_direction.upper()
    valid_directions = ['E', 'W', 'S', 'N']
    if zero_direction not in valid_directions:
        raise ValueError(f"zero_direction must be one of {valid_directions}")
    # calc angle & adjust to the desired zero direction
    x1, y1 = p1
    x2, y2 = p2
    angle_rad = np.arctan2(y1 - y2, x2 - x1)  # counter-clockwise angle between line (p1, p2) and right-facing x-axis
    if zero_direction == 'W':
        angle_rad += np.pi
    elif zero_direction == 'S':
        angle_rad += np.pi / 2
    elif zero_direction == 'N':
        angle_rad -= np.pi / 2
    # make sure the angle is in range [0, 2*pi), and return
    angle_rad = angle_rad % (2 * np.pi)
    if use_radians:
        return angle_rad
    return np.rad2deg(angle_rad)

