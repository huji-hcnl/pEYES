import os
from typing import Tuple, Sequence

import cv2
import numpy as np
from tqdm import trange

import src.pEYES._utils.visualization_utils as vis_utils
from src.pEYES._utils.event_utils import calculate_sampling_rate
import src.pEYES._DataModels.config as cnfg

_DEFAULT_EXTENSION = ".mp4"
_DEFAULT_CODEC = cv2.VideoWriter_fourcc(*"mp4v")


def create_video(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        output_path: str,
        resolution: Tuple[int, int],
        bg_image: np.ndarray = None,
        bg_image_format: str = "BGR",
        label_colors: vis_utils.LabelColormapType = None,
        gaze_radius: int = 10,
        codec: int = _DEFAULT_CODEC,
        extension: str = _DEFAULT_EXTENSION,
        verbose: bool = False,
) -> str:
    """
    Creates a video file from gaze data.

    :param t: array of timestamps (ms)
    :param x: array of x-coordinates (pixels)
    :param y: array of y-coordinates (pixels)
    :param labels: array of event labels
    :param output_path: full path to the output video file
    :param resolution: tuple of (width, height) in pixels

    Optional Parameters:
    :param bg_image: background image (numpy array). If None, a black background will be used (default).
    :param bg_image_format: color format (RGB/BGR) for the background image, if provided. Default is BGR.
    :param label_colors: dictionary mapping event labels to hex/rgb colors. Default is the event color mapping from `config.py`.
    :param gaze_radius: radius of the gaze point in pixels. Default is 10.
    :param codec: codec for the video writer. Default is `mp4v`.
    :param extension: file extension for the output video file. Default is `.mp4`.
    :param verbose: if True, prints progress messages. Default is False.

    :return: full path to the output video file
    """
    assert len(t) == len(x) == len(y) == len(labels), "All input arrays must have the same length."
    fps = round(calculate_sampling_rate(t))
    frames = create_frames(x, y, labels, resolution, bg_image, bg_image_format, label_colors, gaze_radius, verbose)
    return _write_video(frames, output_path, fps, codec, extension, verbose)


def create_frames(
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        resolution: Tuple[int, int],
        bg_image: np.ndarray = None,
        bg_image_format: str = "BGR",
        label_colors: vis_utils.LabelColormapType = None,
        gaze_radius: int = 10,
        verbose: bool = False,
) -> Sequence[np.ndarray]:
    """
    Creates a sequence of frames (numpy arrays) from gaze data.

    :param x: array of x-coordinates (pixels)
    :param y: array of y-coordinates (pixels)
    :param labels: array of event labels
    :param resolution: tuple of (width, height) in pixels

    Optional Parameters:
    :param bg_image: background image (numpy array). If None, a black background will be used (default).
    :param bg_image_format: color format (RGB/BGR) for the background image, if provided. Default is BGR.
    :param label_colors: dictionary mapping event labels to hex/rgb colors. If a label is missing, the default color is used.
    :param gaze_radius: radius of the gaze point in pixels. Default is 10.
    :param verbose: if True, prints progress messages. Default is False.

    :return: list of frames (numpy arrays)
    """
    assert len(x) == len(y) == len(labels), "All input arrays must have the same length."
    frames = []
    n_samples = len(x)
    bg_image = _create_background(resolution, bg_image, bg_image_format)
    label_colors = vis_utils.get_label_colormap(label_colors)
    for i in trange(n_samples, desc="Creating Frames", disable=not verbose):
        curr_img = bg_image.copy()
        curr_x, curr_y = int(x[i]), int(y[i])
        color = label_colors.get(labels[i], label_colors[cnfg.EventLabelEnum.UNDEFINED])
        cv2.circle(curr_img, (curr_x, curr_y), gaze_radius, color, -1)
        frames.append(curr_img)
    return frames


def _create_background(
        resolution: Tuple[int, int],
        image: np.ndarray = None,
        color_format: str = "BGR",
) -> np.ndarray:
    if image is None or not image or image.size == 0:
        bg = np.zeros((*resolution, 3), dtype=np.uint8)
    else:
        if color_format == "BGR":
            bg = image
        elif color_format.lower() == 'rgb':
            bg = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Invalid color format: {color_format}")
    bg = cv2.resize(bg, resolution)
    return bg


def _write_video(
        frames: Sequence[np.ndarray],
        output_path: str,
        fps: int = 30,
        codec: int = _DEFAULT_CODEC,
        extension: str = _DEFAULT_EXTENSION,
        verbose: bool = False,
) -> str:
    if not output_path.endswith(extension):
        output_path += extension
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if verbose:
        print(f"Writing video to: {output_path}")
    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(output_path, codec, fps, (w, h))
    for i in trange(len(frames), desc="Writing Frames", disable=not verbose):
        writer.write(frames[i])
    writer.release()
    if verbose:
        print("Video writing complete.")
    return output_path


