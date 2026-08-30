import os
from typing import Tuple, Sequence

import cv2
import numpy as np
from tqdm import trange

import peyes._DataModels.config as cnfg
import peyes._utils.visualization_utils as vis_utils
from peyes._utils.event_utils import calculate_sampling_rate

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
    if not len(t) == len(x) == len(y) == len(labels):
        raise ValueError("All input arrays must have the same length.")
    fps = round(calculate_sampling_rate(t))
    frames = create_frames(
        x=x, y=y, labels=labels, resolution=resolution,
        bg_image=bg_image, bg_image_format=bg_image_format,
        label_colors=label_colors, gaze_radius=gaze_radius, verbose=verbose,
    )
    return _write_video(frames, output_path, fps, codec, extension, verbose)


def create_frames(
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        resolution: Tuple[int, int],
        bg_image: np.ndarray = None,
        bg_image_format: str = "BGR",
        bg_image_alpha: float = 1,
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
    :param bg_image_alpha: alpha (opacity) value of the background image (range [0, 1]). Default is 1 (100% opacity).
    :param label_colors: dictionary mapping event labels to hex/rgb colors. If a label is missing, the default color is used.
    :param gaze_radius: radius of the gaze point in pixels. Default is 10.
    :param verbose: if True, prints progress messages. Default is False.

    :return: list of frames (numpy arrays)
    """
    if not len(x) == len(y) == len(labels):
        raise ValueError("All input arrays must have the same length.")
    frames = []
    n_samples = len(x)
    bg_image = vis_utils.create_image(resolution, bg_image, bg_image_alpha, bg_image_format, "#000000")
    label_colors = vis_utils.get_label_colormap(label_colors)
    for i in trange(n_samples, desc="Creating Frames", disable=not verbose):
        curr_img = bg_image.copy()
        if not (np.isfinite(x[i]) and np.isfinite(y[i])):
            # missing sample (blink or tracker loss): emit the frame without a gaze marker
            frames.append(curr_img)
            continue
        rgb = label_colors.get(labels[i], label_colors[cnfg.EventLabelEnum.UNDEFINED])
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))   # cv2 expects BGR, the colormap is RGB
        cv2.circle(curr_img, (int(x[i]), int(y[i])), gaze_radius, bgr, -1)
        frames.append(curr_img)
    return frames


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
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if verbose:
        print(f"Writing video to: {output_path}")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(output_path, codec, fps, (w, h))
    for i in trange(len(frames), desc="Writing Frames", disable=not verbose):
        frame = frames[i]
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)     # VideoWriter takes 3-channel BGR
        writer.write(frame)
    writer.release()
    if verbose:
        print("Video writing complete.")
    return output_path


