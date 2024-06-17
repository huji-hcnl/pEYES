from typing import Union, Tuple, Dict, Any, List

import numpy as np
from tqdm import tqdm

import src.pEYES.config as cnfg
from src.pEYES._DataModels.Detector import (
    IVTDetector, IVVTDetector, IDTDetector, EngbertDetector, NHDetector, REMoDNaVDetector
)


def detect(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        viewer_distance: float,
        pixel_size: float,
        detector_name: str,
        missing_value: float = np.nan,
        min_event_duration: float = cnfg.MIN_EVENT_DURATION,
        pad_blinks_time: float = 0,
        include_metadata: bool = False,
        **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Detects gaze events in the given gaze data using the specified detector, and returns the detected label for each
    sample. Optionally, returns the metadata of the detection process.

    :param t: np.ndarray; the timestamps of the gaze data, in milliseconds
    :param x: np.ndarray; the x-coordinates of the gaze data, in pixels
    :param y: np.ndarray; the y-coordinates of the gaze data, in pixels
    :param viewer_distance: float; the distance between the viewer and the screen, in centimeters
    :param pixel_size: float; the size of a pixel on the screen, in centimeters
    :param detector_name: str; the name of the detector to use.
        Must be one of: "IVT", "IVVT", "IDT", "Engbert", "NH", "REMoDNaV"
    :param missing_value: float; the value used to represent missing data in the gaze data
    :param min_event_duration: float; the minimum duration of a gaze event, in milliseconds
    :param pad_blinks_time: float; the duration to pad before and after each detected blink event, in milliseconds
    :param include_metadata: bool; if True, also returns the metadata of the detection process

    Additional keyword arguments required for detector initialization:
    - IVT:
        :keyword saccade_velocity_threshold: float; velocity threshold to separates saccades from fixations (deg/s), default: 45
    - IVVT:
        :keyword saccade_velocity_threshold: float; velocity threshold to separates saccades from smooth pursuits (deg/s), default: 45
        :keyword smooth_pursuit_velocity_threshold: float; the velocity threshold that separates smooth pursuits from fixations (deg/s), default: 5
    - IDT:
        :keyword dispersion_threshold: dispersion threshold that separates saccades from fixations (visual angle degrees), default: 0.5
        :keyword window_duration: the duration of the window used to calculate the dispersion threshold (ms), default: 100
    - Engbert:
        :keyword lambda_param: float; multiplication coefficient used for calculating saccade threshold, default: 5
        :keyword deriv_window_size: int; number of samples used to compute axial velocity, default: 5
    - NH:
        :keyword filter_duration_ms: int; Savitzky-Golay filter's duration (ms), default: 20
        :keyword filter_polyorder: int; Savitzky-Golay filter's polynomial order, default: 2
        :keyword saccade_max_velocity: maximum saccade velocity (deg/s), default: 1000
        :keyword saccade_max_acceleration: maximum saccade acceleration (deg/s^2), default: 100000
        :keyword min_saccade_duration: minimum saccade duration (ms), default: 10
        :keyword min_fixation_duration: minimum fixation duration (ms), default: 50
        :keyword max_pso_duration: maximum PSO duration (ms), default: 80
        :keyword alpha_param: weight of saccade onset threshold when detecting saccade offset, default: 0.7
        :keyword ignore_short_peak_durations: if True, excludes sporadic occurrences of peak velocity when detecting
            saccade peaks, default: True
        :keyword allow_high_psos: if True, includes "high" PSOs, i.e., PSOs with max velocity exceeding saccades' peak threshold but lower than the preceding saccade, default: True
    - REMoDNaV:
        :keyword min_saccade_duration: int; the minimum duration of a saccade (ms), default: 10
        :keyword saccade_initial_velocity_threshold: float; the initial velocity threshold for saccade detection (deg/s), default: 300
        :keyword saccade_context_window_duration: int; the duration of the context window for saccade detection (ms), default: 1000
        :keyword saccade_initial_max_freq: float; the initial maximum frequency for saccade detection (Hz), default: 2.0
        :keyword saccade_onset_threshold_noise_factor: float; the noise factor for saccade onset threshold, default: 5.0
        :keyword min_smooth_pursuit_duration: int; the minimum duration of a smooth pursuit (ms), default: 40
        :keyword smooth_pursuits_lowpass_cutoff_freq: float; the lowpass cutoff frequency for smooth pursuit detection (Hz), default: 4.0
        :keyword smooth_pursuit_drift_velocity_threshold: float; the drift velocity threshold for smooth pursuit detection (deg/s), default: 2.0
        :keyword min_fixation_duration: int; the minimum duration of a fixation (ms), default: 50
        :keyword min_blink_duration: int; the minimum duration of a blink (ms), default: 20
        :keyword max_pso_duration: int; the maximum duration of a PSO (ms), default: 80
        :keyword savgol_filter_polyorder: int; the polynomial order for the Savitzky-Golay filter, default: 2
        :keyword savgol_filter_duration_ms: int; the duration of the Savitzky-Golay filter (ms), default: 19
        :keyword median_filter_duration_ms: int; the duration of the median filter (ms), default: 50
        :keyword max_velocity: float; the maximum velocity of the gaze data (deg/s), default: 1500

    :return:
        labels: np.ndarray; the detected labels for each sample
        metadata: Dict[str, Any]; the metadata of the detection process, returned only if `include_metadata` is True
    """
    detector = _create_detector(detector_name, missing_value, min_event_duration, pad_blinks_time, **kwargs)
    labels, metadata = detector.detect(t, x, y, viewer_distance, pixel_size)
    if include_metadata:
        return labels, metadata
    return labels


def detect_multiple(
        ts: List[np.ndarray],
        xs: List[np.ndarray],
        ys: List[np.ndarray],
        viewer_distance: Union[float, List[float]],
        pixel_size: Union[float, List[float]],
        detector_name: str,
        missing_value: float = np.nan,
        min_event_duration: float = cnfg.MIN_EVENT_DURATION,
        pad_blinks_time: float = 0,
        include_metadata: bool = False,
        verbose: bool = False,
        **kwargs
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[Dict[str, Any]]]]:
    """
    Detects gaze events in multiple gaze data sequences using the specified detector, and returns the detected label
    for each sample in each sequence. Optionally, returns the metadata of the detection process of each sequence.
    Returns a list of np.ndarrays, where each array contains the detected labels of the corresponding gaze data
    sequence. If `include_metadata` is True, also returns a list of dictionaries, where each dictionary contains the
    metadata of the detection process of the corresponding gaze data sequence.
    See `detect` for the list of arguments and keyword arguments.
    """
    if len(ts) != len(xs) != len(ys):
        raise ValueError("The number of timestamp arrays, x-coordinate arrays, and y-coordinate arrays must be equal.")
    detector = _create_detector(detector_name, missing_value, min_event_duration, pad_blinks_time, **kwargs)
    viewer_distance = viewer_distance if isinstance(viewer_distance, list) else [viewer_distance] * len(ts)
    pixel_size = pixel_size if isinstance(pixel_size, list) else [pixel_size] * len(ts)
    labels_list, metadata_list = [], []
    for t, x, y, vd, ps in tqdm(
            zip(ts, xs, ys, viewer_distance, pixel_size), disable=not verbose, desc="Detecting Events"
    ):
        labels, metadata = detector.detect(t, x, y, vd, ps)
        labels_list.append(labels)
        metadata_list.append(metadata)
    if include_metadata:
        return labels_list, metadata_list
    return labels_list


def _create_detector(
        detector_name: str,
        missing_value: float,
        min_event_duration: float,
        pad_blinks_time: float,
        **kwargs
):
    detector_name_lower = detector_name.lower().strip().replace('-', '').removesuffix('detector')
    if detector_name_lower == 'ivt':
        default_params = IVTDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return IVTDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time, **kwargs
        )
    elif detector_name_lower == 'ivvt':
        default_params = IVVTDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return IVVTDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time, **kwargs
        )
    elif detector_name_lower == 'idt':
        default_params = IDTDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return IDTDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time, **kwargs
        )
    elif detector_name_lower == 'engbert':
        default_params = EngbertDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return EngbertDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time, **kwargs
        )
    elif detector_name_lower == 'nh':
        default_params = NHDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return NHDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time, **kwargs
        )
    elif detector_name_lower == 'remodnav':
        default_params = REMoDNaVDetector.get_default_params()
        kwargs = {**default_params, **kwargs}
        return REMoDNaVDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time, **kwargs
        )
    else:
        raise NotImplementedError(f'Detector `{detector_name}` is not implemented.')
