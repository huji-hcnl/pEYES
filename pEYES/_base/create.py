from typing import Union, Sequence

import numpy as np

import pEYES._utils.constants as cnst
import pEYES._DataModels.config as cnfg
from pEYES._utils.event_utils import parse_label
from pEYES._DataModels.Event import BaseEvent, EventSequenceType
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType
from pEYES._DataModels.Detector import BaseDetector
from pEYES._DataModels.Detector import (
    IVTDetector, IVVTDetector, IDTDetector, EngbertDetector, NHDetector, REMoDNaVDetector
)


def create_detector(
        detector_name: str,
        missing_value: float,
        min_event_duration: float,
        pad_blinks_time: float,
        **kwargs
) -> BaseDetector:
    """
    Creates a gaze event detector with the specified parameters.
    :param detector_name: name of the detection algorithm to use
    :param missing_value: value indicating missing data in the gaze data
    :param min_event_duration: minimum duration of a gaze event (ms)
    :param pad_blinks_time: duration to pad before and after each detected blink event (ms)

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

    :return: a detector object
    """
    detector_name_lower = detector_name.lower().strip().replace('-', '').removesuffix('detector')
    if detector_name_lower == 'ivt':
        default_params = IVTDetector.get_default_params()
        return IVTDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif detector_name_lower == 'ivvt':
        default_params = IVVTDetector.get_default_params()
        return IVVTDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif detector_name_lower == 'idt':
        default_params = IDTDetector.get_default_params()
        return IDTDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif detector_name_lower == 'engbert':
        default_params = EngbertDetector.get_default_params()
        return EngbertDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif detector_name_lower == 'nh':
        default_params = NHDetector.get_default_params()
        return NHDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif detector_name_lower == 'remodnav':
        default_params = REMoDNaVDetector.get_default_params()
        return REMoDNaVDetector(
            missing_value=missing_value, min_event_duration=min_event_duration, pad_blinks_ms=pad_blinks_time,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    else:
        raise NotImplementedError(f'Detector `{detector_name}` is not implemented.')


def create_events(
        labels: Union[UnparsedEventLabelType, Sequence[UnparsedEventLabelType]],
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        pupil: np.ndarray,
        viewer_distance: cnfg.VIEWER_DISTANCE,
        pixel_size: cnfg.SCREEN_MONITOR[cnst.PIXEL_SIZE_STR],
) -> Union[BaseEvent, EventSequenceType]:
    """
    Create gaze-events from the given data.
    If `labels` is a single label, creates a single event spanning the entire data.
    Otherwise, `labels` must have the same length as `t` and events will be created for each "chunk" of subsequent
    identical labels.

    :param labels: The event label(s)
    :param t: timestamps (ms)
    :param x: horizontal gaze position (pixels)
    :param y: vertical gaze position (pixels)
    :param pupil: pupil size (mm)
    :param viewer_distance: distance from the viewer's eyes to the screen (cm)
    :param pixel_size: size of a pixel on the screen (cm)
    :return: the generated event(s)
    """
    if isinstance(labels, UnparsedEventLabelType):
        label = parse_label(labels)
        return BaseEvent.make(label, t, x, y, pupil, viewer_distance, pixel_size)
    return BaseEvent.make_multiple(
        np.array([parse_label(l) for l in labels]), t, x, y, pupil, viewer_distance, pixel_size
    )
