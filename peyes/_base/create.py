from typing import Union

import numpy as np

import peyes._utils.constants as cnst
import peyes._DataModels.config as cnfg
from peyes._utils.event_utils import parse_label, calculate_num_samples
from peyes._DataModels.Event import BaseEvent, EventSequenceType
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._DataModels.Detector import BaseDetector
from peyes._DataModels.Detector import (
    IVTDetector, IVVTDetector, IDTDetector, IDVTDetector, EngbertDetector, NHDetector, REMoDNaVDetector
)


def create_detector(
        algorithm: str,
        missing_value: float,
        min_event_duration: float,
        pad_blinks_time: float,
        name: str = None,
        **kwargs
) -> BaseDetector:
    """
    Creates a gaze event detector with the specified parameters.
    :param algorithm: name of the detection algorithm to use
    :param missing_value: value indicating missing data in the gaze data
    :param min_event_duration: minimum duration of a gaze event (ms)
    :param pad_blinks_time: duration to pad before and after each detected blink event (ms)
    :param name: name of the detector. If None, the detector's class name is used.

    Additional keyword arguments required for detector initialization:
    - IVT:
        :keyword saccade_velocity_threshold: float; velocity threshold to separates saccades from fixations (deg/s)
    - IVVT:
        :keyword saccade_velocity_threshold: float; velocity threshold to separates saccades from smooth pursuits (deg/s)
        :keyword smooth_pursuit_velocity_threshold: float; the velocity threshold that separates smooth pursuits from fixations (deg/s)
    - IDT:
        :keyword dispersion_threshold: dispersion threshold that separates saccades from fixations (visual angle degrees)
        :keyword window_duration: the duration of the window used to calculate the dispersion threshold (ms)
    - IDVT:
        :keyword dispersion_threshold: dispersion threshold that separates fixations from smooth pursuits (visual angle degrees)
        :keyword window_duration: the duration of the window used to calculate the dispersion threshold (ms)
        :keyword saccade_velocity_threshold: float; velocity threshold to separates smooth pursuits from saccades (deg/s)
    - Engbert:
        :keyword lambda_param: float; multiplication coefficient used for calculating saccade threshold
        :keyword deriv_window_size: int; number of samples used to compute axial velocity
    - NH:
        :keyword filter_duration_ms: int; Savitzky-Golay filter's duration (ms), default: 2×min saccade duration
        :keyword filter_polyorder: int; Savitzky-Golay filter's polynomial order, default: 2
        :keyword saccade_max_velocity: maximum saccade velocity (deg/s), default: 1000
        :keyword saccade_max_acceleration: maximum saccade acceleration (deg/s^2), default: 100000
        :keyword min_saccade_duration: minimum saccade duration (ms)
        :keyword min_fixation_duration: minimum fixation duration (ms)
        :keyword max_pso_duration: maximum PSO duration (ms)
        :keyword alpha_param: weight of saccade onset threshold when detecting saccade offset, default: 0.7
        :keyword ignore_short_peak_durations: if True, excludes sporadic occurrences of peak velocity when detecting
            saccade peaks, default: True
        :keyword allow_high_psos: if True, includes "high" PSOs, i.e., PSOs with max velocity exceeding saccades' peak
            threshold but lower than the preceding saccade, default: True
    - REMoDNaV:
        :keyword min_saccade_duration: int; the minimum duration of a saccade (ms)
        :keyword saccade_initial_velocity_threshold: float; the initial velocity threshold for saccade detection (deg/s), default: 300
        :keyword saccade_context_window_duration: int; the duration of the context window for saccade detection (ms), default: 1000
        :keyword saccade_initial_max_freq: float; the initial maximum frequency for saccade detection (Hz), default: 2.0
        :keyword saccade_onset_threshold_noise_factor: float; the noise factor for saccade onset threshold, default: 5.0
        :keyword min_smooth_pursuit_duration: int; the minimum duration of a smooth pursuit (ms)
        :keyword smooth_pursuits_lowpass_cutoff_freq: float; the lowpass cutoff frequency for smooth pursuit detection (Hz), default: 4.0
        :keyword smooth_pursuit_drift_velocity_threshold: float; the drift velocity threshold for smooth pursuit detection (deg/s), default: 2.0
        :keyword min_fixation_duration: int; the minimum duration of a fixation (ms)
        :keyword min_blink_duration: int; the minimum duration of a blink (ms)
        :keyword max_pso_duration: int; the maximum duration of a PSO (ms)
        :keyword savgol_filter_polyorder: int; the polynomial order for the Savitzky-Golay filter, default: 2
        :keyword savgol_filter_duration_ms: int; the duration of the Savitzky-Golay filter (ms), default: 19
        :keyword median_filter_duration_ms: int; the duration of the median filter (ms)
        :keyword max_velocity: float; the maximum velocity of the gaze data (deg/s), default: 1500

    :return: a detector object
    """
    algorithm_lower = algorithm.lower().strip().replace('-', '').removesuffix('detector')
    if algorithm_lower == 'ivt':
        default_params = IVTDetector.get_default_params()
        return IVTDetector(
            missing_value=missing_value,
            min_event_duration=min_event_duration,
            pad_blinks_ms=pad_blinks_time,
            name=name,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif algorithm_lower == 'ivvt':
        default_params = IVVTDetector.get_default_params()
        return IVVTDetector(
            missing_value=missing_value,
            min_event_duration=min_event_duration,
            pad_blinks_ms=pad_blinks_time,
            name=name,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif algorithm_lower == 'idt':
        default_params = IDTDetector.get_default_params()
        return IDTDetector(
            missing_value=missing_value,
            min_event_duration=min_event_duration,
            pad_blinks_ms=pad_blinks_time,
            name=name,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif algorithm_lower == 'idvt':
        default_params = IDVTDetector.get_default_params()
        return IDVTDetector(
            missing_value=missing_value,
            min_event_duration=min_event_duration,
            pad_blinks_ms=pad_blinks_time,
            name=name,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif algorithm_lower == 'engbert':
        default_params = EngbertDetector.get_default_params()
        return EngbertDetector(
            missing_value=missing_value,
            min_event_duration=min_event_duration,
            pad_blinks_ms=pad_blinks_time,
            name=name,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif algorithm_lower == 'nh':
        default_params = NHDetector.get_default_params()
        return NHDetector(
            missing_value=missing_value,
            min_event_duration=min_event_duration,
            pad_blinks_ms=pad_blinks_time,
            name=name,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    elif algorithm_lower == 'remodnav':
        default_params = REMoDNaVDetector.get_default_params()
        return REMoDNaVDetector(
            missing_value=missing_value,
            min_event_duration=min_event_duration,
            pad_blinks_ms=pad_blinks_time,
            name=name,
            **{k: kwargs.get(k, default_params[k]) for k in default_params.keys()}
        )
    else:
        raise NotImplementedError(f'Detector `{algorithm}` is not implemented.')


def create_events(
        labels: Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType],
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


def create_boolean_channel(
        channel_type: str,
        data: Union[UnparsedEventLabelSequenceType, EventSequenceType],
        sampling_rate: float = None,
        min_num_samples: int = None,
) -> np.ndarray:
    """
    Converts the given labels or events to a boolean array (MNE-style event channel), indicating either event onsets or
    offsets, depending on the specified channel type.
    Raises a ValueError if neither `labels` and `events` are provided, or if both are provided.

    :param channel_type: either 'start'/'onset' or 'end'/'offset'
    :param data: array-like of event labels or Event objects
    :param sampling_rate: the sampling rate of the recorded data; required if `data` is a series of Event objects
    :param min_num_samples: the number of samples in the output sequence; required if `data` is a series of Event
        objects. If None, the number of samples is determined by the total duration of the provided events.

    :return: array of boolean values, where `True` indicates onsets or offsets
    """
    channel_type_lower = channel_type.lower().strip()
    if channel_type_lower not in {cnst.START_STR, cnst.ONSET_STR, cnst.END_STR, cnst.OFFSET_STR}:
        raise ValueError(f"Invalid channel type: {channel_type}")
    if len(data) == 0:
        # no events or labels provided -> return an empty array
        return np.zeros(np.nanmin(len(data), min_num_samples), dtype=bool)
    if all(isinstance(e, BaseEvent) for e in data):
        # data is of type EventSequenceType
        return _events_to_boolean_channel(channel_type_lower, data, sampling_rate, min_num_samples)
    if all(isinstance(l, UnparsedEventLabelType) for l in data):
        # data is of type UnparsedEventLabelSequenceType
        return _labels_to_boolean_channel(channel_type_lower, data)
    unknown_types = set([
        type(datum) for datum in data if not (isinstance(datum, BaseEvent) or isinstance(datum, UnparsedEventLabelType))
    ])
    raise TypeError(f"Argument `data` contains unknown types: {unknown_types}")


def _labels_to_boolean_channel(
        channel_type: str,
        labels: UnparsedEventLabelSequenceType,
) -> np.ndarray:
    """
    Converts the given labels to a boolean array (MNE-style event channel), indicating either event onsets or offsets.

    :param channel_type: either 'start'/'onset' or 'end'/'offset'
    :param labels: array-like of unparsed event labels

    :return: array of boolean values, where `True` indicates onsets or offsets
    """
    parsed_labels = np.array([parse_label(l) for l in labels])
    bool_channel = np.zeros_like(labels, dtype=bool)
    if channel_type.lower() == cnst.START_STR or channel_type.lower() == cnst.ONSET_STR:
        bool_channel[0] = True
        bool_channel[1:] = np.diff(parsed_labels) != 0
    elif channel_type.lower() == cnst.END_STR or channel_type.lower() == cnst.OFFSET_STR:
        bool_channel[-1] = True
        bool_channel[:-1] = np.diff(parsed_labels) != 0
    else:
        raise ValueError(f"Invalid channel type: {channel_type}")
    bool_channel[parsed_labels == EventLabelEnum.UNDEFINED] = False  # ignore undefined onsets/offsets
    return bool_channel


def _events_to_boolean_channel(
        channel_type: str,
        events: EventSequenceType,
        sampling_rate: float,
        min_num_samples=None,
) -> (np.ndarray, np.ndarray):
    """
    Converts the given events to a boolean array (MNE-style event channel), indicating either event onsets or offsets.

    :param channel_type: either 'start'/'onset' or 'end'/'offset'
    :param events: array-like of Event objects
    :param sampling_rate: the sampling rate of the recorded data
    :param min_num_samples: the number of samples in the output sequence. If None, the number of samples is determined
        by the total duration of the provided events.

    :return: array of boolean values, where `True` indicates the event onset or offset
    """
    global_start_time = min(e.start_time for e in events)
    global_end_time = max(e.end_time for e in events)
    num_samples = calculate_num_samples(global_start_time, global_end_time, sampling_rate, min_num_samples)
    bool_channel = np.zeros(num_samples, dtype=bool)
    for e in events:
        corrected_start_time, corrected_end_time = e.start_time - global_start_time, e.end_time - global_start_time
        if channel_type.lower() == cnst.START_STR or channel_type.lower() == cnst.ONSET_STR:
            start_sample = int(np.round(corrected_start_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
            bool_channel[start_sample] = True
        elif channel_type.lower() == cnst.END_STR or channel_type.lower() == cnst.OFFSET_STR:
            end_sample = int(np.round(corrected_end_time * sampling_rate / cnst.MILLISECONDS_PER_SECOND))
            bool_channel[end_sample - 1] = True
        else:
            raise ValueError(f"Invalid channel type: {channel_type}")
    return bool_channel
