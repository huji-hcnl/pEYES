import copy
from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd
from overrides import override
from scipy.signal import savgol_filter

import src.pEYES.constants as cnst
import src.pEYES.config as cnfg
from src.pEYES._utils.vector_utils import *
from src.pEYES._utils.pixel_utils import *
from src.pEYES._utils.event_utils import calculate_sampling_rate
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum


class BaseDetector(ABC):
    """
    Base class for gaze event detectors, objects that indicate the type of gaze event (fixation, saccade, blink) at each
    sample in the gaze data. All inherited classes must implement the `_detect_impl` method, which is the core of the
    gaze event detection process.

    Detection process:
    1. Detect blinks, including padding them if necessary
    2. Set x and y to nan where blinks are detected
    3. Detect gaze events (using the class-specific logic, implemented in `_detect_impl` method)
    4. Ignore chunks of gaze-events that are shorter than `min_event_duration`
    5. Merge chunks of the same type that are separated by less than `min_event_duration`

    :param missing_value: the value that indicates missing data in the gaze data. Default is np.NaN
    :param min_event_duration: the minimum duration of a gaze event, in milliseconds. Default is 5 ms
    """

    _DEFAULT_MISSING_VALUE = np.nan
    _DEFAULT_MIN_EVENT_DURATION = cnfg.MIN_EVENT_DURATION
    _DEFAULT_PAD_BLINKS = 0  # ms
    _DEFAULT_VIEWER_DISTANCE = np.nan  # cm
    _DEFAULT_PIXEL_SIZE = np.nan  # cm
    _MINIMUM_SAMPLES_IN_EVENT = 2  # minimum number of samples in an event

    def __init__(
            self,
            missing_value: float = _DEFAULT_MISSING_VALUE,
            min_event_duration: float = _DEFAULT_MIN_EVENT_DURATION,
    ):
        self._missing_value = missing_value
        self._min_event_duration = min_event_duration
        self._sr = np.nan  # sampling rate calculated in the detect method
        self._metadata = {}  # additional metadata

    def detect(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            pad_blinks_ms: float = _DEFAULT_PAD_BLINKS,
            viewer_distance_cm: float = _DEFAULT_VIEWER_DISTANCE,
            pixel_size_cm: float = _DEFAULT_PIXEL_SIZE,
    ) -> (np.ndarray, dict):
        t, x, y = self._verify_inputs(t, x, y)
        self._sr = calculate_sampling_rate(t)
        labels = np.full_like(t, EventLabelEnum.UNDEFINED)
        is_blink = self._detect_blinks(x, y, pad_blinks_ms)
        # detect blinks and replace blink-samples with NaN
        labels[is_blink] = EventLabelEnum.BLINK
        x_copy, y_copy = copy.deepcopy(x), copy.deepcopy(y)
        x_copy[is_blink] = self._DEFAULT_MISSING_VALUE
        y_copy[is_blink] = self._DEFAULT_MISSING_VALUE
        labels = self._detect_impl(t, x_copy, y_copy, labels, viewer_distance_cm, pixel_size_cm)
        labels = merge_chunks(labels, self.min_event_samples)
        labels = reset_short_chunks(labels, self.min_event_samples, False)
        self._metadata.update({
            cnst.SAMPLING_RATE_STR: self.sr,
            cnst.PIXEL_SIZE_STR: pixel_size_cm,
            cnst.VIEWER_DISTANCE_STR: viewer_distance_cm,
        })
        return labels, copy.deepcopy(self._metadata)

    @abstractmethod
    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float
    ) -> np.ndarray:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @final
    @property
    def missing_value(self) -> float:
        return self._missing_value

    @final
    @property
    def min_event_duration(self) -> float:
        return self._min_event_duration

    @final
    @property
    def min_event_samples(self) -> int:
        """
        Calculates the minimum number of samples in an event, based on the minimum event duration and the sampling rate.
        Returns the maximum between the calculated number of samples 2 (minimum number of samples in an event).
        """
        num_samples = self._calc_num_samples(self.min_event_duration, self.sr)
        return np.nanmax([num_samples, self._MINIMUM_SAMPLES_IN_EVENT])

    @final
    @property
    def sr(self) -> float:
        return self._sr

    def _detect_blinks(
            self,
            x: np.ndarray,
            y: np.ndarray,
            pad_by_ms: float = _DEFAULT_PAD_BLINKS,
    ) -> np.ndarray:
        """
        Detects blink candidates in the given data:
        1. Identifies samples where x or y are missing as blinks
        2. Ignores chunks of blinks that are shorter than the minimum event duration
        3. Merges consecutive blink chunks separated by less than the minimum event duration
        4. Pads the blink candidates by the amount in `self._dilate_nans_by`

        :param x: x-coordinates
        :param y: y-coordinates
        :return: boolean array indicating blink candidates
        """
        is_blink = np.zeros_like(x, dtype=bool)
        # identify samples where x or y are missing as blinks
        is_missing_x = np.isnan(x) if np.isnan(self.missing_value) else x == self.missing_value
        is_missing_y = np.isnan(y) if np.isnan(self.missing_value) else y == self.missing_value
        is_blink[is_missing_x | is_missing_y] = True
        # merge consecutive blinks if they are close enough to each other:
        is_blink = merge_chunks(is_blink, self.min_event_samples)
        # ignore blinks if they are shorter than the minimum event duration:
        is_blink = reset_short_chunks(is_blink, self.min_event_samples, False)
        # pad remaining blinks by the given amount:
        if pad_by_ms == 0:
            return is_blink
        pad_samples = self._calc_num_samples(pad_by_ms, self.sr)
        for i, val in enumerate(is_blink):
            if val:
                start = max(0, i - pad_samples)
                end = min(len(is_blink), i + pad_samples)
                is_blink[start:end] = True
        return is_blink

    @staticmethod
    def _verify_inputs(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        if not is_one_dimensional(t):
            raise ValueError("`t` must be one-dimensional")
        if not is_one_dimensional(x):
            raise ValueError("`x` must be one-dimensional")
        if not is_one_dimensional(y):
            raise ValueError("`y` must be one-dimensional")
        t = t.reshape(-1)
        x = x.reshape(-1)
        y = y.reshape(-1)
        if len(t) != len(x) or len(t) != len(y) or len(x) != len(y):
            raise ValueError("t, x and y must have the same length")
        return t, x, y

    @staticmethod
    def _calc_num_samples(duration: float, sr: float) -> int:
        """
        :param duration: in ms
        :param sr: in Hz
        :return: int; number of samples
        """
        return round(duration * sr / cnst.MILLISECONDS_PER_SECOND)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class IThresholdDetector(ABC):

    @staticmethod
    def _get_threshold(threshold_deg: float, unit: str, vd: float, ps: float) -> float:
        unit = unit.lower().strip()
        if unit == "deg":
            return threshold_deg
        if unit == "rad":
            return np.deg2rad(threshold_deg)
        if unit == "px":
            return visual_angle_to_pixels(threshold_deg, vd, ps, use_radians=False)
        raise ValueError(f"Invalid unit: {unit}")


class IVTDetector(BaseDetector, IThresholdDetector):
    """
    Implements the I-VT (velocity threshold) gaze event detection algorithm, as described in:
        Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols.
        In Proceedings of the Symposium on Eye Tracking Research & Applications (pp. 71-78).

    General algorithm:
    1. Calculate the angular velocity of the gaze data
    2. Identify saccade candidates as samples with angular velocity greater than the threshold
    3. Assume undefined (non-blink) samples are fixations

    :param velocity_threshold: the threshold for angular velocity, in degrees per second. Default is 45 degrees per-second,
        as suggested in the paper "One algorithm to rule them all? An evaluation and discussion of ten eye
        movement event-detection algorithms" (2016), Andersson et al.
    """

    _DEFAULT_SACCADE_VELOCITY_THRESHOLD = 45  # deg/s
    _SACCADE_VELOCITY_THRESHOLD_STR = "saccade_velocity_threshold"

    def __init__(
            self,
            saccade_velocity_threshold: float = _DEFAULT_SACCADE_VELOCITY_THRESHOLD,
            missing_value: float = BaseDetector._DEFAULT_MISSING_VALUE,
            min_event_duration: float = BaseDetector._DEFAULT_MISSING_VALUE,
    ):
        super().__init__(missing_value, min_event_duration)
        if saccade_velocity_threshold <= 0:
            raise ValueError("Saccade velocity threshold must be positive")
        self._saccade_velocity_threshold = saccade_velocity_threshold

    @property
    def saccade_velocity_threshold_deg(self) -> float:
        return self._saccade_velocity_threshold

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float
    ) -> np.ndarray:
        if not np.isfinite(viewer_distance_cm) or viewer_distance_cm <= 0:
            raise ValueError("Viewer distance must be a positive finite number")
        if not np.isfinite(pixel_size_cm) or pixel_size_cm <= 0:
            raise ValueError("Pixel size must be a positive finite number")
        labels = np.asarray(copy.deepcopy(labels), dtype=EventLabelEnum)
        px_velocities = calculate_velocities(x, y, t)
        px_threshold = self._get_threshold(
            self.saccade_velocity_threshold_deg, "px", viewer_distance_cm, pixel_size_cm
        )
        labels[(labels != EventLabelEnum.BLINK) & (px_velocities > px_threshold)] = EventLabelEnum.SACCADE
        labels[(labels != EventLabelEnum.BLINK) & (px_velocities <= px_threshold)] = EventLabelEnum.FIXATION
        self._metadata.update({
            f"{self._SACCADE_VELOCITY_THRESHOLD_STR}_deg": self.saccade_velocity_threshold_deg,
            f"{self._SACCADE_VELOCITY_THRESHOLD_STR}_px": px_threshold,
        })
        return labels


class IVVTDetector(IVTDetector):
    """
    TODO
    """

    _DEFAULT_SMOOTH_PURSUIT_VELOCITY_THRESHOLD = 15  # deg/s  # TODO: check the default value
    _SMOOTH_PURSUIT_VELOCITY_THRESHOLD_STR = "smooth_pursuit_velocity_threshold"

    def __init__(
            self,
            saccade_velocity_threshold: float = IVTDetector._DEFAULT_SACCADE_VELOCITY_THRESHOLD,
            smooth_pursuit_velocity_threshold: float = _DEFAULT_SMOOTH_PURSUIT_VELOCITY_THRESHOLD,
            missing_value: float = BaseDetector._DEFAULT_MISSING_VALUE,
            min_event_duration: float = BaseDetector._DEFAULT_MISSING_VALUE,
    ):
        super().__init__(saccade_velocity_threshold, missing_value, min_event_duration)
        if smooth_pursuit_velocity_threshold <= 0:
            raise ValueError("Smooth pursuit velocity threshold must be positive")
        self._smooth_pursuit_velocity_threshold = smooth_pursuit_velocity_threshold

    @property
    def smooth_pursuit_velocity_threshold_deg(self) -> float:
        return self._smooth_pursuit_velocity_threshold

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float
    ) -> np.ndarray:
        labels = super()._detect_impl(t, x, y, labels, viewer_distance_cm, pixel_size_cm)
        labels[labels == EventLabelEnum.FIXATION] = EventLabelEnum.UNDEFINED  # reset fixation labels
        px_velocities = calculate_velocities(x, y, t)
        px_threshold = self._get_threshold(
            self.smooth_pursuit_velocity_threshold_deg, "px", viewer_distance_cm, pixel_size_cm
        )
        labels[(labels == EventLabelEnum.UNDEFINED) & (px_velocities > px_threshold)] = EventLabelEnum.SMOOTH_PURSUIT
        labels[(labels == EventLabelEnum.UNDEFINED) & (px_velocities <= px_threshold)] = EventLabelEnum.FIXATION
        self._metadata.update({
            f"{self._SMOOTH_PURSUIT_VELOCITY_THRESHOLD_STR}_deg": self.smooth_pursuit_velocity_threshold_deg,
            f"{self._SMOOTH_PURSUIT_VELOCITY_THRESHOLD_STR}_px": px_threshold,
        })
        return labels


class IDTDetector(BaseDetector, IThresholdDetector):
    """
    Implements the I-DT (dispersion threshold) gaze event detection algorithm, as described in:
        Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols.
        In Proceedings of the Symposium on Eye Tracking Research & Applications (pp. 71-78).

    General algorithm:
    1. Initialize a window spanning the first `window_duration` milliseconds
    2. Calculate the dispersion of the gaze data in the window
    3. If the dispersion is below the threshold, label all samples in the window as fixation and expand the window by a
        single sample. Otherwise, label the current sample as saccade and start a new window in the next sample
    4. Repeat until the end of the gaze data

    :param dispersion_threshold: the threshold for dispersion, in degrees. Default is 0.5 degrees, as suggested in the
        paper "One algorithm to rule them all? An evaluation and discussion of ten eye movement event-detection
        algorithms" (2016), Andersson et al.
    :param window_duration: the duration of the window in milliseconds. Default is 100 ms, as suggested in the paper
        "One algorithm to rule them all? An evaluation and discussion of ten eye movement event-detection algorithms"
        (2016), Andersson et al.
    """

    _DEFAULT_DISPERSION_THRESHOLD = 0.5  # visual degrees
    _DEFAULT_WINDOW_DURATION = 100  # ms
    _DISPERSION_THRESHOLD_STR = "dispersion_threshold"

    def __init__(
            self,
            dispersion_threshold: float = _DEFAULT_DISPERSION_THRESHOLD,
            window_duration: float = _DEFAULT_WINDOW_DURATION,
            missing_value: float = BaseDetector._DEFAULT_MISSING_VALUE,
            min_event_duration: float = BaseDetector._DEFAULT_MISSING_VALUE,
    ):
        super().__init__(missing_value, min_event_duration)
        if dispersion_threshold <= 0:
            raise ValueError("Dispersion threshold must be positive")
        self._dispersion_threshold = dispersion_threshold
        self._window_duration = window_duration

    @property
    def dispersion_threshold_deg(self) -> float:
        return self._dispersion_threshold

    @property
    def window_duration_ms(self) -> float:
        return self._window_duration

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float
    ) -> np.ndarray:
        labels = np.asarray(copy.deepcopy(labels), dtype=EventLabelEnum)
        ws = self._calculate_window_size_samples(t)
        px_threshold = self._get_threshold(self.dispersion_threshold_deg, "px", viewer_distance_cm, pixel_size_cm)
        start_idx, end_idx = 0, ws
        is_fixation = False
        while end_idx <= len(t):
            dispersion = self._calculate_dispersion_length_px(x[start_idx:end_idx], y[start_idx:end_idx])
            if dispersion < px_threshold:
                # label all samples in the window as fixation and expand window to the right
                is_fixation = True
                labels[start_idx: end_idx] = EventLabelEnum.FIXATION
                end_idx += 1
            elif is_fixation:
                # start new window in the end of the old one
                start_idx = end_idx - 1
                end_idx = start_idx + ws
                is_fixation = False
            else:
                # label current sample as saccade and start new window in the next sample
                labels[start_idx] = EventLabelEnum.SACCADE
                start_idx += 1
                end_idx += 1
        self._metadata.update({
            f"{self._DISPERSION_THRESHOLD_STR}_deg": self.dispersion_threshold_deg,
            f"{self._DISPERSION_THRESHOLD_STR}_px": px_threshold,
        })
        return labels

    def _calculate_window_size_samples(self, t: np.ndarray) -> int:
        ws = self._calc_num_samples(self.window_duration_ms, self.sr)
        if ws < 2:
            raise ValueError(f"window_duration={ws}ms is too short for the given sampling_rate={self.sr}Hz")
        if ws >= len(t):
            raise ValueError(f"window_duration={ws}ms is too long for the given input data")
        return ws

    @staticmethod
    def _calculate_dispersion_length_px(xs: np.ndarray, ys: np.ndarray) -> float:
        """ Calculates the dispersion length of the gaze points (px units) """
        return max(xs) - min(xs) + max(ys) - min(ys)

    @staticmethod
    def _calculate_dispersion_area_px(xs: np.ndarray, ys: np.ndarray) -> float:
        """ Calculates the area of the ellipse that fits the window of gaze points (px^2 units) """
        horiz_axis = 0.5 * (max(xs) - min(xs))
        vert_axis = 0.5 * (max(ys) - min(ys))
        return np.pi * horiz_axis * vert_axis


class EngbertDetector(BaseDetector):
    """
    Implements the algorithm described by Engbert, Kliegl, and Mergenthaler in
        "Microsaccades uncover the orientation of covert attention" (2003)
        "Microsaccades are triggered by low retinal image slip" (2006)

    Implementation is based on the following repositories:
        - https://shorturl.at/lyBE2
        - https://shorturl.at/DHJZ6

    General algorithm:
        1. Calculate the velocity of the gaze data in both axes
        2. Calculate the median-standard-deviation of the velocity in both axes
        3. Calculate the noise threshold as the multiple of the median-standard-deviation with the constant `lambda_noise_threshold`
        4. Identify saccade candidates as samples with velocity greater than the noise threshold

    :param lambda: the threshold for the noise, as a multiple of the median-standard-deviation. Default
        is 5, as suggested in the original paper
    :param window_size: the size of the window used to calculate the velocity. Default is 2, as suggested in
        the original paper
    """

    _DEFAULT_LAMBDA_PARAM = 5     # standard deviation multiplier
    _DEFAULT_DERIVATION_WINDOW_SIZE = 5     # number of samples used to calculate axial velocity
    _THRESHOLD_VELOCITY_STR = "threshold_velocity"

    def __init__(
            self,
            lambda_param: float = _DEFAULT_LAMBDA_PARAM,
            deriv_window_size: int = _DEFAULT_DERIVATION_WINDOW_SIZE,
            missing_value: float = BaseDetector._DEFAULT_MISSING_VALUE,
            min_event_duration: float = BaseDetector._DEFAULT_MIN_EVENT_DURATION,
    ):
        super().__init__(missing_value, min_event_duration)
        self._lambda_param = lambda_param
        if self._lambda_param <= 0:
            raise ValueError("Lambda parameter must be positive")
        self._deriv_window_size = deriv_window_size
        if self._deriv_window_size <= 0:
            raise ValueError("Derivation window size must be positive")

    @property
    def lambda_param(self) -> float:
        return self._lambda_param

    @property
    def deriv_window_size(self) -> int:
        return self._deriv_window_size

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float
    ) -> np.ndarray:
        labels = np.asarray(copy.deepcopy(labels), dtype=EventLabelEnum)
        x_velocity = self._axial_velocities_px(x)
        y_velocity = self._axial_velocities_px(y)
        x_thresh = self._median_standard_deviation(x_velocity) * self.lambda_param
        y_thresh = self._median_standard_deviation(y_velocity) * self.lambda_param
        ellipse = (x_velocity / x_thresh) ** 2 + (y_velocity / y_thresh) ** 2
        labels[ellipse < 1] = EventLabelEnum.FIXATION
        labels[ellipse >= 1] = EventLabelEnum.SACCADE
        self._metadata.update({
            f"{cnst.X}_{self._THRESHOLD_VELOCITY_STR}_pxs": x_thresh,
            f"{cnst.Y}_{self._THRESHOLD_VELOCITY_STR}_px": y_thresh,
        })
        return labels

    def _axial_velocities_px(self, arr: np.ndarray) -> np.ndarray:
        """
        Calculates the velocity (px/sec) along a single axis, based on the algorithm described in the original paper:
        1. Sum values in a window of size window_size//2, *before* the current sample:
            sum_before = arr(t-1) + arr(t-2) + ... + arr(t-ws//2)
        2. Sum values in a window of size window_size//2, *after* the current sample:
            sum_after = arr(t+1) + arr(t+2) + ... + arr(t+ws//2)
        3. Calculate the difference of the signal at the current sample:
            diff = sum_after - sum_before
        4. Calculate the velocity as the difference multiplied by sampling rate and divided by the window size:
            velocity = diff * sr / ws
        5. The first and last ws//2 samples are filled with np.NaN
        """
        half_ws = self.deriv_window_size // 2 if self.deriv_window_size % 2 == 0 else self.deriv_window_size // 2 + 1
        velocities = np.full_like(arr, np.nan, dtype=float)
        for idx in range(half_ws, len(arr) - half_ws):
            sum_before = np.sum(arr[idx - half_ws:idx])
            sum_after = np.sum(arr[idx + 1:idx + half_ws + 1])
            diff = sum_after - sum_before
            velocities[idx] = diff * self.sr / self.deriv_window_size
        return velocities

    @staticmethod
    def _median_standard_deviation(arr) -> float:
        """ Calculates the median-based standard-deviation of the input array """
        squared_median = np.power(np.nanmedian(arr), 2)
        median_of_squares = np.nanmedian(np.power(arr, 2))
        sd = np.sqrt(median_of_squares - squared_median)
        return float(np.nanmax([sd, cnfg.EPSILON]))

    @override
    def _verify_inputs(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        t, x, y = super()._verify_inputs(t, x, y)
        if len(x) < 2 * self.deriv_window_size:
            raise ValueError(f"Derivation window ({self.deriv_window_size} samples) is too long for the given data")
        return t, x, y


class NHDetector(BaseDetector):

    _DEFAULT_FILTER_DURATION_MS = 2 * cnfg.EVENT_MAPPING[cnfg.EventLabelEnum.SACCADE][cnst.MIN_DURATION_STR]    # 20ms
    _DEFAULT_FILTER_POLYORDER = 2
    _DEFAULT_SACCADE_MAX_VELOCITY = 1000        # deg/s
    _DEFAULT_SACCADE_MAX_ACCELARATION = 100000  # deg/s^2
    _DEFAULT_ALPHA_PARAM = 0.7  # weight of saccade onset threshold when detecting saccade offset

    def __init__(
            self,
            filter_duration_ms: float = _DEFAULT_FILTER_DURATION_MS,
            filter_polyorder: int = _DEFAULT_FILTER_POLYORDER,
            saccade_max_velocity: float = _DEFAULT_SACCADE_MAX_VELOCITY,
            saccade_max_acceleration: float = _DEFAULT_SACCADE_MAX_ACCELARATION,
            alpha_param: float = _DEFAULT_ALPHA_PARAM,
            allow_high_psos: bool = True,
            missing_value: float = BaseDetector._DEFAULT_MISSING_VALUE,
            min_event_duration: float = BaseDetector._DEFAULT_MIN_EVENT_DURATION,
    ):
        super().__init__(missing_value, min_event_duration)
        self._filter_duration = filter_duration_ms
        if self._filter_duration <= 0:
            raise ValueError("Filter duration must be positive")
        self._filter_polyorder = filter_polyorder
        if self._filter_polyorder <= 0:
            raise ValueError("Filter polyorder must be positive")
        self._saccade_max_velocity = saccade_max_velocity
        if self._saccade_max_velocity <= 0:
            raise ValueError("Saccade max velocity must be positive")
        self._saccade_max_acceleration = saccade_max_acceleration
        if self._saccade_max_acceleration <= 0:
            raise ValueError("Saccade max acceleration must be positive")
        self._alpha_param = alpha_param
        if not 0 <= self._alpha_param <= 1:
            raise ValueError("Alpha parameter must be between 0 and 1")
        self._beta_param = 1 - alpha_param
        self._allow_high_psos = allow_high_psos

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float
    ) -> np.ndarray:
        labels = np.asarray(copy.deepcopy(labels), dtype=EventLabelEnum)

        # detect noise:
        v, a = self._velocities_and_accelerations(x, y, viewer_distance_cm, pixel_size_cm)
        is_noise = self._detect_noise(v, a)

        # denoise the data:
        x_copy, y_copy, v_copy, a_copy = x.copy(), y.copy(), v.copy(), v.copy()
        x_copy[is_noise] = np.nan
        y_copy[is_noise] = np.nan
        v_copy[is_noise] = np.nan
        a_copy[is_noise] = np.nan

        #
        # TODO: implement the rest of the algorithm - start from here
        #

        # detect saccades and PSOs
        peak_threshold, onset_threshold = self._estimate_saccade_thresholds(v_copy)  # global velocity thresholds
        saccades_info = self._detect_saccades(
            v_copy, peak_threshold, onset_threshold
        )   # saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        psos_info = self._detect_psos(
            v_copy, saccades_info, peak_threshold, onset_threshold
        )   # saccade id -> (start_idx, end_idx, pso_type)

        # save results
        candidates = self._classify_samples(is_noise, saccades_info, psos_info)
        df = self.data[cnst.GAZE]
        df[cnst.VELOCITY] = v
        df[cnst.ACCELERATION] = a
        self.data[cnst.GAZE] = df
        self.data["saccade_peak_threshold"] = peak_threshold
        self.data["saccade_onset_threshold"] = onset_threshold
        return candidates

    @property
    def filter_duration_ms(self) -> float:
        return self._filter_duration

    @property
    def filter_polyorder(self) -> int:
        return self._filter_polyorder

    @property
    def saccade_max_velocity_deg(self) -> float:
        return self._saccade_max_velocity

    @property
    def saccade_max_acceleration_deg(self) -> float:
        return self._saccade_max_acceleration

    @property
    def alpha(self) -> float:
        return self._alpha_param

    @property
    def beta(self) -> float:
        return self._beta_param

    @property
    def is_high_psos_allowed(self) -> bool:
        return self._allow_high_psos

    def _velocities_and_accelerations(self, x: np.ndarray, y: np.ndarray, vd, ps) -> (np.ndarray, np.ndarray):
        """
        Calculates the 1st and 2nd derivatives of the gaze data, using the Savitzky-Golay filter. Then calculates the
        velocity and acceleration from the derivatives.

        Note the original paper calculates the velocity and acceleration using:
            v = sr * sqrt[(x')^2 + (y')^2] * pixel-to-angle-constant
            a = sr * sqrt[(x'')^2 + (y'')^2] * pixel-to-angle-constant
        We use `delta=1/self._sr` when calculating the derivatives, to account for sampling time, so we don't need to
        multiply by sr when computing `v` and `a`. See more in the scipy documentation and in the following links:
            - https://stackoverflow.com/q/56168730/8543025
            - https://github.com/scipy/scipy/issues/9910

        :param x: 1D array of x coordinates
        :param y: 1D array of y coordinates
        :return: angular velocity and acceleration of each point
        """
        px_to_deg_constant = pixels_to_visual_angle(1, vd, ps, False)
        ws = self._calc_num_samples(self.filter_duration_ms, self.sr)
        if ws <= self.filter_polyorder:
            raise RuntimeError(
                f"Cannot compute {self.filter_polyorder}-order Savitzky-Golay filter with window of duration {ws}ms " +
                "and sampling rate of {self.sr}Hz"
            )
        # calculate angular velocity (deg/s): v = sqrt((x')^2 + (y')^2) * pixel-to-angle-constant:
        dx = savgol_filter(x, ws, self._filter_polyorder, deriv=1, delta=1/self._sr)
        dy = savgol_filter(y, ws, self._filter_polyorder, deriv=1, delta=1/self._sr)
        v = np.sqrt(dx ** 2 + dy ** 2) * px_to_deg_constant
        # calculate angular acceleration (deg/s^2): a = sqrt((x'')^2 + (y'')^2) * pixel-to-angle-constant:
        ddx = savgol_filter(x, ws, self._filter_polyorder, deriv=2, delta=1/self._sr)
        ddy = savgol_filter(y, ws, self._filter_polyorder, deriv=2, delta=1/self._sr)
        a = np.sqrt(ddx ** 2 + ddy ** 2) * px_to_deg_constant
        return v, a

    def _detect_noise(self, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """
        Detects noise in the gaze data based on the angular velocity and acceleration.
        :param v: angular velocity of the gaze data
        :param a: angular acceleration of the gaze data
        :return: boolean array indicating noise samples
        """
        is_noise = np.zeros(len(v), dtype=bool)
        is_noise[v > self.saccade_max_velocity_deg] = True
        is_noise[a > self.saccade_max_acceleration_deg] = True

        # expand noise periods to include surrounding samples up to the median overall velocity
        median_v = np.nanmedian(v)
        noise_idxs = np.where(is_noise)[0]
        for idx in noise_idxs:
            start, end = idx, idx
            while start > 0 and v[start] > median_v:
                start -= 1
            while end < len(v) and v[end] > median_v:
                end += 1
            is_noise[start:end] = True
        return is_noise
