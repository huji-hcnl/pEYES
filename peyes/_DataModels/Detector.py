import time
from abc import ABC, abstractmethod
from typing import final, Dict

import remodnav
from overrides import override
from scipy.signal import savgol_filter

import peyes._DataModels.config as cnfg
from peyes._utils.vector_utils import *
from peyes._utils.pixel_utils import *
from peyes._utils.event_utils import calculate_sampling_rate, parse_label
from peyes._DataModels.EventLabelEnum import EventLabelEnum, EventLabelSequenceType


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

    :param missing_value: the value that indicates missing data in the gaze data.
    :param min_event_duration: the minimum duration of a gaze event, in milliseconds.
    :param pad_blinks_ms: the duration to pad around detected blinks, in milliseconds.
    """
    _ARTICLES: List[str]

    _MINIMUM_SAMPLES_IN_EVENT = 2   # minimum number of samples in an event

    __MISSING_VALUE_STR = "missing_value"
    __MIN_EVENT_DURATION_STR = "min_event_duration"
    __PAD_BLINKS_MS_STR = "pad_blinks_ms"

    def __init__(
            self,
            missing_value: float,
            min_event_duration: float,
            pad_blinks_ms: float,
    ):
        self._missing_value = missing_value
        self._min_event_duration = min_event_duration
        if self._min_event_duration <= 0:
            raise ValueError("Minimum event duration must be positive")
        self._pad_blinks_ms = pad_blinks_ms
        if self._pad_blinks_ms < 0:
            raise ValueError("Time to pad blinks must be non-negative")
        self._metadata = {}     # additional metadata
        self._sr = np.nan       # sampling rate calculated in the detect method

    @classmethod
    @abstractmethod
    def get_default_params(cls) -> Dict[str, float]:
        raise NotImplementedError

    @final
    def detect(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float,
    ) -> (EventLabelSequenceType, dict):
        start_time = time.time()
        if not np.isfinite(viewer_distance_cm) or viewer_distance_cm <= 0:
            raise ValueError("Viewer distance must be a positive finite number")
        if not np.isfinite(pixel_size_cm) or pixel_size_cm <= 0:
            raise ValueError("Pixel size must be a positive finite number")
        t, x, y = self._reshape_vectors(t, x, y)
        self._sr = calculate_sampling_rate(t)
        labels = np.full_like(t, EventLabelEnum.UNDEFINED, dtype=EventLabelEnum)
        is_blink = self._detect_blinks(x, y)
        # detect blinks and replace blink-samples with NaN
        labels[is_blink] = EventLabelEnum.BLINK
        x_copy, y_copy = copy.deepcopy(x), copy.deepcopy(y)
        x_copy[is_blink] = np.nan
        y_copy[is_blink] = np.nan
        labels = self._detect_impl(t, x_copy, y_copy, labels, viewer_distance_cm, pixel_size_cm)
        labels = merge_chunks(labels, self.min_event_samples)
        labels = reset_short_chunks(labels, self.min_event_samples, EventLabelEnum.UNDEFINED)
        labels = [parse_label(l) for l in labels]
        self._metadata.update({
            cnst.SAMPLING_RATE_STR: self.sr,
            cnst.PIXEL_SIZE_STR: pixel_size_cm,
            cnst.VIEWER_DISTANCE_STR: viewer_distance_cm,
            cnst.RUNTIME_STR: time.time() - start_time,
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
            pixel_size_cm: float,
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
    def sr(self) -> float:
        return self._sr

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
    def pad_blinks_ms(self) -> float:
        return self._pad_blinks_ms

    @final
    @property
    def pad_blinks_samples(self) -> int:
        return self._calc_num_samples(self.pad_blinks_ms, self.sr)

    @classmethod
    @final
    def articles(cls) -> List[str]:
        """ List of articles to cite when using this detector """
        if not cls._ARTICLES:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_{cnst.ARTICLES_STR}`")
        return cls._ARTICLES

    @classmethod
    @final
    def documentation(cls) -> str:
        """ Returns the documentation of the class """
        name = f"Detector:\t{cls.__name__.removesuffix('Detector')}"
        articles = "Articles:\n" + "\n".join([f"- {a}" for a in cls.articles()])
        docstring = cls.__doc__ if cls.__doc__ else ""
        return f"{name}\n{articles}\n{docstring}"

    def _detect_blinks(
            self,
            x: np.ndarray,
            y: np.ndarray,
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
        pad_samples = self.pad_blinks_samples
        if pad_samples == 0:
            return is_blink
        for i, val in enumerate(is_blink):
            if val:
                start = max(0, i - pad_samples)
                end = min(len(is_blink), i + pad_samples)
                is_blink[start:end] = True
        return is_blink

    def _reshape_vectors(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
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

    :param missing_value: the value that indicates missing data in the gaze data.
    :param min_event_duration: the minimum duration of a gaze event, in milliseconds.
    :param pad_blinks_ms: the duration to pad around detected blinks, in milliseconds.
    :param saccade_velocity_threshold: the threshold for angular velocity, in degrees per second. Default is 45 degrees
        per-second, as suggested in the paper "One algorithm to rule them all? An evaluation and discussion of ten eye
        movement event-detection algorithms" (2016), Andersson et al.
    """

    _ARTICLES = [
        "Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. " +
        "In Proceedings of the Symposium on Eye Tracking Research & Applications (pp. 71-78)",
    ]

    _DEFAULT_SACCADE_VELOCITY_THRESHOLD = 45  # deg/s
    _SACCADE_VELOCITY_THRESHOLD_STR = "saccade_velocity_threshold"

    def __init__(
            self,
            missing_value: float,
            min_event_duration: float,
            pad_blinks_ms: float,
            saccade_velocity_threshold: float = _DEFAULT_SACCADE_VELOCITY_THRESHOLD,
    ):
        super().__init__(missing_value, min_event_duration, pad_blinks_ms)
        if saccade_velocity_threshold <= 0:
            raise ValueError("Saccade velocity threshold must be positive")
        self._saccade_velocity_threshold = saccade_velocity_threshold

    @classmethod
    def get_default_params(cls) -> Dict[str, float]:
        return {
            cls._SACCADE_VELOCITY_THRESHOLD_STR: cls._DEFAULT_SACCADE_VELOCITY_THRESHOLD,
        }

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float,
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

    @property
    def saccade_velocity_threshold_deg(self) -> float:
        return self._saccade_velocity_threshold


class IVVTDetector(IVTDetector):
    """
    Implements the I-VVT (dual velocity threshold) gaze event detection algorithm, which builds on the I-VT algorithm
    by adding a second velocity threshold for smooth pursuit detection. The I-VT algorithm is described in:
        Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols.
        In Proceedings of the Symposium on Eye Tracking Research & Applications (pp. 71-78).

    General algorithm:
    1. Calculate the angular velocity of the gaze data
    2. Identify saccade candidates as samples with angular velocity greater than the saccade threshold
    3. Identify smooth pursuit candidates as samples with intermediate angular velocity - lower than the saccade
        threshold and greater than the smooth pursuit threshold.
    4. Assume undefined (non-blink) samples are fixations

    :param missing_value: the value that indicates missing data in the gaze data.
    :param min_event_duration: the minimum duration of a gaze event, in milliseconds.
    :param pad_blinks_ms: the duration to pad around detected blinks, in milliseconds.
    :param saccade_velocity_threshold: the threshold for angular velocity, in degrees per second. Default is 45 degrees
        per-second, as suggested in the paper "One algorithm to rule them all? An evaluation and discussion of ten eye
        movement event-detection algorithms" (2016), Andersson et al.
    :param smooth_pursuit_velocity_threshold: the threshold for angular velocity, in degrees per second, that separates
        smooth pursuit from fixations. Default is 5 degrees per-second.
    """

    _ARTICLES = [
        "Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. " +
        "In Proceedings of the Symposium on Eye Tracking Research & Applications (pp. 71-78)",
    ]

    _DEFAULT_SMOOTH_PURSUIT_VELOCITY_THRESHOLD = 20  # deg/s
    _SMOOTH_PURSUIT_VELOCITY_THRESHOLD_STR = "smooth_pursuit_velocity_threshold"

    def __init__(
            self,
            missing_value: float,
            min_event_duration: float,
            pad_blinks_ms: float,
            saccade_velocity_threshold: float = IVTDetector._DEFAULT_SACCADE_VELOCITY_THRESHOLD,
            smooth_pursuit_velocity_threshold: float = _DEFAULT_SMOOTH_PURSUIT_VELOCITY_THRESHOLD,
    ):
        super().__init__(missing_value, min_event_duration, pad_blinks_ms, saccade_velocity_threshold)
        if smooth_pursuit_velocity_threshold <= 0:
            raise ValueError("Smooth pursuit velocity threshold must be positive")
        self._smooth_pursuit_velocity_threshold = smooth_pursuit_velocity_threshold

    @classmethod
    def get_default_params(cls) -> Dict[str, float]:
        return {
            cls._SACCADE_VELOCITY_THRESHOLD_STR: cls._DEFAULT_SACCADE_VELOCITY_THRESHOLD,
            cls._SMOOTH_PURSUIT_VELOCITY_THRESHOLD_STR: cls._DEFAULT_SMOOTH_PURSUIT_VELOCITY_THRESHOLD
        }

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float,
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

    @property
    def smooth_pursuit_velocity_threshold_deg(self) -> float:
        return self._smooth_pursuit_velocity_threshold


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

    :param missing_value: the value that indicates missing data in the gaze data.
    :param min_event_duration: the minimum duration of a gaze event, in milliseconds.
    :param pad_blinks_ms: the duration to pad around detected blinks, in milliseconds.
    :param dispersion_threshold: the threshold for dispersion, in degrees. Default is 0.5 degrees, as suggested in the
        paper "One algorithm to rule them all? An evaluation and discussion of ten eye movement event-detection
        algorithms" (2016), Andersson et al.
    :param window_duration: the duration of the window in milliseconds. Default is 100 ms, as suggested in the paper
        "One algorithm to rule them all? An evaluation and discussion of ten eye movement event-detection algorithms"
        (2016), Andersson et al.
    """

    _ARTICLES = [
        "Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. " +
        "In Proceedings of the Symposium on Eye Tracking Research & Applications (pp. 71-78)",
    ]

    _DEFAULT_DISPERSION_THRESHOLD = 0.5  # visual degrees
    _DEFAULT_WINDOW_DURATION = 100  # ms
    __DISPERSION_THRESHOLD_STR = "dispersion_threshold"
    __DEFAULT_WINDOW_DURATION_STR = "window_duration"

    def __init__(
            self,
            missing_value: float,
            min_event_duration: float,
            pad_blinks_ms: float,
            dispersion_threshold: float = _DEFAULT_DISPERSION_THRESHOLD,
            window_duration: float = _DEFAULT_WINDOW_DURATION,
    ):
        super().__init__(missing_value, min_event_duration, pad_blinks_ms)
        if dispersion_threshold <= 0:
            raise ValueError("Dispersion threshold must be positive")
        self._dispersion_threshold = dispersion_threshold
        self._window_duration = window_duration

    @classmethod
    def get_default_params(cls) -> Dict[str, float]:
        return {
            cls.__DISPERSION_THRESHOLD_STR: cls._DEFAULT_DISPERSION_THRESHOLD,
            cls.__DEFAULT_WINDOW_DURATION_STR: cls._DEFAULT_WINDOW_DURATION,
        }

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float,
    ) -> np.ndarray:
        labels = np.asarray(copy.deepcopy(labels), dtype=EventLabelEnum)
        ws = self._calculate_window_size_samples(t)
        px_threshold = self._get_threshold(self.dispersion_threshold_deg, "px", viewer_distance_cm, pixel_size_cm)
        start_idx, end_idx = 0, ws
        is_fixation = False
        while end_idx <= len(t):
            dispersion = line_dispersion(x[start_idx:end_idx], y[start_idx:end_idx])
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
            f"{self.__DISPERSION_THRESHOLD_STR}_deg": self.dispersion_threshold_deg,
            f"{self.__DISPERSION_THRESHOLD_STR}_px": px_threshold,
        })
        return labels

    @property
    def dispersion_threshold_deg(self) -> float:
        return self._dispersion_threshold

    @property
    def window_duration_ms(self) -> float:
        return self._window_duration

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
        2. Calculate the median-based-standard-deviation of the velocity in each axis
        3. Calculate the saccade threshold as the multiple of the median-based-standard-deviation with the
            `lambda_param` constant
        4. Identify saccade candidates as samples with velocity greater than the noise threshold

    :param missing_value: the value that indicates missing data in the gaze data.
    :param min_event_duration: the minimum duration of a gaze event, in milliseconds.
    :param pad_blinks_ms: the duration to pad around detected blinks, in milliseconds.
    :param lambda_param: the multiplication coefficient used for calculating saccade threshold. Default is 5, as
        suggested in the original paper
    :param deriv_window_size: number of samples (including the middle sample) used to calculate the velocity, meaning
        the velocity at time t is the difference between the sum of the `ws//2` samples after and before t, divided by
        the window size. Default is 5, as suggested in the original paper
    """

    _ARTICLES = [
        "Engbert, R. & Kliegl, R. (2003). Microsaccades uncover the orientation of covert attention. Vision Research",
        "Engbert, R., Mergenthaler, K., & Purves, D. (Ed.). (2006). Microsaccades are triggered by low retinal " +
        "image slip. PNAS Proceedings of the National Academy of Sciences of the United States of America",
    ]

    _DEFAULT_LAMBDA_PARAM = 5               # standard deviation multiplier
    _DEFAULT_DERIVATION_WINDOW_SIZE = 5     # number of samples used to calculate axial velocity
    __THRESHOLD_VELOCITY_STR = "threshold_velocity"
    __LAMBDA_PARAM_STR = "lambda_param"
    __DERIV_WINDOW_SIZE_STR = "deriv_window_size"

    def __init__(
            self,
            missing_value: float,
            min_event_duration: float,
            pad_blinks_ms: float,
            lambda_param: float = _DEFAULT_LAMBDA_PARAM,
            deriv_window_size: int = _DEFAULT_DERIVATION_WINDOW_SIZE,
    ):
        super().__init__(missing_value, min_event_duration, pad_blinks_ms)
        self._lambda_param = lambda_param
        if self._lambda_param <= 0:
            raise ValueError("Lambda parameter must be positive")
        self._deriv_window_size = deriv_window_size
        if self._deriv_window_size <= 0:
            raise ValueError("Derivation window size must be positive")

    @classmethod
    def get_default_params(cls) -> Dict[str, float]:
        return {
            cls.__LAMBDA_PARAM_STR: cls._DEFAULT_LAMBDA_PARAM,
            cls.__DERIV_WINDOW_SIZE_STR: cls._DEFAULT_DERIVATION_WINDOW_SIZE,
        }

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float,
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
            f"{cnst.X}_{self.__THRESHOLD_VELOCITY_STR}_pxs": x_thresh,
            f"{cnst.Y}_{self.__THRESHOLD_VELOCITY_STR}_px": y_thresh,
        })
        return labels

    @property
    def lambda_param(self) -> float:
        return self._lambda_param

    @property
    def deriv_window_size(self) -> int:
        return self._deriv_window_size

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
    def _reshape_vectors(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        t, x, y = super()._reshape_vectors(t, x, y)
        if len(x) < 2 * self.deriv_window_size:
            raise ValueError(f"Derivation window ({self.deriv_window_size} samples) is too long for the given data")
        return t, x, y


class NHDetector(BaseDetector):
    """
    Implements the algorithm described by Nyström & Holmqvist in
        Nyström, M., Holmqvist, K. An adaptive algorithm for fixation, saccade, and glissade detection in eyetracking
        data. Behavior Research Methods 42, 188–204 (2010).
    The code is based on the Matlab implementation available in https://github.com/dcnieho/NystromHolmqvist2010, which
    was developed for the following article:
        Niehorster, D. C., Siu, W. W., & Li, L. (2015). Manual tracking enhances smooth pursuit eye movements. Journal
        of vision, 15(15), 11-11.
    License: Unspecified

    General algorithm:
        1. Calculate angular velocity & acceleration
        2. Denoise the data using SAVGOL filter
        3. Saccade Detection:
            3a. Detect velocity peaks
            3b. Detect saccade onset and offset surrounding each peak
            3c. Ignore saccades that are too short
        4. PSO (Glissade) Detection:
            4a. Detect samples with velocity exceeding the PSO threshold, that shortly follow a saccade offset
            4b. Find PSO offset
        5. Fixation Detection:
            5a. Detect samples that are not part of a saccade, PSO or noise
            5b. Ignore fixations that are too short

    :param missing_value: the value that indicates missing data in the gaze data.
    :param min_event_duration: the minimum duration of a gaze event, in milliseconds.
    :param pad_blinks_ms: the duration to pad around detected blinks, in milliseconds.
    :param filter_duration_ms: Savitzky-Golay filter's duration (ms)
    :param filter_polyorder: Savitzky-Golay filter's polynomial order
    :param saccade_max_velocity: maximum saccade velocity (deg/s)
    :param saccade_max_acceleration: maximum saccade acceleration (deg/s^2)
    :param min_saccade_duration: minimum saccade duration (ms)
    :param min_fixation_duration: minimum fixation duration (ms)
    :param max_pso_duration: maximum PSO duration (ms)
    :param alpha_param: weight of saccade onset threshold when detecting saccade offset
    :param ignore_short_peak_durations: if True, excludes sporadic instances where velocity is above the PT, when
        detecting saccade peaks
    :param allow_high_psos: if True, includes PSOs with maximum velocity exceeding saccades' peak threshold (PT),
        given that the PSO's max velocity is still lower than the preceding saccade's max velocity
    """

    _ARTICLES = [
        "Nyström, M., Holmqvist, K. An adaptive algorithm for fixation, saccade, and glissade detection in " +
        "eyetracking data. Behavior Research Methods 42, 188–204 (2010)",
        "Niehorster, D. C., Siu, W. W., & Li, L. (2015). Manual tracking enhances smooth pursuit eye movements. " +
        "Journal of vision, 15(15), 11-11",
    ]

    _DEFAULT_FILTER_DURATION_MS = 2 * cnfg.EVENT_MAPPING[EventLabelEnum.SACCADE][cnst.MIN_DURATION_STR]     # 20ms
    _DEFAULT_FILTER_POLYORDER = 2                                                                           # unitless
    _DEFAULT_SACCADE_MAX_VELOCITY = 1000                                                                    # deg/s
    _DEFAULT_SACCADE_MAX_ACCELERATION = 100000                                                              # deg/s^2
    _DEFAULT_MIN_SACCADE_DURATION_MS = cnfg.EVENT_MAPPING[EventLabelEnum.SACCADE][cnst.MIN_DURATION_STR]    # 10ms
    _DEFAULT_MIN_FIXATION_DURATION_MS = cnfg.EVENT_MAPPING[EventLabelEnum.FIXATION][cnst.MIN_DURATION_STR]  # 50ms
    _DEFAULT_MAX_PSO_DURATION_MS = cnfg.EVENT_MAPPING[EventLabelEnum.PSO][cnst.MAX_DURATION_STR]            # 80ms
    _DEFAULT_ALPHA_PARAM = 0.7                                                                              # unitless

    __FILTER_DURATION_MS_STR = "filter_duration_ms"
    __FILTER_POLYORDER_STR = "filter_polyorder"
    __SACCADE_MAX_VELOCITY_STR = "saccade_max_velocity"
    __SACCADE_MAX_ACCELERATION_STR = "saccade_max_acceleration"
    __MIN_SACCADE_DURATION_STR = "min_saccade_duration"
    __MIN_FIXATION_DURATION_STR = "min_fixation_duration"
    __MAX_PSO_DURATION_STR = "max_pso_duration"
    __ALPHA_PARAM_STR = "alpha_param"
    __IGNORE_SHORT_PEAK_DURATIONS_STR = "ignore_short_peak_durations"
    __ALLOW_HIGH_PSOS_STR = "allow_high_psos"
    __SACCADE_PEAK_VELOCITY_THRESHOLD_STR = "saccade_peak_velocity_threshold"
    __SACCADE_ONSET_VELOCITY_THRESHOLD_STR = "saccade_onset_velocity_threshold"

    def __init__(
            self,
            missing_value: float,
            min_event_duration: float,
            pad_blinks_ms: float,
            filter_duration_ms: float = _DEFAULT_FILTER_DURATION_MS,
            filter_polyorder: int = _DEFAULT_FILTER_POLYORDER,
            saccade_max_velocity: float = _DEFAULT_SACCADE_MAX_VELOCITY,
            saccade_max_acceleration: float = _DEFAULT_SACCADE_MAX_ACCELERATION,
            min_saccade_duration: float = _DEFAULT_MIN_SACCADE_DURATION_MS,
            min_fixation_duration: float = _DEFAULT_MIN_FIXATION_DURATION_MS,
            max_pso_duration: float = _DEFAULT_MAX_PSO_DURATION_MS,
            alpha_param: float = _DEFAULT_ALPHA_PARAM,
            ignore_short_peak_durations: bool = True,   # whether to exclude sporadic samples from the calculation (default is True)
            allow_high_psos: bool = True,
    ):
        """
        Initialize a new Nyström & Holmqvist gaze event detector
        :param filter_duration_ms: Savitzky-Golay filter's duration (ms)
        :param filter_polyorder: Savitzky-Golay filter's polynomial order
        :param saccade_max_velocity: maximum saccade velocity (deg/s)
        :param saccade_max_acceleration: maximum saccade acceleration (deg/s^2)
        :param min_saccade_duration: minimum saccade duration (ms)
        :param min_fixation_duration: minimum fixation duration (ms)
        :param max_pso_duration: maximum PSO duration (ms)
        :param alpha_param: weight of saccade onset threshold when detecting saccade offset
        :param ignore_short_peak_durations: if True, excludes sporadic instances where velocity is above the PT, when
            detecting saccade peaks
        :param allow_high_psos: if True, includes PSOs with maximum velocity exceeding saccades' peak threshold (PT),
            given that the PSO's max velocity is still lower than the preceding saccade's max velocity
        :param missing_value: the value that indicates missing data in the gaze data, default is np.NaN
        :param min_event_duration: minimum duration of a gaze event (ms) default is 5 ms
        :param pad_blinks_ms: padding duration for blinks (ms), default is 0 ms
        """
        super().__init__(missing_value, min_event_duration, pad_blinks_ms)
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
        self._min_saccade_duration = max(min_saccade_duration, self._min_event_duration)
        if self._min_saccade_duration < 0:
            raise ValueError("Minimum saccade duration must be non-negative")
        self._min_fixation_duration = max(min_fixation_duration, self._min_event_duration)
        if self._min_fixation_duration < 0:
            raise ValueError("Minimum fixation duration must be non-negative")
        self._max_pso_duration = max(max_pso_duration, self._min_event_duration)
        if self._max_pso_duration < 0:
            raise ValueError("Maximum PSO duration must be non-negative")
        self._alpha_param = alpha_param
        if not 0 <= self._alpha_param <= 1:
            raise ValueError("Alpha parameter must be between 0 and 1")
        self._beta_param = 1 - alpha_param
        self._ignore_short_peak_durations = ignore_short_peak_durations
        self._allow_high_psos = allow_high_psos

    @classmethod
    def get_default_params(cls) -> Dict[str, float]:
        return {
            cls.__FILTER_DURATION_MS_STR: cls._DEFAULT_FILTER_DURATION_MS,
            cls.__FILTER_POLYORDER_STR: cls._DEFAULT_FILTER_POLYORDER,
            cls.__SACCADE_MAX_VELOCITY_STR: cls._DEFAULT_SACCADE_MAX_VELOCITY,
            cls.__SACCADE_MAX_ACCELERATION_STR: cls._DEFAULT_SACCADE_MAX_ACCELERATION,
            cls.__MIN_SACCADE_DURATION_STR: cls._DEFAULT_MIN_SACCADE_DURATION_MS,
            cls.__MIN_FIXATION_DURATION_STR: cls._DEFAULT_MIN_FIXATION_DURATION_MS,
            cls.__MAX_PSO_DURATION_STR: cls._DEFAULT_MAX_PSO_DURATION_MS,
            cls.__ALPHA_PARAM_STR: cls._DEFAULT_ALPHA_PARAM,
            cls.__IGNORE_SHORT_PEAK_DURATIONS_STR: True,
            cls.__ALLOW_HIGH_PSOS_STR: True,
        }

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float,
    ) -> np.ndarray:
        # detect noise:
        v, a = self._velocities_and_accelerations(x, y, viewer_distance_cm, pixel_size_cm)  # deg/s, deg/s^2
        is_noise = self._detect_noise(v, a)

        # denoise the data:
        x_copy, y_copy, v_copy, a_copy = x.copy(), y.copy(), v.copy(), v.copy()
        x_copy[is_noise] = np.nan
        y_copy[is_noise] = np.nan
        v_copy[is_noise] = np.nan
        a_copy[is_noise] = np.nan

        # detect start & end indices of saccades and PSOs:
        peak_threshold, onset_threshold = self._calculate_saccade_thresholds(v_copy)  # PT, OnT velocity thresholds
        saccades_info = self._extract_saccade_info(
            v_copy, peak_threshold, onset_threshold
        )  # saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        psos_info = self._detect_psos(v_copy, saccades_info)  # saccade id -> (start_idx, end_idx, pso_type)

        # save and return results
        labels = self._classify_samples(
            labels, is_noise, saccades_info, psos_info, self._allow_high_psos
        )
        self._metadata.update({
            f"{self.__SACCADE_PEAK_VELOCITY_THRESHOLD_STR}_deg": peak_threshold,
            f"{self.__SACCADE_PEAK_VELOCITY_THRESHOLD_STR}_px": visual_angle_to_pixels(
                peak_threshold, viewer_distance_cm, pixel_size_cm, use_radians=False, keep_sign=False
            ),
            f"{self.__SACCADE_ONSET_VELOCITY_THRESHOLD_STR}_deg": onset_threshold,
            f"{self.__SACCADE_ONSET_VELOCITY_THRESHOLD_STR}_px": visual_angle_to_pixels(
                onset_threshold, viewer_distance_cm, pixel_size_cm, use_radians=False, keep_sign=False
            ),
        })
        return labels

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
    def ignore_short_peak_durations(self) -> bool:
        return self._ignore_short_peak_durations

    @property
    def is_high_psos_allowed(self) -> bool:
        return self._allow_high_psos

    @property
    def min_fixation_samples(self) -> int:
        return self._calc_num_samples(self._min_fixation_duration, self.sr)

    def _velocities_and_accelerations(self, x: np.ndarray, y: np.ndarray, vd, ps) -> (np.ndarray, np.ndarray):
        """
        Calculates the 1st and 2nd derivatives of the gaze data, using the Savitzky-Golay filter. Then calculates the
        velocity and acceleration from the derivatives.

        Note the original paper calculates the velocity and acceleration using:
            v = sr * sqrt[(x')^2 + (y')^2] * pixel-to-angle-constant
            a = sr * sqrt[(x'')^2 + (y'')^2] * pixel-to-angle-constant
        We use `delta=1/self.sr` when calculating the derivatives, to account for sampling time, so we don't need to
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
        dx = savgol_filter(x, ws, self._filter_polyorder, deriv=1, delta=1/self.sr)
        dy = savgol_filter(y, ws, self._filter_polyorder, deriv=1, delta=1/self.sr)
        v = np.sqrt(dx ** 2 + dy ** 2) * px_to_deg_constant
        # calculate angular acceleration (deg/s^2): a = sqrt((x'')^2 + (y'')^2) * pixel-to-angle-constant:
        ddx = savgol_filter(x, ws, self._filter_polyorder, deriv=2, delta=1/self.sr)
        ddy = savgol_filter(y, ws, self._filter_polyorder, deriv=2, delta=1/self.sr)
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

    def _calculate_saccade_thresholds(self, v: np.ndarray, max_iters: int = 100) -> (float, float):
        """
        Calculates the threshold velocities for saccade peaks (PT) and saccade onsets (OnT):
        1. Initialize PT as the maximal value (in range 75-300 deg/s) that has at least one sample with higher velocity.
        2. Iteratively update PT until convergence:
            a. Find samples with velocity below PT.
            b. If enforce_min_dur is True:
                - ignore samples that aren't part of a chunk of length >= min_fixation_samples.
                - chunks of sufficient length are shortened by min_saccade_samples // 3 samples at each edge, to avoid
                    contamination from saccades.
            c. Calculate mean & std of velocity below PT.
            d. Update PT = mean + 6 * std
        3. After convergence, calculate OnT as the mean velocity of samples below PT + 3 * std of velocity below PT.
        See more details in https://shorturl.at/wyCH7.

        :param v: angular velocities of the gaze data (deg/s)
        :param max_iters: maximum number of iterations (default is 100)
        :return:
            pt (float) - threshold velocity for detecting saccade peaks
            ont (float) - threshold velocity for detecting saccade onsets
        """
        # find the starting PT value, by making sure there are at least 1 peak with higher velocity
        start_pt_options = np.arange(300, 74, -25)
        is_v_above_pt = (v > start_pt_options[:, np.newaxis]).any(axis=1)
        pt = start_pt_options[np.argmin(is_v_above_pt)] if any(is_v_above_pt) else np.median(start_pt_options)
        # iteratively update PT value until convergence
        pt_prev = 0
        is_below_pt = v <= pt
        min_chunk_size = self.min_fixation_samples
        num_edge_sample_to_drop = self._calc_num_samples(self._min_saccade_duration // 3, self.sr)
        while abs(pt - pt_prev) > 1 and max_iters > 0:
            pt_prev = pt
            max_iters -= 1
            is_below_pt = v <= pt
            if self.ignore_short_peak_durations:
                # only consider samples that are part of a chunk of length >= min_fixation_samples:
                chunks_below_pt = [
                    ch for ch in get_chunk_indices(is_below_pt) if  is_below_pt[ch[0]] and len(ch) >= min_chunk_size
                ]
                # drop samples at the edges of each chunk to avoid contamination from saccades:
                chunks_below_pt = [ch[num_edge_sample_to_drop: -num_edge_sample_to_drop] for ch in chunks_below_pt]
                if len(chunks_below_pt) > 0:
                    # concatenate the chunks to get the final boolean array:
                    is_below_pt = np.concatenate(chunks_below_pt)
            mu = np.nanmean(v[is_below_pt])
            sigma = np.nanstd(v[is_below_pt])
            pt = mu + 6 * sigma
        if max_iters == 0:
            raise RuntimeError("Failed to converge on PT_1 value for saccade detection")
        ont = np.nanmean(v[is_below_pt]) + 3 * np.nanstd(v[is_below_pt])
        return pt, ont

    def _extract_saccade_info(
            self, v: np.ndarray, pt: float, ont: float, min_peak_samples: int = 2
    ) -> Dict[int, Tuple[int, int, int, float]]:
        """
        Detects saccades in the gaze data based on the angular velocity:
        1. Detect samples with velocity exceeding the saccade peak threshold (PT)
        2. Find the 1st sample preceding each peak with velocity below the onset threshold (OnT) and is a local minimum
        3. Find the 1st sample following each peak with velocity below the offset threshold (OfT) and is a local minimum
        4. Match each saccade peak-idx with its onset-idx, offset-idx and offset-threshold-velocity

        :param v: angular velocity of the gaze data
        :param pt: saccades' peak threshold velocity
        :param ont: saccades' onset threshold velocity
        :param min_peak_samples: minimum number of samples for a peak to be considered a saccade, otherwise ignored
        :return: dict mapping saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        """
        is_above_pt = v > pt
        chunks_above_pt = [     # assume very short peaks are noise
            ch for ch in get_chunk_indices(is_above_pt) if is_above_pt[ch[0]] and len(ch) >= min_peak_samples
        ]
        # each chunk is a possible saccade, find the onset and offset of each saccade
        saccades_info = {}  # saccade_id -> (start_idx, peak_idx, end_idx, offset_threshold)
        for i, chunk in enumerate(chunks_above_pt):
            peak_idx: int = int(chunk[0])
            # find the onset of the saccade: the 1st local minimum preceding the peak with v<=OnT
            onset_idx = self.__find_local_minimum_index(v, peak_idx, ont, move_back=True)
            # calculate the offset threshold: OfT = a * OnT + b * OtT
            # note the locally adaptive term: OtT = mean(v) + 3 * std(v) for the min_fixation_samples before the onset
            window_start_idx = max(0, onset_idx - self.min_fixation_samples)
            window_vel = v[window_start_idx: onset_idx]
            ott = np.nanmean(window_vel) + 3 * np.nanstd(window_vel)
            if np.isfinite(ott) and ott < pt:
                offset_threshold = self.alpha * ont + self.beta * ott
            else:
                offset_threshold = ont
            # save saccade info
            last_peak_idx: int = int(chunk[-1])
            offset_idx = self.__find_local_minimum_index(v, last_peak_idx, offset_threshold, move_back=False)
            saccades_info[i] = (onset_idx, peak_idx, offset_idx, offset_threshold)
        return saccades_info

    def _detect_psos(
            self, v: np.ndarray, saccade_info: Dict[int, Tuple[int, int, int, float]]
    ) -> Dict[int, Tuple[int, int, bool]]:
        """
        Detects PSOs in the gaze data based on the angular velocity:
        1. Determine what velocity threshold to use for PSO detection (depending on value of self._detect_high_psos)
        2. Check if a window of length min fixation duration, succeeding each saccade, has samples above AND below the
              threshold. If so, there is a PSO.
        3. Identify the end of the PSO - the first local velocity-minimum after the last sample above the threshold
        4. Ignore PSOs with amplitude exceeding the preceding saccade

        :param v: angular velocity of the gaze data
        :param saccade_info: dictionary of saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        :return: dict matching saccade id with PSO start-idx, end-idx and PSO type (high or low)
        """
        max_pso_samples = self._calc_num_samples(self._max_pso_duration, self.sr)
        pso_info = {}  # saccade_id -> (start_idx, end_idx, pso_type)

        # extract saccade indices to ready-to-use lists
        saccade_info_list = sorted(saccade_info.items(), key=lambda x: x[0])
        saccade_onset_idxs = np.array([info[1][0] for info in saccade_info_list])
        saccade_peak_idxs = np.array([info[1][1] for info in saccade_info_list])
        saccade_offset_idxs = np.array([info[1][2] for info in saccade_info_list])

        # find PSO start & end idxs after each saccade
        sac_id = 0
        while sac_id < len(saccade_info):
            sac_onset_idx, sac_peak_idx, sac_offset_idx, sac_offset_threshold = saccade_info[sac_id]
            # if a window succeeding a saccade has samples above AND below the offset threshold, there is a PSO
            start_idx, end_idx = 1 + sac_offset_idx, min([1 + sac_offset_idx + self.min_fixation_samples, len(v)])
            window = v[start_idx: end_idx]
            is_above, is_below = window > sac_offset_threshold, window < sac_offset_threshold
            if not (any(is_above) and any(is_below)):
                # no PSO for this saccade
                sac_id += 1
                continue

            # if the window contains samples with velocity above saccades' peak threshold, but below the preceding
            # saccade's max velocity, it is considered "high" PSO and contains saccade peaks
            end_idx = start_idx + np.where(is_above)[0][-1]
            is_high_pso = False
            is_peak_in_window = [start_idx <= p_idx < end_idx for p_idx in saccade_peak_idxs]
            if any(is_peak_in_window):
                last_peak = np.where(is_peak_in_window)[0][-1]
                last_offset_idx = saccade_offset_idxs[last_peak]
                if v[start_idx: last_offset_idx].max() < v[sac_onset_idx: sac_offset_idx].max():
                    # only allow high PSO if its max velocity is below the previous saccade's max velocity
                    is_high_pso = True
                    end_idx = max([end_idx, last_offset_idx])

            # move forward from the last sample above the threshold to find the first local minimum
            window = v[end_idx: len(v)]
            min_idx_in_window = self.__find_local_minimum_index(window, 0, sac_offset_threshold, move_back=False)
            end_idx += min_idx_in_window
            if end_idx - start_idx > max_pso_samples:
                # PSO is too long, ignore it
                sac_id += 1
                continue

            # save PSO info
            pso_info[sac_id] = (start_idx, end_idx, is_high_pso)

            # move to the next saccade
            next_sac_idx = np.where(saccade_onset_idxs > end_idx)[0]
            if len(next_sac_idx):
                sac_id = next_sac_idx[0]
            else:
                break
        return pso_info

    @staticmethod
    def _classify_samples(
            labels: np.ndarray,
            is_noise: np.ndarray,
            saccade_info: Dict[int, Tuple[int, int, int, float]],
            pso_info: Dict[int, Tuple[int, int, bool]],
            allow_high_psos: bool,
    ) -> (np.ndarray, np.ndarray):
        """
        Classifies each sample as either noise, saccade, PSO, fixation or blink. Samples that are not classified as
        noise, blink, saccade or PSO are considered fixations.

        If we allow high PSOs we override the saccade classification when a high PSO was also detected (i.e. we consider
        a saccade immediately following a previous saccade as high-PSO).
        Note the matlab implementation does the opposite, and first merges subsequent saccades that are only separated
        by a few PSO samples, and classifies the union as a saccade. We don't do this here (see in https://shorturl.at/EMOQR)

        :param is_noise: boolean array indicating noise samples
        :param saccade_info: dict of saccade id -> (onset-idx, peak-idx, offset-idx, offset-threshold-velocity)
        :param pso_info: dict of saccade -> (PSO start-idx, PSO end-idx and PSO type (high or low))
        :return: array of classified samples
        """
        labels = np.asarray(copy.deepcopy(labels), dtype=EventLabelEnum)
        for val in saccade_info.values():
            onset_idx, _, offset_idx, _ = val
            labels[onset_idx: offset_idx] = EventLabelEnum.SACCADE
        for val in pso_info.values():
            start_idx, end_idx, is_high = val
            if is_high and not allow_high_psos:
                # high PSO are essentially saccades that immediately follow a previous saccade
                continue
            labels[start_idx: end_idx] = EventLabelEnum.PSO

        is_blinks = labels == EventLabelEnum.BLINK
        is_saccade = labels == EventLabelEnum.SACCADE
        is_pso = labels == EventLabelEnum.PSO
        labels[~(is_noise | is_saccade | is_pso | is_blinks)] = EventLabelEnum.FIXATION
        return labels

    @staticmethod
    def __find_local_minimum_index(arr: np.ndarray, idx: int, min_thresh=np.inf, move_back=False) -> int:
        """
        Finds a local minimum in the array (an element that is smaller than its neighbors) starting from the given index.
        :param arr: the array to search in
        :param idx: the starting index
        :param min_thresh: the minimum value for a local minimum    (default: infinity)
        :param move_back: whether to move back or forward from the starting index   (default: False)
        :return: the index of the local minimum
        """
        while 0 < idx < len(arr):
            if arr[idx] < min_thresh and arr[idx] < arr[idx + 1] and arr[idx] < arr[idx - 1]:
                # idx is a local minimum
                return idx
            idx = idx - 1 if move_back else idx + 1
        return idx


class REMoDNaVDetector(BaseDetector):
    """
    This is a wrapper class that uses an implementation of the REMoDNaV algorithm to detect gaze events in the gaze
        data. This algorithm is based on the NHDetector algorithm, but extends and improves it by adding a more
        sophisticated saccade/pso detection algorithm, and by adding a smooth pursuit detection algorithm.

    See the REMoDNaV paper:
        Dar AH, Wagner AS, Hanke M. REMoDNaV: robust eye-movement classification for dynamic stimulation. Behav Res
        Methods. 2021 Feb;53(1):399-414. doi: 10.3758/s13428-020-01428-x
    See the NH Detector paper:
        Nyström, M., Holmqvist, K. An adaptive algorithm for fixation, saccade, and glissade detection in eyetracking
        data. Behavior Research Methods 42, 188–204 (2010).

    See the REMoDNaV algorithm documentation & implementation:
        https://github.com/psychoinformatics-de/remodnav/tree/master
    License: MIT

    :param missing_value: the value that indicates missing data in the gaze data.
    :param min_event_duration: the minimum duration of a gaze event, in milliseconds.
    :param pad_blinks_ms: the duration to pad around detected blinks, in milliseconds.
    :param min_saccade_duration: the minimum duration of a saccade (ms), default is 4 ms
    :param saccade_initial_velocity_threshold: the initial velocity threshold for saccade detection (deg/s), default is 300
    :param saccade_context_window_duration: the duration of the context window for saccade detection (ms), default is 1000
    :param saccade_initial_max_freq: the initial maximum frequency for saccade detection (Hz), default is 2.0
    :param saccade_onset_threshold_noise_factor: the noise factor for saccade onset threshold, default is 5.0
    :param min_smooth_pursuit_duration: the minimum duration of a smooth pursuit (ms), default is 4 ms
    :param smooth_pursuits_lowpass_cutoff_freq: the lowpass cutoff frequency for smooth pursuit detection (Hz), default is 4.0
    :param smooth_pursuit_drift_velocity_threshold: the drift velocity threshold for smooth pursuit detection (deg/s), default is 2.0
    :param min_fixation_duration: the minimum duration of a fixation (ms), default is 50 ms
    :param min_blink_duration: the minimum duration of a blink (ms), default is 20 ms
    :param max_pso_duration: the maximum duration of a PSO (ms), default is 80 ms
    :param savgol_filter_polyorder: the polynomial order for the Savitzky-Golay filter, default is 2
    :param savgol_filter_duration_ms: the duration of the Savitzky-Golay filter (ms), default is 19 ms
    :param median_filter_duration_ms: the duration of the median filter (ms), default is 50 ms
    :param max_velocity: the maximum velocity of the gaze data (deg/s), default is 1500
    """

    _ARTICLES = [
        "Dar AH, Wagner AS, Hanke M. REMoDNaV: robust eye-movement classification for dynamic stimulation. Behav " +
        "Res Methods. 2021 Feb;53(1):399-414. doi: 10.3758/s13428-020-01428-x",
    ]

    _DEFAULT_MIN_SACCADE_DURATION_MS = cnfg.EVENT_MAPPING[EventLabelEnum.SACCADE][cnst.MIN_DURATION_STR]
    _DEFAULT_MIN_SMOOTH_PURSUIT_DURATION_MS = cnfg.EVENT_MAPPING[EventLabelEnum.SMOOTH_PURSUIT][cnst.MIN_DURATION_STR]
    _DEFAULT_MIN_FIXATION_DURATION_MS = cnfg.EVENT_MAPPING[EventLabelEnum.FIXATION][cnst.MIN_DURATION_STR]
    _DEFAULT_MIN_BLINK_DURATION_MS = cnfg.EVENT_MAPPING[EventLabelEnum.BLINK][cnst.MIN_DURATION_STR]
    _DEFAULT_MAX_PSO_DURATION_MS = cnfg.EVENT_MAPPING[EventLabelEnum.PSO][cnst.MAX_DURATION_STR]

    _DEFAULT_SACCADE_INITIAL_VELOCITY_THRESHOLD = 300       # deg/s
    _DEFAULT_SACCADE_CONTEXT_WINDOW_DURATION_MS = 1000      # ms
    _DEFAULT_SACCADE_INITIAL_MAX_FREQ = 2.0                 # Hz
    _DEFAULT_SACCADE_ONSET_THRESHOLD_NOISE_FACTOR = 5.0     # unitless
    _DEFAULT_SMOOTH_PURSUIT_LOWPASS_CUTOFF_FREQ = 4.0       # Hz
    _DEFAULT_SMOOTH_PURSUIT_DRIFT_VELOCITY_THRESHOLD = 2.0  # deg/s
    _DEFAULT_SAVGOL_POLYORDER = 2                           # unitless
    _DEFAULT_SAVGOL_DURATION_MS = 19                        # ms
    _DEFAULT_MEDIAN_FILTER_DURATION_MS = 50                 # ms
    _DEFAULT_MAX_VELOCITY_DEG = 1500                        # deg/s

    __LABEL_MAPPING = {
        'FIXA': EventLabelEnum.FIXATION, 'SACC': EventLabelEnum.SACCADE, 'ISAC': EventLabelEnum.SACCADE,
        'HPSO': EventLabelEnum.PSO, 'IHPS': EventLabelEnum.PSO, 'LPSO': EventLabelEnum.PSO,
        'ILPS': EventLabelEnum.PSO, 'PURS': EventLabelEnum.SMOOTH_PURSUIT, 'BLNK': EventLabelEnum.BLINK
    }

    __MIN_SACCADE_DURATION_STR = "min_saccade_duration"
    __SACCADE_INITIAL_VELOCITY_THRESHOLD_STR = "saccade_initial_velocity_threshold"
    __SACCADE_CONTEXT_WINDOW_DURATION_STR = "saccade_context_window_duration"
    __SACCADE_INITIAL_MAX_FREQ_STR = "saccade_initial_max_freq"
    __SACCADE_ONSET_THRESHOLD_NOISE_FACTOR_STR = "saccade_onset_threshold_noise_factor"
    __MIN_SMOOTH_PURSUIT_DURATION_STR = "min_smooth_pursuit_duration"
    __SMOOTH_PURSUIT_LOWPASS_CUTOFF_FREQ_STR = "smooth_pursuits_lowpass_cutoff_freq"
    __SMOOTH_PURSUIT_DRIFT_VELOCITY_THRESHOLD_STR = "smooth_pursuit_drift_velocity_threshold"
    __MIN_FIXATION_DURATION_STR = "min_fixation_duration"
    __MIN_BLINK_DURATION_STR = "min_blink_duration"
    __MAX_PSO_DURATION_STR = "max_pso_duration"
    __SAVGOL_POLYORDER_STR = "savgol_filter_polyorder"
    __SAVGOL_DURATION_MS_STR = "savgol_filter_duration_ms"
    __MEDIAN_FILTER_DURATION_MS_STR = "median_filter_duration_ms"
    __MAX_VELOCITY_STR = "max_velocity"

    def __init__(
            self,
            missing_value: float,
            min_event_duration: float,
            pad_blinks_ms: float,
            min_saccade_duration: float = _DEFAULT_MIN_SACCADE_DURATION_MS,
            saccade_initial_velocity_threshold: float = _DEFAULT_SACCADE_INITIAL_VELOCITY_THRESHOLD,
            saccade_context_window_duration: float = _DEFAULT_SACCADE_CONTEXT_WINDOW_DURATION_MS,
            saccade_initial_max_freq: float = _DEFAULT_SACCADE_INITIAL_MAX_FREQ,
            saccade_onset_threshold_noise_factor: float = _DEFAULT_SACCADE_ONSET_THRESHOLD_NOISE_FACTOR,
            min_smooth_pursuit_duration: float = _DEFAULT_MIN_SMOOTH_PURSUIT_DURATION_MS,
            smooth_pursuits_lowpass_cutoff_freq: float = _DEFAULT_SMOOTH_PURSUIT_LOWPASS_CUTOFF_FREQ,
            smooth_pursuit_drift_velocity_threshold: float = _DEFAULT_SMOOTH_PURSUIT_DRIFT_VELOCITY_THRESHOLD,
            min_fixation_duration: float = _DEFAULT_MIN_FIXATION_DURATION_MS,
            min_blink_duration: float = _DEFAULT_MIN_BLINK_DURATION_MS,
            max_pso_duration: float = _DEFAULT_MAX_PSO_DURATION_MS,
            savgol_filter_polyorder: int = _DEFAULT_SAVGOL_POLYORDER,
            savgol_filter_duration_ms: float = _DEFAULT_SAVGOL_DURATION_MS,
            median_filter_duration_ms: float = _DEFAULT_MEDIAN_FILTER_DURATION_MS,
            max_velocity: float = _DEFAULT_MAX_VELOCITY_DEG,
    ):
        super().__init__(missing_value, min_event_duration, pad_blinks_ms)
        self._min_saccade_duration_ms = max(min_saccade_duration, self._min_event_duration)
        self._saccade_initial_velocity_threshold = saccade_initial_velocity_threshold
        self._saccade_context_window_duration = saccade_context_window_duration
        self._saccade_initial_max_freq = saccade_initial_max_freq
        self._saccade_onset_threshold_noise_factor = saccade_onset_threshold_noise_factor
        self._min_smooth_pursuit_duration_ms = max(min_smooth_pursuit_duration, self._min_event_duration)
        self._smooth_pursuit_lowpass_cutoff_freq = smooth_pursuits_lowpass_cutoff_freq
        self._smooth_pursuit_drift_velocity_threshold = smooth_pursuit_drift_velocity_threshold
        self._min_fixation_duration_ms = max(min_fixation_duration, self._min_event_duration)
        self._min_blink_duration_ms = max(min_blink_duration, self._min_event_duration)
        self._max_pso_duration_ms = max(max_pso_duration, self._min_event_duration)
        self._savgol_polyorder = savgol_filter_polyorder
        self._savgol_duration_ms = savgol_filter_duration_ms
        self._median_filter_length = median_filter_duration_ms
        self._max_velocity = max_velocity

    @classmethod
    def get_default_params(cls) -> Dict[str, float]:
        return {
            cls.__MIN_SACCADE_DURATION_STR: cls._DEFAULT_MIN_SACCADE_DURATION_MS,
            cls.__SACCADE_INITIAL_VELOCITY_THRESHOLD_STR: cls._DEFAULT_SACCADE_INITIAL_VELOCITY_THRESHOLD,
            cls.__SACCADE_CONTEXT_WINDOW_DURATION_STR: cls._DEFAULT_SACCADE_CONTEXT_WINDOW_DURATION_MS,
            cls.__SACCADE_INITIAL_MAX_FREQ_STR: cls._DEFAULT_SACCADE_INITIAL_MAX_FREQ,
            cls.__SACCADE_ONSET_THRESHOLD_NOISE_FACTOR_STR: cls._DEFAULT_SACCADE_ONSET_THRESHOLD_NOISE_FACTOR,
            cls.__MIN_SMOOTH_PURSUIT_DURATION_STR: cls._DEFAULT_MIN_SMOOTH_PURSUIT_DURATION_MS,
            cls.__SMOOTH_PURSUIT_LOWPASS_CUTOFF_FREQ_STR: cls._DEFAULT_SMOOTH_PURSUIT_LOWPASS_CUTOFF_FREQ,
            cls.__SMOOTH_PURSUIT_DRIFT_VELOCITY_THRESHOLD_STR: cls._DEFAULT_SMOOTH_PURSUIT_DRIFT_VELOCITY_THRESHOLD,
            cls.__MIN_FIXATION_DURATION_STR: cls._DEFAULT_MIN_FIXATION_DURATION_MS,
            cls.__MIN_BLINK_DURATION_STR: cls._DEFAULT_MIN_BLINK_DURATION_MS,
            cls.__MAX_PSO_DURATION_STR: cls._DEFAULT_MAX_PSO_DURATION_MS,
            cls.__SAVGOL_POLYORDER_STR: cls._DEFAULT_SAVGOL_POLYORDER,
            cls.__SAVGOL_DURATION_MS_STR: cls._DEFAULT_SAVGOL_DURATION_MS,
            cls.__MEDIAN_FILTER_DURATION_MS_STR: cls._DEFAULT_MEDIAN_FILTER_DURATION_MS,
            cls.__MAX_VELOCITY_STR: cls._DEFAULT_MAX_VELOCITY_DEG,
        }

    def _detect_impl(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
            labels: np.ndarray,
            viewer_distance_cm: float,
            pixel_size_cm: float,
            **kwargs,
    ) -> np.ndarray:
        classifier = remodnav.EyegazeClassifier(
            px2deg=pixels_to_visual_angle(1, viewer_distance_cm, pixel_size_cm, use_radians=False),
            sampling_rate=self.sr,
            min_saccade_duration=self.min_fixation_duration_ms / cnst.MILLISECONDS_PER_SECOND,
            min_intersaccade_duration=self.min_inter_saccade_duration_ms / cnst.MILLISECONDS_PER_SECOND,
            saccade_context_window_length=self.saccade_context_window_duration_ms / cnst.MILLISECONDS_PER_SECOND,
            velthresh_startvelocity=self.saccade_initial_velocity_threshold,
            max_initial_saccade_freq=self.saccade_initial_max_freq,
            noise_factor=self.saccade_onset_threshold_noise_factor,
            min_pursuit_duration=self.min_smooth_pursuit_duration_ms / cnst.MILLISECONDS_PER_SECOND,
            pursuit_velthresh=self.smooth_pursuit_drift_velocity_threshold,
            lowpass_cutoff_freq=self.smooth_pursuits_lowpass_cutoff_freq,
            min_fixation_duration=self.min_fixation_duration_ms / cnst.MILLISECONDS_PER_SECOND,
            max_pso_duration=self.max_pso_duration_ms / cnst.MILLISECONDS_PER_SECOND,
        )
        xy = np.rec.fromarrays([x, y], names="{},{}".format(cnst.X, cnst.Y), formats="<f8,<f8")
        pp = classifier.preproc(
            xy,
            max_vel=self.max_velocity,
            savgol_polyord=self.savgol_filter_polyorder,
            savgol_length=self.savgol_filter_duration_ms / cnst.MILLISECONDS_PER_SECOND,
            median_filter_length=self.median_filter_duration_ms / cnst.MILLISECONDS_PER_SECOND,
            min_blink_duration=self.min_blink_duration_ms / cnst.MILLISECONDS_PER_SECOND,
            dilate_nan=self.pad_blinks_ms / cnst.MILLISECONDS_PER_SECOND,
        )
        detected_events = classifier(pp, classify_isp=True, sort_events=True)   # returns a list of dicts, each dict is a single gaze event
        labels = np.asarray(copy.deepcopy(labels), dtype=EventLabelEnum)
        for i, event in enumerate(detected_events):
            start_sample = round(event["start_time"] * self.sr)
            end_sample = round(event["end_time"] * self.sr)
            label = self.__LABEL_MAPPING[event["label"]]
            labels[start_sample:end_sample+1] = label
        return labels

    @property
    def min_saccade_duration_ms(self) -> float:
        return self._min_saccade_duration_ms

    @property
    def min_inter_saccade_duration_ms(self) -> float:
        return min(self._min_blink_duration_ms, self._min_fixation_duration_ms, self._min_smooth_pursuit_duration_ms)

    @property
    def saccade_initial_velocity_threshold(self) -> float:
        return self._saccade_initial_velocity_threshold  # deg/s

    @property
    def saccade_context_window_duration_ms(self) -> float:
        return self._saccade_context_window_duration

    @property
    def saccade_initial_max_freq(self) -> float:
        return self._saccade_initial_max_freq

    @property
    def saccade_onset_threshold_noise_factor(self) -> float:
        return self._saccade_onset_threshold_noise_factor

    @property
    def min_smooth_pursuit_duration_ms(self) -> float:
        return self._min_smooth_pursuit_duration_ms

    @property
    def smooth_pursuits_lowpass_cutoff_freq(self) -> float:
        return self._smooth_pursuit_lowpass_cutoff_freq

    @property
    def smooth_pursuit_drift_velocity_threshold(self) -> float:
        return self._smooth_pursuit_drift_velocity_threshold

    @property
    def min_fixation_duration_ms(self) -> float:
        return self._min_fixation_duration_ms

    @property
    def min_blink_duration_ms(self) -> float:
        return self._min_blink_duration_ms

    @property
    def max_pso_duration_ms(self) -> float:
        return self._max_pso_duration_ms

    @property
    def savgol_filter_polyorder(self) -> int:
        return self._savgol_polyorder

    @property
    def savgol_filter_duration_ms(self) -> float:
        return self._savgol_duration_ms

    @property
    def median_filter_duration_ms(self) -> float:
        return self._median_filter_length

    @property
    def max_velocity(self) -> float:
        return self._max_velocity   # deg/s
