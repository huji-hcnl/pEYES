import traceback
from abc import ABC, abstractmethod
from typing import final

import numpy as np
import pandas as pd

import src.pEYES.constants as cnst
import src.pEYES.config as cnfg
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._utils.vector_utils import is_one_dimensional
from src.pEYES._utils.event_utils import calculate_sampling_rate


class BaseDetector(ABC):
    """
    Base class for gaze event detectors, objects that indicate the type of gaze event (fixation, saccade, blink) at each
    sample in the gaze data. All inherited classes must implement the `_detect_impl` method, which is the core of the
    gaze event detection process.

    Detection process:
    1. Detect blinks, including padding them by the amount in `dilate_nans_by`
    2. Set x and y to nan where blinks are detected
    3. Detect gaze events (using the class-specific logic, implemented in `_detect_impl` method)
    4. Ignore chunks of gaze-events that are shorter than `minimum_event_duration`
    5. Merge chunks of the same type that are separated by less than `minimum_event_duration`

    :param missing_value: the value that indicates missing data in the gaze data. Default is np.nan
    :param viewer_distance: the distance from the viewer to the screen, in centimeters. Default is 60 cm
    :param pixel_size: the size of a single pixel on the screen, in centimeters. Default is the pixel size of the
        screen monitor
    :param minimum_event_duration: the minimum duration of a gaze event, in milliseconds. Default is 5 ms
    :param dilate_nans_by: the amount of time to pad nans by, in milliseconds. Default is 0 ms (no padding)
    """

    __DEFAULT_MISSING_VALUE = np.nan
    __DEFAULT_VIEWER_DISTANCE = 60      # cm
    __DEFAULT_PIXEL_SIZE = 0.027843     # cm  (pixel size in the Tobii screen monitor)
    __DEFAULT_NAN_PADDING = 0           # ms
    __DEFAULT_MIN_EVENT_DURATION = cnfg.MIN_EVENT_DURATION

    def __init__(
            self,
            missing_value=__DEFAULT_MISSING_VALUE,
            viewer_distance=__DEFAULT_VIEWER_DISTANCE,
            pixel_size=__DEFAULT_PIXEL_SIZE,
            dilate_nans_by=__DEFAULT_PIXEL_SIZE,
            min_event_duration=__DEFAULT_MIN_EVENT_DURATION,
            **kwargs  # additional parameters for inherited classes
    ):
        self._missing_value = missing_value
        self._viewer_distance = viewer_distance  # cm
        if self._viewer_distance <= 0:
            raise ValueError("viewer_distance must be positive")
        self._pixel_size = pixel_size  # cm
        if self._pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        self._dilate_nans_by = dilate_nans_by  # ms
        if self._dilate_nans_by < 0:
            raise ValueError("dilate_nans_by must be non-negative")
        if min_event_duration < 0:
            raise ValueError("min_event_duration must be non-negative")
        self._min_event_duration = min_event_duration   # ms
        self._sr = np.nan  # sampling rate
        self._candidates = None  # event candidates
        self.data: dict = {}  # gaze data

    @abstractmethod
    def _detect_impl(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @final
    def detect(
            self,
            t: np.ndarray,
            x: np.ndarray,
            y: np.ndarray,
    ):
        t, x, y = self._verify_inputs(t, x, y)
        sr = calculate_sampling_rate(t)
        candidates = np.full_like(t, EventLabelEnum.UNDEFINED)
        is_blink = self._detect_blinks(x, y)
        candidates[is_blink] = EventLabelEnum.BLINK
        return None

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @final
    def _detect_blinks(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        is_missing_x = np.isnan(x) if np.isnan(self._missing_value) else x == self._missing_value
        is_missing_y = np.isnan(y) if np.isnan(self._missing_value) else y == self._missing_value
        is_blink[is_missing_x | is_missing_y] = True
        is_blink = self._merge_close_events(is_blink)  # ignore short blinks and merge consecutive blinks  # TODO

        # pad blinks by the given amount
        if self._dilate_nans_by == 0:
            return is_blink
        pad_samples = self._calc_num_samples(self._dilate_nans_by)
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

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
