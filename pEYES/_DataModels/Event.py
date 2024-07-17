import warnings
from abc import ABC
from typing import final, List, Optional, Sequence

import pandas as pd

import pEYES._DataModels.config as cnfg
from pEYES._utils.pixel_utils import *
from pEYES._utils.vector_utils import get_chunk_indices
from pEYES._DataModels.EventLabelEnum import EventLabelEnum


class BaseEvent(ABC):

    _LABEL: EventLabelEnum
    _OUTLIER_REASONS_STR = "outlier_reasons"

    def __init__(
            self,
            t: np.ndarray,
            x: np.ndarray = None,
            y: np.ndarray = None,
            pupil: np.ndarray = None,
            viewer_distance: float = cnfg.VIEWER_DISTANCE,
            pixel_size: float = cnfg.SCREEN_MONITOR[cnst.PIXEL_SIZE_STR],
    ):
        _x = x if x is not None else np.full_like(t, np.nan, dtype=float)
        _y = y if y is not None else np.full_like(t, np.nan, dtype=float)
        _pupil = pupil if pupil is not None else np.full_like(t, np.nan, dtype=float)
        assert len(t) == len(_x) == len(_y) == len(_pupil), "t, x, y, and pupil must have the same length"
        assert np.isnan(viewer_distance) or viewer_distance > 0, "viewer_distance must be a positive number"
        assert np.isnan(pixel_size) or pixel_size > 0, "pixel_size must be a positive number"
        samples = np.vstack([t, _x, _y, _pupil]).T
        samples = samples[samples[:, 0].argsort()]  # sort by time
        self._t = samples[:, 0]
        self._x = samples[:, 1]
        self._y = samples[:, 2]
        self._pupil = samples[:, 3]
        self._viewer_distance = viewer_distance
        self._pixel_size = pixel_size

    @staticmethod
    @final
    def make(
            label: EventLabelEnum,
            t: np.ndarray,
            x: np.ndarray = None,
            y: np.ndarray = None,
            pupil: np.ndarray = None,
            viewer_distance: float = cnfg.VIEWER_DISTANCE,
            pixel_size: float = cnfg.SCREEN_MONITOR[cnst.PIXEL_SIZE_STR],
    ) -> Optional["BaseEvent"]:
        """  Creates a single Event from the given data.  """
        if label == EventLabelEnum.UNDEFINED:
            return None
        if label == EventLabelEnum.FIXATION:
            return FixationEvent(t, x, y, pupil, viewer_distance, pixel_size)
        if label == EventLabelEnum.SACCADE:
            return SaccadeEvent(t, x, y, pupil, viewer_distance, pixel_size)
        if label == EventLabelEnum.PSO:
            return PSOEvent(t, x, y, pupil, viewer_distance, pixel_size)
        if label == EventLabelEnum.SMOOTH_PURSUIT:
            return SmoothPursuitEvent(t, x, y, pupil, viewer_distance, pixel_size)
        if label == EventLabelEnum.BLINK:
            return BlinkEvent(t, x, y, pupil, viewer_distance, pixel_size)
        raise ValueError(f"Invalid event type: {label}")

    @staticmethod
    @final
    def make_multiple(
            labels: np.ndarray,
            t: np.ndarray,
            x: np.ndarray = None,
            y: np.ndarray = None,
            pupil: np.ndarray = None,
            viewer_distance: float = cnfg.VIEWER_DISTANCE,
            pixel_size: float = cnfg.SCREEN_MONITOR[cnst.PIXEL_SIZE_STR],
    ) -> list["BaseEvent"]:
        if len(labels) != len(t):
            raise ValueError("Length of `labels` and `t` must be the same")
        same_label_chunk_idxs = get_chunk_indices(labels)
        events = []
        for idx_group in same_label_chunk_idxs:
            event_label = EventLabelEnum(labels[idx_group[0]])
            if event_label == EventLabelEnum.UNDEFINED:
                # ignore undefined samples
                continue
            new_event = BaseEvent.make(
                label=event_label,
                t=t[idx_group],
                x=x[idx_group] if x is not None else None,
                y=y[idx_group] if y is not None else None,
                pupil=pupil[idx_group] if pupil is not None else None,
                viewer_distance=viewer_distance,
                pixel_size=pixel_size
            )
            if new_event is not None:
                events.append(new_event)
        return events

    @final
    def velocities(self, unit: str = 'px') -> np.ndarray:
        """
        Calculates the velocity (distance per second) of each sample within the event.
        :param unit: determines the nominator units
        :return: np.ndarray of velocities (units are one of px/s, deg/s, rad/s)
        """
        px_velocities = calculate_velocities(self._x, self._y, self._t)
        unit = unit.lower()
        if unit in {"px", "pixel", "pixels", "px/sec"}:
            return px_velocities
        px_to_visual_angle_vec = np.vectorize(pixels_to_visual_angle)
        if unit in {"deg", "degree", "degrees", "deg/sec"}:
            return px_to_visual_angle_vec(px_velocities, self._viewer_distance, self._pixel_size)
        if unit in {"rad", "radian", "radians", "rad/sec"}:
            return px_to_visual_angle_vec(px_velocities, self._viewer_distance, self._pixel_size, use_radians=True)
        raise ValueError(f"unit must be one of `px`, `deg`, or `rad`, not `{unit}`")

    def get_outlier_reasons(self) -> List[str]:
        reasons = []
        if self.duration < self.get_min_duration():
            reasons.append(cnst.MIN_DURATION_STR)
        if self.duration > self.get_max_duration():
            reasons.append(cnst.MAX_DURATION_STR)
        if np.any(self._x < 0) or np.any(self._y < 0):
            reasons.append("negative_pixel")
        if (
                np.any(self._x > cnfg.SCREEN_MONITOR[cnst.RESOLUTION_STR][0]) or
                np.any(self._y > cnfg.SCREEN_MONITOR[cnst.RESOLUTION_STR][1])
        ):
            reasons.append("pixel_outside_screen")
        # TODO: check min, max velocity, acceleration, dispersion, etc.
        return reasons

    def summary(self) -> pd.Series:
        d = {
            cnst.LABEL_STR: self.label,
            cnst.START_TIME_STR: self.start_time, cnst.END_TIME_STR: self.end_time, cnst.DURATION_STR: self.duration,
            cnst.DISTANCE_STR: self.distance, cnst.AMPLITUDE_STR: self.amplitude, cnst.AZIMUTH_STR: self.azimuth,
            cnst.PEAK_VELOCITY_STR: self.peak_velocity, cnst.MEDIAN_VELOCITY_STR: self.median_velocity,
            cnst.CUMULATIVE_DISTANCE_STR: self.cumulative_distance, cnst.CUMULATIVE_AMPLITUDE_STR: self.cumulative_amplitude,
            cnst.CENTER_PIXEL_STR: self.center_pixel, cnst.PIXEL_STD_STR: self.pixel_std, cnst.ELLIPSE_AREA_STR: self.ellipse_area,
            cnst.IS_OUTLIER_STR: self.is_outlier, self._OUTLIER_REASONS_STR: self.get_outlier_reasons()
        }
        return pd.Series(d)

    @final
    def time_overlap(self, other: "BaseEvent", normalize: bool = True) -> float:
        """  Calculates the time overlap between this event and another event, in milliseconds.  """
        if self.duration == 0 or other.duration == 0:
            return 0
        start_time = max(self.start_time, other.start_time)
        end_time = min(self.end_time, other.end_time)
        total_overlap = max([0, end_time - start_time])
        if normalize:
            return total_overlap / self.duration
        return total_overlap

    @final
    def time_iou(self, other: "BaseEvent") -> float:
        """
        Calculates the intersection-over-union (IoU) between times of this event and another event (unitless).
        See Startsev & Zemblys (2023) for more information.
        """
        total_overlap = self.time_overlap(other, normalize=False)
        total_union = self.duration + other.duration - total_overlap
        return total_overlap / total_union

    @final
    def time_l2(self, other: "BaseEvent") -> float:
        """
        Calculates the l2-norm between the onset- and offset-differences of this event and another event.
        See Kothari et al. (2020) for more details.
        """
        return np.linalg.norm([self.start_time - other.start_time, self.end_time - other.end_time])

    @final
    def center_distance(self, other: "BaseEvent") -> float:
        """  Calculates the Euclidean distance between the center pixel of this event and another event.  """
        x1, y1 = self.center_pixel
        x2, y2 = other.center_pixel
        return np.linalg.norm([x1 - x2, y1 - y2])

    @classmethod
    @final
    def get_min_duration(cls) -> float:
        return cnfg.EVENT_MAPPING[cls._LABEL][cnst.MIN_DURATION_STR]

    @classmethod
    @final
    def set_min_duration(cls, min_duration: float):
        event_type = cls._LABEL.name.capitalize()
        if min_duration < 0:
            raise ValueError(f"min_duration for {event_type} must be a positive number")
        max_duration = cnfg.EVENT_MAPPING[cls._LABEL][cnst.MAX_DURATION_STR]
        if min_duration > max_duration:
            raise ValueError(f"min_duration for {event_type} must be less than or equal to max_duration")
        cnfg.EVENT_MAPPING[cls._LABEL][cnst.MIN_DURATION_STR] = min_duration

    @classmethod
    @final
    def get_max_duration(cls) -> float:
        return cnfg.EVENT_MAPPING[cls._LABEL][cnst.MAX_DURATION_STR]

    @classmethod
    @final
    def set_max_duration(cls, max_duration: float):
        event_type = cls._LABEL.name.capitalize()
        if max_duration < 0:
            raise ValueError(f"max_duration for {event_type} must be a positive number")
        min_duration = cnfg.EVENT_MAPPING[cls._LABEL][cnst.MIN_DURATION_STR]
        if max_duration < min_duration:
            raise ValueError(f"max_duration for {event_type} must be greater than or equal to min_duration")
        cnfg.EVENT_MAPPING[cls._LABEL][cnst.MAX_DURATION_STR] = max_duration

    @final
    @property
    def label(self) -> EventLabelEnum:
        return self.__class__._LABEL

    @final
    @property
    def num_samples(self) -> int:
        return len(self._t)

    @final
    @property
    def is_outlier(self) -> bool:
        return len(self.get_outlier_reasons()) > 0

    @final
    @property
    def start_time(self) -> float:
        return float(self._t[0])

    @final
    @property
    def start_pixel(self) -> Tuple[float, float]:
        return float(self._x[0]), float(self._y[0])

    @final
    @property
    def end_time(self) -> float:
        return float(self._t[-1])

    @final
    @property
    def end_pixel(self) -> Tuple[float, float]:
        return float(self._x[-1]), float(self._y[-1])

    @final
    @property
    def duration(self) -> float:
        return float(self.end_time - self.start_time)

    @final
    @property
    def distance(self) -> float:
        """  Euclidean distance between the start and end pixels of the event (pixel units)  """
        start_x, start_y = self.start_pixel
        end_x, end_y = self.end_pixel
        return np.linalg.norm([end_x - start_x, end_y - start_y])

    @final
    @property
    def amplitude(self) -> float:
        """  Euclidean distance between the start and end pixels of the event (visual degree units)  """
        return pixels_to_visual_angle(self.distance, self._viewer_distance, self._pixel_size)

    @final
    @property
    def azimuth(self) -> float:
        """  Angle between the start and end pixels of the event (degrees)"""
        return calculate_azimuth(self.start_pixel, self.end_pixel, use_radians=False)

    @final
    @property
    def cumulative_distance(self) -> float:
        """  Cumulative distance traveled during the event (pixel units)  """
        return np.sum(np.linalg.norm(np.diff(np.column_stack((self._x, self._y)), axis=0), axis=1))

    @final
    @property
    def cumulative_amplitude(self) -> float:
        """  Cumulative distance traveled during the event (visual degree units)  """
        return pixels_to_visual_angle(self.cumulative_distance, self._viewer_distance, self._pixel_size)

    @final
    @property
    def center_pixel(self) -> Tuple[float, float]:
        """  Returns the mean coordinates of the event on the X,Y axes  """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_mean = float(np.nanmean(self._x))
            y_mean = float(np.nanmean(self._y))
            return x_mean, y_mean

    @final
    @property
    def pixel_std(self) -> Tuple[float, float]:
        """  Returns the standard deviation of the event (in pixel units)  """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            x_std = float(np.nanstd(self._x))
            y_std = float(np.nanstd(self._y))
            return x_std, y_std

    @final
    @property
    def top_pixel(self) -> Tuple[float, float]:
        """  Returns the top pixel of the event (assuming the screen's top-left corner is (0,0))  """
        min_y_idx = np.argmin(self._y)
        return float(self._x[min_y_idx]), float(self._y[min_y_idx])

    @final
    @property
    def bottom_pixel(self) -> Tuple[float, float]:
        """  Returns the bottom pixel of the event (assuming the screen's top-left corner is (0,0))  """
        max_y_idx = np.argmax(self._y)
        return float(self._x[max_y_idx]), float(self._y[max_y_idx])

    @final
    @property
    def left_pixel(self) -> Tuple[float, float]:
        """  Returns the leftmost pixel of the event (assuming the screen's top-left corner is (0,0))  """
        min_x_idx = np.argmin(self._x)
        return float(self._x[min_x_idx]), float(self._y[min_x_idx])

    @final
    @property
    def right_pixel(self) -> Tuple[float, float]:
        """  Returns the rightmost pixel of the event (assuming the screen's top-left corner is (0,0))  """
        max_x_idx = np.argmax(self._x)
        return float(self._x[max_x_idx]), float(self._y[max_x_idx])

    @final
    @property
    def x_dispersion(self) -> float:
        """  Returns the horizontal dispersion of the event (visual degree units)  """
        return pixels_to_visual_angle(np.nanmax(self._x) - np.nanmin(self._x), self._viewer_distance, self._pixel_size)

    @final
    @property
    def y_dispersion(self) -> float:
        """  Returns the vertical dispersion of the event (visual degree units)  """
        return pixels_to_visual_angle(np.nanmax(self._y) - np.nanmin(self._y), self._viewer_distance, self._pixel_size)

    @final
    @property
    def ellipse_area(self) -> float:
        """  Returns the area of the ellipse that encloses the event (visual degree units)  """
        x_radius = self.x_dispersion / 2
        y_radius = self.y_dispersion / 2
        return np.pi * x_radius * y_radius

    @final
    @property
    def peak_velocity(self) -> float:
        """  Returns the maximum velocity during the event (visual degree / second)  """
        return np.nanmax(self.velocities(unit='deg'))

    @final
    @property
    def median_velocity(self) -> float:
        """  Returns the median velocity during the event (visual degree / second)  """
        return np.nanmedian(self.velocities(unit='deg'))

    def __hash__(self):
        return hash((
            self.label,
            self._t.tobytes(),
            self._x.tobytes(),
            self._y.tobytes(),
            self._pupil.tobytes(),
            self._viewer_distance,
            self._pixel_size
        ))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if not self.label == other.label:
            return False
        if not np.array_equal(self._t, other._t, equal_nan=True):
            return False
        if not np.array_equal(self._x, other._x, equal_nan=True):
            return False
        if not np.array_equal(self._y, other._y, equal_nan=True):
            return False
        if not np.array_equal(self._pupil, other._pupil, equal_nan=True):
            return False
        if not np.array_equal(self._viewer_distance, other._viewer_distance, equal_nan=True):
            return False
        if not np.array_equal(self._pixel_size, other._pixel_size, equal_nan=True):
            return False
        return True

    def __str__(self) -> str:
        return f"{self.label.name}({self.duration}ms)"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return self.num_samples


class FixationEvent(BaseEvent):
    _LABEL = EventLabelEnum.FIXATION


class SaccadeEvent(BaseEvent):
    _LABEL = EventLabelEnum.SACCADE


class PSOEvent(BaseEvent):
    _LABEL = EventLabelEnum.PSO


class SmoothPursuitEvent(BaseEvent):
    _LABEL = EventLabelEnum.SMOOTH_PURSUIT


class BlinkEvent(BaseEvent):
    _LABEL = EventLabelEnum.BLINK


EventSequenceType = Sequence[BaseEvent]
