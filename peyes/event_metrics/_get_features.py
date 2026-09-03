from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import peyes._utils.constants as cnst
from peyes._DataModels.Event import BaseEvent, EventSequenceType, EventLabelEnum


def features_by_labels(events: EventSequenceType) -> pd.DataFrame:
    """
    Aggregates the given events into a DataFrame, where each row is an event-label and columns are event features.
    Values of the same event-label are grouped together as lists (e.g. aggregated.loc["SACCADE", "amplitude"] is a list
    of all saccade amplitudes in the given events).
    """
    # Build with the full schema so an empty input still has every feature column: callers compare an
    # inlier frame against an outlier frame, and either group can legitimately be empty.
    summary = pd.DataFrame([e.summary() for e in events], columns=BaseEvent.summary_columns())
    aggregated = summary.groupby(cnst.LABEL_STR).agg(list)      # rows are event labels, columns are features
    for l in EventLabelEnum:
        if l not in aggregated.index:
            aggregated.loc[l] = [[] for _ in range(len(aggregated.columns))]
    # remove undefined events, these shouldn't exist in the first place:
    aggregated.drop(index=EventLabelEnum.UNDEFINED, inplace=True, errors="ignore")
    aggregated[cnst.COUNT_STR] = aggregated[cnst.DURATION_STR].map(len)
    return aggregated.sort_index()


# M-14: aliases for the handful of summary columns whose common/short name differs from BaseEvent.summary_columns()'s
# own name for them. Every other feature name is looked up as-is against summary_columns().
_ALIASES = {"onset": cnst.START_TIME_STR, "offset": cnst.END_TIME_STR, "center": cnst.CENTER_PIXEL_STR}


def get_features(
        events: EventSequenceType, *features: str, verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extracts the specified features from the given events.
    :param events: the events to extract features from
    :param features: the features to extract - any name from `BaseEvent.summary_columns()` (event_type, label,
        start_time, end_time, duration, distance, amplitude, azimuth, peak_velocity, median_velocity,
        min_velocity, cumulative_distance, cumulative_amplitude, start_x, start_y, end_x, end_y, center_pixel,
        pixel_std, dispersion, ellipse_area, is_outlier, outlier_reasons), plus the aliases 'onset' (start_time),
        'offset' (end_time), and 'center' (center_pixel)
    :param verbose: if True, display a progress bar while extracting the features
    :return: a dictionary mapping each requested feature name to its numpy array of values
    :raises ValueError: if a requested feature name isn't recognized
    """
    resolved_names = [_resolve_feature_name(feat) for feat in features]
    # summarize each event once, regardless of how many features are requested, rather than recomputing a
    # full summary per requested feature
    summaries = [event.summary() for event in tqdm(events, desc="Event Features", disable=not verbose)]
    return {
        feat: np.array([s[name] for s in summaries]) for feat, name in zip(features, resolved_names)
    }


def _resolve_feature_name(feature: str) -> str:
    feature_lower = feature.lower().strip().replace(" ", "_").replace("-", "_")
    if feature_lower not in BaseEvent.summary_columns() and feature_lower not in _ALIASES:
        # only strip a trailing plural "s" as a fallback, so a feature name that legitimately ends
        # in "s" isn't silently mangled if it's already an exact match (M-15)
        feature_lower = feature_lower.removesuffix('s')
    resolved = _ALIASES.get(feature_lower, feature_lower)
    if resolved not in BaseEvent.summary_columns():
        raise ValueError(f"Unknown event feature '{feature}'")
    return resolved
