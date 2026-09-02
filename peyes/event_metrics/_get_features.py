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


def get_features(
        events: EventSequenceType, *features: str, verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extracts the specified features from the given events.
    :param events: the events to extract features from
    :param features: the features to extract. Supported features are:
        - 'start_time' or 'onset': the start time of the event (ms)
        - 'end_time' or 'offset': the end time of the event (ms)
        - 'duration': the duration of the event (ms)
        - 'amplitude': the amplitude of the event (deg)
        - 'azimuth': the azimuth of the event (deg)
        - 'center_pixel' or 'center': the center pixel of the event
    :param verbose: if True, display a progress bar while extracting the features
    :return: a dictionary mapping each requested feature name to its numpy array of values
    """
    results: Dict[str, np.ndarray] = {}
    for feat in tqdm(features, desc="Event Features", disable=not verbose):
        results[feat] = _get_features_impl(events, feat)
    return results


def _get_features_impl(events: EventSequenceType, feature: str) -> np.ndarray:
    feature_lower = feature.lower().strip().replace(" ", "_").replace("-", "_")
    _recognized = {"start_time", "onset", "end_time", "offset", "duration", "amplitude", "azimuth", "center_pixel", "center"}
    if feature_lower not in _recognized:
        # only strip a trailing plural "s" as a fallback, so a future feature name that legitimately ends
        # in "s" isn't silently mangled if it's already an exact match (M-15)
        feature_lower = feature_lower.removesuffix('s')
    if feature_lower == "start_time" or feature_lower == "onset":
        return np.array([event.start_time for event in events])
    if feature_lower == "end_time" or feature_lower == "offset":
        return np.array([event.end_time for event in events])
    if feature_lower == "duration":
        return np.array([event.duration for event in events])
    if feature_lower == "amplitude":
        return np.array([event.amplitude for event in events])
    if feature_lower == "azimuth":
        return np.array([event.azimuth for event in events])
    if feature_lower == "center_pixel" or feature_lower == "center":
        return np.array([event.center_pixel for event in events])
    raise ValueError(f"Unknown event feature '{feature}'")
