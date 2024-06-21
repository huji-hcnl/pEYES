from typing import Sequence, Dict, Union

import numpy as np
from tqdm import tqdm

from src.pEYES._DataModels.Event import BaseEvent


def get_features(
        events: Sequence[BaseEvent], *features: str, verbose: bool = False
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
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
    :return: the extracted features as a numpy array (for a single feature) or a dictionary of numpy arrays (for
        multiple features).
    """
    results: Dict[str, np.ndarray] = {}
    for feat in tqdm(features, desc="Event Features", disable=not verbose):
        results[feat] = _get_features_impl(events, feat)
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def _get_features_impl(events: Sequence[BaseEvent], feature: str) -> np.ndarray:
    feature_lower = feature.lower().strip().replace(" ", "_").replace("-", "_")
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
