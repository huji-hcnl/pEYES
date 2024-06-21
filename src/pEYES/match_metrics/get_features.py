from typing import Dict, Union

import numpy as np
from tqdm import tqdm

from src.pEYES._DataModels.EventMatcher import OneToOneEventMatchesType


def get_features(
        matches: OneToOneEventMatchesType, *features: str, verbose: bool = False
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Calculates the difference in the specified features within each pair of matched events.
    :param matches: a one-to-one mapping from (subset of) ground truth events to (subset_of) predicted events
    :param features: the features to calculate the difference of. Supported features are:
        - 'onset': the difference in start time of the events (ms)
        - 'offset': the difference in end time of the events (ms)
        - 'duration': the difference in duration of the events (ms)
        - 'amplitude': the difference in amplitude of the events (deg)
        - 'azimuth': the difference in azimuth of the events (deg)
        - 'center_pixel_distance': the distance between the center pixels of the events (deg)
        - 'time_overlap': the overlap in time between the events (ms)
        - 'time_iou': the intersection over union in time between the events
        - 'time_l2': the L2 norm of time differences (onset, offset) between the events
    :param verbose: if True, display a progress bar while calculating the features
    :return: the calculated feature differences as a numpy array (for a single feature) or a dictionary of numpy arrays
        (for multiple features).
    """
    results: Dict[str, np.ndarray] = {}
    for feat in tqdm(features, desc="Match Features", disable=not verbose):
        results[feat] = _get_features_impl(matches, feat)
    if len(results) == 1:
        return next(iter(results.values()))
    return results


def _get_features_impl(matches: OneToOneEventMatchesType, feature: str,) -> np.ndarray:
    feature_name = feature.lower().strip().replace(" ", "_").replace("-", "_")
    feature_name = feature_name.removesuffix("s").removesuffix("_difference")   # remove _difference(s)
    if feature_name == "onset":
        return np.array([gt.start_time - pred.start_time for gt, pred in matches.items()])
    if feature_name == "offset":
        return np.array([gt.end_time - pred.end_time for gt, pred in matches.items()])
    if feature_name == "duration":
        return np.array([gt.duration - pred.duration for gt, pred in matches.items()])
    if feature_name == "amplitude":
        return np.array([gt.amplitude - pred.amplitude for gt, pred in matches.items()])
    if feature_name == "azimuth":
        return np.array([gt.azimuth - pred.azimuth for gt, pred in matches.items()])
    if feature_name == "center_pixel_distance":
        return np.array([gt.center_distance(pred) for gt, pred in matches.items()])
    if feature_name == "time_overlap":
        return np.array([gt.time_overlap(pred) for gt, pred in matches.items()])
    if feature_name == "time_iou":
        return np.array([gt.time_iou(pred) for gt, pred in matches.items()])
    if feature_name == "time_l2":
        return np.array([gt.time_l2(pred) for gt, pred in matches.items()])
    raise ValueError(f"Unknown feature: {feature}")
