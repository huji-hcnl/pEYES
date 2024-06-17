from typing import Sequence, Set, Dict, Union

import numpy as np
from tqdm import tqdm

from src.pEYES._DataModels.Event import BaseEvent
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._DataModels.EventMatcher import EventMatcher, EventMatchesType, OneToOneEventMatchesType


def match(
        ground_truth: Sequence[BaseEvent],
        prediction: Sequence[BaseEvent],
        match_by: str,
        ignore_events: Set[EventLabelEnum] = None,
        allow_xmatch: bool = False,
        **match_kwargs,
) -> EventMatchesType:
    """
    Match events based on the given matching criteria, ignoring specified event-labels.
    Matches can be one-to-one or one-to-many depending on the matching criteria and the specified parameters.

    :param ground_truth: a sequence of BaseEvent objects representing the ground truth events.
    :param prediction: a sequence of BaseEvent objects representing the predicted events.
    :param match_by: the matching criteria to use:
        - 'first' or 'first overlap': match the first predicted event that overlaps with each ground-truth event
        - 'last' or 'last overlap': match the last predicted event that overlaps with each ground-truth event
        - 'max' or 'max overlap': match the predicted event with maximum overlap with each ground-truth event
        - 'longest overlap': match the longest predicted event that overlaps with each ground-truth event
        - 'iou' or 'intersection over union': match the predicted event with maximum intersection-over-union
        - 'onset' or 'onset difference': match the predicted event with least onset difference
        - 'offset' or 'offset difference': match the predicted event with least offset difference
        - 'window' or 'window based': match the predicted event within a specified onset- and offset-latency window
        - 'l2' or 'l2 timing': match the predicted event with minimum timing l2 norm
    :param ignore_events: a set of event-labels to ignore during the matching process, default is None.
    :param allow_xmatch: if True, allows cross-matching between detectors/raters, default is False.
    :param match_kwargs: additional keyword arguments to pass to the matching function.
    :return: a dictionary matching each ground-truth event to event(s) from the predictions.
    """
    ignore_events = ignore_events or set()
    match_by = match_by.lower().replace("_", " ").replace("-", " ").strip()
    allow_xmatch = allow_xmatch or match_kwargs.pop("allow_xmatch", None) or match_kwargs.pop("allow_cross_match", False)
    ground_truth = [e for e in ground_truth if e.label not in ignore_events]
    prediction = [e for e in prediction if e.label not in ignore_events]
    if match_by == "first" or match_by == "first overlap":
        return EventMatcher.first_overlap(
            ground_truth, prediction, min_overlap=match_kwargs.pop("min_overlap", 0), allow_cross_matching=allow_xmatch
        )
    if match_by == "last" or match_by == "last overlap":
        return EventMatcher.last_overlap(
            ground_truth, prediction, min_overlap=match_kwargs.pop("min_overlap", 0), allow_cross_matching=allow_xmatch
        )
    if match_by == "max" or match_by == "max overlap":
        return EventMatcher.max_overlap(
            ground_truth, prediction, min_overlap=match_kwargs.pop("min_overlap", 0), allow_cross_matching=allow_xmatch
        )
    if "longest" in match_by and "overlap" in match_by:
        return EventMatcher.longest_overlapping_event(
            ground_truth, prediction, min_overlap=match_kwargs.pop("min_overlap", 0), allow_cross_matching=allow_xmatch
        )
    if match_by == "iou" or match_by == "intersection over union":
        return EventMatcher.iou(
            ground_truth, prediction, min_iou=match_kwargs.pop("min_iou", 0), allow_cross_matching=allow_xmatch
        )
    if match_by == "onset" or match_by == "onset difference":
        return EventMatcher.onset_difference(
            ground_truth,
            prediction,
            max_onset_difference=match_kwargs.pop("max_onset_difference", 0),
            allow_cross_matching=allow_xmatch
        )
    if match_by == "offset" or match_by == "offset max_onset_difference":
        return EventMatcher.offset_difference(
            ground_truth,
            prediction,
            max_offset_difference=match_kwargs.pop("max_offset_difference", 0),
            allow_cross_matching=allow_xmatch
        )
    if match_by == "window" or match_by == "window based":
        return EventMatcher.window_based(
            ground_truth,
            prediction,
            max_onset_difference=match_kwargs.pop("max_onset_difference", 0),
            max_offset_difference=match_kwargs.pop("max_offset_difference", 0),
            allow_cross_matching=allow_xmatch
        )
    if "l2" in match_by:
        return EventMatcher.l2_timing(
            ground_truth, prediction, max_l2=match_kwargs.pop("max_l2", 0), allow_cross_matching=allow_xmatch
        )
    return EventMatcher.generic_matching(
        ground_truth, prediction, allow_cross_matching=allow_xmatch, **match_kwargs
    )


def match_multiple(
        ground_truth: Sequence[BaseEvent],
        predictions: Dict[str, Sequence[BaseEvent]],
        match_by: str,
        ignore_events: Set[EventLabelEnum] = None,
        allow_xmatch: bool = False,
        verbose: bool = False,
        **match_kwargs,
) -> Dict[str, EventMatchesType]:
    """
    Matched between each of the predicted event sequences and the ground-truth event sequence, using the specified
    matching criteria and ignoring specified event-labels. Matching can be one-to-one or one-to-many depending on the
    matching criteria and the specified parameters. If `verbose` is True, a progress bar tracks the matching process for
    predicted sequences.
    Returns a dictionary mapping prediction names to their respective matching results.
    See `match_events` function for more details on the matching criteria and parameters.
    """
    matches = {}
    for name, pred in tqdm(predictions.items(), desc="Matching", disable=not verbose):
        matches[name] = match(ground_truth, pred, match_by, ignore_events=ignore_events, allow_xmatch=allow_xmatch,
                              **match_kwargs)
    return matches


def calculate_metrics(
        matches: OneToOneEventMatchesType, metrics: Union[str, Sequence[str]], verbose: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    if isinstance(metrics, str):
        return _calculate_metric_impl(matches, metrics)
    results = {}
    for metric in tqdm(metrics, desc="Calculating metrics", disable=not verbose):
        results[metric] = _calculate_metric_impl(matches, metric)
    return results


def _calculate_metric_impl(matches: OneToOneEventMatchesType, metric: str,) -> np.ndarray:
    metric_name = metric.lower().strip().replace(" ", "_").replace("-", "_")
    metric_name = metric_name.removesuffix("_difference")
    if metric_name == "onset":
        return np.array([gt.start_time - pred.start_time for gt, pred in matches.items()])
    if metric_name == "offset":
        return np.array([gt.end_time - pred.end_time for gt, pred in matches.items()])
    if metric_name == "duration":
        return np.array([gt.duration - pred.duration for gt, pred in matches.items()])
    if metric_name == "amplitude":
        return np.array([gt.amplitude - pred.amplitude for gt, pred in matches.items()])
    if metric_name == "azimuth":
        return np.array([gt.azimuth - pred.azimuth for gt, pred in matches.items()])
    if metric_name == "center_pixel_distance":
        return np.array([gt.center_distance(pred) for gt, pred in matches.items()])
    if metric_name == "time_overlap":
        return np.array([gt.time_overlap(pred) for gt, pred in matches.items()])
    if metric_name == "time_iou":
        return np.array([gt.time_iou(pred) for gt, pred in matches.items()])
    if metric_name == "time_l2":
        return np.array([gt.time_l2(pred) for gt, pred in matches.items()])
    raise ValueError(f"Unknown metric: {metric}")
