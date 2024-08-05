from typing import Set, Dict

from tqdm import tqdm

from peyes._DataModels.Event import EventSequenceType
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._DataModels.EventMatcher import EventMatcher, EventMatchesType


def match(
        ground_truth: EventSequenceType,
        prediction: EventSequenceType,
        match_by: str,
        ignore_events: Set[EventLabelEnum] = None,
        allow_xmatch: bool = False,
        **kwargs,
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

    :keyword min_overlap: minimum overlap required for 'first', 'last', 'max', and 'longest overlap' matching.
    :keyword min_iou: minimum intersection-over-union required for 'iou' matching.
    :keyword max_onset_difference: maximum onset difference allowed for 'onset' or 'window' matching.
    :keyword max_offset_difference: maximum offset difference allowed for 'offset' or 'window' matching.
    :keyword max_l2: maximum l2 norm allowed for 'l2' matching.

    :return: a dictionary matching each ground-truth event to event(s) from the predictions.
    """
    ignore_events = ignore_events or set()
    match_by = match_by.lower().replace("_", " ").replace("-", " ").strip()
    allow_xmatch = allow_xmatch or kwargs.pop("allow_xmatch", None) or kwargs.pop("allow_cross_match", False)
    ground_truth = [e for e in ground_truth if e.label not in ignore_events]
    prediction = [e for e in prediction if e.label not in ignore_events]
    if match_by == "first" or match_by == "first overlap":
        return EventMatcher.first_overlap(
            ground_truth, prediction, min_overlap=kwargs.pop("min_overlap", 0), allow_cross_matching=allow_xmatch
        )
    if match_by == "last" or match_by == "last overlap":
        return EventMatcher.last_overlap(
            ground_truth, prediction, min_overlap=kwargs.pop("min_overlap", 0), allow_cross_matching=allow_xmatch
        )
    if match_by == "max" or match_by == "max overlap" or match_by == "max_overlap":
        return EventMatcher.max_overlap(
            ground_truth, prediction, min_overlap=kwargs.pop("min_overlap", 0), allow_cross_matching=allow_xmatch
        )
    if "longest" in match_by and "overlap" in match_by:
        return EventMatcher.longest_overlapping_event(
            ground_truth, prediction, min_overlap=kwargs.pop("min_overlap", 0), allow_cross_matching=allow_xmatch
        )
    if match_by == "iou" or match_by == "intersection over union":
        return EventMatcher.iou(
            ground_truth, prediction, min_iou=kwargs.pop("min_iou", 0), allow_cross_matching=allow_xmatch
        )
    if match_by == "onset" or match_by == "onset difference":
        return EventMatcher.onset_difference(
            ground_truth,
            prediction,
            max_onset_difference=kwargs.pop("max_onset_difference", 0),
            allow_cross_matching=allow_xmatch
        )
    if match_by == "offset" or match_by == "offset max_onset_difference":
        return EventMatcher.offset_difference(
            ground_truth,
            prediction,
            max_offset_difference=kwargs.pop("max_offset_difference", 0),
            allow_cross_matching=allow_xmatch
        )
    if match_by == "window" or match_by == "window based":
        return EventMatcher.window_based(
            ground_truth,
            prediction,
            max_onset_difference=kwargs.pop("max_onset_difference", 0),
            max_offset_difference=kwargs.pop("max_offset_difference", 0),
            allow_cross_matching=allow_xmatch
        )
    if "l2" in match_by:
        return EventMatcher.l2_timing(
            ground_truth, prediction, max_l2=kwargs.pop("max_l2", 0), allow_cross_matching=allow_xmatch
        )
    return EventMatcher.generic_matching(ground_truth, prediction, allow_cross_matching=allow_xmatch, **kwargs)


def match_multiple(
        ground_truth: EventSequenceType,
        predictions: Dict[str, EventSequenceType],
        match_by: str,
        ignore_events: Set[EventLabelEnum] = None,
        allow_xmatch: bool = False,
        verbose: bool = False,
        **kwargs,
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
        matches[name] = match(
            ground_truth, pred, match_by, ignore_events=ignore_events, allow_xmatch=allow_xmatch, **kwargs
        )
    return matches

