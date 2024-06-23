from abc import ABC
from typing import Sequence, Dict, Union, Callable

from src.pEYES._DataModels.Event import BaseEvent, EventSequenceType

OneToOneEventMatchesType = Dict[BaseEvent, BaseEvent]
OneToManyEventMatchesType = Dict[BaseEvent, EventSequenceType]
EventMatchesType = Union[OneToOneEventMatchesType, OneToManyEventMatchesType]
EventMatchingFunctionType = Callable[[EventSequenceType, EventSequenceType], EventMatchesType]


class EventMatcher(ABC):
    """
    Implementation of different methods to match two sequences of gaze-events, that may have been detected by different
    human annotators or detection algorithms, as discussed in section "Event Matching Methods" in the article:
        Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art
        Behav Res 55, 1653–1714 (2023). https://doi.org/10.3758/s13428-021-01763-7
    """

    @staticmethod
    def generic_matching(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            allow_cross_matching: bool,
            min_overlap: float = 0,
            min_iou: float = - float("inf"),
            max_l2_timing_offset: float = float("inf"),
            max_onset_difference: float = float("inf"),
            max_offset_difference: float = float("inf"),
            reduction: str = "all"
    ) -> EventMatchesType:
        """
        Match each ground-truth event to a predicted event(s) that satisfies the specified criteria.

        :param ground_truth: sequence of ground-truth events
        :param predictions: sequence of predicted events
        :param allow_cross_matching: if True, a ground-truth event can match a predicted event of a different type
        :param min_overlap: minimum overlap time (normalized to GT duration) to consider a possible match (between 0 and 1)
        :param min_iou: minimum intersection-over-union to consider a possible match
        :param max_l2_timing_offset: maximum L2-timing-offset to consider a possible match
        :param max_onset_difference: maximum absolute difference (in ms) between the start times of the GT and predicted events
        :param max_offset_difference: maximum absolute difference (in ms) between the end times of the GT and predicted events
        :param reduction: name of reduction function used to choose a predicted event(s) from multiple matching ones:
            - 'all': return all matched events
            - 'first': return the first matched event
            - 'last': return the last matched event
            - 'longest': return the longest matched event
            - 'max overlap': return the matched event with maximum overlap with the GT event (normalized by GT duration)
            - 'iou': return the matched event with the maximum intersection-over-union with the GT event
            - 'l2': return the matched event with the minimum L2-timing-offset with the GT event
            - 'onset difference': return the matched event with the least onset-time difference
            - 'offset difference': return the matched event with the least offset-time difference
        :return: dictionary, where keys are ground-truth events and values are their matched predicted event(s)
        :raises NotImplementedError: if the reduction function is not implemented
        """
        reduction = reduction.lower().replace("_", " ").replace("-", " ").strip()
        matches = {}
        matched_predictions = set()
        for gt in ground_truth:
            unmatched_predictions = [p for p in predictions if p not in matched_predictions]
            possible_matches = EventMatcher.__find_matches(
                gt=gt,
                predictions=unmatched_predictions,
                allow_cross_matching=allow_cross_matching,
                min_overlap=min_overlap,
                min_iou=min_iou,
                max_l2_timing_offset=max_l2_timing_offset,
                max_onset_latency=max_onset_difference,
                max_offset_latency=max_offset_difference
            )
            p = EventMatcher.__choose_match(gt, possible_matches, reduction)
            if len(p):
                matches[gt] = p
            if reduction != "all":
                # If reduction is not 'all', cannot allow multiple matches for the same prediction
                matched_predictions.update(p)

        # verify output integrity
        if reduction != "all":
            assert all(len(v) == 1 for v in matches.values()), "Multiple matches for a GT event"
            matches = {k: v[0] for k, v in matches.items()}
            assert len(matches.values()) == len(set(matches.values())), "Matched predictions are not unique"
        return matches

    @staticmethod
    def first_overlap(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            min_overlap: float = 0,
            allow_cross_matching: bool = True
    ) -> OneToOneEventMatchesType:
        """
        Matches the first predicted event that overlaps with each ground-truth event, above a minimal overlap time.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching, min_overlap=min_overlap, reduction="first"
        )

    @staticmethod
    def last_overlap(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            min_overlap: float = 0,
            allow_cross_matching: bool = True
    ) -> Dict[BaseEvent, BaseEvent]:
        """
        Matches the last predicted event that overlaps with each ground-truth event, above a minimal overlap time.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching, min_overlap=min_overlap, reduction="last"
        )

    @staticmethod
    def max_overlap(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            min_overlap: float = 0,
            allow_cross_matching: bool = True
    ) -> OneToOneEventMatchesType:
        """
        Matches the predicted event with maximum overlap with each ground-truth event, above a minimal overlap time.
        Overlap-time is normalized by the duration of the ground-truth event, so values are between 0 and 1.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching, min_overlap=min_overlap, reduction="max overlap"
        )

    @staticmethod
    def longest_overlapping_event(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            min_overlap: float = 0,
            allow_cross_matching: bool = True
    ) -> OneToOneEventMatchesType:
        """
        Matches the longest predicted event that overlaps with each ground-truth event, above a minimal overlap time.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching, min_overlap=min_overlap, reduction="longest"
        )

    @staticmethod
    def iou(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            min_iou: float = 0,
            allow_cross_matching: bool = True
    ) -> OneToOneEventMatchesType:
        """
        Matches the predicted event with maximum intersection-over-union with each ground-truth event, above a minimal value.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching, min_iou=min_iou, reduction="iou"
        )

    @staticmethod
    def l2_timing(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            max_l2: float = 0,
            allow_cross_matching: bool = True
    ) -> OneToOneEventMatchesType:
        """
        Matches the predicted event with minimum L2-timing-offset with each ground-truth event, below a maximum l2 value.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching, max_l2_timing_offset=max_l2, reduction="l2"
        )

    @staticmethod
    def onset_difference(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            max_onset_difference: float = 0,
            allow_cross_matching: bool = True
    ) -> OneToOneEventMatchesType:
        """
        Matches the predicted event with least onset difference with each ground-truth event, below a maximum latency.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching,
            max_onset_difference=max_onset_difference, reduction="onset difference"
        )

    @staticmethod
    def offset_difference(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            max_offset_difference: float = 0,
            allow_cross_matching: bool = True
    ) -> OneToOneEventMatchesType:
        """
        Matches the predicted event with least offset latency with each ground-truth event, below a maximum latency.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching,
            max_offset_difference=max_offset_difference, reduction="offset difference"
        )

    @staticmethod
    def window_based(
            ground_truth: EventSequenceType,
            predictions: EventSequenceType,
            max_onset_difference: float = 0,
            max_offset_difference: float = 0,
            allow_cross_matching: bool = True,
            reduction: str = "onset difference"
    ) -> OneToOneEventMatchesType:
        """
        Finds all predicted events with onset- and offset-latencies within a specified window for each ground-truth event,
        and chooses the best gt-prediction match based on the specified reduction function.
        """
        return EventMatcher.generic_matching(
            ground_truth, predictions, allow_cross_matching,
            max_onset_difference=max_onset_difference, max_offset_difference=max_offset_difference,
            reduction=reduction
        )

    @staticmethod
    def __find_matches(
            gt: BaseEvent,
            predictions: EventSequenceType,
            allow_cross_matching: bool,
            min_overlap: float,
            min_iou: float,
            max_l2_timing_offset: float,
            max_onset_latency: float,
            max_offset_latency: float
    ) -> EventSequenceType:
        """
        Find predicted events that are possible matches for the ground-truth event, based on the specified criteria.

        :param gt: ground-truth event
        :param predictions: sequence of predicted events
        :param allow_cross_matching: if True, a GT event can match a predicted event of a different type
        :param min_overlap: minimum overlap time to consider a possible match
        :param min_iou: minimum intersection-over-union to consider a possible match
        :param max_l2_timing_offset: maximum L2-timing-offset to consider a possible match
        :param max_onset_latency: maximum absolute difference between the start times of the GT and predicted events
        :param max_offset_latency: maximum absolute difference between the end times of the GT and predicted events
        :return: sequence of predicted events that are possible matches for the ground-truth event
        """
        if not allow_cross_matching:
            predictions = [p for p in predictions if p.label == gt.label]
        predictions = [p for p in predictions if
                       gt.time_overlap(p) >= min_overlap and
                       gt.time_iou(p) >= min_iou and
                       gt.time_l2(p) <= max_l2_timing_offset and
                       abs(p.start_time - gt.start_time) <= max_onset_latency and
                       abs(p.end_time - gt.end_time) <= max_offset_latency]
        return predictions

    @staticmethod
    def __choose_match(gt: BaseEvent, matches: EventSequenceType, reduction: str) -> EventSequenceType:
        """
        Choose predicted event(s) matching the ground-truth event, based on the reduction function.
        Possible reduction functions:
            - 'all': return all matched events
            - 'first': return the first matched event
            - 'last': return the last matched event
            - 'longest': return the longest matched event
            - 'max overlap': return the matched event with maximum overlap with the GT event (normalized by GT duration)
            - 'iou': return the matched event with the maximum intersection-over-union with the GT event
            - 'l2': return the matched event with the minimum L2-timing-offset with the GT event
            - 'onset difference': return the matched event with the least onset difference
            - 'offset difference': return the matched event with the least offset difference

        :param gt: ground-truth event
        :param matches: sequence of predicted events matching with the GT event
        :param reduction: reduction function to choose a predicted event from multiple matching ones

        :return: predicted event(s) matching the ground-truth event
        :raises NotImplementedError: if the reduction function is not implemented
        """
        reduction = reduction.lower().replace("_", " ").replace("-", " ").strip()
        if len(matches) == 0:
            return []
        if len(matches) == 1:
            return [matches[0]]
        if reduction == "all":
            return matches
        if reduction == "first":
            return [min(matches, key=lambda e: e.start_time)]
        if reduction == "last":
            return [max(matches, key=lambda e: e.start_time)]
        if reduction == "longest":
            return [max(matches, key=lambda e: e.duration)]
        if reduction == "max overlap":
            return [max(matches, key=lambda e: gt.time_overlap(e))]
        if reduction == "iou":
            return [max(matches, key=lambda e: gt.time_iou(e))]
        if reduction in {"l2", "timing l2", "l2 timing offset"}:
            return [min(matches, key=lambda e: gt.time_l2(e))]
        if reduction == "onset difference":
            return [min(matches, key=lambda e: abs(e.start_time - gt.start_time))]
        if reduction == "offset difference":
            return [min(matches, key=lambda e: abs(e.end_time - gt.end_time))]
        raise NotImplementedError(f"Reduction function '{reduction}' is not implemented")
