from typing import Optional

import pandas as pd
import sklearn.metrics as met

from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelSequenceType

from peyes._utils.event_utils import parse_label as _parse_label
from peyes._utils.metric_utils import transition_matrix as _transition_matrix

_GROUND_TRUTH_STR = "Ground Truth"
_PREDICTION_STR = "Prediction"


def label_counts(
        seq: UnparsedEventLabelSequenceType
) -> pd.Series:
    """
    Counts the occurrences of each event-label in the given sequence.
    Returns a pandas Series mapping each event-label to its count.
    """
    if seq is None:
        return pd.Series({l: 0 for l in EventLabelEnum})
    seq = [_parse_label(l) for l in seq]
    counts = pd.Series(seq).value_counts()
    return counts.reindex(EventLabelEnum, fill_value=0)


def transition_matrix(
        seq: UnparsedEventLabelSequenceType,
        normalize_rows: bool = False
) -> pd.DataFrame:
    """
    Calculates the transition matrix from a sequence of event-labels and returns a DataFrame where rows indicate the
    origin label and columns indicate the destination label. If `normalize_rows` is True, the matrix will be normalized
    by the sum of each row, i.e. contains transition **probabilities** and not **counts**.

    Returns a DataFrame where rows indicate the origin label and columns indicate the destination label.
    """
    seq = [_parse_label(l) for l in seq]
    return _transition_matrix(seq, normalize_rows)


def confusion_matrix(
        ground_truth: UnparsedEventLabelSequenceType,
        prediction: UnparsedEventLabelSequenceType,
        labels: Optional[UnparsedEventLabelSequenceType] = None,
) -> pd.DataFrame:
    """
    Calculates the confusion matrix from the given ground-truth and predicted event-labels.
    If `labels` is not None, the matrix will be restricted to the given labels; otherwise it will contain all unique labels.
    """
    ground_truth = [_parse_label(l) for l in ground_truth]
    prediction = [_parse_label(l) for l in prediction]
    labels = list(EventLabelEnum) if labels is None else list(set(_parse_label(l) for l in labels))
    conf = met.confusion_matrix(ground_truth, prediction, labels=labels)
    df = pd.DataFrame(conf, index=labels, columns=labels)
    df.index.name = _GROUND_TRUTH_STR
    df.columns.name = _PREDICTION_STR
    return df
