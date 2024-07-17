from typing import Optional

import pandas as pd
import sklearn.metrics as met

from pEYES._DataModels.EventLabelEnum import EventLabelSequenceType, EventLabelEnum
from pEYES._utils.metric_utils import transition_matrix as _transition_matrix

_GROUND_TRUTH_STR = "Ground Truth"
_PREDICTION_STR = "Prediction"


def label_counts(
        seq: EventLabelSequenceType
) -> pd.Series:
    """
    Counts the occurrences of each event-label in the given sequence.
    Returns a pandas Series mapping each event-label to its count.
    """
    if seq is None:
        return pd.Series({l: 0 for l in EventLabelEnum})
    counts = pd.Series(seq).value_counts()
    return counts.reindex(EventLabelEnum, fill_value=0)


def transition_matrix(
        seq: EventLabelSequenceType,
        normalize_rows: bool = False
) -> pd.DataFrame:
    """
    Calculates the transition matrix from a sequence of event-labels.
    If `normalize_rows` is True, the matrix will be normalized by the sum of each row, i.e. contains transition probabilities.
    Returns a DataFrame where rows indicate the origin label and columns indicate the destination label.
    """
    return _transition_matrix(seq, normalize_rows)


def confusion_matrix(
        ground_truth: EventLabelSequenceType,
        prediction: EventLabelSequenceType,
        labels: Optional[EventLabelSequenceType] = None,
) -> pd.DataFrame:
    """
    Calculates the confusion matrix from the given ground-truth and predicted event-labels.
    If `labels` is not None, the matrix will be restricted to the given labels; otherwise it will contain all unique labels.
    """
    labels = list(set(EventLabelEnum)) if labels is None else list(set(labels))
    conf = met.confusion_matrix(ground_truth, prediction, labels=labels)
    df = pd.DataFrame(conf, index=labels, columns=labels)
    df.index.name = _GROUND_TRUTH_STR
    df.columns.name = _PREDICTION_STR
    return df
