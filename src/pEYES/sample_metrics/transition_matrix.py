from typing import Sequence

import pandas as pd

from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._utils.metric_utils import transition_matrix as _transition_matrix


def transition_matrix(
        seq: Sequence[EventLabelEnum],
        normalize_rows: bool = False
) -> pd.DataFrame:
    """
    Calculates the transition matrix from a sequence of event-labels.
    If `normalize_rows` is True, the matrix will be normalized by the sum of each row, i.e. contains transition probabilities.
    Returns a DataFrame where rows indicate the origin label and columns indicate the destination label.
    """
    return _transition_matrix(seq, normalize_rows)
