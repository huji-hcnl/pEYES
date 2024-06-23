from typing import Sequence

import pandas as pd

from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES._utils.metric_utils import transition_matrix as _transition_matrix


def transition_matrix(
        seq: EventSequenceType,
        normalize_rows: bool = False
) -> pd.DataFrame:
    """
    Calculates the transition matrix from a sequence of event.
    If `normalize_rows` is True, the matrix will be normalized by the sum of each row, i.e. contains transition probabilities.
    Returns a DataFrame where rows indicate the origin event-label and columns indicate the destination event-label.
    """
    return _transition_matrix([e.label for e in seq], normalize_rows)
