from typing import Sequence

import pandas as pd

from src.pEYES._DataModels.Event import BaseEvent
from src.pEYES._utils.event_utils import count_labels as _count


def counts(events: Sequence[BaseEvent]) -> pd.Series:
    """
    Count the number of occurrences of each event-label within the given events.
    Returns a pandas Series mapping each event-label to its count.
    """
    return _count(events)
