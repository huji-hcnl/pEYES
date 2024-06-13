from typing import Union, Sequence

import pandas as pd

from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._DataModels.Event import BaseEvent


def parse_label(
        val: Union[EventLabelEnum, BaseEvent, int, str, float],
        safe: bool = True
) -> EventLabelEnum:
    """
    Parses a gaze label from the original dataset's type to an EventLabelEnum
    :param val: the value to parse
    :param safe: if True, returns EventLabelEnum.UNDEFINED when the parsing fails; otherwise, raises an exception
    :return: the parsed gaze label
    """
    try:
        if isinstance(val, EventLabelEnum):
            return val
        if isinstance(val, BaseEvent):
            return val.label
        if isinstance(val, int):
            return EventLabelEnum(val)
        if isinstance(val, str):
            return EventLabelEnum[val.upper()]
        if isinstance(val, float):
            if not val.is_integer():
                raise ValueError(f"Invalid value: {val}")
            return EventLabelEnum(int(val))
        raise TypeError(f"Incompatible type: {type(val)}")
    except Exception as err:
        if safe and (isinstance(err, ValueError) or isinstance(err, TypeError)):
            return EventLabelEnum.UNDEFINED
        raise err


def count_labels(data: Sequence[Union[EventLabelEnum, BaseEvent]]) -> pd.Series:
    """
    Counts the number of same-type labels/events in the given data, and fills in missing labels with 0 counts.
    Returns a Series with the counts of each GazEventTypeEnum label.
    """
    if data is None:
        return pd.Series({l: 0 for l in EventLabelEnum})
    labels = pd.Series([e.label if isinstance(e, BaseEvent) else e for e in data])
    counts = labels.value_counts()
    if counts.empty:
        return pd.Series({l: 0 for l in EventLabelEnum})
    if len(counts) == len(EventLabelEnum):
        return counts
    missing_labels = pd.Series({l: 0 for l in EventLabelEnum if l not in counts.index})
    return pd.concat([counts, missing_labels]).sort_index()
