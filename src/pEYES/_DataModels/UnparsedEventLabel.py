from typing import Union, Sequence

from src.pEYES._DataModels.Event import BaseEvent
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

UnparsedEventLabelType = Union[EventLabelEnum, BaseEvent, int, str, float]
UnparsedEventLabelSequenceType = Sequence[UnparsedEventLabelType]
