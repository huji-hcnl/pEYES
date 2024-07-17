from typing import Union, Sequence

from pEYES._DataModels.Event import BaseEvent
from pEYES._DataModels.EventLabelEnum import EventLabelEnum

UnparsedEventLabelType = Union[EventLabelEnum, BaseEvent, int, str, float]
UnparsedEventLabelSequenceType = Sequence[UnparsedEventLabelType]
