from typing import Union, Sequence

import numpy as np

from peyes._DataModels.Event import BaseEvent
from peyes._DataModels.EventLabelEnum import EventLabelEnum

UnparsedEventLabelType = Union[EventLabelEnum, BaseEvent, int, str, float, np.number]
UnparsedEventLabelSequenceType = Sequence[UnparsedEventLabelType]
