from typing import Union, Sequence

import numpy as np

from pEYES._DataModels.Event import BaseEvent
from pEYES._DataModels.EventLabelEnum import EventLabelEnum

UnparsedEventLabelType = Union[EventLabelEnum, BaseEvent, int, str, float, np.number]
UnparsedEventLabelSequenceType = Sequence[UnparsedEventLabelType]
