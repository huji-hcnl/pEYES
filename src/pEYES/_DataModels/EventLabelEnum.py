from enum import IntEnum
from typing import Sequence


class EventLabelEnum(IntEnum):
    UNDEFINED = 0
    FIXATION = 1
    SACCADE = 2
    PSO = 3
    SMOOTH_PURSUIT = 4
    BLINK = 5


EventLabelSequenceType = Sequence[EventLabelEnum]
