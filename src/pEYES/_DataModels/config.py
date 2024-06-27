import src.pEYES._utils.constants as cnst
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

EPSILON = 1e-8

EVENT_MAPPING = {
    EventLabelEnum.UNDEFINED: {
        cnst.LABEL_STR: EventLabelEnum.UNDEFINED.name,
        cnst.COLOR_STR: "#dddddd",
        cnst.MIN_DURATION_STR: 0,
        cnst.MAX_DURATION_STR: 1e6
    },
    EventLabelEnum.FIXATION: {
        cnst.LABEL_STR: EventLabelEnum.FIXATION.name,
        cnst.COLOR_STR: "#1f78b4",
        cnst.MIN_DURATION_STR: 50,
        cnst.MAX_DURATION_STR: 2000
    },
    EventLabelEnum.SACCADE: {
        cnst.LABEL_STR: EventLabelEnum.SACCADE.name,
        cnst.COLOR_STR: "#33a02c",
        cnst.MIN_DURATION_STR: 10,
        cnst.MAX_DURATION_STR: 200
    },
    EventLabelEnum.PSO: {
        cnst.LABEL_STR: EventLabelEnum.PSO.name,
        cnst.COLOR_STR: "#b2df8a",
        cnst.MIN_DURATION_STR: 10,
        cnst.MAX_DURATION_STR: 80
    },
    EventLabelEnum.SMOOTH_PURSUIT: {
        cnst.LABEL_STR: EventLabelEnum.SMOOTH_PURSUIT.name,
        cnst.COLOR_STR: "#fb9a99",
        cnst.MIN_DURATION_STR: 40,
        cnst.MAX_DURATION_STR: 2000
    },
    EventLabelEnum.BLINK: {
        cnst.LABEL_STR: EventLabelEnum.BLINK.name,
        cnst.COLOR_STR: "#222222",
        cnst.MIN_DURATION_STR: 20,
        cnst.MAX_DURATION_STR: 2000
    }
}

MIN_EVENT_DURATION = min([
    EVENT_MAPPING[e][cnst.MIN_DURATION_STR] for e in EVENT_MAPPING.keys() if e != EventLabelEnum.UNDEFINED
])
