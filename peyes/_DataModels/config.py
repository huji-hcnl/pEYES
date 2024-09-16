import peyes._utils.constants as cnst
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._utils.pixel_utils import calculate_pixel_size as _calc_ps

EPSILON = 1e-8

VIEWER_DISTANCE = 60    # cm

SCREEN_MONITOR = {
    cnst.WIDTH_STR: cnst.TOBII_WIDTH,
    cnst.HEIGHT_STR: cnst.TOBII_HEIGHT,
    cnst.RESOLUTION_STR: cnst.TOBII_RESOLUTION,
    cnst.PIXEL_SIZE_STR: _calc_ps(cnst.TOBII_WIDTH, cnst.TOBII_HEIGHT, cnst.TOBII_RESOLUTION),
}

EVENT_MAPPING = {
    EventLabelEnum.UNDEFINED: {
        cnst.LABEL_STR: EventLabelEnum.UNDEFINED.name,
        cnst.COLOR_STR: "#dddddd",
        cnst.MIN_DURATION_STR: 0,   # ms
        cnst.MAX_DURATION_STR: 1e6  # ms
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
        cnst.MIN_DURATION_STR: 50,
        cnst.MAX_DURATION_STR: 2000
    },
    EventLabelEnum.BLINK: {
        cnst.LABEL_STR: EventLabelEnum.BLINK.name,
        cnst.COLOR_STR: "#222222",
        cnst.MIN_DURATION_STR: 50,
        cnst.MAX_DURATION_STR: 2000
    }
}
