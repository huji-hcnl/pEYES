import peyes._utils.constants as cnst
from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._utils.pixel_utils import calculate_pixel_size as _calc_ps

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
        cnst.MAX_DURATION_STR: 1e9,  # ms
    },
    EventLabelEnum.FIXATION: {
        cnst.LABEL_STR: EventLabelEnum.FIXATION.name,
        cnst.COLOR_STR: "#1f78b4",
        cnst.MIN_DURATION_STR: 55,
        cnst.MAX_DURATION_STR: 2500,
        cnst.MAX_ACCELERATION_STR: 50000,  # deg/s^2; https://doi.org/10.1167/tvst.11.2.35
    },
    EventLabelEnum.SACCADE: {
        cnst.LABEL_STR: EventLabelEnum.SACCADE.name,
        cnst.COLOR_STR: "#33a02c",
        cnst.MIN_DURATION_STR: 10,
        cnst.MAX_DURATION_STR: 200,
        cnst.MAX_VELOCITY_STR: 1000,  # deg/s; https://doi.org/10.1186/s41235-025-00657-y
        # cnst.MAX_VELOCITY_STR: 500,  # deg/s; https://doi.org/10.1371/journal.pone.0229177
        # cnst.MAX_VELOCITY_STR: 1500,  # deg/s; macaques, https://doi.org/10.1523/ENEURO.0086-23.2023
        # cnst.MAX_ACCELERATION_STR: 120000,  # deg/s^2; macaques, https://doi.org/10.1523/ENEURO.0086-23.2023
        # max amplitude 20 deg; https://doi.org/10.1037/pag0000718 (no amplitude-threshold check implemented yet)
    },
    EventLabelEnum.PSO: {
        cnst.LABEL_STR: EventLabelEnum.PSO.name,
        cnst.COLOR_STR: "#b2df8a",
        cnst.MIN_DURATION_STR: 4,
        cnst.MAX_DURATION_STR: 40,
    },
    EventLabelEnum.SMOOTH_PURSUIT: {
        cnst.LABEL_STR: EventLabelEnum.SMOOTH_PURSUIT.name,
        cnst.COLOR_STR: "#fb9a99",
        cnst.MIN_DURATION_STR: 40,
        cnst.MAX_DURATION_STR: 5000,
    },
    EventLabelEnum.BLINK: {
        cnst.LABEL_STR: EventLabelEnum.BLINK.name,
        cnst.COLOR_STR: "#222222",
        cnst.MIN_DURATION_STR: 20,
        cnst.MAX_DURATION_STR: 2500,
    }
}
