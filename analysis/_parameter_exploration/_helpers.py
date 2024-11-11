import os

import numpy as np
import peyes

import analysis.utils as u

DATASET_NAME = "lund2013"
STIMULUS_TYPE = peyes.constants.IMAGE_STR
GT1, GT2 = "RA", "MN"
MULTI_COMP = "bonferroni"   # method for multiple comparisons correction: bonferroni, fdr_bh, holm, etc.
ALPHA = 0.05

_PARAMS_EXPLORATION_STR = "parameter_exploration"
_FIGURES_STR = "figures"
PROCESSED_DATA_DIR = os.path.join(u.OUTPUT_DIR, _PARAMS_EXPLORATION_STR)
FIGURES_DIR = os.path.join(u.OUTPUT_DIR, _PARAMS_EXPLORATION_STR, DATASET_NAME, _FIGURES_STR)
os.makedirs(FIGURES_DIR, exist_ok=True)

_default_detector_params = dict(missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)
DETECTORS = {
    "ivt_andersson": peyes.create_detector(     # Andersson et al. (2017)
        algorithm="ivt", saccade_velocity_threshold=45, **_default_detector_params
    ),
    "ivt_birawo": peyes.create_detector(        # Birawo & Kasprowski (2022)
        algorithm="ivt", saccade_velocity_threshold=16.5, **_default_detector_params
    ),
    # "ivt_salvucci": peyes.create_detector(    # Salvucci & Goldberg (2000)
    #     algorithm="ivt", saccade_velocity_threshold=100, **_default_detector_params
    # ),
    "ivvt": peyes.create_detector(
        algorithm="ivvt", saccade_velocity_threshold=45, smooth_pursuit_velocity_threshold=26, **_default_detector_params
    ),
    "idt_andersson": peyes.create_detector(
        algorithm="idt", dispersion_threshold=2.7, window_duration=55, **_default_detector_params
    ),
    # "idt_birawo": peyes.create_detector(
    #         # Birawo & Kasprowski (2022) use dispersion of 3.5px and window of 5 samples (10ms)
    #     algorithm="idt", dispersion_threshold=0.1, window_duration=10, **_default_detector_params
    # ),
    "idt_salvucci": peyes.create_detector(
        algorithm="idt", dispersion_threshold=1, window_duration=100, **_default_detector_params
    ),
    "idvt": peyes.create_detector(
        algorithm="idvt", **_default_detector_params
    ),
    "engbert_5": peyes.create_detector(
        algorithm="engbert", lambda_param=5, **_default_detector_params
    ),
    "engbert_6": peyes.create_detector(
        algorithm="engbert", lambda_param=6, **_default_detector_params
    ),
    # "nh": peyes.create_detector(
    #     algorithm="nh", **_default_detector_params
    # ),
    # "remodnav": peyes.create_detector(
    #     algorithm="remodnav", show_warnings=False, **_default_detector_params
    # ),
}
for key, detector in DETECTORS.items():
    detector.name = key
