import os

import numpy as np
import peyes

import analysis.utils as u

DATASET_NAME = "lund2013"
STIMULUS_TYPE = peyes.constants.IMAGE_STR
GT1, GT2 = "RA", "MN"
MULTI_COMP = "bonferroni"   # method for multiple comparisons correction: bonferroni, fdr_bh, holm, etc.
ALPHA, MARGINAL_ALPHA = 0.05, 0.075


_ARTICLE_RESULTS_STR = "article_results"
_FIGURES_STR = "figures"
PROCESSED_DATA_DIR = os.path.join(u.OUTPUT_DIR, _ARTICLE_RESULTS_STR)
FIGURES_DIR = os.path.join(u.OUTPUT_DIR, _ARTICLE_RESULTS_STR, DATASET_NAME, _FIGURES_STR)
os.makedirs(FIGURES_DIR, exist_ok=True)

_default_detector_params = dict(missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)
DETECTORS = {
    "ivt": peyes.create_detector(
        algorithm="ivt", saccade_velocity_threshold=45, **_default_detector_params
    ),
    "ivvt": peyes.create_detector(
        algorithm="ivvt", saccade_velocity_threshold=45, smooth_pursuit_velocity_threshold=26, **_default_detector_params
    ),
    "idt": peyes.create_detector(
        algorithm="idt", dispersion_threshold=2.7, **_default_detector_params
    ),
    # "idt_salvucci": peyes.create_detector(        # Salvucci & Goldberg (2000)
    #     algorithm="idt", dispersion_threshold=1, window_duration=100, **_default_detector_params
    # ),
    "idvt": peyes.create_detector(
        algorithm="idvt", dispersion_threshold=2.7, **_default_detector_params
    ),
    # "idvt_komogortsev": peyes.create_detector(    # Komogortsev & Karpov (2013)
    #     algorithm="idvt",
    #     saccade_velocity_threshold=45,
    #     dispersion_threshold=2,
    #     window_duration=110,
    #     **_default_detector_params
    # ),
    "engbert": peyes.create_detector(
        algorithm="engbert", lambda_param=6, **_default_detector_params
    ),
    "nh": peyes.create_detector(
        algorithm="nh", **_default_detector_params
    ),
    "remodnav": peyes.create_detector(
        algorithm="remodnav", show_warnings=False, **_default_detector_params
    ),
}
for key, detector in DETECTORS.items():
    detector.name = key

## FIGURE CONFIG ##
LABELER_PLOTTING_CONFIG = {
    # labeler -> (order, color, line-style)
    'Other Human': (0, "#bab0ac", 'dot'),
    'RA': (1, u.DEFAULT_DISCRETE_COLORMAP[0], 'dot'),
    'MN': (2, u.DEFAULT_DISCRETE_COLORMAP[1], 'dot'),
    **{key: (i+2 ,u.DEFAULT_DISCRETE_COLORMAP[i], None) for i, key in enumerate(DETECTORS.keys())}
}
