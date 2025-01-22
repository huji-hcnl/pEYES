import os

import numpy as np
import peyes

import analysis.utils as u

DATASET_NAME = "lund2013"
STIMULUS_TYPE = peyes.constants.IMAGE_STR
GT1, GT2 = "RA", "MN"
MULTI_COMP = "fdr_bh"   # FDR Benjamini-Hochberg correction for multiple comparisons
ALPHA = 0.05

_DEFAULT_VALUES_STR = "default_values"
_FIGURES_STR = "figures"
PROCESSED_DATA_DIR = os.path.join(u.OUTPUT_DIR, _DEFAULT_VALUES_STR)
FIGURES_DIR = os.path.join(u.OUTPUT_DIR, _DEFAULT_VALUES_STR, DATASET_NAME, _FIGURES_STR)
os.makedirs(FIGURES_DIR, exist_ok=True)

_default_detector_params = dict(missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)
DEFAULTS_DETECTORS = {
    "ivt": peyes.create_detector("ivt", **_default_detector_params),
    "ivvt": peyes.create_detector("ivvt", **_default_detector_params),
    "idt": peyes.create_detector("idt", **_default_detector_params),
    "idvt": peyes.create_detector("idvt", **_default_detector_params),
    "engbert": peyes.create_detector("engbert", **_default_detector_params),
    "nh": peyes.create_detector("nh", **_default_detector_params),
    "remodnav": peyes.create_detector("remodnav", show_warnings=False, **_default_detector_params),
}
