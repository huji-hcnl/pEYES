import os

import analysis.utils as u

DATASET_NAME = "lund2013"
GT1, GT2 = "RA", "MN"
ALPHA = 0.05

_DEFAULT_VALUES_STR = "default_values"
_FIGURES_STR = "figures"
PROCESSED_DATA_DIR = os.path.join(u.OUTPUT_DIR, _DEFAULT_VALUES_STR)
FIGURES_DIR = os.path.join(u.OUTPUT_DIR, _DEFAULT_VALUES_STR, DATASET_NAME, _FIGURES_STR)
os.makedirs(FIGURES_DIR, exist_ok=True)