import os

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
