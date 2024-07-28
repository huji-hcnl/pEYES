import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u

pio.renderers.default = "browser"

###################

GT1, GT2 = "RA", "MN"


###################
## Matched SDT Metrics ; 1st iteration ; image trials
matched_sdt_metrics = pd.read_pickle(os.path.join(u.OUTPUT_DIR, "default_values", "lund2013", "matches_metrics", "fixation_sdt_metrics.pkl"))
iter1_matched_sdt_metrics = matched_sdt_metrics.xs(1, level=peyes.constants.ITERATION_STR, axis=1)
image_iter1_matched_sdt_metrics = iter1_matched_sdt_metrics.loc[:, iter1_matched_sdt_metrics.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(image_trials)]

mean_match_ratio_across_schemes = image_iter1_matched_sdt_metrics.xs(u.MATCH_RATIO_STR, level=peyes.constants.METRIC_STR, axis=0).mean(axis=0)
mean_match_ratio_across_schemes.name = u.MATCH_RATIO_STR
mean_match_ratio_across_schemes = mean_match_ratio_across_schemes.to_frame().T

statistics, pvalues, dunns, Ns = kruskal_wallis_with_posthoc_dunn(mean_match_ratio_across_schemes, GT_COLS, multi_comp="fdr_bh")
fig = sample_metrics_figure(mean_match_ratio_across_schemes, GT1, GT2, title=f"Sample Metrics")
fig.show()

