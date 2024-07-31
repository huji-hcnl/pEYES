import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import pEYES as peyes
import analysis.utils as u

pio.renderers.default = "browser"

######################

DATASET_NAME = "lund2013"
LABEL = 1
STIMULUS_TYPE = peyes.constants.IMAGE_STR
GT1, GT2 = "RA", "MN"
MULTI_COMP = "fdr_bh"

# %%
######################
##  Sample Metrics  ##

import analysis.statistics.sample_metrics as sm

sample_metrics = sm.load(
    DATASET_NAME, os.path.join(u.OUTPUT_DIR, "default_values"), label=LABEL, stimulus_type=STIMULUS_TYPE, metric=None
)
sm_statistics, sm_pvalues, sm_dunns, sm_Ns = sm.kruskal_wallis_dunns(
    sample_metrics, [GT1, GT2], multi_comp=MULTI_COMP
)
sample_metrics_fig = sm.distributions_figure(sample_metrics, GT1, gt2=GT2, only_box=False)
sample_metrics_fig.show()

# %%
##########################
##  Channel Time Diffs  ##

import analysis.statistics.channel_time_diffs as ctd

time_diffs = ctd.load(
    DATASET_NAME, os.path.join(u.OUTPUT_DIR, "default_values"), label=LABEL, stimulus_type=STIMULUS_TYPE
)
ctd_statistics, ctd_pvalues, ctd_dunns, ctd_Ns = ctd.kruskal_wallis_dunns(
    time_diffs, [GT1, GT2], multi_comp=MULTI_COMP
)
time_diffs_fig = ctd.distributions_figure(time_diffs, GT1, gt2=GT2, only_box=False)
time_diffs_fig.show()

# %%
###########################
##  Channel SDT Metrics  ##

import analysis.statistics.channel_sdt as csdt

CHANNEL_TYPE = "onset"
THRESHOLD = 10  # samples

sdt_metrics = csdt.load(
    dataset_name=DATASET_NAME,
    output_dir=os.path.join(u.OUTPUT_DIR, "default_values"),
    label=LABEL,
    stimulus_type=STIMULUS_TYPE,
    channel_type=None
)

csdt_statistics, csdt_pvalues, csdt_dunns, csdt_Ns = csdt.kruskal_wallis_dunns(
    sdt_metrics, CHANNEL_TYPE, THRESHOLD, [GT1, GT2], multi_comp=MULTI_COMP
)

threshold_fig = csdt.single_threshold_figure(sdt_metrics, CHANNEL_TYPE, THRESHOLD, GT1, gt2=GT2)

csdt_figs = csdt.multi_threshold_figures(sdt_metrics, CHANNEL_TYPE, show_err_bands=True)
csdt_figs[GT1].show()


# %%
###############################
##  Matched-Events Features  ##

import analysis.statistics.matched_features as mf

SCHEME = "window"
ALPHA = 0.05

matched_features = mf.load(
    dataset_name=DATASET_NAME,
    output_dir=os.path.join(u.OUTPUT_DIR, "default_values"),
    label=None,
    stimulus_type=STIMULUS_TYPE,
    matching_schemes=None,
)

mf_statistics, mf_pvalues, mf_dunns, mf_Ns = mf.kruskal_wallis_dunns(
    matched_features=matched_features, matching_scheme=SCHEME, gt_cols=[GT1, GT2], features=None, multi_comp=MULTI_COMP
)
mf_statistics2, mf_pvalues2, mf_corrected_pvalues2, mf_Ns2 = mf.wilcoxon_signed_rank(
    matched_features=matched_features, matching_scheme=SCHEME, gt_col=GT1, features=None, multi_comp=MULTI_COMP, alpha=ALPHA
)

mf_fig = mf.distributions_figure(matched_features, SCHEME, GT1, GT2)
mf_fig.show()


# %%
########################
## Matched-Events SDT ##

import analysis.statistics.matches_sdt as msdt

SCHEME = "window_10"

matched_sdt = msdt.load(
    dataset_name=DATASET_NAME, output_dir=os.path.join(u.OUTPUT_DIR, "default_values"),
    label=LABEL, stimulus_type=STIMULUS_TYPE, matching_schemes=None, metrics=None
)

msdt_statistics, msdt_pvalues, msdt_dunns, msdt_Ns = msdt.kruskal_wallis_dunns(
    matched_sdt, SCHEME, [GT1, GT2], metrics=None, multi_comp=MULTI_COMP
)

msdt_single_fig = msdt.single_scheme_figure(
    matched_sdt, SCHEME, GT1, gt2=GT2, metrics=None, only_box=False
)
msdt_single_fig.show()

msdt_figs = msdt.multi_threshold_figures(matched_sdt, SCHEME, metrics=None, show_err_bands=True)
msdt_figs[GT1].show()
