import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import peyes
import analysis.utils as u

pio.renderers.default = "browser"

######################

DATASET_NAME = "lund2013"
OUTPUT_DIR = os.path.join(u.OUTPUT_DIR, "default_values")

LABEL = 1
STIMULUS_TYPE = peyes.constants.IMAGE_STR
GT1, GT2 = "RA", "MN"
MULTI_COMP = "fdr_bh"

# %%
##########################
##  Feature Comparison  ##

stim_trial_ids = u.get_trials_for_stimulus_type(DATASET_NAME, STIMULUS_TYPE)

all_events = pd.read_pickle(os.path.join(OUTPUT_DIR, DATASET_NAME, "events.pkl"))
all_events = all_events.xs(1, level=peyes.constants.ITERATION_STR, axis=1)
all_events = all_events.loc[:, all_events.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(stim_trial_ids)]
all_events = all_events.dropna(axis=0, how="all")

all_labelers = all_events.columns.get_level_values(peyes.constants.LABELER_STR).unique()
events_by_labelers = {
    lblr: all_events.xs(lblr, level=peyes.constants.LABELER_STR, axis=1).stack().dropna() for lblr in all_labelers
}

fixations_comparison_figure = peyes.visualize.feature_comparison(
    [
        peyes.constants.DURATION_STR,
        peyes.constants.AMPLITUDE_STR,
        peyes.constants.PEAK_VELOCITY_STR,
        peyes.constants.MEDIAN_VELOCITY_STR,
        peyes.constants.COUNT_STR,
     ],
    *[vals[vals.apply(lambda e: e.label == 1)] for vals in events_by_labelers.values()],
    labels=events_by_labelers.keys(),
    title="Fixation Features Comparison",
)
fixations_comparison_figure.show()

saccades_comparison_figure = peyes.visualize.feature_comparison(
    [
        peyes.constants.DURATION_STR,
        peyes.constants.AMPLITUDE_STR,
        peyes.constants.PEAK_VELOCITY_STR,
        peyes.constants.MEDIAN_VELOCITY_STR,
        peyes.constants.COUNT_STR,
     ],
    *[vals[vals.apply(lambda e: e.label == 2)] for vals in events_by_labelers.values()],
    labels=events_by_labelers.keys(),
    title="Saccade Features Comparison",
)
saccades_comparison_figure.show()

# %%
##########################
##  Events Summary Plot  ##

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    events_summary_figures = {
        lblr: peyes.visualize.event_summary(
            events_by_labelers[lblr],
            show_outliers=True,
            title=f"{lblr} :: Events Summary",
        ) for lblr in all_labelers
    }
    fixations_summary_figures = {
        lblr: peyes.visualize.fixation_summary(
            events_by_labelers[lblr],
            show_outliers=True,
            title=f"{lblr} :: Fixations Summary",
        ) for lblr in all_labelers
    }
    saccades_summary_figures = {
        lblr: peyes.visualize.saccade_summary(
            events_by_labelers[lblr],
            show_outliers=True,
            title=f"{lblr} :: Saccades Summary",
        ) for lblr in all_labelers
    }

events_summary_figures[GT1].show()
fixations_summary_figures[GT1].show()
saccades_summary_figures[GT1].show()


# %%
######################
##  Sample Metrics  ##

import analysis.statistics.sample_metrics as sm

sample_global_metrics = sm.load_global_metrics(
    DATASET_NAME, OUTPUT_DIR, stimulus_type=STIMULUS_TYPE, metric=None
)
sm_global_statistics, sm_global_pvalues, sm_global_dunns, sm_global_Ns = sm.kruskal_wallis_dunns(
    sample_global_metrics, [GT1, GT2], multi_comp=MULTI_COMP
)
sm_global_metrics_fig = sm.global_metrics_distributions_figure(sample_global_metrics, GT1, gt2=GT2, only_box=False)
sm_global_metrics_fig.show()

###

sample_sdt_metrics = sm.load_sdt(
    DATASET_NAME, OUTPUT_DIR, label=LABEL, stimulus_type=STIMULUS_TYPE, metric=None
)
sm_sdt_statistics, sm_sdt_pvalues, sm_sdt_dunns, sm_sdt_Ns = sm.kruskal_wallis_dunns(
    sample_sdt_metrics, [GT1, GT2], multi_comp=MULTI_COMP
)
sample_sdt_metrics_fig = sm.sdt_distributions_figure(sample_sdt_metrics, GT1, gt2=GT2, only_box=False)
sample_sdt_metrics_fig.show()

# %%
##########################
##  Channel Time Diffs  ##

import analysis.statistics.channel_time_diffs as ctd

time_diffs = ctd.load(
    DATASET_NAME, OUTPUT_DIR, label=LABEL, stimulus_type=STIMULUS_TYPE
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

THRESHOLD = 10  # samples

###
CHANNEL_TYPE = "onset"

sdt_onset_metrics = csdt.load(
    dataset_name=DATASET_NAME,
    output_dir=OUTPUT_DIR,
    label=LABEL,
    stimulus_type=STIMULUS_TYPE,
    channel_type=None
)

csdt_onset_statistics, csdt_onset_pvalues, csdt_onset_dunns, csdt_onset_Ns = csdt.kruskal_wallis_dunns(
    sdt_onset_metrics, CHANNEL_TYPE, THRESHOLD, [GT1, GT2], multi_comp=MULTI_COMP
)

threshold_onset_fig = csdt.single_threshold_figure(sdt_onset_metrics, CHANNEL_TYPE, THRESHOLD, GT1, gt2=GT2)
threshold_onset_fig.show()

csdt_onset_figs = csdt.multi_threshold_figures(sdt_onset_metrics, CHANNEL_TYPE, show_err_bands=True)
csdt_onset_figs[GT1].show()

###
CHANNEL_TYPE = "offset"

sdt_offset_metrics = csdt.load(
    dataset_name=DATASET_NAME,
    output_dir=OUTPUT_DIR,
    label=LABEL,
    stimulus_type=STIMULUS_TYPE,
    channel_type=None
)

csdt_offset_statistics, csdt_offset_pvalues, csdt_offset_dunns, csdt_offset_Ns = csdt.kruskal_wallis_dunns(
    sdt_offset_metrics, CHANNEL_TYPE, THRESHOLD, [GT1, GT2], multi_comp=MULTI_COMP
)

threshold_offset_fig = csdt.single_threshold_figure(sdt_offset_metrics, CHANNEL_TYPE, THRESHOLD, GT1, gt2=GT2)
threshold_offset_fig.show()

csdt_offset_figs = csdt.multi_threshold_figures(sdt_offset_metrics, CHANNEL_TYPE, show_err_bands=True)
csdt_offset_figs[GT1].show()


# %%
###############################
##  Matched-Events Features  ##

import analysis.statistics.matched_features as mf

SCHEME = "window_10"
ALPHA = 0.05

matched_features = mf.load(
    dataset_name=DATASET_NAME,
    output_dir=OUTPUT_DIR,
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

import analysis.statistics.matched_sdt as msdt

matched_sdt = msdt.load(
    dataset_name=DATASET_NAME, output_dir=OUTPUT_DIR,
    label=LABEL, stimulus_type=STIMULUS_TYPE, matching_schemes=None, metrics=None
)

msdt_statistics, msdt_pvalues, msdt_dunns, msdt_Ns = msdt.kruskal_wallis_dunns(
    matched_sdt, SCHEME, [GT1, GT2], metrics=None, multi_comp=MULTI_COMP
)

msdt_single_fig = msdt.single_scheme_figure(
    matched_sdt, SCHEME, GT1, gt2=GT2, metrics=None, only_box=False
)
msdt_single_fig.show()

msdt_figs = msdt.multi_threshold_figures(matched_sdt, "window", metrics=None, show_err_bands=True)
msdt_figs[GT1].show()
