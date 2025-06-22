#
# This script is used to explore the effect of the lambda SNR parameter on the Engbert detection algorithm.
# The following lambdas are compared:
#   - Engbert & Kliegl (2003) + Andersson et al. (2017): 6.0
#   - Engbert & Mergenthaler (2006): 5.0
#
# We evaluate detection performance compared to human coders on the lund2013+image dataset (only image stimuli).
# The following metrics are used:
#   1) overall sample-level agreement - balanced accuracy, Cohen's Kappa, MCC, 1-NLD
#   2) fixation onset/offset discriminability (d')
#   3) saccade onset/offset discriminability (d')
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
import pandas as pd
import plotly.io as pio

import peyes

import analysis._parameter_exploration._helpers as h

pio.renderers.default = "browser"

COMPARED_DETECTORS = [det for det in h.DETECTORS.keys() if "engbert" in det.lower()]

# %%
# SAMPLE-LEVEL AGREEMENT (GLOBAL MEASURES)

import analysis.statistics.sample_metrics as sm

# Load Data
global_metrics = sm.load_global_metrics(
    h.DATASET_NAME, h.PROCESSED_DATA_DIR, stimulus_type=h.STIMULUS_TYPE, metric=None, iteration=1,
)
global_metrics = pd.concat(
    [global_metrics.xs(det, axis=1, level='pred', drop_level=False) for det in COMPARED_DETECTORS], axis=1
).drop(index=peyes.constants.ACCURACY_STR, inplace=False)    # drop Accuracy metric

global_statistics, global_pvalues, global_Ns = sm.wilcoxon(
    global_metrics, [h.GT1, h.GT2], method='exact'
)

global_distribution_figure = sm.global_metrics_distributions_figure(global_metrics, h.GT1, h.GT2,)
global_distribution_figure.show()

print("There is statistical difference between λ=5.0 and λ=6.0!")

# %%
# ONSET & OFFSET DETECTION (FIXATIONS)

import analysis.statistics.channel_sdt as ch_sdt

LABEL = 1       # EventLabelEnum.FIXATION.value
THRESHOLD = 5   # samples

# Load Data
fixation_csdt_metrics = ch_sdt.load(
    dataset_name=h.DATASET_NAME,
    output_dir=h.PROCESSED_DATA_DIR,
    label=LABEL,
    stimulus_type=h.STIMULUS_TYPE,
    channel_type=None
)
fixation_csdt_metrics = pd.concat(
    [fixation_csdt_metrics.xs(det, axis=1, level='pred', drop_level=False) for det in COMPARED_DETECTORS], axis=1
)
fixation_csdt_metrics.drop(index=['P', 'PP', 'TP', 'N'], level=peyes.constants.METRIC_STR, inplace=True)    # drop irrelevant metrics

# calc stats
fix_onset_w_stat, fix_onset_pvalues, fix_onset_Ns = ch_sdt.wilcoxon(
    fixation_csdt_metrics, "onset", THRESHOLD, [h.GT1, h.GT2], method='exact',
)

fix_offset_W_stat, fix_offset_pvalues, fix_offset_Ns = ch_sdt.wilcoxon(
    fixation_csdt_metrics, "offset", THRESHOLD, [h.GT1, h.GT2], method='exact',
)

# show figures
fixation_dprime_figure = ch_sdt.multi_channel_figure(
    fixation_csdt_metrics,
    metric=peyes.constants.D_PRIME_STR,
    yaxis_title=r"$d'$", show_other_gt=True, show_err_bands=True
)
fixation_dprime_figure.update_layout(width=1400, height=500,)
fixation_dprime_figure.show()

print("There is statistical difference between λ=5.0 and λ=6.0, but only for fixation offsets.")

# %%
# ONSET & OFFSET DETECTION (SACCADES)

import analysis.statistics.channel_sdt as ch_sdt

LABEL = 2       # EventLabelEnum.SACCADE.value
THRESHOLD = 5   # samples

# Load Data
saccade_csdt_metrics = ch_sdt.load(
    dataset_name=h.DATASET_NAME,
    output_dir=h.PROCESSED_DATA_DIR,
    label=LABEL,
    stimulus_type=h.STIMULUS_TYPE,
    channel_type=None
)
saccade_csdt_metrics = pd.concat(
    [saccade_csdt_metrics.xs(det, axis=1, level='pred', drop_level=False) for det in COMPARED_DETECTORS], axis=1
)
saccade_csdt_metrics.drop(index=['P', 'PP', 'TP', 'N'], level=peyes.constants.METRIC_STR, inplace=True)    # drop irrelevant metrics

# calc stats
sacc_onset_w_stat, sacc_onset_pvalues, sacc_onset_Ns = ch_sdt.wilcoxon(
    saccade_csdt_metrics, "onset", THRESHOLD, [h.GT1, h.GT2], method='exact',
)

sacc_offset_w_stat, sacc_offset_pvalues, sacc_offset_Ns = ch_sdt.wilcoxon(
    saccade_csdt_metrics, "offset", THRESHOLD, [h.GT1, h.GT2], method='exact',
)

# show figures
saccade_dprime_figure = ch_sdt.multi_channel_figure(
    saccade_csdt_metrics,
    metric=peyes.constants.D_PRIME_STR,
    yaxis_title=r"$d'$", show_other_gt=True, show_err_bands=True
)
saccade_dprime_figure.update_layout(width=1400, height=500,)
saccade_dprime_figure.show()

print("There is statistical difference between λ=5.0 and λ=6.0!")
