#
# The following scripts are meant to compare velocity thresholds of the I-DT algorithm:
#   - Andersson et al. (2017) - fixation dispersion criteria: 2.7 deg + 55 ms; saccade velocity threshold: 45 deg/s
#   - Komoogortsev & Karpov (2013) - fixation dispersion criteria: 2 deg + 110 ms; saccade velocity threshold: 45 deg/s
#
# We evaluate detection performance compared to human coders on the lund2013+image dataset (only image stimuli).
# The following metrics are used:
#   1) overall sample-level agreement - balanced accuracy, Cohen's Kappa, MCC, 1-NLD
#   2) fixation onset/offset discriminability (d')
#   3) saccade onset/offset discriminability (d')
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

import pandas as pd
import plotly.io as pio

import peyes

import analysis._parameter_exploration._helpers as h

pio.renderers.default = "browser"

COMPARED_DETECTORS = [det for det in h.DETECTORS.keys() if "idvt" in det.lower()]

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

global_statistics, global_pvalues, global_dunns, global_Ns = sm.kruskal_wallis_dunns(
    global_metrics, [h.GT1, h.GT2], multi_comp=h.MULTI_COMP
)

global_distribution_figure = sm.global_metrics_distributions_figure(global_metrics, h.GT1, h.GT2,)
global_distribution_figure.show()

print("No significant differences in global metrics")

# %%
# ONSET & OFFSET DETECTION (FIXATIONS)

import analysis.statistics.channel_sdt as ch_sdt

LABEL = 1       # EventLabelEnum.FIXATION.value
THRESHOLD = 5   # samples

# Load Data
fixation_csdt_metrics = ch_sdt.load(
    dataset_name=h.DATASET_NAME, output_dir=h.PROCESSED_DATA_DIR, label=LABEL, stimulus_type=h.STIMULUS_TYPE,
)
fixation_csdt_metrics = pd.concat(
    [fixation_csdt_metrics.xs(det, axis=1, level='pred', drop_level=False) for det in COMPARED_DETECTORS], axis=1
)

# calc stats (Kruskal-Wallis)
fix_onset_statistics, fix_onset_pvalues, fix_onset_dunns, fix_onset_Ns = ch_sdt.kruskal_wallis_dunns(
    fixation_csdt_metrics, "onset", THRESHOLD, [h.GT1, h.GT2], multi_comp=h.MULTI_COMP
)

fix_offset_statistics, fix_offset_pvalues, fix_offset_dunns, fix_offset_Ns = ch_sdt.kruskal_wallis_dunns(
    fixation_csdt_metrics, "offset", THRESHOLD, [h.GT1, h.GT2], multi_comp=h.MULTI_COMP
)

# Show Figures
fixation_dprime_figure = ch_sdt.multi_channel_figure(
    fixation_csdt_metrics, metric=peyes.constants.D_PRIME_STR,
    yaxis_title=r"$d'$", show_other_gt=True, show_err_bands=True
)
fixation_dprime_figure.update_layout(width=1400, height=500,)
fixation_dprime_figure.show()

print("No significant differences in fixation onset/offset detection")

# %%
# ONSET & OFFSET DETECTION (SACCADES)

import analysis.statistics.channel_sdt as ch_sdt

LABEL = 2       # EventLabelEnum.SACCADE.value
THRESHOLD = 5   # samples

# Load Data
saccade_csdt_metrics = ch_sdt.load(
    dataset_name=h.DATASET_NAME, output_dir=h.PROCESSED_DATA_DIR, label=LABEL, stimulus_type=h.STIMULUS_TYPE,
)
saccade_csdt_metrics = pd.concat(
    [saccade_csdt_metrics.xs(det, axis=1, level='pred', drop_level=False) for det in COMPARED_DETECTORS], axis=1
)

# calc stats (Kruskal-Wallis)
sac_onset_statistics, sac_onset_pvalues, sac_onset_dunns, sac_onset_Ns = ch_sdt.kruskal_wallis_dunns(
    saccade_csdt_metrics, "onset", THRESHOLD, [h.GT1, h.GT2], multi_comp=h.MULTI_COMP
)

sac_offset_statistics, sac_offset_pvalues, sac_offset_dunns, sac_offset_Ns = ch_sdt.kruskal_wallis_dunns(
    saccade_csdt_metrics, "offset", THRESHOLD, [h.GT1, h.GT2], multi_comp=h.MULTI_COMP
)

# Show Figures
saccade_dprime_figure = ch_sdt.multi_channel_figure(
    saccade_csdt_metrics, metric=peyes.constants.D_PRIME_STR,
    yaxis_title=r"$d'$", show_other_gt=True, show_err_bands=True
)
saccade_dprime_figure.update_layout(width=1400, height=500,)
saccade_dprime_figure.show()

print(
    "saccade onset using I-DVT with Andersson params is *significantly* worse than with Komogorob & "
    "Komogorov-Birawo params (which are the same)"
)
