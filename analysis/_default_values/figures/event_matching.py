import plotly.io as pio

import peyes
from analysis._default_values._helpers import *
import analysis.statistics.matched_sdt as msdt

pio.renderers.default = "browser"

THRESHOLD = 5  # samples
MATCHING_SCHEME = "window"
METRICS = [peyes.constants.MATCH_RATIO_STR, peyes.constants.F1_STR, peyes.constants.D_PRIME_STR]

# %%
######################
## Fixation Matches ##

LABEL = 1       # EventLabelEnum.FIXATION.value

matched_fixations_sdt = msdt.load(
    dataset_name=DATASET_NAME, output_dir=PROCESSED_DATA_DIR,
    label=LABEL, stimulus_type=STIMULUS_TYPE, matching_schemes=None, metrics=None
)

fixation_statistics, fixation_pvalues, fixation_dunns, fixation_Ns = msdt.kruskal_wallis_dunns(
    matched_fixations_sdt,
    f"{MATCHING_SCHEME}_{THRESHOLD}",
    [GT1, GT2],
    metrics=METRICS,
    multi_comp=MULTI_COMP
)

### Show Figures

fixation_single_threshold_fig = msdt.single_scheme_figure(
    matched_fixations_sdt, f"{MATCHING_SCHEME}_{THRESHOLD}", GT1, gt2=GT2, metrics=METRICS, only_box=False
)
fixation_single_threshold_fig.show()

fixation_multi_threshold_figs = msdt.multi_metric_figure(
    matched_fixations_sdt,
    MATCHING_SCHEME,
    [peyes.constants.MATCH_RATIO_STR, peyes.constants.F1_STR, peyes.constants.D_PRIME_STR],
    show_other_gt=True, show_err_bands=True
)
fixation_multi_threshold_figs.show()

# %%
#####################
## Saccade Matches ##

LABEL = 2       # EventLabelEnum.SACCADE.value

matched_saccade_sdt = msdt.load(
    dataset_name=DATASET_NAME, output_dir=PROCESSED_DATA_DIR,
    label=LABEL, stimulus_type=STIMULUS_TYPE, matching_schemes=None, metrics=None
)

saccade_statistics, saccade_pvalues, saccade_dunns, saccade_Ns = msdt.kruskal_wallis_dunns(
    matched_fixations_sdt,
    f"{MATCHING_SCHEME}_{THRESHOLD}",
    [GT1, GT2],
    metrics=METRICS,
    multi_comp=MULTI_COMP
)

### Show Figures

saccade_single_threshold_fig = msdt.single_scheme_figure(
    matched_fixations_sdt, f"{MATCHING_SCHEME}_{THRESHOLD}", GT1, gt2=GT2, metrics=METRICS, only_box=False
)
saccade_single_threshold_fig.show()

saccade_multi_threshold_figs = msdt.multi_metric_figure(
    matched_fixations_sdt,
    MATCHING_SCHEME,
    [peyes.constants.MATCH_RATIO_STR, peyes.constants.F1_STR, peyes.constants.D_PRIME_STR],
    show_other_gt=True, show_err_bands=True
)
saccade_multi_threshold_figs.show()

# %%
######################
## Save Figures

peyes.visualize.save_figure(
    fixation_single_threshold_fig, "single_threshold-fixation", FIGURES_DIR,
    as_png=True, as_html=False, as_json=False
)
peyes.visualize.save_figure(
    fixation_multi_threshold_figs, "multi_threshold-fixation", FIGURES_DIR,
    as_png=True, as_html=False, as_json=False
)

peyes.visualize.save_figure(
    saccade_single_threshold_fig, "single_threshold-saccade", FIGURES_DIR,
    as_png=True, as_html=False, as_json=False
)
peyes.visualize.save_figure(
    saccade_multi_threshold_figs, "multi_threshold-saccade", FIGURES_DIR,
    as_png=True, as_html=False, as_json=False
)
