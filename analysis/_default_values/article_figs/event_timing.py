import plotly.io as pio

import peyes
from analysis._default_values._helpers import *
import analysis.statistics.channel_sdt as ch_sdt

pio.renderers.default = "browser"
THRESHOLD = 5  # samples

# %%
#######################
## Fixation Channels ##

LABEL = 1       # EventLabelEnum.FIXATION.value

fixation_csdt_metrics = ch_sdt.load(
    dataset_name=DATASET_NAME,
    output_dir=PROCESSED_DATA_DIR,
    label=LABEL,
    stimulus_type=STIMULUS_TYPE,
    channel_type=None
)

## Stats

fix_onset_statistics, fix_onset_pvalues, fix_onset_dunns, fix_onset_Ns = ch_sdt.kruskal_wallis_dunns(
    fixation_csdt_metrics, "onset", THRESHOLD, [GT1, GT2], multi_comp=MULTI_COMP
)

fix_offset_statistics, fix_offset_pvalues, fix_offset_dunns, fix_offset_Ns = ch_sdt.kruskal_wallis_dunns(
    fixation_csdt_metrics, "offset", THRESHOLD, [GT1, GT2], multi_comp=MULTI_COMP
)

### Show Figures

fixation_dprime_figure = ch_sdt.multi_channel_figure(
    fixation_csdt_metrics,
    metric=peyes.constants.D_PRIME_STR,
    yaxis_title=r"$d'$", show_other_gt=True, show_err_bands=True
)
fixation_dprime_figure.show()

fixation_criterion_figure = ch_sdt.multi_channel_figure(
    fixation_csdt_metrics,
    metric=peyes.constants.CRITERION_STR,
    yaxis_title="Criterion", show_other_gt=True, show_err_bands=True
)
fixation_criterion_figure.show()

### Save Figures

peyes.visualize.save_figure(
    fixation_dprime_figure, "fixation_dprime", FIGURES_DIR, as_png=True, as_html=False, as_json=False
)
peyes.visualize.save_figure(
    fixation_criterion_figure, "fixation_criterion", FIGURES_DIR, as_png=True, as_html=False, as_json=False
)

# %%
######################
## Saccade Channels ##

LABEL = 2       # EventLabelEnum.FIXATION.value

saccade_csdt_metrics = ch_sdt.load(
    dataset_name=DATASET_NAME,
    output_dir=PROCESSED_DATA_DIR,
    label=LABEL,
    stimulus_type=STIMULUS_TYPE,
    channel_type=None
)

## Stats

sac_onset_statistics, sac_onset_pvalues, sac_onset_dunns, sac_onset_Ns = ch_sdt.kruskal_wallis_dunns(
    saccade_csdt_metrics, "onset", THRESHOLD, [GT1, GT2], multi_comp=MULTI_COMP
)

sac_offset_statistics, sac_offset_pvalues, sac_offset_dunns, sac_offset_Ns = ch_sdt.kruskal_wallis_dunns(
    saccade_csdt_metrics, "offset", THRESHOLD, [GT1, GT2], multi_comp=MULTI_COMP
)

### Show Figures

saccade_dprime_figure = ch_sdt.multi_channel_figure(saccade_csdt_metrics, metric=peyes.constants.D_PRIME_STR,
                                                    yaxis_title=r"$d'$", show_other_gt=True, show_err_bands=True)
saccade_dprime_figure.show()

saccade_criterion_figure = ch_sdt.multi_channel_figure(saccade_csdt_metrics, metric=peyes.constants.CRITERION_STR,
                                                       yaxis_title="Criterion", show_other_gt=True, show_err_bands=True)
saccade_criterion_figure.show()

# %%
######################
## Save Figures

peyes.visualize.save_figure(
    fixation_dprime_figure, "channel_metrics-fixation-dprime", FIGURES_DIR,
    as_png=True, as_html=False, as_json=False
)
peyes.visualize.save_figure(
    fixation_criterion_figure, "channel_metrics-fixation-criterion", FIGURES_DIR,
    as_png=True, as_html=False, as_json=False
)
peyes.visualize.save_figure(
    saccade_dprime_figure, "channel_metrics-saccade-dprime", FIGURES_DIR,
    as_png=True, as_html=False, as_json=False
)
peyes.visualize.save_figure(
    saccade_criterion_figure, "channel_metrics-saccade-criterion", FIGURES_DIR,
    as_png=True, as_html=False, as_json=False
)
