import pandas as pd
import plotly.io as pio
from patsy.util import widen

import peyes
import analysis.statistics.sample_metrics as sm
from analysis._default_values._helpers import *
from peyes._DataModels.EventLabelEnum import EventLabelEnum

pio.renderers.default = "browser"

# GLOBAL METRICS
######################

# Load Data
sample_global_metrics = sm.load_global_metrics(
    DATASET_NAME, PROCESSED_DATA_DIR, stimulus_type=STIMULUS_TYPE, metric=None
)
sample_global_metrics.drop(index=peyes.constants.ACCURACY_STR, inplace=True)    # Drop Accuracy metric

# Statistics: Kruskal-Wallis and Dunn's Post-Hoc
sm_global_statistics, sm_global_pvalues, sm_global_dunns, sm_global_Ns = sm.kruskal_wallis_dunns(
    sample_global_metrics, [GT1, GT2], multi_comp=MULTI_COMP
)

# Show Figure
sm_global_metrics_fig = sm.global_metrics_distributions_figure(sample_global_metrics, GT1, gt2=GT2, only_box=False)
sm_global_metrics_fig.update_layout(width=1600, height=550,)
sm_global_metrics_fig.show()

# Save Figure
peyes.visualize.save_figure(
    sm_global_metrics_fig, "sample_metrics-global", FIGURES_DIR, as_png=True, as_html=False, as_json=False
)

# SDT METRICS
######################

# Load Data for fixations & saccades
fixation_sdt = sm.load_sdt(
    DATASET_NAME, PROCESSED_DATA_DIR, label=EventLabelEnum.FIXATION.value, stimulus_type=STIMULUS_TYPE, metric=None
)
fixation_sdt = fixation_sdt.loc[[peyes.constants.D_PRIME_STR, peyes.constants.F1_STR]]  # Keep only d' and f1 metrics
fixation_sdt = fixation_sdt.rename(index=lambda idx: f"fixation_{idx}")     # Rename index

saccade_sdt = sm.load_sdt(
    DATASET_NAME, PROCESSED_DATA_DIR, label=EventLabelEnum.SACCADE.value, stimulus_type=STIMULUS_TYPE, metric=None
)
saccade_sdt = saccade_sdt.loc[[peyes.constants.D_PRIME_STR, peyes.constants.F1_STR]]  # Keep only d' and f1 metrics
saccade_sdt = saccade_sdt.rename(index=lambda idx: f"saccade_{idx}")     # Rename index

fixation_saccade_sdt = pd.concat([fixation_sdt, saccade_sdt], axis=0)
del fixation_sdt, saccade_sdt


# Statistics: Kruskal-Wallis and Dunn's Post-Hoc
sdt_statistics, sdt_pvalues, sdt_dunns, sdt_Ns = sm.kruskal_wallis_dunns(
    fixation_saccade_sdt, [GT1, GT2], multi_comp=MULTI_COMP
)

# Show Figures
sdt_metrics_fig = sm.sdt_distributions_figure(
    fixation_saccade_sdt, GT1, GT2, title="SDT Metrics (fixations & saccades)", only_box=False,
)
sdt_metrics_fig.update_layout(width=1600, height=550,)
sdt_metrics_fig.show()

# Save Figure
peyes.visualize.save_figure(
    sdt_metrics_fig, "sample_metrics-sdt-fixations&saccades", FIGURES_DIR, as_png=True, as_html=False, as_json=False
)
