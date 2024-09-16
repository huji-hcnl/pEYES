import os

import pandas as pd
import plotly.io as pio

import peyes
import analysis.utils as u
from analysis._default_values._helpers import DATASET_NAME, STIMULUS_TYPE, PROCESSED_DATA_DIR, FIGURES_DIR

pio.renderers.default = "browser"

######################
# Load Events Data

stim_trial_ids = u.get_trials_for_stimulus_type(DATASET_NAME, STIMULUS_TYPE)

all_events = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, DATASET_NAME, "events.pkl"))
all_events = all_events.xs(1, level=peyes.constants.ITERATION_STR, axis=1)
all_events = all_events.loc[:, all_events.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(stim_trial_ids)]
all_events = all_events.dropna(axis=0, how="all")

all_labelers = all_events.columns.get_level_values(peyes.constants.LABELER_STR).unique()
events_by_labelers = {
    lblr: all_events.xs(lblr, level=peyes.constants.LABELER_STR, axis=1).stack().dropna() for lblr in all_labelers
}

######################
# Fixation Features Comparison

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

######################
# Saccade Features Comparison

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

######################
# Save Figures

peyes.visualize.save_figure(
    fixations_comparison_figure, "features_comparison-fixation", FIGURES_DIR, as_png=True, as_html=False, as_json=False
)

peyes.visualize.save_figure(
    saccades_comparison_figure, "features_comparison-saccade", FIGURES_DIR, as_png=True, as_html=False, as_json=False
)
