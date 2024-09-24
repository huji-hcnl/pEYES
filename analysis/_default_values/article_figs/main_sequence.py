import os

import pandas as pd
import plotly.io as pio

import peyes
import analysis.utils as u
from analysis._default_values._helpers import DATASET_NAME, STIMULUS_TYPE, PROCESSED_DATA_DIR, FIGURES_DIR, GT1

from peyes._DataModels.EventLabelEnum import EventLabelEnum

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
# Main Sequence for each Labeler

main_sequence_figures = {
    lblr: peyes.visualize.main_sequence(
        saccades=events_by_labelers[lblr][events_by_labelers[lblr].apply(lambda e: e.label == EventLabelEnum.SACCADE)],
        include_outliers=False,
    ) for lblr in all_labelers
}

######################
# Show Stat Results & Figure for Single Labeler

main_sequence_figures[GT1][0].show()
print(main_sequence_figures[GT1][1].iloc[0, 0].summary())    # print the statistical results for the inliers' m.s.

######################
# Save Figures
peyes.visualize.save_figure(
    main_sequence_figures[GT1][0], "main_sequence", FIGURES_DIR, as_png=True, as_html=False, as_json=False
)
