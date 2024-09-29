import os
import warnings

import pandas as pd
import plotly.io as pio

import peyes
import analysis.utils as u
from analysis._default_values._helpers import DATASET_NAME, STIMULUS_TYPE, PROCESSED_DATA_DIR, FIGURES_DIR, GT1

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
# Create Figures for All Labelers

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
        ).update_annotations(
            # move the "Azimuth" subtitle a bit up to avoid overlap with the 90° symbol
            y=0.4, selector={'text': peyes.constants.AZIMUTH_STR.title()}
        ) for lblr in all_labelers
    }

######################
# Show Figures Single Labeler

for lblr in all_labelers:
    if lblr != GT1:
        # avoid showing all labelers' figures to avoid cluttering the browser
        # TODO: remove this condition if you want to see all labelers' figures
        continue
    events_summary_figures[lblr].show()
    fixations_summary_figures[lblr].show()
    saccades_summary_figures[lblr].show()

######################
# Save Figures Single Labeler

for lblr in all_labelers:
    if lblr != GT1:
        # avoid saving all labelers' figures to avoid cluttering the filesystem
        # TODO: remove this condition if you want to save all labelers' figures
        continue
    peyes.visualize.save_figure(
        events_summary_figures[lblr],
        f"events_summary_{lblr}",
        FIGURES_DIR,
        as_png=True, as_html=False, as_json=False
    )
    peyes.visualize.save_figure(
        fixations_summary_figures[lblr],
        f"fixations_summary_{lblr}",
        FIGURES_DIR,
        as_png=True, as_html=False, as_json=False
    )
    peyes.visualize.save_figure(
        saccades_summary_figures[lblr],
        f"saccades_summary_{lblr}",
        FIGURES_DIR,
        as_png=True, as_html=False, as_json=False
    )
