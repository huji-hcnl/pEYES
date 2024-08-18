import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import peyes
import analysis.utils as u
import analysis.statistics._helpers as h

from peyes._DataModels.EventLabelEnum import EventLabelEnum
import peyes._utils.visualization_utils as viz

pio.renderers.default = "browser"

######################

DATASET_NAME = "lund2013"
SAVED_DATA_DIR = os.path.join(u.OUTPUT_DIR, "default_values")
OUTPUT_DIR = r'Z:\jonathan.nir\ISTRC Scholarship'

LABEL = 1
STIMULUS_TYPE = peyes.constants.IMAGE_STR
GT1, GT2 = "RA", "MN"

# %%
###################
## Event Counts  ##

stim_trial_ids = u.get_trials_for_stimulus_type(DATASET_NAME, STIMULUS_TYPE)

all_events = pd.read_pickle(os.path.join(SAVED_DATA_DIR, DATASET_NAME, "events.pkl"))
all_events = all_events.xs(1, level=peyes.constants.ITERATION_STR, axis=1)
all_events = all_events.loc[:, all_events.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(stim_trial_ids)]
all_events = all_events.dropna(axis=0, how="all")

all_labelers = all_events.columns.get_level_values(peyes.constants.LABELER_STR).unique()
events_by_labelers = {
    lblr: all_events.xs(lblr, level=peyes.constants.LABELER_STR, axis=1).stack().dropna() for lblr in all_labelers
}
event_counts_by_labelers = pd.DataFrame({
    lblr: evnts.apply(lambda e: e.label).value_counts() for lblr, evnts in events_by_labelers.items()
}).fillna(0).T.rename(columns=lambda l: peyes.parse_label(l).name)

del stim_trial_ids, all_events, all_labelers, events_by_labelers

fig1 = px.bar(event_counts_by_labelers, title="Event Counts", text_auto=True)
fig1.update_layout(
    xaxis_title="Labeler",
    yaxis_title="Count",
    # legend_title="Event",
    width=1200, height=600,
    legend=dict(title="Event:", orientation="h", yanchor="top", y=1.07, xanchor="left", x=0.21)
)
fig1.show()

peyes.visualize.save_figure(fig1, "event_counts", OUTPUT_DIR, as_png=True)

# %%
######################
##  Sample Metrics  ##

import analysis.statistics.sample_metrics as sm

sample_metrics = pd.concat([
    sm.load_global_metrics(DATASET_NAME, SAVED_DATA_DIR, stimulus_type=STIMULUS_TYPE, metric=None),
    sm.load_sdt(DATASET_NAME, SAVED_DATA_DIR, label=LABEL, stimulus_type=STIMULUS_TYPE, metric=None)
], axis=0).loc[[peyes.constants.BALANCED_ACCURACY_STR, peyes.constants.COHENS_KAPPA_STR, peyes.constants.D_PRIME_STR]]
fig2 = h.distributions_figure(data=sample_metrics, gt1=GT1, gt2=GT2, title="Agreement Metrics (global)")
fig2.update_layout(
    yaxis2=dict(range=[0, 1]),
    width=1000, height=500,
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0.125)
)
fig2.show()

peyes.visualize.save_figure(fig2, "sample_metrics", OUTPUT_DIR, as_png=True)

# %%
######################
## Channel Metrics  ##

import analysis.statistics.channel_sdt as csdt

CHANNEL_TYPES = ["onset", "offset"]

sdt_metrics = csdt.load(
    dataset_name=DATASET_NAME,
    output_dir=SAVED_DATA_DIR,
    label=LABEL,
    stimulus_type=STIMULUS_TYPE,
    channel_type=None
).xs(peyes.constants.D_PRIME_STR, level=peyes.constants.METRIC_STR, axis=0)

fig3 = make_subplots(
    rows=2, cols=2, shared_xaxes="all", shared_yaxes="all", row_titles=CHANNEL_TYPES, column_titles=[GT1, GT2],
    vertical_spacing=0.025, horizontal_spacing=0.01
)

for i, gt in enumerate([GT1, GT2]):
    gt_data = sdt_metrics.xs(gt, level=u.GT_STR, axis=1)
    for j, ch_type in enumerate(CHANNEL_TYPES):
        ch_gt_data = gt_data.xs(ch_type, level=peyes.constants.CHANNEL_TYPE_STR, axis=0)
        detectors = sorted(
            ch_gt_data.columns.get_level_values(u.PRED_STR).unique(),
            key=lambda d: u.LABELERS_CONFIG[d.strip().lower().removesuffix("detector")][1]
        )
        for k, det in enumerate(detectors):
            data = ch_gt_data.xs(det, level=u.PRED_STR, axis=1)
            thresholds = data.index.get_level_values(peyes.constants.THRESHOLD_STR).unique()
            mean = data.mean(axis=1)
            sem = data.std(axis=1) / np.sqrt(data.count(axis=1))
            if det in [GT1, GT2]:
                det_name = "Other Human"
                det_color = "#bab0ac"
                dash = "dot"
            else:
                det_name = det.strip().removesuffix("Detector")
                det_color = u.LABELERS_CONFIG[det_name.lower()][2]
                dash = None
            fig3.add_trace(
                row=j + 1, col=i + 1, trace=go.Scatter(
                        x=thresholds, y=mean, error_y=dict(type="data", array=sem),
                        name=det_name, legendgroup=det_name,
                        mode="lines+markers",
                        marker=dict(size=5, color=det_color),
                        line=dict(color=det_color, dash=dash),
                        showlegend=(i == 0 and j == 0),
                    )
            )
        if j == 0:
            fig3.update_yaxes(title_text=r"$d'$", row=i+1, col=1)
        if i == 1:
            fig3.update_xaxes(title_text="Absolute Difference (samples)", row=2, col=j+1)

del i, gt, gt_data, j, ch_type, ch_gt_data, detectors, k, det, data, thresholds, mean, sem, det_name, det_color, dash


fig3.update_layout(
    title="Detection Sensitivity for Varying Differences",
    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0.25),
    width=1400, height=450,
)
fig3.show()

peyes.visualize.save_figure(fig3, "sensitivity_metrics", OUTPUT_DIR, as_png=True)

# %%

dataset = u.load_dataset(DATASET_NAME)
stim_trial_ids = u.get_trials_for_stimulus_type(DATASET_NAME, STIMULUS_TYPE)
stim_dataset = dataset[dataset[peyes.constants.TRIAL_ID_STR].isin(stim_trial_ids)]
stim_names = stim_dataset[peyes.constants.STIMULUS_NAME_STR].unique()
subj_ids = stim_dataset[peyes.constants.SUBJECT_ID_STR].unique()
is_na_trial = stim_dataset.groupby(peyes.constants.TRIAL_ID_STR)[[GT1, GT2]].apply(lambda c: c.isna().any())
trial_lengths = stim_dataset.groupby(peyes.constants.TRIAL_ID_STR).apply(lambda t: t[peyes.constants.T].max())
