import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

import src.pEYES as peyes
import src.pEYES._utils.constants as cnst
from src.pEYES._utils.event_utils import parse_label

CWD = os.getcwd()
pio.renderers.default = "browser"

#########

from src.pEYES.event_metrics import features_by_labels
from src.pEYES.visualize.todo_event_summary import event_summary, fixation_summary

dataset = peyes.datasets.lund2013(directory=os.path.join(CWD, "output", "datasets"), save=True, verbose=True)
multi_trial_events = []
for i in range(1, 21):
    trial = dataset[dataset[peyes.TRIAL_ID_STR] == i]
    ra_events = peyes.create_events(
        labels=trial['RA'].values,
        t=trial[peyes.T].values,
        x=trial[peyes.X].values,
        y=trial[peyes.Y].values,
        pupil=trial[peyes.PUPIL].values,
        viewer_distance=trial[peyes.VIEWER_DISTANCE_STR].iloc[0],
        pixel_size=trial[peyes.PIXEL_SIZE_STR].iloc[0]
    )
    multi_trial_events.extend(ra_events)
summary = peyes.summarize_events(multi_trial_events)
features = features_by_labels(multi_trial_events)
labels = [parse_label(l) for l in features.index]

# fig = event_summary(multi_trial_events, show_outliers=True)
# fig.show()

fix_fig = fixation_summary(multi_trial_events, show_outliers=True)
fix_fig.show()

del i, trial, ra_events

###############

num_bins = 16
half_bin = 360 / num_bins / 2
edges = np.linspace(0, 360, num_bins + 1, endpoint=True)
centers = (edges[1:] + edges[:-1]) / 2 - half_bin

fig = make_subplots(
    cols=6, rows=2, specs=[[{'type': 'polar'}] * 6, [{'colspan': 6}, None, None, None, None, None]],
    subplot_titles=[f"{l}" for l in labels] + ["Azimuth Distribution"]
)
for i in range(2):
    if i == 1:
        pass
    else:
        for j, evnt in enumerate(labels):
            azimuths = (np.array(features.loc[evnt, cnst.AZIMUTH_STR]) + half_bin) % 360
            counts, _ = np.histogram(azimuths, bins=edges)
            fig.add_trace(
                col=j + 1, row=i + 1,
                trace=go.Barpolar(
                    r=counts,
                )
            )
fig.show()
