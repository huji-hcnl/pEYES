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
from src.pEYES.visualize._event_summary import event_summary, fixation_summary, saccade_summary

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

fig = event_summary(multi_trial_events, show_outliers=True)
fig.show()

fix_fig = fixation_summary(multi_trial_events, show_outliers=True)
fix_fig.show()

sac_fig = saccade_summary(multi_trial_events, show_outliers=True)
sac_fig.show()

del i, trial, ra_events
