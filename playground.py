import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import src.pEYES as peyes

CWD = os.getcwd()
pio.renderers.default = "browser"

#########

from src.pEYES.event_metrics import features_by_labels
from src.pEYES.visualize.todo_event_summary import event_summary

dataset = peyes.datasets.lund2013(directory=os.path.join(CWD, "output", "datasets"), save=True, verbose=True)
multi_trial_events = []
for i in range(1, 11):
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

fig = event_summary(multi_trial_events, title="RA Event Summary")
fig.show()

