import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import src.pEYES as peyes
import src.pEYES._utils.constants as cnst
from src.pEYES._utils.event_utils import parse_label

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

fig = event_summary(multi_trial_events, title="RA Event Summary", show_outliers=True)
fig.show()

fig.update_layout(violingroupgap=0.1)
fig.show()


#########

fig.add_trace(
    col=1, row=2,
    trace=go.Violin(x0=0, y=[np.nan], name='placeholder', side='positive')
)
fig.add_trace(
    col=1, row=2,
    trace=go.Violin(x0=0, y=[np.nan]*100, name='placeholder', side='negative')
)

fig.update_xaxes(
    col=1, row=2, title_text='Event Labels',
    type='category', categoryorder='array', categoryarray=list(range(6))
)

fig.update_layout(sharex=True)

fig.show()


#############

import plotly.graph_objects as go
from plotly.subplots import make_subplots

measure1 = {
    'a': [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
    'b': [1, 2, 1, 3, 1, 4, 1, 5, 1, 1],
    'c': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}
measure2 = {
    'a': [5, 4, 3, 2, 1, 1, 2, 3, 4, 5],
    'b': [2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    'c': []
}

fig = make_subplots(cols=1, rows=2, shared_xaxes=True)

for i, key in enumerate(measure1.keys()):
    fig.add_trace(
        col=1, row=1,
        trace=go.Violin(
            name=key, legendgroup=key, x0=key,
            y=measure1[key], showlegend=True, side='positive'
        )
    )
    fig.add_trace(
        col=1, row=2,
        trace=go.Violin(
            name=key, legendgroup=key, x0=key,
            y=measure2[key], showlegend=False, side=None
        )
    )

fig.show()


