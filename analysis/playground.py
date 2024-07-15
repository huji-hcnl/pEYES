import os
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

import src.pEYES as peyes
import src.pEYES._utils.constants as cnst
from src.pEYES._utils.event_utils import parse_label
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

import analysis.utils as u

pio.renderers.default = "browser"

###################

dataset = u.load_dataset("lund2013", verbose=True)
labels, events, metadata = u.process_dataset(
    dataset,
    detectors=u.DEFAULT_DETECTORS,
    annotators=u.DATASET_ANNOTATORS["lund2013"],
    num_iterations=4,
    overwrite_label=EventLabelEnum.SACCADE,
    verbose=False
)

###################

# dataset = peyes.datasets.lund2013(directory=u.DATASETS_DIR, save=False, verbose=True)
# multi_trial_events = []
# for i in range(1, 21):
#     trial = dataset[dataset[peyes.TRIAL_ID_STR] == i]
#     ra_events = peyes.create_events(
#         labels=trial['RA'].values,
#         t=trial[peyes.T].values,
#         x=trial[peyes.X].values,
#         y=trial[peyes.Y].values,
#         pupil=trial[peyes.PUPIL].values,
#         viewer_distance=trial[peyes.VIEWER_DISTANCE_STR].iloc[0],
#         pixel_size=trial[peyes.PIXEL_SIZE_STR].iloc[0]
#     )
#     multi_trial_events.extend(ra_events)
# summary = peyes.summarize_events(multi_trial_events)
# features = peyes.event_metrics.features_by_labels(multi_trial_events)
# labels = [parse_label(l) for l in features.index]
#
# del i, trial, ra_events

###################
