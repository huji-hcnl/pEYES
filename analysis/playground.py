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

dataset, labels, events, metadata = u.default_load_or_process("lund2013", verbose=False)

###################

for tr in labels.index.get_level_values(level=peyes.TRIAL_ID_STR):
    for gt_labeler in u.DATASET_ANNOTATORS['lund2013']:
        gt_labels = labels.loc[tr, gt_labeler].dropna().values.flatten()
        for other_labeler in labels.columns.get_level_values(u.LABELER_STR).unique():
            if other_labeler == gt_labeler or other_labeler == "MN":
                continue
            other_labels_df = labels.loc[tr, other_labeler]
            for it in other_labels_df.columns.get_level_values(peyes.ITERATION_STR).unique():
                other_labels = other_labels_df[it].dropna().values.flatten()
                res = peyes.sample_metrics.calculate(gt_labels, other_labels)
                break
            break
        break
    break
