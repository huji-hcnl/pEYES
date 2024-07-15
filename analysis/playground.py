import os
import copy
import warnings
from typing import List, Dict, Union, Optional

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
from src.pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.preprocess as pp

pio.renderers.default = "browser"

###################

dataset, labels, events, metadata = pp.run_default("lund2013", verbose=False)

###################

gt_labelers = u.DATASET_ANNOTATORS["lund2013"]
pred_labelers = events.columns.get_level_values(level=u.LABELER_STR).unique()
matching_schemes = {
    'onset': dict(max_onset_difference=15),
    'offset': dict(max_offset_difference=15),
    'window': dict(max_onset_difference=15, max_offset_difference=15),
    'l2': dict(max_l2=15),
    'iou': dict(min_iou=1/3),
    'max_overlap': dict(min_overlap=0.5),
}

results = dict()
for tr in tqdm(events.index.get_level_values(level=peyes.TRIAL_ID_STR).unique(), desc="Trials"):
    for gt_labeler in gt_labelers:
        gt_events = events.loc[tr, gt_labeler].dropna().values.flatten()
        if gt_events.size == 0:
            continue
        for pred_labeler in pred_labelers:
            if gt_labeler == pred_labeler:
                continue
            pred_events = events.loc[tr, pred_labeler].dropna().values.flatten()
            if pred_events.size == 0:
                continue
            for match_by, kwargs in matching_schemes.items():
                matches = peyes.match(gt_events, pred_events, match_by, allow_xmatch=False, **kwargs)
                results[(tr, gt_labeler, pred_labeler, match_by)] = matches

# TODO: rewrite this as a function in `preprocess.py` and use it in the pipeline
