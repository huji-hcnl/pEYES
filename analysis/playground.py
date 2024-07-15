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

#########

dataset_name = "lund2013"
num_iterations = 4
verbose = True


def _load_dataset(dataset_name: str, verbose: bool = True) -> pd.DataFrame:
    if dataset_name == "lund2013":
        dataset = peyes.datasets.lund2013(directory=u.DATASETS_DIR, save=True, verbose=verbose)
    elif dataset_name == "irf":
        dataset = peyes.datasets.irf(directory=u.DATASETS_DIR, save=True, verbose=verbose)
    elif dataset_name == "hfc":
        dataset = peyes.datasets.hfc(directory=u.DATASETS_DIR, save=True, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset


def process_dataset(
        dataset_name: str, num_iterations: int = 4, verbose: bool = True
):
    dataset = _load_dataset(dataset_name, verbose=verbose)
    detectors = [peyes.create_detector(det, missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)
                 for det in u.DETECTOR_NAMES]
    labels, events, metadata = dict(), dict(), dict()
    for tr in tqdm(dataset[peyes.TRIAL_ID_STR].unique(), desc="Trials", leave=True, disable=not verbose):
        if tr == 2:
            break
        tr_labels, tr_events, tr_metadata = dict(), dict(), dict()
        trial = dataset[dataset[peyes.TRIAL_ID_STR] == tr]
        t = trial[peyes.T].values
        x = trial[peyes.X].values
        y = trial[peyes.Y].values
        p = trial[peyes.PUPIL].values
        vd = trial[peyes.VIEWER_DISTANCE_STR].iloc[0]
        ps = trial[peyes.PIXEL_SIZE_STR].iloc[0]

        for annot in tqdm(u.DATASET_ANNOTATORS[dataset_name], desc="\tAnnotators", leave=False, disable=not verbose):
            tr_labels[(annot, 1)] = pd.Series(trial[annot].values, name=(annot))
            tr_events[(annot, 1)] = peyes.create_events(
                labels=trial[annot].values, t=t, x=x, y=y, pupil=p, viewer_distance=vd, pixel_size=ps
            )

        for det in tqdm(detectors, desc="\tDetectors", leave=False, disable=not verbose):
            det_labels, det_events, det_metadata = dict(), dict(), dict()
            det_name = det.name
            x_copy, y_copy, p_copy = copy.deepcopy(x), copy.deepcopy(y), copy.deepcopy(p)
            for it in trange(num_iterations, desc="\t\tIterations", leave=False, disable=not verbose):
                it_labels, it_metadata = det.detect(t, x_copy, y_copy, viewer_distance_cm=vd, pixel_size_cm=ps)
                it_events = peyes.create_events(
                    labels=it_labels, t=t, x=x_copy, y=y_copy, pupil=p_copy, viewer_distance=vd, pixel_size=ps
                )
                det_labels[it+1] = it_labels
                det_events[it+1] = it_events
                det_metadata[it+1] = it_metadata
                x_copy[it_labels == EventLabelEnum.SACCADE] = np.nan
                y_copy[it_labels == EventLabelEnum.SACCADE] = np.nan
                p_copy[it_labels == EventLabelEnum.SACCADE] = np.nan
            det_labels = pd.DataFrame(det_labels)
            det_events = pd.DataFrame(det_events)
            det_metadata = pd.DataFrame(det_metadata)
            tr_labels[det_name] = det_labels
            tr_events[det_name] = det_events
            tr_metadata[det_name] = det_metadata

        tr_labels = pd.concat(tr_labels, axis=1).T
        tr_events = pd.concat(tr_events, axis=1).T
        tr_metadata = pd.concat(tr_metadata, axis=1).T

        labels[tr] = tr_labels
        events[tr] = tr_events
        metadata[tr] = tr_metadata


    return None



dataset = peyes.datasets.lund2013(directory=u.DATASETS_DIR, save=False, verbose=True)
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
features = peyes.event_metrics.features_by_labels(multi_trial_events)
labels = [parse_label(l) for l in features.index]

del i, trial, ra_events

#########
