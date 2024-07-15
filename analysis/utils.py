import os
import copy
import warnings
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

import src.pEYES as peyes
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum
from src.pEYES._DataModels.Detector import BaseDetector


CWD = os.getcwd()
DATASETS_DIR = os.path.join(CWD, "output", "datasets")

DATASET_ANNOTATORS = {
    "lund2013": ["RA", "MN"],
    "irf": ['RZ'],
    "hfc": ['DN', 'IH', 'JB', 'JF', 'JV', 'KH', 'MN', 'MS', 'PZ', 'RA', 'RH', 'TC']
}
DETECTOR_NAMES = ["ivt", "ivvt", "idt", "engbert", "nh", "remodnav"]
DEFAULT_DETECTORS = [
    peyes.create_detector(det, missing_value=np.nan, min_event_duration=4, pad_blinks_time=0) for det in DETECTOR_NAMES
]

###########################################


def load_dataset(dataset_name: str, verbose: bool = True) -> pd.DataFrame:
    if dataset_name == "lund2013":
        dataset = peyes.datasets.lund2013(directory=DATASETS_DIR, save=True, verbose=verbose)
    elif dataset_name == "irf":
        dataset = peyes.datasets.irf(directory=DATASETS_DIR, save=True, verbose=verbose)
    elif dataset_name == "hfc":
        dataset = peyes.datasets.hfc(directory=DATASETS_DIR, save=True, verbose=verbose)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset


def process_dataset(
        dataset: pd.DataFrame,
        detectors: List[BaseDetector],
        annotators: List[str],
        num_iterations: int = 4,
        overwrite_label: EventLabelEnum = EventLabelEnum.SACCADE,
        verbose: bool = True,
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels, events, metadata = dict(), dict(), dict()
        for tr in tqdm(dataset[peyes.TRIAL_ID_STR].unique(), desc="Trials", leave=True, disable=False, position=0):
            tr_labels, tr_events, tr_metadata = dict(), dict(), dict()
            trial = dataset[dataset[peyes.TRIAL_ID_STR] == tr]
            t = trial[peyes.T].values
            x = trial[peyes.X].values
            y = trial[peyes.Y].values
            p = trial[peyes.PUPIL].values
            vd = trial[peyes.VIEWER_DISTANCE_STR].iloc[0]
            ps = trial[peyes.PIXEL_SIZE_STR].iloc[0]

            for annot in tqdm(annotators, desc="\tAnnotators", leave=False, disable=not verbose, position=1):
                tr_labels[annot] = pd.Series(trial[annot].values).to_frame()
                annot_events = peyes.create_events(
                    labels=trial[annot].values, t=t, x=x, y=y, pupil=p, viewer_distance=vd, pixel_size=ps
                )
                tr_events[annot] = pd.Series(annot_events).to_frame()

            for det in tqdm(detectors, desc="\tDetectors", leave=False, disable=not verbose, position=1):
                det_labels, det_events, det_metadata = dict(), dict(), dict()
                det_name = det.name
                x_copy, y_copy, p_copy = copy.deepcopy(x), copy.deepcopy(y), copy.deepcopy(p)
                for it in trange(num_iterations, desc="\t\tIterations", leave=False, disable=not verbose, position=2):
                    it_labels, it_metadata = det.detect(t, x_copy, y_copy, viewer_distance_cm=vd, pixel_size_cm=ps)
                    it_events = peyes.create_events(
                        labels=it_labels, t=t, x=x_copy, y=y_copy, pupil=p_copy, viewer_distance=vd, pixel_size=ps
                    )
                    det_labels[it+1] = it_labels
                    det_events[it+1] = it_events
                    det_metadata[it+1] = it_metadata
                    to_overwrite = np.array(it_labels) == overwrite_label
                    x_copy[to_overwrite] = np.nan
                    y_copy[to_overwrite] = np.nan
                    p_copy[to_overwrite] = np.nan
                det_labels = pd.DataFrame.from_dict(det_labels, orient='index').T
                det_events = pd.DataFrame.from_dict(det_events, orient='index').T
                det_metadata = pd.DataFrame.from_dict(det_metadata, orient='index').T
                tr_labels[det_name] = det_labels
                tr_events[det_name] = det_events
                tr_metadata[det_name] = det_metadata

            tr_labels = pd.concat(tr_labels, axis=1)
            tr_events = pd.concat(tr_events, axis=1)
            tr_metadata = pd.concat(tr_metadata, axis=1)
            labels[tr] = tr_labels
            events[tr] = tr_events
            metadata[tr] = tr_metadata

        labels = pd.concat(labels, axis=0)
        labels.index.names = [peyes.TRIAL_ID_STR, peyes.SAMPLE_STR]
        events = pd.concat(events, axis=0)
        events.index.names = [peyes.TRIAL_ID_STR, f"{peyes.EVENT_STR}_id"]
        metadata = pd.concat(metadata, axis=0)
        metadata.index.names = [peyes.TRIAL_ID_STR, "field_name"]
        labels.columns.names = events.columns.names = metadata.columns.names = ["labeler", "iteration"]
        return labels, events, metadata
