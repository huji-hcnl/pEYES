from typing import Union, List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.pEYES._DataModels.Event import EventSequenceType
from src.pEYES._DataModels.EventMatcher import EventMatchesType

import src.pEYES.constants as cnst
import src.pEYES.process as process

_DEFAULT_PARSE_PARAMS = {
    "time_column": cnst.T,
    "x_column": cnst.X,
    "y_column": cnst.Y,
    "pupil_column": cnst.PUPIL,
    "viewer_distance_column": cnst.VIEWER_DISTANCE_STR,
    "pixel_size_column": cnst.PIXEL_SIZE_STR,
    "missing_data": np.nan,
}


def preprocess(
        data: Union[str, np.ndarray, pd.DataFrame],
        detectors: Union[str, List[str]],
        human_raters: Union[str, List[str]] = None,
        group_trials_by: Union[str, List[str]] = cnst.TRIAL_ID_STR,
        match_to: str = None,
        parse_params: Dict[str, Any] = None,
        detection_params: dict = None,
        match_params: dict = None,
        verbose: bool = False,
):
    """

    :param data:
    :param detectors:
    :param human_raters:
    :param group_trials_by:
    :param match_to:
    :param parse_params:
        - time_column:
        - x_column:
        - y_column:
        - pupil_column:
        - viewer_distance_column:
        - pixel_size_column:
        - missing_data:
    :param detection_params:
    :param match_params:
    :param verbose:
    :return:
    """
    data = process.parse(data, **{**_DEFAULT_PARSE_PARAMS, **(parse_params or {})})

    human_raters = [] if human_raters is None else [human_raters] if isinstance(human_raters, str) else human_raters
    detection_params = {} or detection_params
    match_params = {} or match_params
    labels, metadata, events, matches = {}, {}, {}, {}
    for i, (idxs, trial_data) in tqdm(
            enumerate(data.groupby(group_trials_by)), desc="Preprocessing Trials", disable=not verbose
    ):
        if verbose:
            print(f"Trial {i + 1}:\tDetecting Events...")
        t = trial_data[cnst.T].values
        x = trial_data[cnst.X].values
        y = trial_data[cnst.Y].values
        p = trial_data[cnst.PUPIL].values
        vd = trial_data[cnst.VIEWER_DISTANCE_STR].iloc[0]
        ps = trial_data[cnst.PIXEL_SIZE_STR].iloc[0]

        trial_labels, trial_metadata, trial_events = _detect_trial(t, x, y, p, vd, ps, detectors, detection_params)
        for hr in human_raters:
            rater_events = process.create_events(
                labels=trial_data[hr], t=t, x=x, y=y, pupil=p, viewer_distance=vd, pixel_size=ps
            )
            trial_events[hr] = rater_events
        labels[i] = trial_labels
        metadata[i] = trial_metadata
        events[i] = trial_events

        if match_to:
            if verbose:
                print(f"Trial {i + 1}:\tMatching Events...")
            trial_matches = _match_trial(trial_events, match_to, match_params)
            matches[i] = trial_matches
    return labels, metadata, events, matches


def _data_to_frame(
        dataset: Union[str, np.ndarray, pd.DataFrame],
        directory: str = None,
        time_column: str = cnst.T,
        x_column: str = cnst.X,
        y_column: str = cnst.Y,
        pupil_column: str = cnst.PUPIL,
        viewer_distance_column: str = cnst.VIEWER_DISTANCE_STR,
        pixel_size_column: str = cnst.PIXEL_SIZE_STR,
        verbose: bool = False,
) -> pd.DataFrame:
    if isinstance(dataset, str):
        dataset = process.load_dataset(dataset, directory=directory, verbose=verbose)
    elif isinstance(dataset, np.ndarray):
        dataset = pd.DataFrame(dataset)
    elif isinstance(dataset, pd.DataFrame):
        dataset = dataset.copy(deep=True)   # avoid overwriting the original dataset
    else:
        raise TypeError(f"Invalid dataset type: {type(dataset)}")
    dataset.rename(
        columns={
            time_column: cnst.T, x_column: cnst.X, y_column: cnst.Y, pupil_column: cnst.PUPIL,
            viewer_distance_column: cnst.VIEWER_DISTANCE_STR, pixel_size_column: cnst.PIXEL_SIZE_STR
        },
        inplace=True
    )
    return dataset


def _detect_trial(
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        vd: float,
        ps: float,
        detectors: List[str],
        detection_params: dict,
) -> (Dict[str, np.ndarray], Dict[str, dict], Dict[str, EventSequenceType]):
    assert len(t) == len(x) == len(y) == len(p), "Input arrays must have the same length"
    labels, metadata, events = {}, {}, {}
    for det in detectors:
        detector_labels, detector_metadata = process.detect(
            t=t, x=x, y=y, viewer_distance=vd, pixel_size=ps, detector_name=det, include_metadata=True,
            **detection_params
        )
        detector_events = process.create_events(
            labels=detector_labels, t=t, x=x, y=y, pupil=p, viewer_distance=vd, pixel_size=ps
        )
        labels[det] = detector_labels
        metadata[det] = detector_metadata
        events[det] = detector_events
    return labels, metadata, events


def _match_trial(
        events,
        match_to: str,
        match_params: dict,
) -> Dict[str, EventMatchesType]:
    assert match_to in events.keys(), f"Invalid match_to value: {match_to}"
    gt_events = events[match_to]
    matches = {}
    for key in events.keys():
        if key == match_to:
            continue
        pred_events = events[key]
        match = process.match(gt_events, pred_events, **match_params)
        matches[key] = match
    return matches
