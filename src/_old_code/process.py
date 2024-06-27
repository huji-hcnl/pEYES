from typing import Union, List

import numpy as np
import pandas as pd

import src.pEYES.constants as cnst
import src.pEYES.config as cnfg
import src.pEYES.process as preprocess
import src.pEYES.datasets as datasets


def process_trial(
        trial_data: Union[np.ndarray, pd.DataFrame],
        detectors: Union[str, list[str]],
        annotators: Union[str, list[str]] = None,
        verbose: bool = False,
        **kwargs,
):
    t, x, y, pupil, v_d, px_s = _extract_raw_data(trial_data, **kwargs)
    labels, metadata, events = {}, {}, {}

    # algorithm detection
    detectors = [detectors] if isinstance(detectors, str) else detectors
    for det in detectors:
        det_obj = preprocess.create_detector(
            det,
            missing_value=kwargs.get("missing_data", np.nan),
            min_event_duration=kwargs.get("min_event_duration", cnfg.MIN_EVENT_DURATION),
            pad_blinks_time=kwargs.get("pad_blinks_time", 0),
            **kwargs
        )
        det_labels, det_metadata = det_obj.detect(t, x, y, viewer_distance_cm=v_d, pixel_size_cm=px_s)
        det_events = preprocess.create_events(det_labels, t, x, y, pupil, v_d, px_s)
        labels[det_obj.name], metadata[det_obj.name], events[det_obj.name] = det_labels, det_metadata, det_events

    # process human annotations
    annotators = [] if annotators is None else [annotators] if isinstance(annotators, str) else annotators
    for annotator in annotators:
        ann_labels = trial_data[annotator] if isinstance(trial_data, np.ndarray) else trial_data[annotator].values
        ann_events = preprocess.create_events(ann_labels, t, x, y, pupil, v_d, px_s)
        labels[annotator], events[annotator] = ann_labels, ann_events

    # todo: match



    return None


def _extract_raw_data(
        trial_data: Union[np.ndarray, pd.DataFrame],
        **kwargs,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float):
    trial_data = preprocess.parse(trial_data, time_name=kwargs.get("time_name", cnst.T),
                                  x_name=kwargs.get("x_name", cnst.X), y_name=kwargs.get("y_name", cnst.Y),
                                  pupil_name=kwargs.get("pupil_name", cnst.PUPIL),
                                  viewer_distance_name=kwargs.get("viewer_distance_name", cnst.VIEWER_DISTANCE_STR),
                                  pixel_size_name=kwargs.get("pixel_size_name", cnst.PIXEL_SIZE_STR),
                                  missing_data_value=kwargs.get("missing_data", np.nan))
    t, x, y, pupil = trial_data[cnst.T], trial_data[cnst.X], trial_data[cnst.Y], trial_data[cnst.PUPIL]
    viewer_distance = trial_data[cnst.VIEWER_DISTANCE_STR].iloc[0]
    pixel_size = trial_data[cnst.PIXEL_SIZE_STR].iloc[0]
    return t, x, y, pupil, viewer_distance, pixel_size


