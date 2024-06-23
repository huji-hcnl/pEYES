import os

import numpy as np

import src.pEYES as pEYES

CWD = os.getcwd()

#########
dataset = pEYES.process.load_dataset(
    "lund2013", directory=os.path.join(CWD, "output", "datasets"), save=True, verbose=True
)
trial1 = dataset[dataset[pEYES.constants.TRIAL_ID_STR] == 1]

#########
detectors = ["IVT", "IVVT", "IDT", "ENGBERT", "NH", "REMODNAV"]
detection_results = {
    detector: pEYES.process.detect(
        t=trial1[pEYES.constants.T].values,
        x=trial1[pEYES.constants.X].values,
        y=trial1[pEYES.constants.Y].values,
        viewer_distance=trial1[pEYES.constants.VIEWER_DISTANCE_STR].iloc[0],
        pixel_size=trial1[pEYES.constants.PIXEL_SIZE_STR].iloc[0],
        detector_name=detector,
        include_metadata=True,
    ) for detector in detectors
}

#########
events = pEYES.process.create_events(labels=detection_results["ENGBERT"][0], t=trial1[pEYES.constants.T].values,
                                     x=trial1[pEYES.constants.X].values, y=trial1[pEYES.constants.Y].values,
                                     pupil=np.full_like(trial1[pEYES.constants.T].values, np.nan),
                                     viewer_distance=trial1[pEYES.constants.VIEWER_DISTANCE_STR].iloc[0],
                                     pixel_size=trial1[pEYES.constants.PIXEL_SIZE_STR].iloc[0])
labels = pEYES.process.events_to_labels(events, sampling_rate=500)

#########
ra_events = pEYES.process.create_events(labels=trial1["RA"], t=trial1[pEYES.constants.T].values,
                                        x=trial1[pEYES.constants.X].values, y=trial1[pEYES.constants.Y].values,
                                        pupil=np.full_like(trial1[pEYES.constants.T].values, np.nan),
                                        viewer_distance=trial1[pEYES.constants.VIEWER_DISTANCE_STR].iloc[0],
                                        pixel_size=trial1[pEYES.constants.PIXEL_SIZE_STR].iloc[0])

#########
matches = pEYES.process.match(events, ra_events, "onset", allow_xmatch=False, max_onset_difference=200)

#########
event_durations = pEYES.event_metrics.durations(events)
sample_kappa = pEYES.sample_metrics.cohen_kappa(trial1["RA"], detection_results["ENGBERT"][0])
dprime = pEYES.match_metrics.d_prime(ra_events, events, matches, 2)
p, r, f1 = pEYES.match_metrics.precision_recall_f1(ra_events, events, matches, 2)
