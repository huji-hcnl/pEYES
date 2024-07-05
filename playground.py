import os

import numpy as np
import pandas as pd

import src.pEYES as peyes

CWD = os.getcwd()

#########
dataset = peyes.datasets.lund2013(directory=os.path.join(CWD, "output", "datasets"), save=True, verbose=True)
trial1 = dataset[dataset[peyes.TRIAL_ID_STR] == 1]
t, x, y = trial1[peyes.T].values, trial1[peyes.X].values, trial1[peyes.Y].values
vd, ps = trial1[peyes.VIEWER_DISTANCE_STR].iloc[0], trial1[peyes.PIXEL_SIZE_STR].iloc[0]
pupil = trial1[peyes.PUPIL].values

ra_events = peyes.create_events(trial1['RA'].values, t, x, y, pupil=pupil, viewer_distance=vd, pixel_size=ps)
summary = peyes.summarize_events(ra_events)
aggregated = aggregate_events(ra_events)

####

engbert = peyes.create_detector(
    'Engbert', missing_value=np.nan, min_event_duration=5, pad_blinks_time=0
)
eng_labels, eng_metadata = engbert.detect(t, x, y, viewer_distance_cm=vd, pixel_size_cm=ps)
eng_events = peyes.create_events(eng_labels, t, x, y, pupil=pupil, viewer_distance=vd, pixel_size=ps)
table = peyes.summarize_events(eng_events)

nh = peyes.create_detector('NH', missing_value=np.nan, min_event_duration=5, pad_blinks_time=0)
nh_labels, nh_metadata = nh.detect(t, x, y, viewer_distance_cm=vd, pixel_size_cm=ps)
nh_events = peyes.create_events(nh_labels, t, x, y, pupil=pupil, viewer_distance=vd, pixel_size=ps)

matches = peyes.match(
    eng_events, nh_events, match_by='onset', allow_xmatch=False, verbose=True, max_onset_difference=20
)

#########

zzz = peyes.sample_metrics.precision(eng_labels, nh_labels, pos_labels=[peyes.Labels.FIXATION])

eng_durations = peyes.event_metrics.durations(eng_events)
nh_durations = peyes.event_metrics.durations(nh_events)
matches_durations = peyes.event_metrics.durations(matches)

cohen_kappa = peyes.sample_metrics.cohen_kappa(eng_labels, nh_labels)
nld = peyes.sample_metrics.complement_nld(eng_labels, nh_labels)

fix_dprime = peyes.match_metrics.d_prime(eng_events, nh_events, matches, positive_label=peyes.Labels.FIXATION)
