import os

import numpy as np
import plotly.graph_objects as go

import peyes

import analysis.utils as u

DATASET_NAME = "hfc"
STIMULUS_TYPE = "free_viewing"
GT1 = "IH"      # 24 years of experience; lab "Exp Psy Utrecht"
GT2 = "DN"      # 10 years of experience; lab "Humlab Lund"
GT3 = "JV"      # 10 years of experience; lab "Exp Psy Utrecht"
GT4, GT5 = "RA", "MN"   # annotators of the lund2013 dataset; lab "Humlab Lund"
GT1, GT2 = GT4, GT5     # for now, use the same annotators as in lund2013 dataset   # TODO: remove this line

MULTI_COMP = "bonferroni"   # method for multiple comparisons correction: bonferroni, fdr_bh, holm, etc.
ALPHA, MARGINAL_ALPHA = 0.05, 0.075

_ARTICLE_RESULTS_STR = "article_results"
_FIGURES_STR = "figures"
PROCESSED_DATA_DIR = os.path.join(u.OUTPUT_DIR, _ARTICLE_RESULTS_STR)
FIGURES_DIR = os.path.join(u.OUTPUT_DIR, _ARTICLE_RESULTS_STR, DATASET_NAME, _FIGURES_STR)
os.makedirs(FIGURES_DIR, exist_ok=True)

_default_detector_params = dict(missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)
DETECTORS = {
    "ivt": peyes.create_detector(
        algorithm="ivt", saccade_velocity_threshold=45, **_default_detector_params
    ),
    "ivvt": peyes.create_detector(
        algorithm="ivvt", saccade_velocity_threshold=45, smooth_pursuit_velocity_threshold=26, **_default_detector_params
    ),
    "idt": peyes.create_detector(
        algorithm="idt", dispersion_threshold=2.7, **_default_detector_params
    ),
    # "idt_salvucci": peyes.create_detector(        # Salvucci & Goldberg (2000)
    #     algorithm="idt", dispersion_threshold=1, window_duration=100, **_default_detector_params
    # ),
    "idvt": peyes.create_detector(
        algorithm="idvt", dispersion_threshold=2.7, **_default_detector_params
    ),
    # "idvt_komogortsev": peyes.create_detector(    # Komogortsev & Karpov (2013)
    #     algorithm="idvt",
    #     saccade_velocity_threshold=45,
    #     dispersion_threshold=2,
    #     window_duration=110,
    #     **_default_detector_params
    # ),
    "engbert": peyes.create_detector(
        algorithm="engbert", lambda_param=6, **_default_detector_params
    ),
    "nh": peyes.create_detector(
        algorithm="nh", **_default_detector_params
    ),
    "remodnav": peyes.create_detector(
        algorithm="remodnav", show_warnings=False, **_default_detector_params
    ),
}
for key, detector in DETECTORS.items():
    detector.name = key


_MATCH_BY_STR = "match_by"
MATCHING_SCHEMES = {
    # 'iou': {_MATCH_BY_STR: 'iou', 'min_iou': 1/3},
    # 'max_overlap': {_MATCH_BY_STR: 'max_overlap', 'min_overlap': 0.5},
    # 'onset': {_MATCH_BY_STR: 'onset', 'max_onset_difference': 15},
    # 'offset': {_MATCH_BY_STR: 'offset', 'max_offset_difference': 15},
    'l2': {_MATCH_BY_STR: 'l2', 'max_l2': 15},
    # 'window': {_MATCH_BY_STR: 'window', 'max_onset_difference': 15, 'max_offset_difference': 15},
    **{f"window_{w}": {
        _MATCH_BY_STR: 'window', 'max_onset_difference': w, 'max_offset_difference': w} for w in np.arange(0, 21, 5)
    },
    # TODO: consider re-writing this to have `threshold` another argument
    # **{f"onset_{o}": {_MATCH_BY_STR: 'onset', 'max_onset_difference': o} for o in np.arange(21)},
    # **{f"iou_{iou:.1f}": {_MATCH_BY_STR: 'iou', 'min_iou': iou} for iou in np.arange(0.1, 1.01, 0.1)},
    # **{f"overlap_{ov:.1f}": {_MATCH_BY_STR: 'max_overlap', 'min_overlap': ov} for ov in np.arange(0.1, 1.01, 0.1)},
}

## FIGURE CONFIG ##
LABELER_PLOTTING_CONFIG = {
    # labeler -> (order, color, line-style)
    'Other Human': (0, "#bab0ac", 'dot'),
    'RA': (1, u.DEFAULT_DISCRETE_COLORMAP[len(DETECTORS) + 1], 'dot'),
    'MN': (2, u.DEFAULT_DISCRETE_COLORMAP[len(DETECTORS) + 2], 'dot'),
    "IH": (1, u.DEFAULT_DISCRETE_COLORMAP[len(DETECTORS) + 3], 'dot'),
    "DN": (2, u.DEFAULT_DISCRETE_COLORMAP[len(DETECTORS) + 4], 'dot'),
    "JV": (3, u.DEFAULT_DISCRETE_COLORMAP[len(DETECTORS) + 5], 'dot'),
    **{key: (i+3 ,u.DEFAULT_DISCRETE_COLORMAP[i], None) for i, key in enumerate(DETECTORS.keys())}
}


def save_fig(fig: go.Figure, fig_id: int, panel_id: str, panel_name: str, is_supp: bool):
    fig_path = os.path.join(FIGURES_DIR, f"{'supp-' if is_supp else ''}fig{fig_id}")
    os.makedirs(fig_path, exist_ok=True)
    file_name = f"{panel_id}_{panel_name}"
    peyes.visualize.save_figure(fig, file_name, fig_path, as_png=True, as_eps=False, as_json=True)
