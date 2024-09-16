import os

import numpy as np
import pandas as pd
import cv2
import plotly.io as pio

import peyes
import analysis.utils as u
from analysis._default_values._helpers import DATASET_NAME, PROCESSED_DATA_DIR, FIGURES_DIR

pio.renderers.default = "browser"

STIMULUS_TYPE = peyes.constants.IMAGE_STR
STIMULUS_DIR = os.path.join(u.BASE_DIR, "stimuli", DATASET_NAME.capitalize(), STIMULUS_TYPE)

######################
# Load Raw Data

dataset = u.load_dataset(DATASET_NAME, verbose=False)
img_trial_id = u.get_trials_for_stimulus_type(DATASET_NAME, STIMULUS_TYPE)[5]
trial_data = dataset[dataset[peyes.constants.TRIAL_ID_STR] == img_trial_id]

img_name = trial_data[peyes.constants.STIMULUS_NAME_STR].values[0]
img_name += ".png"
img = cv2.imread(os.path.join(STIMULUS_DIR, img_name))
resolution = (img.shape[1], img.shape[0])

pixel_size = trial_data[peyes.constants.PIXEL_SIZE_STR].values[0]
viewer_distance = trial_data[peyes.constants.VIEWER_DISTANCE_STR].values[0]
t = trial_data[peyes.constants.T].values
x = trial_data[peyes.constants.X].values
y = trial_data[peyes.constants.Y].values

x[(x < 0) | (x >= resolution[0])] = np.nan
y[(y < 0) | (y >= resolution[1])] = np.nan

######################
# Load Labels

labels_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, DATASET_NAME, peyes.constants.LABELS_STR + ".pkl"))
labels_df = labels_df.xs(1, level=peyes.constants.ITERATION_STR, axis=1)  # Only use first iteration
labels_df = labels_df.xs(img_trial_id, level=peyes.constants.TRIAL_ID_STR, axis=1)    # Only use the relevant trial
labels_df = labels_df.dropna(how="all", axis=0)   # Drop rows with all NaNs

labeler_names = sorted(
    labels_df.columns.get_level_values(peyes.constants.LABELER_STR).unique(),
    key=lambda d: u.LABELERS_CONFIG[d.strip().lower().removesuffix("detector")][1]
)

######################

top_fig = peyes.visualize.gaze_trajectory(
    x=x, y=y, resolution=resolution, title="Gaze Trajectory",
    bg_image=img, bg_image_format='rgb',
    t=t, colorscale='Jet'
)
top_fig.update_layout(
    title=None,
)
top_fig.show()


middle_fig = peyes.visualize.gaze_over_time(
    x=x, y=y, t=t, resolution=resolution, title="Gaze Over Time",
    v=peyes._utils.pixel_utils.calculate_velocities(x, y, t), v_measure='px/s'
)
middle_fig.update_layout(
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    height=top_fig.layout.height // 2,
    width=top_fig.layout.width,
    title=None,
)
middle_fig.show()

bottom_fig = peyes.visualize.scarfplot_comparison_figure(
    t,
    *[labels_df[labeler_name] for labeler_name in labeler_names],
    names=labeler_names,
)
bottom_fig.update_layout(
    height=top_fig.layout.height // 2,
    width=top_fig.layout.width,
    title=None,
)
bottom_fig.show()

######################
# Save Figures

peyes.visualize.save_figure(top_fig, "single_tria-gaze_trajectory", FIGURES_DIR, as_png=True, as_html=False, as_json=False)
peyes.visualize.save_figure(middle_fig, "single_tria-gaze_over_time", FIGURES_DIR, as_png=True, as_html=False, as_json=False)
peyes.visualize.save_figure(bottom_fig, "single_tria-scarfplot", FIGURES_DIR, as_png=True, as_html=False, as_json=False)
