import os

import numpy as np
import cv2
from plotly.subplots import make_subplots
import plotly.io as pio

import peyes
import analysis.utils as u
from analysis._default_values._helpers import DATASET_NAME, FIGURES_DIR

pio.renderers.default = "browser"

######################

STIMULUS_TYPE = peyes.constants.IMAGE_STR
STIMULUS_DIR = os.path.join(u.BASE_DIR, "stimuli", DATASET_NAME.capitalize(), STIMULUS_TYPE)

######################

dataset = u.load_dataset(DATASET_NAME, verbose=False)
img_trial_id = u.get_trials_for_stimulus_type(DATASET_NAME, STIMULUS_TYPE)[5]
trial_data = dataset[dataset[peyes.constants.TRIAL_ID_STR] == img_trial_id]

img_name = trial_data[peyes.constants.STIMULUS_NAME_STR].values[0]
img_name += ".png"
img = cv2.imread(os.path.join(STIMULUS_DIR, img_name))
resolution = (img.shape[1], img.shape[0])

t = trial_data[peyes.constants.T].values
x = trial_data[peyes.constants.X].values
y = trial_data[peyes.constants.Y].values
pixel_size = trial_data[peyes.constants.PIXEL_SIZE_STR].values[0]
viewer_distance = trial_data[peyes.constants.VIEWER_DISTANCE_STR].values[0]

x[(x < 0) | (x >= resolution[0])] = np.nan
y[(y < 0) | (y >= resolution[1])] = np.nan

######################

t_fig = peyes.visualize.gaze_trajectory(
    x=x, y=y, resolution=resolution, title="Gaze Trajectory",
    bg_image=img, bg_image_format='rgb',
    t=t, colorscale='Jet'
)
t_fig.update_layout(
    title=None,
)
t_fig.show()


b_fig = peyes.visualize.gaze_over_time(
    x=x, y=y, t=t, resolution=resolution, title="Gaze Over Time",
    v=peyes._utils.pixel_utils.calculate_velocities(x, y, t), v_measure='px/s'
)
b_fig.update_layout(
    paper_bgcolor='rgba(0, 0, 0, 0)',
    plot_bgcolor='rgba(0, 0, 0, 0)',
    height=t_fig.layout.height // 2,
    width=t_fig.layout.width,
    title=None,
)
b_fig.show()

######################

peyes.visualize.save_figure(t_fig, "1A_gaze_trajectory", FIGURES_DIR, as_png=True, as_html=False, as_json=False)
peyes.visualize.save_figure(b_fig, "1B_gaze_over_time", FIGURES_DIR, as_png=True, as_html=False, as_json=False)

######################

# combined_fig = make_subplots(cols=1, rows=2, shared_xaxes=False, shared_yaxes=False)
#
# [combined_fig.add_trace(fig_data, row=1, col=1) for fig_data in t_fig.data]
# [combined_fig.add_trace(fig_data, row=2, col=1) for fig_data in b_fig.data]
# combined_fig.show()
