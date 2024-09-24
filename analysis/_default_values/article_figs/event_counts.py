import pandas as pd
import plotly.io as pio
import plotly.express as px

import peyes
from analysis._default_values._helpers import *

pio.renderers.default = "browser"

######################
# Load Events Data

stim_trial_ids = u.get_trials_for_stimulus_type(DATASET_NAME, STIMULUS_TYPE)

all_events = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, DATASET_NAME, "events.pkl"))
all_events = all_events.xs(1, level=peyes.constants.ITERATION_STR, axis=1)  # Keep only the first iteration
all_events = all_events.loc[:, all_events.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(stim_trial_ids)]
all_events = all_events.dropna(axis=0, how="all")

######################
# Count Events by Labeler

all_labelers = all_events.columns.get_level_values(peyes.constants.LABELER_STR).unique()
events_by_labelers = {
    lblr: all_events.xs(lblr, level=peyes.constants.LABELER_STR, axis=1).stack().dropna() for lblr in all_labelers
}
event_counts_by_labelers = pd.DataFrame({
    lblr: evnts.apply(lambda e: e.label).value_counts() for lblr, evnts in events_by_labelers.items()
}).fillna(0).T.rename(columns=lambda l: peyes.parse_label(l).name)

del stim_trial_ids, all_events, all_labelers, events_by_labelers

######################
# Create Bar Plot

fig = px.bar(event_counts_by_labelers, title="Event Counts", text_auto=True)
fig.update_layout(
    title_x=0.5,
    xaxis_title=peyes.constants.LABELER_STR,
    yaxis_title=peyes.constants.COUNT_STR,
    width=1200, height=600,
    legend=dict(title="Event:", orientation="h", yanchor="top", y=1.07, xanchor="left", x=0.21)
)
fig.show()

######################
# Save Figure

peyes.visualize.save_figure(
    fig, "event_counts_per_labeler", FIGURES_DIR, as_png=True, as_html=False, as_json=False
)
