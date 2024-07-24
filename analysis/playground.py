import os

import numpy as np
import pandas as pd
import plotly.io as pio

import pEYES as peyes
import analysis.utils as u
from analysis.pipeline.full_pipeline import run

pio.renderers.default = "browser"

###################

results = run(output_dir=os.path.join(u.OUTPUT_DIR, "default_values"), dataset_name="lund2013", verbose=False)
(dataset, labels, metadata, events, matches, sample_mets, time_diffs, channel_sdt_metrics, matched_features, matches_sdt_metrics) = results
del results

###################
