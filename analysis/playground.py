import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import pEYES as peyes
import analysis.utils as u

from analysis.process.full_pipeline import run
import analysis.statistics.channel_sdt as csdt

pio.renderers.default = "browser"

################

DATASET_NAME = "lund2013"
GT1, GT2 = "RA", "MN"
MULTI_COMP = "fdr_bh"

sdt_metrics = csdt.load(
    dataset_name=DATASET_NAME,
    output_dir=os.path.join(u.OUTPUT_DIR, "default_values"),
    label=[1, 2],
    stimulus_type=peyes.constants.IMAGE_STR,
    channel_type=None
)

figs = csdt.multi_threshold_figures(sdt_metrics, "onset", show_err_bands=True)
figs[GT1].show()

onset_fa = sdt_metrics.xs(("onset", "false_alarm_rate"), level=(u.CHANNEL_TYPE_STR, peyes.constants.METRIC_STR), axis=0)

