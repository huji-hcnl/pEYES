import os
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

import src.pEYES as peyes
import src.pEYES._utils.constants as cnst
from src.pEYES._utils.event_utils import parse_label
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum

import analysis.utils as u

pio.renderers.default = "browser"

###################

dataset, labels, events, metadata = u.default_load_or_process("lund2013", verbose=False)

###################
