
__version__ = '0.1.0'  # TODO: update automatically

import src.pEYES.constants as constants
import src.pEYES.config as config

from src.pEYES.load_dataset import load_dataset
from src.pEYES.detect import detect, detect_multiple

import src.pEYES.events as events
import src.pEYES.event_metrics as event_metrics
import src.pEYES.sample_metrics as sample_metrics
import src.pEYES.match_metrics as match_metrics
