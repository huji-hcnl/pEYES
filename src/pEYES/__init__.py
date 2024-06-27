
__version__ = '0.1.0'  # TODO: update automatically

import src.pEYES._base.config as config
from src.pEYES._base.parse import parse
from src.pEYES._base.create import create_detector, create_events
from src.pEYES._base.match import match
from src.pEYES._base.postprocess_events import events_to_labels, summarize_events

import src.pEYES.datasets as datasets
import src.pEYES.event_metrics as event_metrics
import src.pEYES.sample_metrics as sample_metrics
import src.pEYES.match_metrics as match_metrics
import src.pEYES.visualization as visualization

