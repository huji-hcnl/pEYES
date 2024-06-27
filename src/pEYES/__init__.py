
__version__ = '0.1.0'  # TODO: update automatically

from src.pEYES._base_scripts.parse import parse
from src.pEYES._base_scripts.create import create_detector, create_events
from src.pEYES._base_scripts.match import match
from src.pEYES._base_scripts.postprocess_events import events_to_labels, summarize_events

import src.pEYES.config as config
import src.pEYES.constants as constants
import src.pEYES.datasets as datasets
import src.pEYES.event_metrics as event_metrics
import src.pEYES.sample_metrics as sample_metrics
import src.pEYES.match_metrics as match_metrics
import src.pEYES.visualization as visualization

