
__version__ = '0.1.0'  # TODO: update automatically

from src.pEYES._utils.constants import *

from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum as Labels
import src.pEYES._DataModels.config as config

from src.pEYES._base.parse import *
from src.pEYES._base.create import *
from src.pEYES._base.match import *
from src.pEYES._base.postprocess_events import *

import src.pEYES.datasets as datasets
import src.pEYES.event_metrics as event_metrics
import src.pEYES.sample_metrics as sample_metrics
import src.pEYES.match_metrics as match_metrics
import src.pEYES.visualization as visualization

