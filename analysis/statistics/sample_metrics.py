import os
from typing import List, Optional, Union

import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.statistics._helpers as h

pio.renderers.default = "browser"

###################


def get_sample_metrics(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        stimulus_type: Optional[Union[str, List[str]]] = None,
        metric: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    return h.get_data_impl(
        dataset_name=dataset_name,
        output_dir=output_dir,
        label=label,
        stimulus_type=stimulus_type,
        sub_index=metric,
        data_dir_name=f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}",
        filename_suffix="",
        filename_prefix="",
        iteration=1,
    )


###################
## EXAMPLE USAGE ##

GT1, GT2 = "RA", "MN"

sample_metrics = get_sample_metrics(
    "lund2013", os.path.join(u.OUTPUT_DIR, "default_values"), label=None, stimulus_type="image", metric=None
)
sm_statistics, sm_pvalues, sm_dunns, sm_Ns = h.statistical_analysis(sample_metrics, [GT1, GT2], multi_comp="fdr_bh")
sample_metrics_fig = h.distributions_figure(sample_metrics, GT1, gt2=GT2, title=f"Sample Metrics", only_box=False)
sample_metrics_fig.show()
