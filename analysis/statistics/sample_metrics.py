import os
from typing import List, Optional, Union, Tuple

import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.statistics._helpers as h

pio.renderers.default = "browser"

###################


def load(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        stimulus_type: Optional[Union[str, List[str]]] = None,
        metric: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    return h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}", label=label,
        iteration=1, stimulus_type=stimulus_type, sub_index=metric
    )


def stats(
        data: pd.DataFrame,
        gt_cols: List[str],
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return h.statistical_analysis(data, gt_cols, multi_comp)


def distributions_figure(
        data: pd.DataFrame,
        gt1: str,
        gt2: str,
        title: str = "Samples :: Metric Distributions",
        only_box: bool = False,
) -> go.Figure:
    return h.distributions_figure(data, gt1=gt1, gt2=gt2, title=title, only_box=only_box)


###################

DATASET_NAME = "lund2013"
GT1, GT2 = "RA", "MN"
MULTI_COMP = "fdr_bh"

####################
## Sample Metrics ##

sample_metrics = load("lund2013", os.path.join(u.OUTPUT_DIR, "default_values"), label=None, stimulus_type="image",
                      metric=None)
sm_statistics, sm_pvalues, sm_dunns, sm_Ns = stats(sample_metrics, [GT1, GT2], multi_comp=MULTI_COMP)
sample_metrics_fig = distributions_figure(sample_metrics, GT1, gt2=GT2, only_box=False)
sample_metrics_fig.show()
