import os
from typing import Optional, Union, Tuple, Sequence, List

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

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
        stimulus_type: Optional[Union[str, Sequence[str]]] = None,
        channel_type: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    return h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{peyes.constants.SAMPLES_STR}_{u.CHANNEL_STR}", label=label,
        filename_suffix="timing_differences", iteration=1, stimulus_type=stimulus_type,
        sub_index=channel_type
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
        title: str = "Samples Channel :: Difference Distributions",
        only_box: bool = False,
) -> go.Figure:
    return h.distributions_figure(data, gt1=gt1, gt2=gt2, title=title, only_box=only_box)


##################

DATASET_NAME = "lund2013"
GT1, GT2 = "RA", "MN"
MULTI_COMP = "fdr_bh"

##################
##  Time Diffs  ##

time_diffs = load(
    DATASET_NAME, os.path.join(u.OUTPUT_DIR, "default_values"), label=None, stimulus_type=peyes.constants.IMAGE_STR
)
statistics, pvalues, dunns, Ns = h.statistical_analysis(time_diffs, ["RA", "MN"], multi_comp=MULTI_COMP)
time_diffs_fig = distributions_figure(time_diffs, GT1, gt2=GT2, only_box=False)
time_diffs_fig.show()
