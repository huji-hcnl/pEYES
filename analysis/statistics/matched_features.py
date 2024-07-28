import os
from typing import Optional, Union, Tuple, Sequence, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.statistics._helpers as h

pio.renderers.default = "browser"

########################


def load(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        stimulus_type: Optional[Union[str, Sequence[str]]] = None,
        matching_schemes: Optional[Union[str, Sequence[str]]] = None,
        features: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    if matching_schemes is None or len(matching_schemes) == 0:
        matching_schemes = None
    elif isinstance(matching_schemes, str):
        matching_schemes = [matching_schemes]
    matched_features = h.load_data(
        dataset_name=dataset_name, output_dir=output_dir,
        data_dir_name=f"{u.MATCHES_STR}_{peyes.constants.METRICS_STR}",
        label=label, stimulus_type=stimulus_type, sub_index=matching_schemes,
        filename_suffix=f"matched_{peyes.constants.FEATURES_STR}", iteration=1,
    )
    if features is None or len(features) == 0:
        return matched_features
    if isinstance(features, str):
        features = [features]
    is_features = matched_features.index.get_level_values(peyes.constants.FEATURE_STR).isin(features)
    return matched_features.loc[is_features]


def kruskal_wallis_dunns(
        matched_features: pd.DataFrame,
        matching_scheme: str,
        gt_cols: Sequence[str],
        features: Union[str, Sequence[str]] = None,
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each unique value in the input DataFrame's index (feature names) and each of the GT labelers, performs
    Kruskal-Wallis test with post-hoc Dunn's test for multiple comparisons.
    Returns the KW-statistic, KW-p-value, Dunn's-p-values and number of samples for each (index, GT labeler) pair.
    """
    subframe = h.extract_subframe(
        matched_features, level=u.MATCHING_SCHEME_STR, value=matching_scheme, axis=0, drop_single_values=True
    )
    if features:
        subframe = h.extract_subframe(
            subframe, level=peyes.constants.FEATURE_STR, value=features, axis=0, drop_single_values=False
        )
    statistics, pvalues, dunns, Ns = h.kruskal_wallis_dunns(subframe, gt_cols, multi_comp)
    return statistics, pvalues, dunns, Ns


def distributions_figure(
        matched_features: pd.DataFrame,
        matching_scheme: str,
        gt1: str,
        gt2: str,
        features: Union[str, Sequence[str]] = None,
        title: Optional[str] = None,
) -> go.Figure:
    subframe = h.extract_subframe(matched_features, level=u.MATCHING_SCHEME_STR, value=matching_scheme, axis=0,
                                  drop_single_values=True)
    if features:
        subframe = h.extract_subframe(subframe, level=peyes.constants.FEATURE_STR, value=features, axis=0,
                                      drop_single_values=False)
    title = title if title else (
            "Matched Events :: Features <br>" + f"<sup>(Matching Scheme: {matching_scheme})</sup>"
    )
    fig = h.distributions_figure(subframe, gt1=gt1, gt2=gt2, title=title)
    return fig


########################

DATASET_NAME = "lund2013"
GT1, GT2 = "RA", "MN"
MULTI_COMP = "fdr_bh"

########################
##  Matched Features  ##

matched_features = load(
    dataset_name=DATASET_NAME,
    output_dir=os.path.join(u.OUTPUT_DIR, "default_values"),
    label=None,
    stimulus_type=peyes.constants.IMAGE_STR,
    matching_schemes=None,
)

statistics, pvalues, dunns, Ns = kruskal_wallis_dunns(matched_features, "window", [GT1, GT2])
fig = distributions_figure(matched_features, "window", GT1, GT2)
fig.show()


###################
##  Matched SDT  ##
# TODO:
# matched_sdt_metrics = pd.read_pickle(os.path.join(u.OUTPUT_DIR, "default_values", "lund2013", "matches_metrics", "fixation_sdt_metrics.pkl"))
# iter1_matched_sdt_metrics = matched_sdt_metrics.xs(1, level=peyes.constants.ITERATION_STR, axis=1)
# image_iter1_matched_sdt_metrics = iter1_matched_sdt_metrics.loc[:, iter1_matched_sdt_metrics.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(image_trials)]
