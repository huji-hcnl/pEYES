from typing import Optional, Union, Tuple, Sequence

import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as multi
import plotly.graph_objects as go

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u
import analysis.statistics._helpers as h

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
        data_dir_name=f"{peyes.constants.MATCHES_STR}_{peyes.constants.METRICS_STR}",
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
        gt_cols: Union[str, Sequence[str]],
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


def wilcoxon_signed_rank(
        matched_features: pd.DataFrame,
        matching_scheme: str,
        gt_col: str,
        features: Union[str, Sequence[str]] = None,
        multi_comp: str = "fdr_bh",
        alpha: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert 0 < alpha < 1, "Argument `alpha` must be in the open interval (0, 1)"
    scheme_matched_features = h.extract_subframe(
        matched_features, level=u.MATCHING_SCHEME_STR, value=matching_scheme, axis=0, drop_single_values=True
    )
    if features:
        scheme_matched_features = h.extract_subframe(
            scheme_matched_features, level=peyes.constants.FEATURE_STR, value=features, axis=0, drop_single_values=False
        )
    features = sorted(
        scheme_matched_features.index.unique(),
        key=lambda met: u.METRICS_CONFIG[met][1] if met in u.METRICS_CONFIG else ord(met[0])
    )
    gt_columns = scheme_matched_features.columns.get_level_values(u.GT_STR).unique()
    statistics, pvalues, Ns = {}, {}, {}
    for i, feat in enumerate(features):
        alternative = 'two-sided' if peyes.constants.DIFFERENCE_STR in feat else 'greater'
        gt_series = scheme_matched_features.xs(gt_col, level=u.GT_STR, axis=1).loc[feat]
        gt_df = gt_series.unstack().drop(columns=gt_columns, errors='ignore')
        for j, det in enumerate(gt_df.columns):
            vals = gt_df[det].explode().dropna().values.astype(float)
            if feat in [f"{peyes.constants.TIME_STR}_overlap", f"{peyes.constants.TIME_STR}_iou"]:
                # for IoU & (normalized) Overlap, the best values are 1, and Wilcoxon's needs to compare against 0
                vals = 1 - vals
            statistic, pvalue = stats.wilcoxon(x=vals, alternative=alternative, zero_method='zsplit')
            statistics[(feat, det)] = statistic
            pvalues[(feat, det)] = pvalue
            Ns[(feat, det)] = len(vals)
    statistics = pd.Series(statistics).unstack()
    pvalues = pd.Series(pvalues).unstack()
    Ns = pd.Series(Ns).unstack()
    corrected_pvalues = pd.DataFrame(
        index=pvalues.index,
        data={col: multi.multipletests(pvalues[col], method=multi_comp, alpha=alpha)[1] for col in pvalues.columns}
    )
    return statistics, pvalues, corrected_pvalues, Ns
# TODO: add similar function comparing (GT1, Pred) against (GT1, GT2) for each trial/detector


def distributions_figure(
        matched_features: pd.DataFrame,
        matching_scheme: str,
        gt1: str,
        gt2: str,
        features: Union[str, Sequence[str]] = None,
        title: Optional[str] = None,
) -> go.Figure:
    subframe = h.extract_subframe(
        matched_features, level=u.MATCHING_SCHEME_STR, value=matching_scheme, axis=0, drop_single_values=True
    )
    if features:
        subframe = h.extract_subframe(
            subframe, level=peyes.constants.FEATURE_STR, value=features, axis=0, drop_single_values=False
        )
    title = title if title else (
            "Matched Events :: Features <br>" + f"<sup>(Matching Scheme: {matching_scheme})</sup>"
    )
    fig = h.distributions_figure(subframe, gt1=gt1, gt2=gt2, title=title)
    return fig
