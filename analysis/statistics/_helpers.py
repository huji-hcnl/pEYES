import os
from typing import List, Optional, Union, Tuple, Sequence, Callable

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import plotly.graph_objects as go

import peyes

import analysis.utils as u
from peyes._utils.visualization_utils import make_empty_figure
from peyes._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType


def extract_subframe(
        data: pd.DataFrame,
        level: Union[str, int],
        value: Union[str, int, Sequence[str], Sequence[int]],
        axis: int = 0,
        drop_single_values: bool = True,  # drop level if only one value is selected
) -> pd.DataFrame:
    if isinstance(value, str) or isinstance(value, int):
        value = [value]
    if axis == 0:
        is_values = data.index.get_level_values(level).isin(value)
        subframe = data.loc[is_values]
    elif axis == 1:
        is_values = data.columns.get_level_values(level).isin(value)
        subframe = data.loc[:, is_values]
    else:
        raise ValueError("`axis` must be 0 or 1.")
    if drop_single_values and len(value) == 1:
        subframe = subframe.droplevel(level, axis=axis)
    return subframe


def load_data(
        dataset_name: str,
        output_dir: str,
        data_dir_name: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        filename_prefix: str = "",
        filename_suffix: str = "",
        iteration: Optional[int] = 1,
        stimulus_type: Optional[Union[str, List[str]]] = None,
        sub_index: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    data_dir_path = os.path.join(output_dir, dataset_name, data_dir_name)
    fullpath = os.path.join(data_dir_path, u.get_filename_for_labels(
        label, prefix=filename_prefix, suffix=filename_suffix, extension="pkl"
    ))
    try:
        data = pd.read_pickle(fullpath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find `{fullpath}`. Please preprocess the dataset first.")
    if iteration is not None:
        data = data.xs(iteration, level=peyes.constants.ITERATION_STR, axis=1, drop_level=True)
    if stimulus_type:
        stim_trial_ids = u.get_trials_for_stimulus_type(dataset_name, stimulus_type)
        is_trial_ids = data.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(stim_trial_ids)
        data = data.loc[:, is_trial_ids]
    if sub_index:
        data = extract_subframe(data, level=0, value=sub_index, axis=0, drop_single_values=False)
    return data


def mann_whitney(
        data: pd.DataFrame,
        gt_cols: Union[str, Sequence[str]],
        alternative: str = "two-sided",
        method: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the Mann-Whitney U-test for each (metric, GT labeler) pair.
    Mann-Whitney is the non-parametric equivalent of two-sampled t-test (using ranks instead of means), which assumes
    independent measures between groups.

    Returns the U-statistic, p-value and number of samples for each pair.
    Read the docstring for `_statistical_analysis` for implementation details.
    """
    pred = data.columns.get_level_values(u.PRED_STR).unique()
    if len(pred) > 2:
        raise ValueError(
            f"Mann-Whitney's U-test requires exactly 2 predictors, got {len(pred)}. " +
            "Consider using Kruskal-Wallis test instead."
        )
    mw_test = lambda vals: stats.mannwhitneyu(*vals, alternative=alternative, method=method, nan_policy='omit')
    u_stats, p_vals, _, Ns = _statistical_analysis(data, gt_cols, mw_test)
    return u_stats, p_vals, Ns


def wilcoxon(data: pd.DataFrame,
             gt_cols: Union[str, Sequence[str]],
             alternative: str = "two-sided",
             method: str = "auto",
             zero_method: str = 'wilcox',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the Wilcoxon signed-rank for each (metric, GT labeler) pair.
    Wilcoxon is the non-parametric equivalent of paired-samples t-test (using ranks instead of means), which assumes
    repeated measures between groups.

    Returns the W-statistic, p-value and number of samples for each pair.
    Read the docstring for `_statistical_analysis` for implementation details.
    """
    def wilcoxon_test(vals):
        assert 0 < len(vals) <=2, f"Wilcoxon test can only run on 1 set of differences or 2 sets of samples, {len(vals)} given."
        if len(vals) == 1:
            return stats.wilcoxon(
                vals, alternative=alternative, method=method, zero_method=zero_method, nan_policy='omit'
            )
        vals1, vals2 = vals[0], vals[1]
        assert len(vals1) == len(vals2), f"length mismatch: |group 1|={len(vals1)} and |group 2|={len(vals2)}"
        return stats.wilcoxon(
            x=vals1, y=vals2, alternative=alternative, method=method, zero_method=zero_method, nan_policy='omit'
        )

    w_stats, p_vals, _, Ns = _statistical_analysis(data, gt_cols, wilcoxon_test)
    return w_stats, p_vals, Ns


def kruskal_wallis_dunns(
        data: pd.DataFrame,
        gt_cols: Union[str, Sequence[str]],
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the Kruskal-Wallis test with Dunn's post-hoc test for multiple comparisons, for each (metric, GT labeler) pair.
    Kruskal-Wallis is the non-parametric equivalent to one-way ANOVA (using ranks instead of means), which assumes
    independent samples between groups. Dunn's test is the common post-hoc test used after a significant KW, to
    determine which pairs of groups are significantly different.

    Returns the KW-statistic, KW-p-value, Dunn's-p-values and number of samples for each pair.
    Read the docstring for `_statistical_analysis` for implementation details.
    """
    kw_test = lambda vals: stats.kruskal(*vals, nan_policy='omit')
    dunns_test = lambda vals: sp.posthoc_dunn(a=list(vals), p_adjust=multi_comp)
    h_stat, p_vals, dunn_p_vals, Ns = _statistical_analysis(data, gt_cols, kw_test, dunns_test)
    return h_stat, p_vals, dunn_p_vals, Ns


def friedman_nemenyi(
        data: pd.DataFrame,
        gt_cols: Union[str, Sequence[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the Friedman test with Nemenyi's post-hoc test for multiple comparisons, for each (metric, GT labeler) pair.
    Friedman is the non-parametric equivalent to repeated measures ANOVA (using ranks instead of means), which assumes
    paired samples between groups. Nemenyi's test is the common post-hoc test used after a significant Friedman, to
    determine which pairs of groups are significantly different.
    Note Nemenyi's test doesn't require correcting for multiple comparisons.

    Returns the Friedman's Fr-statistic, p-value, Nemenyi's p-values and number of samples for each pair.
    Read the docstring for `_statistical_analysis` for implementation details.
    """
    friedman_test = lambda vals: stats.friedmanchisquare(*vals, nan_policy='omit')
    nemenyi_test = lambda vals: sp.posthoc_nemenyi(a=list(vals))
    q_stat, p_vals, nemenyi_p_vals, Ns = _statistical_analysis(data, gt_cols, friedman_test, nemenyi_test)
    return q_stat, p_vals, nemenyi_p_vals, Ns


def create_post_hoc_table(
        post_hoc_results: pd.DataFrame,
        metric: str,
        *gt_names: str,
        alpha: float = 0.05,
        marginal_alpha: Optional[float] = 0.075,
) -> (pd.DataFrame, pd.DataFrame):
    assert 0 < alpha < 1, f"parameter `alpha` must be in range (0, 1), {alpha: .3f} given."
    assert marginal_alpha is None or alpha < marginal_alpha < 1, f"parameter `marginal_alpha` must be in range ({alpha: .3f}, 1), {marginal_alpha: .3f} given."
    pvals = pd.concat(
        [post_hoc_results.loc[metric, gt] for gt in gt_names], keys=gt_names
    ).reorder_levels([1, 0])
    pvals.sort_index(
        level=0, key=lambda idx: idx.map(lambda det: list(pvals.columns).index(det)), inplace=True
    )
    pvals.index.names = [u.PRED_STR, u.GT_STR]
    pvals.columns.names = [u.PRED_STR]

    table = np.full_like(pvals, "n.s.", dtype=np.dtypes.StringDType())
    if marginal_alpha is not None:
        table[pvals <= marginal_alpha] = '†'
    table[pvals <= alpha] = '*'
    table[pvals <= alpha / 5] = '**'
    table[pvals <= alpha / 50] = '***'
    table = pd.DataFrame(table, index=pvals.index, columns=pvals.columns)

    for c, det1 in enumerate(pvals.index.get_level_values(0).unique()):
        for r, det2 in enumerate(pvals.index.get_level_values(0).unique()):
            if c == r:
                table.loc[det1, det2] = "--"
            elif r > c:
                table.loc[det2, det1] = ""
            else:
                for gt in gt_names:
                    table.loc[(det1, gt), det2] = f"{pvals.loc[(det1, gt), det2] :.4f}"
    return table



def distributions_figure(
        data: pd.DataFrame,
        gt1: str,
        title: str,
        colors: u.COLORMAP_TYPE = None,
        gt2: Optional[str] = None,
        only_box: bool = False,
        show_other_gt: bool = False,
) -> go.Figure:
    """
    Creates a violin/box subplot for each unique value in the input DataFrame's index. Each subplot (index value)
    contains violins/boxes for all predictor labelers (detection algorithms), compared to the specified GT labeler(s).
    If two GT labelers are provided, the plot will show GT1 in the positive (right) side and GT2 in the negative (left) side.

    :param data: DataFrame; Should have the following MultiIndex structure:
        - Index :: single level
        - Columns :: 1st level: trial id
        - Columns :: 2nd level: GT labeler
        - Columns :: 3rd level: detector
    :param gt1: name of the first GT labeler to compare.
    :param title: optional; title for the plot.
    :param colors: optional; list of hex colors to mark different labelers in the violins/boxes.
    :param gt2: optional; name of the second GT labeler to compare.
    :param only_box: if True, only box plots will be shown.
    :param show_other_gt: if True, a separate violin/box plot will be shown for the other GT labeler.

    :return: Plotly figure with the violin/box plot.
    """
    gt_cols = list(filter(None, [gt1, gt2]))
    assert 0 < len(gt_cols) <= 2
    indices = sorted(
        data.index.unique(),
        key=lambda met: u.METRICS_CONFIG[met][1] if met in u.METRICS_CONFIG else ord(met[0])
    )
    fig, nrows, ncols = make_empty_figure(
        subtitles=list(map(lambda idx: u.METRICS_CONFIG[idx][0] if idx in u.METRICS_CONFIG else idx, indices)),
        sharex=False, sharey=False,
    )
    for i, idx in enumerate(indices):
        r, c = (i, 0) if ncols == 1 else divmod(i, ncols)
        for j, gt_col in enumerate(gt_cols):
            violin_side = None if len(gt_cols) == 1 else 'positive' if j == 0 else 'negative'
            opacity = 0.75 if len(gt_cols) == 1 else 0.75 if j == 0 else 0.25
            gt_df = (data.xs(gt_col, level=u.GT_STR, axis=1).loc[idx]).unstack()
            detectors = u.sort_labelers(gt_df.columns.get_level_values(u.PRED_STR).unique())
            for k, det in enumerate(detectors):
                if det not in gt_cols:
                    det_name = det.removesuffix("Detector")
                    det_color = u.get_labeler_color(det_name, k, colors)
                elif det in gt_cols and show_other_gt:
                    det_name = "Other GT"
                    det_color = "#bab0ac"
                else:
                    continue
                if only_box:
                    fig.add_trace(
                        row=r + 1, col=c + 1,
                        trace=go.Box(
                            x0=det_name, y=gt_df[det].explode().dropna().values,
                            name=f"{gt_col}, {det_name}", legendgroup=det_name,
                            marker_color=det_color, line_color=det_color,
                            opacity=opacity, boxmean='sd', showlegend=i == 0,
                        )
                    )
                else:
                    fig.add_trace(
                        row=r + 1, col=c + 1,
                        trace=go.Violin(
                            x0=det_name, y=gt_df[det].explode().dropna().values,
                            side=violin_side, opacity=opacity, spanmode='hard',
                            fillcolor=det_color, line_color='black',
                            name=f"{gt_col}, {det_name}", legendgroup=det_name, scalegroup=idx, showlegend=i == 0,
                            box_visible=True, meanline_visible=True,
                        ),
                    )
        y_range = u.METRICS_CONFIG[idx][2] if idx in u.METRICS_CONFIG else None
        fig.update_yaxes(row=r + 1, col=c + 1, range=y_range)
    fig.update_layout(
        title=title,
        violinmode='overlay',
        boxmode='group',
        boxgroupgap=0,
        boxgap=0,
    )
    return fig


def calc_error_bars(sub_data: pd.DataFrame, error_bars: Optional[str] = None) -> pd.Series:
    error_bars = error_bars.strip().lower() if error_bars else None
    if error_bars is None:
        return None
    if error_bars == 'ci':
        # see https://stackoverflow.com/a/69841368/8543025
        ci = sub_data.apply(    # symmetric CI around 0
            lambda x: stats.t.interval(0.95, len(x) - 1, loc=0, scale=stats.sem(x)), axis=1
        )
        return ci.apply(lambda x: x[1])     # only the top of the CI
    std = sub_data.std(axis=1)
    if error_bars == "std":
        return std
    elif error_bars == "sem":
        return std / np.sqrt(sub_data.count(axis=1))
    raise ValueError(f"Invalid error bars type: {error_bars}, choose from: sem, std, ci")


def _statistical_analysis(
        data: pd.DataFrame,
        gt_cols: Union[str, Sequence[str]],
        test: Callable[[Sequence[Sequence[float]]], Tuple[float, float]],
        post_hoc_test: Optional[Callable[[Sequence[Sequence[float]]], pd.DataFrame]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the statistical test and post-hoc test for each (metric, GT labeler) pair in the input DataFrame.
    For each unique value in the input DataFrame's index (metric names) and each of the GT labelers, we:
    1. Extract the values for each predictor labeler (detection algorithm) over all trials, so that we have a list of
        values for each predictor.
    2. Run the statistical test with the extracted values.
    3. If a post-hoc test is provided, we run it with the extracted values.
    4. Store the results in the output DataFrames.

    Raises a ValueError if there are less than 2 predictors.
    Returns four DataFrames containing results for the statistical analysis:
    - statistics: index is metric name, columns are GT labelers, values are the test statistic (float).
    - pvalues: index is metric name, columns are GT labelers, values are the test p-value (float).
    - post_hoc_res: index is metric name, columns are GT labelers, values are the post-hoc test results: a DataFrame
        with predictors as index and columns, and p-values as values. If no post-hoc test is provided, this DataFrame
        will be empty.
    - Ns: index is metric name, columns are multiindex with pairs of (GT, Pred) labelers, values are the number of
        measurements (trials) for each pair.

    :param data: DataFrame; Should have the following MultiIndex structure:
        - Index :: single level metric names
        - Columns :: 1st level: trial id
        - Columns :: 2nd level: GT labeler
        - Columns :: 3rd level: Pred labeler (detection algorithm)
    :param gt_cols: GT labeler(s) to compare against predictors (detection algorithms).
    :param test: Statistical test to run with the extracted values.
    :param post_hoc_test: Optional; post-hoc test to run with the extracted values.

    :return: Four DataFrames - statistics, pvalues, post_hoc_res, Ns.
    """
    predictors = data.columns.get_level_values(u.PRED_STR).unique()
    if len(predictors) < 2:
        raise ValueError(f"Not enough predictors for a statistical test: {len(predictors)}")
    if len(predictors) == 2 and post_hoc_test is not None:
        raise RuntimeError("You don't need a post-hoc test for only 2 predictors.")
    gt_cols = gt_cols if isinstance(gt_cols, list) else [gt_cols]
    metrics = sorted(
        data.index.unique(),
        key=lambda met: u.METRICS_CONFIG[met][1] if met in u.METRICS_CONFIG else ord(met[0])
    )
    Ns = {}                                         # (metric, GT, Pred) -> num_trials
    statistics, pvalues, post_hoc_res = {}, {}, {}      # (metric, GT) -> value
    for _i, gt_col in enumerate(gt_cols):
        for _j, met in enumerate(metrics):
            gt_series = data.xs(gt_col, level=u.GT_STR, axis=1).loc[met]
            if pd.isna(gt_series).all():
                continue
            gt_df = gt_series.unstack().drop(columns=gt_cols, errors='ignore')  # drop other GT labelers

            # extract per-detector values
            detectors = u.sort_labelers(gt_df.columns.get_level_values(u.PRED_STR).unique())
            detector_values = {}
            for det in detectors:
                vals = gt_df[det].explode().values.astype(float)
                n = sum(~np.isnan(vals))
                Ns[(met, gt_col, det)] = n
                if n > 0:
                    detector_values[det] = vals

            # calculate statistical test
            res = test(list(detector_values.values()))
            statistic, pvalue = res.statistic, res.pvalue
            statistics[(met, gt_col)] = statistic
            pvalues[(met, gt_col)] = pvalue

            # calculate post-hoc test
            if post_hoc_test is not None:
                post_result = pd.DataFrame(
                    post_hoc_test(list(detector_values.values())).values,
                    index=list(detector_values.keys()), columns=list(detector_values.keys())
                )
            else:
                post_result = pd.DataFrame(np.array([]))
            post_result.index.names = post_result.columns.names = [u.PRED_STR]
            post_hoc_res[(met, gt_col)] = post_result

    # create outputs
    statistics = pd.Series(statistics).unstack()
    pvalues = pd.Series(pvalues).unstack()
    post_hoc_res = pd.Series(post_hoc_res).unstack()
    statistics.index.names = pvalues.index.names = post_hoc_res.index.names = [peyes.constants.METRIC_STR]
    statistics.columns.names = pvalues.columns.names = post_hoc_res.columns.names = [u.GT_STR]
    Ns = pd.Series(Ns)
    Ns.index.names = [peyes.constants.METRIC_STR, u.GT_STR, u.PRED_STR]
    Ns = Ns.unstack([u.GT_STR, u.PRED_STR])
    return statistics, pvalues, post_hoc_res, Ns
