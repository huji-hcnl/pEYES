import os
from typing import List, Optional, Union, Any, Tuple, Sequence

import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pEYES as peyes
from pEYES._DataModels.UnparsedEventLabel import UnparsedEventLabelType, UnparsedEventLabelSequenceType

import analysis.utils as u


def load_data(
        dataset_name: str,
        output_dir: str,
        data_dir_name: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        filename_prefix: str = "",
        filename_suffix: str = "",
        iteration: int = 1,
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
    data = data.xs(iteration, level=peyes.constants.ITERATION_STR, axis=1)
    if isinstance(stimulus_type, str):
        stimulus_type = [stimulus_type]
    if stimulus_type:
        stimulus_type = [stmtp.lower().strip() for stmtp in stimulus_type if isinstance(stmtp, str)]
        trial_ids = _trial_ids_by_condition(dataset_name, key=peyes.constants.STIMULUS_TYPE_STR, values=stimulus_type)
        is_trial_ids = data.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(trial_ids)
        data = data.loc[:, is_trial_ids]
    if isinstance(sub_index, str):
        sub_index = [sub_index]
    if sub_index:
        sub_index = [sub_idx.lower().strip() for sub_idx in sub_index if isinstance(sub_idx, str)]
        data = data.loc[sub_index]
    return data


def statistical_analysis(
        data: pd.DataFrame,
        gt_cols: Sequence[str],
        multi_comp: Optional[str] = "fdr_bh",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each unique value in the input DataFrame's index and each of the GT labelers, performs Kruskal-Wallis test with
    post-hoc Dunn's test for multiple comparisons. Returns the KW-statistic, KW-p-value, Dunn's-p-values and number of
    samples for each (index, GT labeler) pair.

    :param data: DataFrame; Should have the following MultiIndex structure:
        - Index :: single level
        - Columns :: 1st level: trial id
        - Columns :: 2nd level: GT labeler
        - Columns :: 3rd level: Pred labeler (detection algorithm)
    :param gt_cols: List of GT labelers to compare.
    :param multi_comp: Method for multiple comparisons correction when performing Dunn's post-hoc test.

    :return: Four DataFrames containing results for the statistical analysis.
        - statistics: KW statistic; index is metric name, columns are GT labelers.
        - pvalues: KW p-value; index is metric name, columns are GT labelers.
        - dunns: DataFrame with Dunn's p-values for pair-wise comparisons; index and columns are Pred labelers.
        - Ns: Number of data-points (trials) for each (metric, GT labeler, Pred labeler) pair; index is metric name,
            columns multiindex with pairs of (GT, Pred) labelers.
    """
    indices = sorted(
        data.index.unique(),
        key=lambda met: u.METRICS_CONFIG[met][1] if met in u.METRICS_CONFIG else ord(met[0])
    )
    statistics, pvalues, dunns, Ns = {}, {}, {}, {}
    for i, idx in enumerate(indices):
        for j, gt_col in enumerate(gt_cols):
            gt_series = data.xs(gt_col, level=u.GT_STR, axis=1).loc[idx]
            gt_df = gt_series.unstack().drop(columns=gt_cols, errors='ignore')
            detector_values = {col: gt_df[col].explode().dropna().values for col in gt_df.columns}
            statistic, pvalue = stats.kruskal(*detector_values.values(), nan_policy='omit')
            dunn = pd.DataFrame(
                sp.posthoc_dunn(a=list(detector_values.values()), p_adjust=multi_comp).values,
                index=gt_df.columns, columns=gt_df.columns
            )
            statistics[(idx, gt_col)] = statistic
            pvalues[(idx, gt_col)] = pvalue
            dunns[(idx, gt_col)] = dunn
            Ns.update({(idx, gt_col, det): det_vals.shape[0] for det, det_vals in detector_values.items()})

    # create outputs
    statistics = pd.Series(statistics).unstack()
    pvalues = pd.Series(pvalues).unstack()
    statistics.index.names = pvalues.index.names = [peyes.constants.METRIC_STR]
    statistics.columns.names = pvalues.columns.names = [u.GT_STR]
    dunns = pd.Series(dunns).unstack()
    dunns.index.names = dunns.columns.names = [u.PRED_STR]
    Ns = pd.Series(Ns)
    Ns.index.names = [peyes.constants.METRIC_STR, u.GT_STR, u.PRED_STR]
    Ns = Ns.unstack([u.GT_STR, u.PRED_STR])
    return statistics, pvalues, dunns, Ns


def distributions_figure(
        data: pd.DataFrame,
        gt1: str,
        title: str,
        gt2: Optional[str] = None,
        only_box: bool = False,
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
    :param gt2: optional; name of the second GT labeler to compare.
    :param only_box: if True, only box plots will be shown.

    :return: Plotly figure with the violin/box plot.
    """
    gt_cols = list(filter(None, [gt1, gt2]))
    assert 0 < len(gt_cols) <= 2
    indices = sorted(
        data.index.unique(),
        key=lambda met: u.METRICS_CONFIG[met][1] if met in u.METRICS_CONFIG else ord(met[0])
    )
    fig, nrows, ncols = _make_empty_figure(indices, sharex=False, sharey=False)
    for i, idx in enumerate(indices):
        r, c = (i, 0) if ncols == 1 else divmod(i, ncols)
        for j, gt_col in enumerate(gt_cols):
            gt_series = data.xs(gt_col, level=u.GT_STR, axis=1).loc[idx]
            gt_df = gt_series.unstack().drop(columns=gt_cols, errors='ignore')
            detectors = sorted(gt_df.columns, key=lambda det: u.DETECTORS_CONFIG[det][1])
            for k, det in enumerate(detectors):
                det_name = det.removesuffix("Detector")
                det_color = u.DETECTORS_CONFIG[det_name.lower()][2]
                if len(gt_cols) == 1:
                    violin_side = None
                    opacity = 0.75
                else:
                    violin_side = 'positive' if j == 0 else 'negative'
                    opacity = 0.75 if j == 0 else 0.25
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


def _trial_ids_by_condition(dataset_name: str, key: str, values: Union[Any, List[Any]]) -> List[int]:
    dataset = u.load_dataset(dataset_name, verbose=False)
    if not isinstance(values, list):
        values = [values]
    all_trial_ids = dataset[peyes.constants.TRIAL_ID_STR]
    is_condition = dataset[key].isin(values)
    return all_trial_ids[is_condition].unique().tolist()


def _make_empty_figure(metrics: Sequence[str], sharex=False, sharey=False) -> Tuple[go.Figure, int, int]:
    ncols = 1 if len(metrics) <= 3 else 2
    nrows = len(metrics) if len(metrics) <= 3 else sum(divmod(len(metrics), ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols,
        shared_xaxes=sharex, shared_yaxes=sharey,
        subplot_titles=list(map(lambda met: u.METRICS_CONFIG[met][0] if met in u.METRICS_CONFIG else met, metrics)),
    )
    return fig, nrows, ncols
