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
from analysis.pipeline.preprocess import load_dataset

pio.renderers.default = "browser"

###################


def get_data(
        dataset_name: str,
        output_dir: str,
        label: Optional[Union[UnparsedEventLabelType, UnparsedEventLabelSequenceType]] = None,
        iteration: int = 1,
        stimulus_type: Optional[Union[str, List[str]]] = None,
        metric: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    sample_metrics_dir = os.path.join(output_dir, dataset_name, f"{peyes.constants.SAMPLE_STR}_{peyes.constants.METRICS_STR}")
    fullpath = os.path.join(sample_metrics_dir, u.get_filename_for_labels(label, extension="pkl"))
    try:
        sample_metrics = pd.read_pickle(fullpath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Couldn't find `{fullpath}`. Please preprocess the dataset first.")
    sample_metrics = sample_metrics.xs(iteration, level=peyes.constants.ITERATION_STR, axis=1)
    if stimulus_type:
        dataset = load_dataset(dataset_name, verbose=False)
        trial_ids = u.trial_ids_by_condition(dataset, key=peyes.constants.STIMULUS_TYPE_STR, values=stimulus_type)
        is_trial_ids = sample_metrics.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(trial_ids)
        sample_metrics = sample_metrics.loc[:, is_trial_ids]
    if metric:
        sample_metrics = sample_metrics.loc[metric]
    return sample_metrics


def statistical_analysis(
        sample_metrics: pd.DataFrame,
        gt_cols: List[str],
        multi_comp: Optional[str] = "fdr_bh",
):
    """
    For each of the metrics in the input DataFrame and each of the GT labelers, performs Kruskal-Wallis test with
    post-hoc Dunn's test for multiple comparisons. Returns the KW-statistic, KW-p-value, Dunn's-p-values and number of
    samples for each (metric, GT labeler) pair.

    :param sample_metrics: DataFrame with sample metrics. Should have the following MultiIndex structure:
        - Index :: 1st level: metric name
        - Columns :: 1st level: trial id
        - Columns :: 2nd level: GT labeler
        - Columns :: 3rd level: detector
    :param gt_cols: List of GT labelers to compare.
    :param multi_comp: Method for multiple comparisons correction when performing Dunn's post-hoc test.

    :return: Four DataFrames containing results for the statistical analysis.
        - statistics: KW statistic; index is metric name, columns are GT labelers.
        - pvalues: KW p-value; index is metric name, columns are GT labelers.
        - dunns: DataFrame with Dunn's p-values for pair-wise comparisons; index and columns are Pred labelers.
        - Ns: Number of data-points (trials) for each (metric, GT labeler, Pred labeler) pair; index is metric name,
            columns multiindex with pairs of (GT, Pred) labelers.
    """
    metrics = sorted(
        sample_metrics.index.unique(),
        key=lambda met: u.METRICS_CONFIG[met][1] if met in u.METRICS_CONFIG else ord(met[0])
    )
    statistics, pvalues, dunns, Ns = {}, {}, {}, {}
    for m, metric in enumerate(metrics):
        for gt, gt_col in enumerate(gt_cols):
            gt_series = sample_metrics.xs(gt_col, level=u.GT_STR, axis=1).loc[metric]
            gt_df = gt_series.unstack().drop(columns=gt_cols, errors='ignore')
            detector_values = {col: gt_df[col].explode().dropna().values for col in gt_df.columns}
            statistic, pvalue = stats.kruskal(*detector_values.values(), nan_policy='omit')
            dunn = pd.DataFrame(
                sp.posthoc_dunn(a=list(detector_values.values()), p_adjust=multi_comp).values,
                index=gt_df.columns, columns=gt_df.columns
            )
            statistics[(metric, gt_col)] = statistic
            pvalues[(metric, gt_col)] = pvalue
            dunns[(metric, gt_col)] = dunn
            Ns.update({(metric, gt_col, det): det_vals.shape[0] for det, det_vals in detector_values.items()})
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


def sample_metrics_figure(
        sample_metrics: pd.DataFrame,
        gt1: str,
        gt2: Optional[str] = None,
        title: str = "Sample Metrics",
        only_box: bool = False,
) -> go.Figure:
    """
    Creates a violin/box plot for each metric in the input DataFrame, comparing the detectors for the GT labeler(s).
    If two GT labelers are provided, the plot will show GT1 in the positive (right) side and GT2 in the negative (left) side.

    :param sample_metrics: DataFrame with sample metrics. Should have the following MultiIndex structure:
        - Index :: 1st level: metric name
        - Columns :: 1st level: trial id
        - Columns :: 2nd level: GT labeler
        - Columns :: 3rd level: detector
    :param gt1: name of the first GT labeler to compare.
    :param gt2: optional; name of the second GT labeler to compare.
    :param title: optional; title for the plot.
    :param only_box: if True, only box plots will be shown.
    :return:
    """
    gt_cols = list(filter(None, [gt1, gt2]))
    assert 0 < len(gt_cols) <= 2
    metrics = sorted(
        sample_metrics.index.unique(),
        key=lambda met: u.METRICS_CONFIG[met][1] if met in u.METRICS_CONFIG else ord(met[0])
    )
    if len(metrics) <= 3:
        ncols = 1
        nrows = len(metrics)
    else:
        ncols = 2
        nrows = sum(divmod(len(metrics), ncols))
    fig = make_subplots(
        rows=nrows, cols=ncols,
        shared_xaxes=False,
        subplot_titles=list(map(lambda met: u.METRICS_CONFIG[met][0] if met in u.METRICS_CONFIG else met, metrics)),
    )
    for m, metric in enumerate(metrics):
        r, c = (m, 0) if ncols == 1 else divmod(m, ncols)
        for gt, gt_col in enumerate(gt_cols):
            gt_series = sample_metrics.xs(gt_col, level=u.GT_STR, axis=1).loc[metric]
            gt_df = gt_series.unstack().drop(columns=gt_cols, errors='ignore')
            for d, detector in enumerate(gt_df.columns):
                det_name = detector.removesuffix("Detector")
                if len(gt_cols) == 1:
                    violin_side = None
                    opacity = 0.75
                else:
                    violin_side = 'positive' if gt == 0 else 'negative'
                    opacity = 0.75 if gt == 0 else 0.25
                if only_box:
                    fig.add_trace(
                        row=r + 1, col=c + 1,
                        trace=go.Box(
                            x0=det_name, y=gt_df[detector],
                            name=f"{gt_col}, {det_name}", legendgroup=det_name,
                            marker_color=u.DEFAULT_DISCRETE_COLORMAP[d], line_color='black',
                            opacity=opacity, boxmean='sd', showlegend=m == 0,
                        )
                    )
                else:
                    fig.add_trace(
                        row=r + 1, col=c + 1,
                        trace=go.Violin(
                            x0=det_name, y=gt_df[detector],
                            side=violin_side, opacity=opacity, spanmode='hard',
                            fillcolor=u.DEFAULT_DISCRETE_COLORMAP[d],
                            name=f"{gt_col}, {det_name}", legendgroup=det_name, scalegroup=metric, showlegend=m == 0,
                            box_visible=True, meanline_visible=True, line_color='black',
                        ),
                    )
        y_range = u.METRICS_CONFIG[metric][2] if metric in u.METRICS_CONFIG else None
        fig.update_yaxes(row=r + 1, col=c + 1, range=y_range)
    fig.update_layout(
        title=title,
        violinmode='overlay',
        boxmode='group',
        boxgroupgap=0,
        boxgap=0,
    )
    return fig


###################
## EXAMPLE USAGE ##

GT1, GT2 = "RA", "MN"

sample_metrics = get_data("lund2013", os.path.join(u.OUTPUT_DIR, "default_values"), label=None, iteration=1, stimulus_type="image")

sm_statistics, sm_pvalues, sm_dunns, sm_Ns = statistical_analysis(sample_metrics, [GT1, GT2], multi_comp="fdr_bh")
sample_metrics_fig = sample_metrics_figure(sample_metrics, GT1, GT2, title=f"Sample Metrics")
sample_metrics_fig.show()
