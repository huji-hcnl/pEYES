import os
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import pEYES as peyes

import analysis.utils as u

pio.renderers.default = "browser"

###################


def kruskal_wallis_with_posthoc_dunn(
        sample_metrics: pd.DataFrame,
        gt_cols: List[str],
        multi_comp: Optional[str] = "fdr_bh",
):
    metrics = sorted(
        sample_metrics.index.unique(),
        key=lambda met: u.SAMPLE_METRICS[met][1] if met in u.SAMPLE_METRICS else ord(met[0])
    )
    statistics, pvalues, dunns, Ns = {}, {}, {}, {}
    for m, metric in enumerate(metrics):
        for gt, gt_col in enumerate(gt_cols):
            gt_series = sample_metrics.xs(gt_col, level=u.GT_STR, axis=1).loc[metric]
            gt_df = gt_series.unstack().drop(columns=GT_COLS, errors='ignore')
            N, _ = gt_df.shape
            detector_values = {col: gt_df[col].values for col in gt_df.columns}
            statistic, pvalue = stats.kruskal(*detector_values.values(), nan_policy='omit')
            dunn = pd.DataFrame(
                sp.posthoc_dunn(a=list(detector_values.values()), p_adjust=multi_comp).values,
                index=gt_df.columns, columns=gt_df.columns
            )
            dunn.index.name = dunn.columns.name = None
            statistics[(metric, gt_col)] = statistic
            pvalues[(metric, gt_col)] = pvalue
            dunns[(metric, gt_col)] = dunn
            Ns[(metric, gt_col)] = N
    statistics = pd.Series(statistics).unstack()
    pvalues = pd.Series(pvalues).unstack()
    dunns = pd.Series(dunns).unstack()
    Ns = pd.Series(Ns).unstack()
    return statistics, pvalues, dunns, Ns


def sample_metrics_figure(
        sample_metrics: pd.DataFrame,
        gt1: str,
        gt2: Optional[str] = None,
        title: str = "Sample Metrics",
        only_box: bool = False,
) -> go.Figure:
    gt_cols = list(filter(None, [gt1, gt2]))
    assert 0 < len(gt_cols) <= 2
    metrics = sorted(
        sample_metrics.index.unique(),
        key=lambda met: u.SAMPLE_METRICS[met][1] if met in u.SAMPLE_METRICS else ord(met[0])
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
        subplot_titles=list(map(lambda met: u.SAMPLE_METRICS[met][0] if met in u.SAMPLE_METRICS else met, metrics)),
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
        y_range = u.SAMPLE_METRICS[metric][2] if metric in u.SAMPLE_METRICS else None
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

GT1, GT2 = "RA", "MN"
GT_COLS = list(filter(None, [GT1, GT2]))

dataset = pd.read_pickle(os.path.join(u.DATASETS_DIR, "lund2013.pkl"))
image_trials = u.trial_ids_by_condition(dataset, key=peyes.constants.STIMULUS_TYPE_STR, values="image")

## Sample Metrics ; 1st iteration ; image trials
sample_mets = pd.read_pickle(os.path.join(u.OUTPUT_DIR, "default_values", "lund2013", "sample_metrics", "all_labels.pkl"))
iter1_sample_mets = sample_mets.xs(1, level=peyes.constants.ITERATION_STR, axis=1)
image_iter1_sample_mets = iter1_sample_mets.loc[:, iter1_sample_mets.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(image_trials)]

statistics, pvalues, dunns, Ns = kruskal_wallis_with_posthoc_dunn(image_iter1_sample_mets, GT_COLS, multi_comp="fdr_bh")
fig = sample_metrics_figure(image_iter1_sample_mets, GT1, GT2, title=f"Sample Metrics")
fig.show()

## Matched SDT Metrics ; 1st iteration ; image trials
matched_sdt_metrics = pd.read_pickle(os.path.join(u.OUTPUT_DIR, "default_values", "lund2013", "matches_metrics", "all_labels_sdt_metrics.pkl"))
iter1_matched_sdt_metrics = matched_sdt_metrics.xs(1, level=peyes.constants.ITERATION_STR, axis=1)
image_iter1_matched_sdt_metrics = iter1_matched_sdt_metrics.loc[:, iter1_matched_sdt_metrics.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(image_trials)]

mean_match_ratio_across_schemes = image_iter1_matched_sdt_metrics.xs(u.MATCH_RATIO_STR, level=peyes.constants.METRIC_STR, axis=0).mean(axis=0)
mean_match_ratio_across_schemes.name = u.MATCH_RATIO_STR
mean_match_ratio_across_schemes = mean_match_ratio_across_schemes.to_frame().T

statistics, pvalues, dunns, Ns = kruskal_wallis_with_posthoc_dunn(mean_match_ratio_across_schemes, GT_COLS, multi_comp="fdr_bh")
fig = sample_metrics_figure(mean_match_ratio_across_schemes, GT1, GT2, title=f"Sample Metrics")
fig.show()

## Channel SDT Metrics ; 1st iteration ; image trials
# TODO: run this for specific thresholds & channel types
channel_sdt_metrics = pd.read_pickle(os.path.join(u.OUTPUT_DIR, "default_values", "lund2013", "channel_metrics", "all_labels_sdt_metrics.pkl"))
iter1_channel_sdt_metrics = channel_sdt_metrics.xs(1, level=peyes.constants.ITERATION_STR, axis=1)
image_iter1_channel_sdt_metrics = iter1_channel_sdt_metrics.loc[:, iter1_channel_sdt_metrics.columns.get_level_values(peyes.constants.TRIAL_ID_STR).isin(image_trials)]

onset_metrics = image_iter1_channel_sdt_metrics.xs("onset", level=u.CHANNEL_TYPE_STR, axis=0)
statistics, pvalues, dunns, Ns = kruskal_wallis_with_posthoc_dunn(onset_metrics, GT_COLS, multi_comp="fdr_bh")
fig = sample_metrics_figure(onset_metrics, GT1, GT2, title=f"Sample Metrics")
fig.show()

