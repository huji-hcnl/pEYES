# pEYES User Guide

Tutorial notebooks for the `peyes` package, covering its public API end to end. Read them in order — each one
assumes what came before, and each covers 1-2 functionalities in increasing depth.

Prerequisites: `pip install -e .` from the repository root (see the main [README](../../README.md)), and a
Jupyter kernel with that environment. The first notebook to load a given dataset downloads it and caches it under
`data/` (ignored by git); later notebooks reuse the cache.

| # | Notebook | Covers |
|---|---|---|
| 1 | [Quick Start](<1 Quick Start.ipynb>) | The full pipeline once, end to end: load data → detect → build events → summarize. |
| 2 | [Datasets](<2 Datasets.ipynb>) | The four built-in human-annotated datasets (`lund2013`, `irf`, `hfc`, `gazecom`). |
| 3 | [Parsing Custom Data & Configuration](<3 Parsing Custom Data & Configuration.ipynb>) | Bringing your own recordings in with `parse_data`; global config (`set_viewer_distance`, `set_screen_monitor`, `set_event_configurations`). |
| 4 | [Detection Algorithms](<4 Detection Algorithms.ipynb>) | `create_detector` in depth via IVT and Engbert, plus a tour of the other five algorithms. |
| 5 | [Sample-Level Evaluation](<5 Sample-Level Evaluation.ipynb>) | `sample_metrics` — comparing two label sequences sample by sample. |
| 6 | [Events - Construction & Properties](<6 Events - Construction & Properties.ipynb>) | `create_events`, `Event` properties, `summarize_events`, `events_to_labels`. |
| 7 | [Event Metrics](<7 Event Metrics.ipynb>) | `event_metrics` — rates, counts, feature distributions, transition matrices. |
| 8 | [Event Matching & Match Evaluation](<8 Event Matching & Match Evaluation.ipynb>) | `match` strategies and `match_metrics` on the resulting matches. |
| 9 | [Temporal Alignment Evaluation](<9 Temporal Alignment Evaluation.ipynb>) | `channel_metrics` — onset/offset timing precision without an explicit event match. |
| 10 | [Visualizing Gaze & Events](<10 Visualizing Gaze & Events.ipynb>) | `visualize.gaze_trajectory/gaze_heatmap/gaze_over_time` and single-sequence event summaries. |
| 11 | [Visualizing Comparisons & Video](<11 Visualizing Comparisons & Video.ipynb>) | Comparing sequences (feature/scarfplot comparisons) and exporting gaze video. |

[`_helpers.py`](_helpers.py) is a small shared module (`load_example_trial`) used from notebook 1 onward to skip
the boilerplate of extracting one trial from a loaded dataset — not part of the `peyes` public API itself.
