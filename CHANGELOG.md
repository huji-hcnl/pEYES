# Changelog

All notable changes to pEYES are documented here. This project uses [semantic versioning](https://semver.org);
while the major version is `0`, breaking changes bump the **minor** version.

## [Unreleased]

## [0.2.1] - not yet released

Declares and validates Python 3.14 support alongside the existing 3.12 floor. No breaking changes, no
dependency floor changes, and no functional code changes at all relative to 0.2.0 - confirmed by diff: only
`peyes/__init__.py`'s version string, `tests/`, CI configuration, and documentation changed. Publication
results ([Nir & Deouell, 2026](https://doi.org/10.3758/s13428-026-02983-5)) are therefore still unchanged from
v0.1.0, exactly as in 0.2.0 (which was independently verified bit-for-bit against real article data - see that
entry below). A dedicated regression harness (see Added below) additionally confirmed the floor-pinned and
latest dependency versions produce identical detection/matching/metrics output on real data. Safe upgrade from
either 0.1.0 or 0.2.0 - no new caveats beyond what's already documented for 0.2.0 below.

### Added
- `tests/regression_tests/`: runs the real detect/match/metrics pipeline (IVT, IDT, IDVT, Engbert, NH,
  REMoDNaV) on a small, real, checked-in Lund2013 slice and compares against a golden reference generated
  under floor-pinned dependencies - guards against a future dependency bump silently changing results.
- CI now tests Python 3.14 (latest dependencies) alongside Python 3.12 (floor-pinned dependencies).

### Changed
- `README.md` now states tested support for both Python 3.12 and 3.14.

## [0.2.0] - 2026-09-02

A correctness release. Fixes dozens of findings from a full review of the package plus several follow-up
passes, some of which change values that previous versions returned. **Read the breaking changes before
upgrading**: some of them alter results silently rather than raising. Publication results
([Nir & Deouell, 2026](https://doi.org/10.3758/s13428-026-02983-5)) are unchanged - verified bit-for-bit
against real article data.

### Breaking changes

| What changed | Before | After |
|---|---|---|
| `match(..., match_by="offset difference")` | fell through to generic matching and returned one-to-many `list` values | returns one-to-one `Event` values, like every other alias |
| `match(...)` with `"onset"`, `"offset"`, `"l2"`, `"window"` and no explicit tolerance | tolerances defaulted to `0`, requiring exact onset/offset equality, so almost nothing matched | tolerances default to unbounded, matching `EventMatcher.generic_matching` |
| `create_boolean_channel(..., "offset", events)` | offsets marked one sample early; disagreed with the labels-based path | offsets land on the event's last sample; both paths now agree |
| `create_boolean_channel` / `events_to_labels` length | one sample short of the input | matches the input; the labels → events → labels round trip is lossless |
| `events_to_labels` | each event lost its final sample | every sample of every event is labeled |
| `summarize_events(...)` | 19 columns | 23 columns (adds `start_x`, `start_y`, `end_x`, `end_y`) |
| `summarize_events([])` | `(0, 0)` frame with no columns | `(0, 23)` frame carrying the full schema |
| `event_metrics.features_by_labels([])` | no feature columns, no `count` | full schema with zero counts |
| `event_rate`, `microsaccade_rate` | divided by the last event's absolute end time | divided by the recording duration, so the rate no longer depends on where the clock starts |
| `sample_metrics.transition_matrix`, `event_metrics.transition_matrix` | only the labels present in the sequence | always the full `EventLabelEnum`, so matrices are comparable across sequences |
| `create_detector(...)` with an unrecognised keyword | silently ignored, default used | raises `TypeError` naming the supported keywords |
| `match_metrics.match_ratio` with one-to-many matches | `AttributeError` | `TypeError` explaining which matching schemes are appropriate |
| `match_metrics.*` with `positive_label=None` | `TypeError` from iterating `None` | `ValueError` stating the argument is required |
| Invalid arguments to `Event(...)`, `set_viewer_distance`, `set_screen_monitor`, `calculate_velocities`, `sample_metrics.calculate` | `AssertionError`, and nothing at all under `python -O` | `ValueError`, including under `-O` |
| `events_to_labels([])` | `ValueError` from `min()` on an empty generator | `ValueError` naming the problem |
| `sample_metrics.calculate`, `event_metrics.get_features`, `match_metrics.get_features` | returned a bare value when one metric/feature was requested, a dict when several were | always return a `Dict[str, ...]`, even for one requested metric/feature |

`match_metrics` true-positive counts now require **both** the ground-truth event and its matched prediction to
carry a positive label. This only changes results when cross-matching is enabled (`allow_xmatch=True`); with
the default `False`, matched predictions already share their ground-truth label and the counts are identical.

The single-metric/feature convenience functions (`sample_metrics.accuracy`, `event_metrics.durations`,
`match_metrics.onset_difference`, and their siblings) are unaffected by the `calculate`/`get_features` change
and still return a bare `float`/`np.ndarray` as before - only the lower-level entry points changed.

### Added

- `start_x` / `start_y` / `end_x` / `end_y` in `summary()` and `summarize_events()`, so a saccade's endpoints
  are recoverable - previously only the midpoint was reported ([#24]).
- `BaseEvent.summary_columns()`, the summary schema as a single source of truth.
- `peyes.__version__`.
- Optional dependency extras: `analysis` (for the scripts under `analysis/`) and `dev`.
- Ruff configuration and a GitHub Actions workflow running lint plus the test suite.
- `BaseEvent.get_outlier_reasons()` now also checks peak velocity and peak acceleration against configurable
  per-label thresholds (`set_event_configurations(..., min_velocity=, max_velocity=, min_acceleration=,
  max_acceleration=)`), in addition to the existing duration and screen-bounds checks. Defaults: saccade
  `max_velocity=1000` deg/s, fixation `max_acceleration=50000` deg/s^2 (both literature-sourced; see
  `_DataModels/config.py` for the alternative thresholds considered and their references). A threshold that's
  `None`/`NaN` disables that specific check. Closes [#26]. Note: given the "angular velocity is computed with
  the wrong transform" known issue below, these particular default values aren't reachable in practice yet -
  the mechanism itself is correct and works with any threshold actually within the current 0-180 deg range.

### Fixed

**Visualization**

- `visualization.create_video` passed its arguments to `create_frames` positionally and misaligned three of
  them; the function could not run on any input.
- `create_frames` raised `ValueError` on any NaN sample - i.e. on any recording with blinks or tracker loss.
- Gaze markers rendered with red and blue exchanged (RGB colours drawn into a BGR frame).
- `create_image` dereferenced the background image before the `None` check, so every call without a
  background image raised `AttributeError` - the documented default for `gaze_heatmap`, `gaze_trajectory`
  and `create_frames`.
- `feature_comparison` raised when given more than six event sequences, and silently used event-label colours
  below that.
- `gaze_heatmap` raised on off-screen gaze, and silently wrapped negative coordinates to the opposite edge.
- `event_summary(show_outliers=True)` raised `KeyError` on any trial with no outliers.
- `_write_video` raised when given a bare filename with no directory component.

**Detection and matching**

- `channel_metrics.onset_detection_metrics` / `offset_detection_metrics` raised `TypeError` for the
  documented scalar `threshold`.
- `create_boolean_channel(..., min_num_samples=None)` - the documented default - raised `TypeError`.
- `min_num_samples` is now a floor on the output length rather than an exact value.
- `REMoDNaVDetector` validated none of its parameters; a non-positive threshold now raises.
- `IDVTDetector` bypassed the saccade-velocity-threshold validation its parent performs.
- Blink padding was asymmetric, adding one fewer sample after each blink than before it.
- The `remodnav` logger level was raised process-wide and never restored.

**Metrics and data model**

- `_dprime_rates` normalised the correction name and then compared the raw string, so `"log linear"` and
  `"Log-Linear"` raised `ValueError` while `"loglinear"` worked.
- `top_pixel` / `bottom_pixel` / `left_pixel` / `right_pixel` returned `(nan, nan)` if any sample was NaN.
- `get_chunk_indices`, `merge_chunks` and `reset_short_chunks` silently returned wrong results for the
  `(1, n)` and `(n, 1)` shapes that `is_one_dimensional` accepts.
- `set_viewer_distance` and `set_screen_monitor` were silent no-ops for events built with default
  `viewer_distance` / `pixel_size`, because those defaults were bound at import time.
- `summarize_events([])` returned a column-less frame ([#25]).

**Packaging**

- `authors` carried a `website` key, which PEP 621 does not permit.
- No `requires-python`, so pip would install on 3.8 and fail at import.
- Dependency pins were near-meaningless (`numpy~=1.2` admits numpy 1.2 while excluding numpy 2.x).
- `scikit_posthocs` was a runtime dependency but is only used by `analysis/`.
- `kaleido==0.1.0post1` had no Linux wheel, so `pip install -e .` failed outright on Linux (including CI).
  Bumped to `kaleido>=1.0,<2` (requires `plotly>=6.1.1`, also bumped); callers of `write_image()`/`save_figure`
  should call `kaleido.get_chrome_sync()` once per environment first, since kaleido otherwise drives whatever
  Chrome/Chromium is already installed, which isn't always compatible - see `save_figure()`'s docstring ([#15]).
- `opencv-python` had no upper version bound (`>=4.9`), unlike every other pinned dependency here; capped at `<6`.

### Documentation

- `BaseEvent.duration` now states the sample-count convention: duration is `end_time - start_time`, so an
  n-sample event spans `(n - 1) * dt` and a single-sample event has duration 0.
- `summary()` notes that `is_outlier` and `outlier_reasons` are derived at call time from mutable global
  configuration, so the same event can summarise differently after `set_event_configurations` ([#27]).

### Known issues

This release does **not** fix everything found in the review. The most important outstanding item:

- **Angular velocity is computed with the wrong transform.** `velocities(unit='deg')` converts a px/s rate
  through a transform meant for a spatial extent, so `peak_velocity`, `median_velocity` and `min_velocity`
  saturate at 180 deg/s and are compressed well below that - a true 500 deg/s saccade reports about
  156 deg/s. **Velocity features from this release are not trustworthy in absolute terms.** A fix is planned
  for a future release.
- `NHDetector` raises at sampling rates of roughly 120 Hz and below, and produces an all-fixation result with
  no warning in a band around 150-166 Hz.
- `is_outlier` still doesn't check dispersion (velocity and acceleration were added - see Added, [#26]).

## [0.1.0] - 2026

Initial release, as used for [Nir & Deouell (2026)](https://doi.org/10.3758/s13428-026-02983-5).

[#24]: https://github.com/huji-hcnl/pEYES/issues/24
[#25]: https://github.com/huji-hcnl/pEYES/issues/25
[#26]: https://github.com/huji-hcnl/pEYES/issues/26
[#27]: https://github.com/huji-hcnl/pEYES/issues/27
