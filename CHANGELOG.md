# Changelog

All notable changes to pEYES are documented here. This project uses [semantic versioning](https://semver.org);
while the major version is `0`, breaking changes bump the **minor** version.

## [0.2.0] - 2026-08-31

A correctness release. It fixes 47 findings from a full review of the package (see
[docs/CODE_REVIEW.md](docs/CODE_REVIEW.md)), several of which change values that previous versions returned.
**Read the breaking changes before upgrading**: some of them alter results silently rather than raising.

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

`match_metrics` true-positive counts now require **both** the ground-truth event and its matched prediction to
carry a positive label. This only changes results when cross-matching is enabled (`allow_xmatch=True`); with
the default `False`, matched predictions already share their ground-truth label and the counts are identical.

### Added

- `start_x` / `start_y` / `end_x` / `end_y` in `summary()` and `summarize_events()`, so a saccade's endpoints
  are recoverable - previously only the midpoint was reported ([#24]).
- `BaseEvent.summary_columns()`, the summary schema as a single source of truth.
- `peyes.__version__`.
- Optional dependency extras: `analysis` (for the scripts under `analysis/`) and `dev`.
- Ruff configuration and a GitHub Actions workflow running lint plus the test suite.

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
  156 deg/s. **Velocity features from this release are not trustworthy in absolute terms.** Fixing it changes
  values published in Nir & Deouell (2026), so it is being handled separately. Tracked as C-1 in
  [docs/CODE_REVIEW.md](docs/CODE_REVIEW.md).
- `NHDetector` raises at sampling rates around 120 Hz and below, and produces an all-fixation result with no
  warning in a band around 150-166 Hz (D-1, D-6).
- `is_outlier` checks only event duration and screen bounds, not velocity, acceleration or dispersion, so it
  is narrower than its name suggests ([#26], C-6).

The full list of open findings, with reasons, is in
[docs/CODE_REVIEW.md](docs/CODE_REVIEW.md) under "Open after phases A-D".

## [0.1.0] - 2026

Initial release, as used for [Nir & Deouell (2026)](https://doi.org/10.3758/s13428-026-02983-5).

[#24]: https://github.com/huji-hcnl/pEYES/issues/24
[#25]: https://github.com/huji-hcnl/pEYES/issues/25
[#26]: https://github.com/huji-hcnl/pEYES/issues/26
[#27]: https://github.com/huji-hcnl/pEYES/issues/27
