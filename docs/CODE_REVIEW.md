# pEYES Code Review

**Scope:** `peyes/` (the installable package) + `tests/`. `analysis/` and `docs/User Guide/` excluded by agreement.
**Reviewed at:** branch `claude/peyes-code-review-4b4661`, base commit `34ab816`.
**Priority ordering of concerns:** correctness > robustness/design > efficiency.

## How to read this

Findings are grouped by **scope** (the module you'd open to fix them), and tagged with a **priority**:

| Tag | Meaning |
|---|---|
| **CRIT** | Silently wrong scientific results, or a public entry point that cannot work at all. |
| **HIGH** | Wrong results or a crash on a common, documented input path. |
| **MED** | Wrong/crashing on a plausible-but-narrower path; or a design flaw that will keep producing bugs. |
| **LOW** | Edge case, poor error behaviour, or an efficiency problem with real cost. |
| **NIT** | Cosmetic, naming, dead code, doc typos. |

Each finding has a stable ID (`C-1`, `D-3`, ...) so it can be referenced in issues/commits.

## Caveats on verification

- **Findings were executed and confirmed.** A populated venv is now available and every CRIT/HIGH finding, plus most MED ones, was reproduced against this branch. Confirmed findings quote their observed output. The few still unverified are marked **[verify]**.
- **The environment is not the declared one**, which is itself evidence for P-1: Python **3.14** (README says 3.12), **numpy 2.5.2** and **pandas 3.0.5** — but `pyproject.toml` pins `numpy~=1.2`, which *excludes* numpy 2.x. The installed environment already violates the declared constraints. The editable `peyes` install also points at a different worktree, so verification was run with `PYTHONPATH` forced to this branch.
- **Upstream issues** (`huji-hcnl/pEYES`) were read and are mapped in §7; `JonNir1/pEYES` has issues disabled.
- **`analysis/` was not reviewed**, but it *was* read to determine which findings could have reached the published results (§8). That triage reflects `analysis/` as it stands at this commit, which may differ from what was actually run for the article.

## Summary, most urgent first

The **Article?** column is the §8 triage: whether the finding could have reached the published results.

| ID | Priority | Article? | Scope | One-liner |
|---|---|---|---|---|
| C-1 | CRIT | **YES** | core / Event | `velocities(unit='deg')` applies a non-linear px to deg map to a px/s rate; deg/s velocities saturate at 180 and are badly compressed. |
| V-1 | CRIT | no | visualize | `create_video` passes args to `create_frames` positionally and misaligns three of them; the function cannot work. |
| D-1 | CRIT | no (latent) | detectors / NH | `num_edge_sample_to_drop == 0` at ~100 Hz makes `ch[0:-0]` empty, so PT/OnT become NaN and NH silently labels everything a fixation. |
| D-2 | HIGH | **YES** | detectors / NH | `np.argmin` where `argmax` is meant: the adaptive peak threshold always initialises to 300 deg/s. |
| C-2 | HIGH | no | core / postprocess | `events_to_labels` is off by one: output is one sample short and each event loses its last sample. |
| C-3 | HIGH | no | core / create | `_events_to_boolean_channel` marks offsets one sample early, disagreeing with the labels-based path. |
| C-4 | HIGH | no | core / create+postprocess | `min_num_samples=None` (the documented default) raises `TypeError` via `max(int(res), None)`. |
| D-3 | HIGH | no | detectors / NH | `__find_local_minimum_index` reads `arr[idx+1]` at the last index, raising `IndexError`. |
| M-1 | HIGH | no | metrics / alignment | `max(threshold)` runs before the int-to-list normalisation, raising `TypeError` for the documented scalar form. |
| M-2 | HIGH | no | metrics / event | `event_rate` divides by absolute end time, not recording duration. |
| B-1 | HIGH | no | core / match | `"offset max_onset_difference"` typo: `match_by="offset difference"` silently falls through to generic matching. |
| B-2 | HIGH | no | core / match | Matcher wrappers default their tolerances to `0`, so `match(..., "onset"/"offset"/"l2")` matches almost nothing. |
| V-2 | HIGH | no | visualize | `create_image` dereferences `image.ndim` before the `image is None` check. |
| V-3 | HIGH | no | visualize | `create_frames` calls `int(x[i])`, raising `ValueError` on any NaN sample. |
| P-1 | HIGH | no | packaging | `authors` carries a non-PEP-621 `website` key; no `requires-python`; version pins are near-meaningless. |
| T-1 | HIGH | no | tests | `test_event.test_init` asserts `"FIXATION(19.00ms)"` but `__str__` yields `"FIXATION(19.0ms)"`. |
| T-5 | HIGH | no | tests | `test_pixel_utils` asserts a stale literal `0.027844`; the correct value is `0.0276855`. |

Everything else is MED and below, listed in the per-scope sections. Several MED findings *do* reach the published results (D-5, D-10, D-11, D-16, C-6, V-4/V-5) and are collected in §8.

## 1. Core data models and pipeline

### `peyes/_DataModels/Event.py`

**C-1 · CRIT · Angular velocity is computed with the wrong transform**
`velocities()` computes px/s first (`calculate_velocities`), then maps the *rate* through `pixels_to_visual_angle`, which is `2*arctan(v*pixel_size / 2d)`, a transform for a spatial *extent*, not a rate. Because `arctan` saturates, the result is bounded by **180 deg/s** (`2*arctan(x) -> pi` rad) and is severely compressed well below that.
*Measured on this branch:* a synthetic saccade at 21,000 px/s (`pixel_size=0.0277`, `d=60`) reports `peak_velocity = 156.7 deg/s`; the linear-constant conversion used by `NHDetector` gives **555.5 deg/s** on the same data. `pixels_to_visual_angle(1e9, 60, 0.0277)` returns `180.000`, confirming the asymptote.
This propagates to `peak_velocity`, `median_velocity`, `min_velocity`, hence `summary()` / `summarize_events()`, `event_summary`, `saccade_summary`, `fixation_summary`, and the peak-velocity main sequence.
Note the correct pattern already exists in the codebase: `NHDetector._velocities_and_accelerations` uses `px_to_deg_constant = pixels_to_visual_angle(1, vd, ps)` and multiplies, i.e. a linear per-pixel constant.
*Fix:* convert per-sample displacement (small, so the transform is accurate) and divide by dt; or simply `px_velocities * pixels_to_visual_angle(1, vd, ps)`.
*Caution:* `tests/unit_tests/data_models/test_event.py::test_velocity` currently asserts the buggy behaviour (`np.vectorize(pixels_to_visual_angle)(expected_px_vel, ...)`) and must be updated with the fix. Since the article's results were produced with this code, changing it changes published feature values, so confirm intent with the maintainer before landing.

**Reaches the published results, and there is a cheap way to confirm it.**
`analysis/_article_results/lund2013/rater_analysis/event-features.ipynb` takes `event.peak_velocity` quantiles and screens saccades against literature thresholds:

```
('peak_velocity', 'min'): 45,     # Andersson et al. (2016)
('peak_velocity', 'max'): 1000,   # Nyström & Holmqvist (2010)
```

Under the saturation, `peak_velocity` **cannot exceed 180 deg/s** for any event, so the 1000 deg/s criterion can never fire, the 45 deg/s screens operate on compressed values, and any saccade whose true peak exceeds 180 deg/s — i.e. most real saccades — is reported below its actual velocity. *Confirmation test:* take the saccade features already produced for the article and check `max(peak_velocity)`. If nothing exceeds 180, the bug is confirmed on the article's own data with no re-run needed.
The peak-velocity panel of `visualize.event_summary` (AppendixA fig 1b) carries the same distortion. `visualize.main_sequence` is called with `y_feature=DURATION_STR`, so that figure is unaffected.

**C-5 · MED · `summary()` omits `start_pixel` / `end_pixel`** — upstream issue #24. `center_pixel` alone cannot reconstruct a saccade's landing site. Both properties already exist (`Event.py:228`, `:238`). Prefer four scalar columns over 2-tuples.

**C-6 · MED · `get_outlier_reasons` checks only duration and screen bounds** — upstream issue #26, with an in-code `TODO` at `Event.py:132`. `is_outlier` reads as a general plausibility filter but does not check velocity/acceleration/dispersion. Either implement the checks or narrow the docstring; leaving it as-is has already cost the reporter debugging time.

**C-7 · LOW · `summary()` re-derives `outlier_reasons` from mutable global config at call time** — upstream issue #27. Behaviour is probably intended (config is global by design); the fix is a docstring note on `summary()` and `is_outlier` that the value is computed at call time, not at construction.

**C-8 · MED · `top/bottom/left/right_pixel` use `argmin`/`argmax`, not the nan-aware variants**
`Event.py:302-324`. With any NaN in `x`/`y`, which is routine after blink handling, `np.argmin` returns the index of the NaN, so these properties return `(nan, nan)`. Use `np.nanargmin`/`np.nanargmax`, guarding the all-NaN case.

**C-9 · MED · Input validation uses bare `assert`**
`Event.__init__:30-32`, and likewise in `set_config.py`, `create_video`, and `EventMatcher.generic_matching`'s integrity checks. Under `python -O` these vanish, so mismatched-length arrays or a negative `pixel_size` proceed silently. Raise `ValueError` instead; keep `assert` only for genuine internal invariants.

**C-10 · LOW · `duration` convention is `end - start`, so an n-sample event has duration `(n-1)*dt`**
Self-consistent within `Event`, but it is the root of the round-trip off-by-one in C-2/C-3, and it makes a single-sample event have `duration == 0` (which then short-circuits `time_overlap`). Worth documenting explicitly on the `duration` property, and worth deciding once whether the package's convention is `n*dt` or `(n-1)*dt`. Both off-by-one bugs below stem from this being unstated.

**C-11 · NIT** — `peak_velocity`/`min_velocity` return `np.float64` while `median_velocity` returns `float`; `make_multiple` assumes `x`/`y`/`pupil` are `np.ndarray` (a list raises `TypeError` on fancy indexing).

### `peyes/_base/postprocess_events.py`

**C-2 · HIGH · `events_to_labels` is off by one, twice**
`num_samples = ceil((end-start)*sr/1000)` under-counts by one sample, and `out[start_sample:end_sample]` excludes each event's final sample.
Worked example, labels `[F,F,F,S,S]` at 500 Hz with `t=[0,2,4,6,8]`: round-tripping through `create_events` then `events_to_labels` yields `[F,F,UNDEFINED,S]` (4 samples) instead of `[F,F,F,S,S]`.
*Fix:* `num_samples = calculate_num_samples(...) + 1` and `out[start_sample:end_sample + 1]`, or settle C-10 first and derive both consistently.

**C-12 · MED · `summarize_events([])` returns a `(0, 0)` frame with no columns** — upstream issue #25. Downstream this breaks `feature_comparison` (`summary_df["is_outlier"]` raises `KeyError`) and `pd.concat` of two empty results. *Fix:* hoist the `summary()` key list into a module constant or a `BaseEvent.summary_columns()` classmethod and use it for the empty branch.

**C-13 · LOW · `events_to_labels([])` raises `ValueError` from `min()` on an empty generator** — needs an explicit empty guard.

**C-14 · LOW · Return type is not what's annotated** — `EventLabelSequenceType` promises `Sequence[EventLabelEnum]`, but `np.full(n, EventLabelEnum.UNDEFINED)` yields an integer array, so elements are `np.int64`.

### `peyes/_base/create.py`

**C-4 · HIGH · `min_num_samples=None` crashes**
`_events_to_boolean_channel:263` and `events_to_labels:158` both pass `min_num_samples` positionally into `calculate_num_samples(..., min_samples)`, which ends in `max(int(res), min_samples)`. With `None`, which is the parameter default and explicitly documented as "If None, the number of samples is determined by the total duration", this is `max(int, None)` and raises `TypeError`. Verified in isolation on Python 3.14. *Fix:* `min_samples=1 if min_num_samples is None else min_num_samples`.

**C-3 · HIGH · `_events_to_boolean_channel` marks offsets one sample early**
`create.py:272` writes `bool_channel[end_sample - 1] = True` while onsets are written at `bool_channel[start_sample]` with no adjustment. With the `(n-1)*dt` duration convention, `end_sample` is already the index of the event's last sample, so the `-1` is spurious. The labels-based path (`_labels_to_boolean_channel`) marks the true last sample of each chunk, so **the two paths disagree by one sample**, and `alignment_metrics` (channel SDT, timing differences) accepts either as input. Any comparison mixing events and labels carries a systematic one-sample offset.

**C-15 · MED · `create_boolean_channel` empty branch is broken**
`create.py:205`: `np.zeros(np.nanmin(len(data), min_num_samples), dtype=bool)`. `np.nanmin`'s second positional argument is `axis`, not a second operand. With `min_num_samples=None` it happens to return `0`; with any integer it raises `AxisError`. *Fix:* `np.zeros(min_num_samples or 0, dtype=bool)`.

**C-16 · MED · No bounds checking when writing onsets/offsets**
`create.py:269, 272` index `bool_channel` with a computed sample index. If `min_num_samples` is smaller than the events' span this raises `IndexError`; a negative index silently writes at the wrong end.

**C-17 · MED · `create_detector` is seven copy-pasted branches and silently drops unknown kwargs**
`create.py:81-146`. `**{k: kwargs.get(k, default_params[k]) for k in default_params}` means a misspelled keyword (`saccade_velocity_treshold`) is silently ignored and the default is used, which is a very easy way to run an experiment with the wrong parameters. *Fix:* a `{name: class}` registry plus one construction path; raise on kwargs not in `get_default_params()`.

**C-18 · LOW · `_labels_to_boolean_channel` builds its output from the raw `labels` argument** (`np.zeros_like(labels, dtype=bool)`) rather than from `parsed_labels`; works for list/array input, fragile otherwise.

**C-19 · NIT · `algorithm.lower().strip().replace('-','').removesuffix('detector')`** — `"IVT Detector"` normalises to `"ivt "` and fails to match. Strip again after `removesuffix`, or normalise whitespace first.

### `peyes/_base/match.py` and `peyes/_DataModels/EventMatcher.py`

**B-1 · HIGH · Copy-paste typo silently changes the matching algorithm**
`match.py:78`: `if match_by == "offset" or match_by == "offset max_onset_difference":`. The second alias is meant to be `"offset difference"`, which `match()`'s own docstring advertises. Today `match(gt, pred, match_by="offset difference")` falls through every branch and lands on `EventMatcher.generic_matching` with `reduction="all"`, returning one-to-many matches instead of offset-difference one-to-one matches. Callers get a structurally different result with no error.

**B-2 · HIGH · Tolerance defaults of `0` make three matchers no-ops**
`EventMatcher.onset_difference(max_onset_difference=0)`, `offset_difference(max_offset_difference=0)`, `l2_timing(max_l2=0)`, `window_based(...=0, ...=0)`, and the corresponding `kwargs.pop(..., 0)` in `match.py:75, 82, 89-90, 95`. `generic_matching` sensibly defaults these to `inf`, but every wrapper overrides with `0`, requiring *exact* onset/offset equality. `peyes.match(gt, pred, "onset")` with no extra kwargs returns (near) nothing, silently. *Fix:* default to `inf` in the wrappers and in `match()`, matching `generic_matching`.

**B-3 · MED · `matched_predictions` is a `set` of `BaseEvent`, which defines value-based `__eq__`/`__hash__`**
`EventMatcher.py:58, 60, 76`. Two distinct predicted events with byte-identical `t/x/y/pupil` are indistinguishable, so consuming one removes both from the candidate pool. Rare, but possible with short synthetic or degenerate events. *Fix:* track by `id()` or by index.

**B-4 · LOW · `EventMatcher.py:72` — `if len(p):` after `__choose_match`** conflates "no match" with an empty reduction result; and `generic_matching`'s output-integrity `assert`s (`:80, :82`) vanish under `-O`.

**B-5 · LOW · Dead branch** — `match.py:48`: `kwargs.pop("allow_xmatch", ...)` can never fire, since `allow_xmatch` is a named parameter and is consumed before `**kwargs`.

**B-6 · NIT** — `match_multiple` is not exported from `peyes/__init__.py`; its docstring refers to a `match_events` function that does not exist. `EventMatcher` is an `ABC` containing only `@staticmethod`s, i.e. a module masquerading as a class.

### `peyes/_utils/`

**C-20 · MED · `is_one_dimensional` accepts `(1, n)` / `(n, 1)`, but `get_chunk_indices`, `merge_chunks` and `reset_short_chunks` only handle true 1-D**
`vector_utils.py:231-236`: `np.arange(len(arr))` on a `(1, n)` array gives length 1, and `np.nonzero(np.diff(arr))[0]` returns row indices. Results are silently wrong rather than raising. *Fix:* `arr = np.asarray(arr).reshape(-1)` at the top of each, or tighten the guard.

**C-21 · LOW · `parse_label` does not handle the case its own comment claims**
`event_utils.py:352-354`: the comment says it handles `"1.0", "2,0"`, but `float("2,0")` raises `ValueError` (verified), so comma-decimal strings become `UNDEFINED` under `safe=True`. Either implement the comma replacement or drop the claim.

**C-22 · LOW · `parse_label(True)` returns `FIXATION`** — `bool` is an `int` subclass and is checked before anything else could reject it.

**C-23 · LOW · `cast_to_integers` truncates toward zero, not down** — `pixel_utils.py:20` uses `astype(int)`; the docstring says "rounding down to the nearest smaller integer", which differs for negative coordinates. Use `np.floor`.

**C-24 · NIT · Duplicate `microsaccade_ratio`** — `event_utils.py:371` (strict `<`, no `zero_division` parameter, unused anywhere) versus `event_metrics/_rates_and_transitions.py:115` (`<=`, public). Delete the former.

**C-25 · NIT · `from peyes._utils.pixel_utils import *` inside `Event.py` and `Detector.py`** — this is how `np`, `cnst`, `Tuple`, `List` reach those modules. It makes the dependency graph invisible and any rename in `pixel_utils` a silent breakage. Import explicitly.

## 2. Detectors (`peyes/_DataModels/Detector.py`)

### NHDetector

**D-1 · CRIT · Threshold estimation degenerates to NaN at ~100 Hz**
`Detector.py:1104` (`_calculate_saccade_thresholds`):

```python
chunks_below_pt = [ch[num_edge_sample_to_drop: -num_edge_sample_to_drop] for ch in chunks_below_pt]
```

`num_edge_sample_to_drop = _calc_num_samples(min_saccade_duration // 3, sr)`. With the default `min_saccade_duration = 10` ms, `10 // 3 = 3` ms, and at 100 Hz `round(3 * 100 / 1000) = 0`. Then `ch[0:-0]` is `ch[0:0]`, i.e. **empty for every chunk**. `np.concatenate` of empties gives an empty selector, `np.nanmean([])` is NaN, `pt` becomes NaN, the `while abs(pt - pt_prev) > 1` condition is False so the loop exits without hitting the `max_iters == 0` guard, and `ont` is NaN. Downstream `v > NaN` is all-False, so no saccades and no PSOs are found and `_classify_samples` labels everything `FIXATION`, with no error and no warning.
*Confirmed on this branch, with a narrower trigger than first stated.* Measured `num_edge_sample_to_drop` by sampling rate: **500 Hz -> 2, 300 -> 1, 200 -> 1, 150 -> 0, 100 -> 0, 60 -> 0**, and `ch[0:-0]` is indeed empty. But at 100 and 120 Hz the run never reaches this code: `_velocities_and_accelerations` raises first, because the Savitzky-Golay window rounds to 2, which is not greater than `polyorder` (see D-6). The **silent** NaN path therefore fires in a band around **150-166 Hz**, where `edge == 0` but `savgol_ws == 3` passes the guard. At or below ~120 Hz the failure is loud instead.
*Fix:* guard with `if num_edge_sample_to_drop > 0:` before trimming (or use `ch[k: len(ch)-k]`), and make a NaN `pt` raise rather than silently exit the loop.
*Article impact: none — latent.* The article's datasets are Lund2013 at 500 Hz (seven trials at 200 Hz) and HFC at 300 Hz, giving `num_edge_sample_to_drop` of 2, 1 and 1 respectively. It is rated CRIT because it is silent, not because it fired: any future user at 100 Hz gets an all-fixation result with no warning.

**D-2 · HIGH · `argmin` should be `argmax` when seeding the peak threshold**
`Detector.py:1078`: `pt = start_pt_options[np.argmin(is_v_above_pt)]`. `start_pt_options` descends `[300, 275, ..., 75]`, so `is_v_above_pt` is `[False...False, True...True]`; `argmin` returns the first `False`, which is index 0 whenever any `False` exists, giving `pt = 300` always. The docstring says "the maximal value ... that has at least one sample with higher velocity", i.e. the first `True`, i.e. `np.argmax`. As written the documented adaptive initialisation never happens.

**D-3 · HIGH · Out-of-bounds read in `__find_local_minimum_index`**
`Detector.py:1267-1278`: the guard is `while 0 < idx < len(arr)` but the body reads `arr[idx + 1]`. At `idx == len(arr) - 1` this is `IndexError`. Triggered by any saccade whose offset search runs to the end of the recording. *Fix:* `while 0 < idx < len(arr) - 1`.

**D-4 · MED · `a_copy = v.copy()` should be `a.copy()`**
`Detector.py:942`. Currently harmless only because `a_copy` is never read afterwards, which is itself a sign that the denoising of acceleration was dropped. Fix both: assign `a.copy()` and either use it or delete the variable.

**D-5 · MED · Saccade/PSO spans overwrite BLINK labels**
`_classify_samples` (`Detector.py:1250-1259`) writes `labels[onset_idx:offset_idx] = SACCADE` unconditionally, and only afterwards computes `is_blinks = labels == BLINK`. Blink samples that fall inside a saccade span are permanently relabelled. Compare `IVTDetector._detect_impl`, which correctly guards with `labels != BLINK`. Same class of issue in `REMoDNaVDetector._detect_impl:1483` and `EngbertDetector._detect_impl:712-713` (Engbert is saved only incidentally, because blink samples are NaN so the ellipse statistic is NaN).

**D-6 · MED · Savitzky-Golay window length may be even**
`_velocities_and_accelerations:1026` computes `ws = round(filter_duration_ms * sr / 1000)` and passes it straight to `savgol_filter`. *Confirmed:* at 100 Hz the default 20 ms filter gives `ws = 2`, which fails the `ws <= polyorder` guard and raises `RuntimeError`, so `NHDetector` is simply unusable at low sampling rates. Force an odd window and pick a minimum that satisfies the polyorder constraint. The raised message also exposes D-9 verbatim: it prints the literal text `{self.sr}Hz`.

**D-7 · LOW · `.max()` on possibly-empty / NaN-containing slices** — `_detect_psos:1197` uses `v[a:b].max()`; an empty slice raises `ValueError`, a NaN makes the comparison silently False. Use `np.nanmax` with an emptiness guard.

**D-8 · LOW · Unbounded fallback in the onset search** — if `__find_local_minimum_index(..., move_back=True)` finds nothing it returns `0`, and `_classify_samples` then labels `[0, offset_idx)` as one giant saccade.

**D-9 · NIT** — missing `f` prefix on the error string at `Detector.py:1029-1030` (prints a literal `{self.sr}`); `ws` described as "ms" in the message but is in samples; `num_edge_sample_to_drop` computes `duration_ms // 3` then converts, while the docstring says `min_saccade_samples // 3`.

### IDT / IDVT

**D-10 · MED · The last `window_size - 1` samples are never labelled**
`IDTDetector._detect_impl:493` loops `while end_idx <= len(t)`, so a tail of `window_size - 1` samples can never be reached.
*Confirmed:* on a 300-sample 500 Hz trace ending in high dispersion, `ws = 28` and the last **27** samples come back `UNDEFINED`, both from `_detect_impl` directly and after the full `detect()` post-processing (`merge_chunks` / `reset_short_chunks` do not rescue them, since the trailing chunk is longer than `min_event_samples`).
*Trigger condition:* only when the loop exits from the non-fixation branch. A trace ending inside an expanding fixation window has its tail covered by the final `labels[start_idx:end_idx]` write, so this does not fire on every trial — it fires on trials ending in a saccade or in high-dispersion data.

**D-11 · MED · IDVT classifies undefined-velocity samples as smooth pursuit**
`Detector.py:624`: `is_smooth_pursuit = ~is_fixation & ~is_saccade`. `calculate_velocities` returns NaN for the first sample and for every sample adjacent to a blink, so those are neither fixation nor saccade and fall into smooth pursuit by default. *Fix:* require a finite velocity, or make smooth pursuit an explicit positive condition.

**D-12 · MED · IDVT bypasses saccade-threshold validation**
`Detector.py:598`: `self._saccade_velocity_threshold = saccade_velocity_threshold` is assigned directly, skipping the `<= 0` check that `IVTDetector.__init__` performs. The diamond MRO (`IDVTDetector(IDTDetector, IVTDetector)`) is also fragile: `IDTDetector.__init__`'s bare `super().__init__(...)` happens to land on `IVTDetector.__init__` with defaults, which is then overwritten. Prefer composition, or pass all parameters explicitly through the chain.

**D-13 · MED · Detector defaults are frozen at import and drift from `set_event_configurations`**
`_DEFAULT_WINDOW_DURATION`, `_DEFAULT_MIN_SACCADE_DURATION_MS`, `_DEFAULT_MIN_FIXATION_DURATION_MS`, `_DEFAULT_MAX_PSO_DURATION_MS`, NH's `_DEFAULT_FILTER_DURATION_MS`, and REMoDNaV's four config-derived defaults are all evaluated at class-definition time from `cnfg.EVENT_MAPPING`. `set_event_configurations(...)` mutates that dict but the class attributes never update, contradicting docstrings such as "Default is the minimal fixation duration from the configuration file". *Fix:* resolve these inside `get_default_params()` / `__init__` using a `None` sentinel.

**D-14 · LOW · Docstring/code mismatch on the IDVT dispersion default** — the docstring says "Default is 2.0 DVA, as used in ... Komogortsev & Karpov (2013)", but the signature uses `IDTDetector._DEFAULT_DISPERSION_THRESHOLD = 0.5`.

**D-15 · NIT** — `IDTDetector._calculate_dispersion_length_px` is dead code (and uses non-nan-aware `max`/`min`); `_calculate_window_size_samples`'s error messages print a sample count labelled "ms"; `__DEFAULT_WINDOW_DURATION_STR` is a key name, not a default.

### Engbert

**D-16 · MED · Axial velocity deviates from the published formula, and from its own docstring**
`_axial_velocities_px:1042-1059`. For the default `window_size=5`, `half_ws = 5//2 + 1 = 3`, so the code computes `(x[n+1]+x[n+2]+x[n+3] - x[n-1]-x[n-2]-x[n-3]) * sr / 5`. Engbert & Kliegl's estimator is `(x[n+2]+x[n+1]-x[n-1]-x[n-2]) / (6*dt)`, i.e. a different span *and* a different normaliser. The docstring also says the sums span `window_size // 2` samples, which is only true for even windows. This shifts the noise threshold and therefore every Engbert detection. Confirm the intended definition with the maintainer before changing, since it affects published results.

**D-17 · MED · Negative variance silently becomes a near-zero threshold**
`_median_standard_deviation:1073`: `sqrt(median(v^2) - median(v)^2)` can be NaN when the radicand is negative. `np.nanmax([nan, 1e-10])` *ignores* the NaN and returns `1e-10`, so the threshold collapses to ~0 and every sample is classified `SACCADE`. Detect the negative radicand explicitly and raise or warn.

**D-18 · LOW · Pure-Python O(n*ws) loop** for axial velocities. A `cumsum`-based sliding sum, or `np.convolve` with a `[+1...0...-1]` kernel, is a one-liner and orders of magnitude faster on long recordings.

**D-19 · NIT** — metadata keys are inconsistent: `"x_threshold_velocity_pxs"` versus `"y_threshold_velocity_px"`.

### BaseDetector / REMoDNaV

**D-20 · MED · `_metadata` accumulates across `detect()` calls**
`BaseDetector.__init__:53` creates `self._metadata = {}` once; `detect()` and every `_detect_impl` only ever `.update()` it. Reusing one detector across trials, which is the normal usage, leaves stale keys from previous trials in the returned metadata. *Fix:* reset `self._metadata = {}` at the top of `detect()`.

**D-21 · MED · Blink padding is asymmetric**
`_detect_blinks:210-212`: `start = max(0, i - pad)` but `end = min(len, i + pad)`, and the slice end is exclusive, so `pad` samples are added before but only `pad - 1` after. Use `i + pad + 1`. The whole loop is also O(n*pad); `scipy.ndimage.binary_dilation` does it in one call.

**D-22 · LOW · REMoDNaV skips all parameter validation** — every other detector validates its arguments in `__init__`; `REMoDNaVDetector.__init__:1400-1417` validates none.

**D-23 · LOW · The remodnav logger level is raised globally and never restored** — `Detector.py:1450-1452`, when `show_warnings=False`. Use a context manager, or restore the prior level.

**D-24 · LOW · REMoDNaV writes label spans inclusively (`[start:end+1]`) while NH writes them exclusively** — consecutive events overlap by one sample and the later one wins. Pick one convention (see C-10).

**D-25 · LOW · `detect()` never checks that `t` is monotonic**, and `calculate_sampling_rate` uses the *mean* inter-sample interval, so a single gap silently distorts the sampling rate used for every duration-to-samples conversion.

**D-26 · LOW · Detector objects carry per-call state (`self._sr`, `self._metadata`)** — not reentrant or thread-safe; parallelising over trials with a shared detector will interleave.

**D-27 · NIT · `dtype=EventLabelEnum` in `np.full_like` / `np.asarray`** (`Detector.py:78`, and in each `_detect_impl`) is not a numpy dtype. *Confirmed on numpy 2.5.2:* it resolves to **`dtype=object`**, so every label array carries per-element Python objects and every mask operation on it dispatches through Python — which is presumably why the `[parse_label(l) for l in labels]` clean-up exists at `Detector.py:88`. It does not raise, so this is a memory and performance cost rather than a crash. Prefer an explicit integer dtype and convert once at the boundary.

## 3. Metrics

### `peyes/alignment_metrics/`

**M-1 · HIGH · `max(threshold)` runs before `threshold` is normalised**
`_signal_detection_metrics.py:117` calls `max(threshold) + 1`; the `isinstance(threshold, int)` to `[threshold]` normalisation is at `:122`, five lines later. `onset_detection_metrics(gt, pred, threshold=5)`, the documented scalar form (`:33` "int or array-like of int"), raises `TypeError: 'int' object is not iterable`. Move the normalisation above the `timing_differences` call. Note a numpy scalar (`np.int64`) fails at the *other* branch, so widen the check to `numbers.Integral`.

**M-3 · LOW · `N` is a non-integer window count** — `:130` computes `n = (len(gt_channel) - (2t+1)*p) / (2t+1)` as a float and feeds it to `dprime_and_criterion` as if it were a count. The `TODO` at `:95` about false-alarm rates exceeding 1 is the visible symptom of this framing; worth documenting the derivation.

**M-4 · NIT** — the sign convention of `timing_differences` (`pred_idx - gt_idx`) is not stated in the docstring.

### `peyes/sample_metrics/` and `peyes/_utils/metric_utils.py`

**M-5 · MED · `_dprime_rates` checks the raw `correction` for the log-linear branch instead of the normalised `corr`**
`metric_utils.py:98`: `if correction in {"ll", "loglinear", "log_linear", "hautus"}`, but `corr` (built at `:84` precisely to normalise spacing/dashes/case) is what the Macmillan branch at `:87` uses. So `correction="log linear"` or `"Log-Linear"` normalises to `"log_linear"`, misses the raw comparison, and falls through to `raise ValueError(f"Invalid correction: ...")`. Existing tests only pass the already-normalised spelling, so this is uncaught. *Fix:* use `corr` on line 98.

**M-6 · MED · Multi-label d-prime counts a hit only on exact label equality**
`_calculate_metrics.py:232`: `tp = count(pred == gt and gt in pos_labels)`, while `pp = count(pred in pos_labels)` at `:231` uses the set-membership framing. With `pos_labels={SACCADE, PSO}`, a `gt=SACCADE / pred=PSO` sample is positive-predicted *and* positive-actual, i.e. an SDT hit, but is counted as a miss. `tp` and `pp` are computed under two different definitions of "positive". *Fix:* `tp = count(gt in pos_labels and pred in pos_labels)`.

**M-7 · MED · `pos_labels=None` defaults to *all* labels**, making `n = 0` and every SDT rate NaN (`_calculate_metrics.py:216-217`). Compare `match_metrics._extract_contingency_values:195`, which correctly raises on the all-labels case. Make the sample-metrics path raise too.

**M-8 · MED · `transition_matrix` is not reindexed to the full label set**
`metric_utils.py:28`: `unstack(fill_value=0)` produces only rows/columns for labels present in the sequence, so matrices from different detectors have different shapes and cannot be stacked, subtracted, or compared. Reindex to `list(EventLabelEnum)`. It also raises on sequences shorter than 2.

**M-9 · LOW · `confusion_matrix` passes `list(set(...))` as `labels`** (`_counts_and_matrices.py:291`) — silently deduplicates and discards the caller's ordering. `list(dict.fromkeys(...))` preserves both.

**M-10 · LOW · Return type varies with the number of requested metrics** — `calculate(...)` returns a bare float for one metric and a dict for several (`_calculate_metrics.py:183`); the same pattern is in `event_metrics.get_features` and `match_metrics.get_features`. Callers must special-case. Also `results` is keyed by the caller's raw string, so requesting the same metric twice collapses to one entry and silently changes the return *type*.

**M-11 · LOW · The whole of `calculate()` runs under `warnings.simplefilter("ignore", UserWarning)`** — this suppresses sklearn's "ill-defined metric" warnings, which are exactly the signal a user needs when a class is absent.

**M-12 · LOW · `normalized_levenshtein_distance` divides by `len(gt)` with no zero guard.**

### `peyes/event_metrics/`

**M-2 · HIGH · `event_rate` and `microsaccade_rate` divide by absolute end time, not recording duration**
`_rates_and_transitions.py:96-100` and `:112`: `len(label_events) / events[-1].end_time * 1000`. If the recording's timestamps do not start at 0, which is routine for dataset timestamps and for any trial-sliced sequence, the denominator is inflated and the rate is silently too low. *Fix:* `events[-1].end_time - events[0].start_time`. Both also assume `events` is sorted by time and index `events[-1]` without an emptiness guard.

**M-13 · MED · `features_by_labels` drops the `count` column when the input is empty**
`_get_features.py:27-29`: the `if aggregated.empty: return` early-exit happens before `count` is assigned. `visualize.event_summary` with `show_outliers=True` and no outliers then raises `KeyError: 'count'` at `_event_summary.py:70`.

**M-14 · LOW · `get_features` supports only 6 of the 19 features `summary()` produces** — no `peak_velocity`, `dispersion`, `ellipse_area` and so on, though `feature_relationship` reads them straight off the summary frame. The two feature vocabularies should be unified.

**M-15 · NIT · `feature_lower.removesuffix('s')`** (`_get_features.py:60`) mangles any feature name legitimately ending in `s`.

### `peyes/match_metrics/`

**M-16 · MED · `positive_label=None` raises `TypeError` despite being typed `Optional`**
`_match_evaluation.py:192-194`: `None` is not in `UnparsedEventLabelType`, so it reaches `set(parse_label(l) for l in positive_label)` and fails on iteration. Either handle `None` explicitly or drop `Optional` from the four public signatures that advertise it.

**M-17 · MED · `tp` counts matched *predictions* only**
`_match_evaluation.py:200`: `tp = len([e for e in matches.values() if e.label in positive_label])`. When cross-matching is enabled a negative-label GT event can be matched to a positive-label prediction and still be counted as a true positive. Check both sides via `matches.items()`.

**M-18 · LOW · `match_ratio` assumes one-to-one matches** — `matches.values()` yields lists under `reduction="all"`, and `.label` then fails. Given B-1 routes `"offset difference"` into exactly that path, this is reachable.

## 4. Visualization (`peyes/visualize/`, `peyes/_utils/visualization_utils.py`)

**V-1 · CRIT · `create_video` misaligns three arguments**
`_video.py:334`:

```python
frames = create_frames(x, y, labels, resolution, bg_image, bg_image_format, label_colors, gaze_radius, verbose)
```

`create_frames`'s signature is `(x, y, labels, resolution, bg_image, bg_image_format, bg_image_alpha, label_colors, gaze_radius, verbose)`. So `label_colors` lands in `bg_image_alpha`, `gaze_radius` in `label_colors`, `verbose` in `gaze_radius`, and `verbose` is never passed. The first thing `create_frames` does is `create_image(resolution, bg_image, <a dict or None>, ...)`, then `get_label_colormap(<an int>)`, which calls `.items()` on an `int`. The function cannot succeed on any input. *Fix:* call with keyword arguments.

**V-2 · HIGH · `create_image` dereferences `image` before the `None` check**
`visualization_utils.py:82`: `if image.ndim != 4 and (alpha < 0 or alpha > 1)`, but `image=None` is the default and the branch at `:86` exists precisely to handle it. Any call without a background image raises `AttributeError: 'NoneType' object has no attribute 'ndim'`, and that is the documented default path for `gaze_heatmap`, `gaze_trajectory` and `create_frames`. *Fix:* validate `alpha` unconditionally and move the check below the `None` handling. (`ndim != 4` is also the wrong test: images are 2-D or 3-D.)

**V-3 · HIGH · `create_frames` crashes on NaN gaze samples**
`_video.py:375`: `curr_x, curr_y = int(x[i]), int(y[i])` raises `ValueError: cannot convert float NaN to integer`. Blinks and tracker loss guarantee NaNs in real data, and the pipeline itself NaNs blink samples. Skip non-finite samples, or draw the previous position.

**V-4 · MED · Scarfplot colours are wrong unless all six labels are present**
`_scarfplot.py:240-241, 264-271`: `label_colors` is filtered to the labels actually in the sequence, and `_discrete_colormap` spreads those `k` colours evenly over the normalised `[0, 1]` colourscale, but the heatmap keeps `zmin=min(EventLabelEnum)`, `zmax=max(EventLabelEnum)` (0-5). With, say, only `FIXATION` and `SACCADE` present, `z=1` maps to 0.2 and `z=2` to 0.4, both landing in the first colour bin, so saccades render in the fixation colour. *Fix:* build the colourscale over the full 0-5 range and select colours by label value, not by position among the present labels.

**V-5 · MED · Scarfplot colourbar ticks are placed in arbitrary units**
`_scarfplot.py:244`: `tick_centers = tick_centers * colorbar_length * (max - min) / len(tick_centers)`. The tick values must be in `z` units (0-5) to line up with the colour bands; this expression produces neither `z` units nor `[0, 1]`. The commented-out "Normalize tick centers to [0,1]" note above it suggests this was known-unfinished. **[verify]** visually. Fix alongside V-4.

**V-6 · MED · `feature_comparison` colour fallback indexes a label-keyed dict by position**
`_features.py:332-335`: `colors[i]` where `colors` is `get_label_colormap(...)`, keyed by `EventLabelEnum` and by detector-name strings. For `i <= 5` this silently returns *event-label* colours for what are supposed to be per-sequence colours; for `i >= 6` it raises `KeyError`. And since `labels` defaults to `list(range(n))` (ints), the `seq_name.strip()` fallbacks on the same line raise `AttributeError` for `i >= 6`. So comparing 7 or more sequences without an explicit `colors=` always fails. *Fix:* use a dedicated qualitative sequence (`_DISCRETE_COLORMAP`) indexed modulo its length.

**V-7 · MED · `__pixel_counts` does not bound-check**
`_gaze.py:277`: `counts[int(y_), int(x_)] = count`. Coordinates outside the screen either raise `IndexError` or, for negatives, wrap silently to the opposite edge, putting off-screen gaze in the wrong place on the heatmap. Filter to the valid range first.

**V-8 · MED · Video colours are channel-swapped** — `get_label_colormap` returns RGB tuples, but the frame is BGRA (`create_image` converts everything to BGRA) and `cv2.circle` expects BGR. Gaze markers render with R and B exchanged. Convert at the call site in `_video.py:376-377`.

**V-9 · LOW · `_visualize_gaze_trajectory`'s own default `marker_color` is unusable** — `_gaze.py:212`: `np.full_like(x, "#000000")` tries to parse a hex string into a float array. Only `gaze_trajectory`'s always-array `marker_color=t` keeps the public path alive.

**V-10 · LOW · `gaze_over_time` reads `vert_line_color` but documents `vert_line_colors`** (`_gaze.py:154` versus `:105`), so the documented keyword is silently ignored. The loop at `:158` also rebinds `v`, shadowing the velocity parameter. The x-axis is labelled `"time (sample)"` though `t` is in ms.

**V-11 · LOW · Unguarded divisions in figure code** — `_gaze.py:57` (flat heatmap gives 0/0), `_gaze.py:148` (`xy_max / v_max` with `v_max == 0`), `_features.py:465` (`px.get_trendline_results` when no trendline was requested) **[verify]**.

**V-12 · LOW · `_write_video` calls `os.makedirs(os.path.dirname(output_path))`** — `dirname` of a bare filename is `""`, which raises `FileNotFoundError`.

**V-13 · LOW · `gaze_heatmap`'s `scale` keyword has no effect** — `_gaze.py:54-57` multiplies the counts by `scale` and then min-max normalises, cancelling it exactly. Either remove the keyword or apply it after normalisation.

**V-14 · LOW · `_create_single_event_figure` indexes `events[0]` without an emptiness guard** (`_event_summary.py:183`); `show_legend=i == 1` (`:214`) silently assumes `num_cols == 2`.

**V-15 · NIT · `get_label_colormap` does not parse its keys** — a user passing `{"fixation": "#abc123"}` gets no override, because the defaults are keyed by `EventLabelEnum`. Run keys through `parse_label`.

**V-16 · NIT · Open `TODO`s** — `_gaze.py:14` (labelled x-y trajectory figure, never implemented) and `_features.py:461` (`color_discrete_map` for default event colours; this is also the cleanest fix for V-6's cousin in `feature_relationship`).

## 5. Datasets (`peyes/_DataModels/DatasetLoader.py`)

**S-1 · MED · Redundant O(n^2) loop in the HFC parser**
`DatasetLoader.py:463-472`: `for _, row in annotations.iterrows():` never uses `row`. The body rebuilds `interp1d` and re-applies it to the *entire* `annotations` frame each pass, producing the same `labels` array `len(annotations)` times. The result is correct; the cost is quadratic in the annotation count, per rater, per trial. *Fix:* hoist `f = interp1d(...)` out and drop the loop entirely.

**S-2 · MED · Extrapolated annotation indices are written without bounds checking**
Same block, `:472`: `interp1d(..., bounds_error=False, fill_value="extrapolate")` can return indices below 0 or at/above `len(data)`; `labels[list(fixation_samples)] = 1` then raises or wraps. Clip to `[0, l-1]`.

**S-3 · MED · IRF reconstructs coordinates with per-axis pixel sizes but reports a single diagonal pixel size**
`__correct_coordinates:353-368` uses `width_cm / width_px` for x and `height_cm / height_px` for y (non-square pixels), while `__PIXEL_SIZE_CM_VAL`, the value written into the dataframe and used by every downstream visual-angle calculation, is the diagonal-based square-pixel size from `calculate_pixel_size`. Positions and angles are therefore derived under two different pixel models. Worth confirming which is intended; if the per-axis one is correct, downstream needs per-axis sizes too. The `.apply()` over every sample is also a Python-level loop over the whole dataset and is worth vectorising.

**S-4 · MED · No timeout and no streaming on dataset downloads**
`BaseDatasetLoader.download:66`: `req.get(cls._URL)` with no `timeout=` can hang indefinitely, and `io.BytesIO(response.content)` holds the entire archive in memory, which GazeCom's own docstring warns is "extremely large". Add a timeout and stream to a temporary file.

**S-5 · LOW · Lund2013 merges raters by index alignment**
`:219`: `existing_df.loc[:, rater] = gaze_data.loc[:, rater]`. If two raters' files for the same trial differ in length, pandas aligns on the index and silently fills NaN rather than failing.

**S-6 · LOW · `load()` uses exception flow for the cache-miss path** — `:45-52` catches only `FileNotFoundError`/`TypeError`, so a truncated or unpickleable cache file propagates an opaque error. Check `directory` explicitly and `os.path.isfile` first. Note also that this path calls `pd.read_pickle` on a caller-supplied path; fine for a cache the package wrote, but worth a docstring caveat.

**S-7 · LOW · `GazeComDatasetLoader.load_zipfile` uses `posixpath` for filesystem paths** (`:351-355`) — works on Windows by accident (`os.stat` tolerates mixed separators) but should be `os.path`.

**S-8 · NIT** — `download()` validates via `cls.url()` then requests `cls._URL` (`:65-66`); `tqdm(enumerate(...))` in all four loaders yields no progress percentage (wrap the sequence, not the enumerate).

## 6. Packaging, tests, and repo hygiene

**P-1 · HIGH · `pyproject.toml` problems**

- `authors` entries carry a `website` key. PEP 621 permits only `name` and `email` in `authors`. Move the URL into `[project.urls]`. **[verify]** — `hatchling` is not installed in the available venv, so whether it hard-rejects the key or silently drops it is still untested; run `python -m build` to settle it.
- **No `requires-python`** (confirmed absent from `pyproject.toml`). The package uses `str.removesuffix` (3.9+) and PEP 585 builtin generics in annotations; pip will happily install it on 3.8 and fail at import. Add `requires-python = ">=3.12"` to match the README.
- **Version pins are effectively unbounded below.** `numpy~=1.2` means `>=1.2, <2`, i.e. it admits numpy 1.2 (2008) and *excludes* numpy 2.x, which the issue reporters are actually running. Same shape for `scipy~=1.1`, `statsmodels~=0.1` (`>=0.1, <1`), `python-Levenshtein~=0.2`. Pin to the minor versions actually tested (`numpy~=1.26`, etc.) and decide explicitly about numpy 2.
- `scikit_posthocs` is not imported anywhere under `peyes/`; it is an `analysis/` dependency and should not be a runtime requirement of the installed package. (`statsmodels` *is* needed, transitively, for plotly's `trendline="ols"`; worth a comment saying so.)

**T-1 · HIGH · A unit test asserts a string the code cannot produce**
`tests/unit_tests/data_models/test_event.py:26` expects `"FIXATION(19.00ms)"`; `BaseEvent.__str__` (`Event.py:401`) is `f"{self.label.name}({self.duration}ms)"` with `duration` a float, giving `"FIXATION(19.0ms)"`. *Confirmed:* `AssertionError: 'FIXATION(19.0ms)' != 'FIXATION(19.00ms)'`. Either fix the expectation or format the duration in `__str__`.

**T-5 · HIGH · A second unit test fails, on a stale magic number**
`tests/unit_tests/utils/test_pixel_utils.py:44` asserts `np.isclose(0.027844, calculate_pixel_size(TOBII_WIDTH, TOBII_HEIGHT, TOBII_RESOLUTION))`. The actual value is **0.0276855**; the test fails.
`calculate_pixel_size` is not at fault — the two exact assertions above it in the same test (`1x1 @ 1x1 -> 1`, `1x1 @ 2x2 -> 0.5`) pass, and `53.1 x 30.0 cm` over `1920 x 1080` genuinely gives 0.0276855 (a 60.99 cm diagonal). The literal `0.027844` corresponds to a 61.34 cm diagonal and matches no configured geometry. `git log -L` shows `TOBII_WIDTH, TOBII_HEIGHT = 53.1, 30.0` was introduced once and never changed, so the constant is not the thing that drifted — the expected value was simply wrong when written. Replace it with the computed value, or assert against the formula rather than a literal.

**T-6 · NIT · Stray `print()` in the test suite** — `tests/unit_tests/utils/test_pixel_utils.py:74` prints a velocity array on every run.

**T-2 · HIGH · Test coverage is confined to `_utils` and `Event`**
Measured on this branch: **28 tests across 6 modules, 2 failing** (T-1, T-5), one module (`test_visualization_utils`) running zero tests. There are no tests at all for `Detector` (1548 lines, seven algorithms), `EventMatcher`, `DatasetLoader`, any of the four metrics packages, `_base/{create,match,parse,postprocess_events,set_config}.py`, or `visualize/` — which is precisely where the CRIT/HIGH findings above live. Highest-value additions, in order:

1. A labels-to-events-to-labels round-trip (catches C-2, C-3, C-10).
2. `create_boolean_channel` onset/offset parity between the labels path and the events path (C-3, C-15, C-16).
3. Each `match_by` alias in `match()` returning the matcher it names (B-1, B-2).
4. A synthetic 100 Hz trace through every detector, asserting a non-degenerate label distribution (D-1, D-2, D-10, D-11).
5. `summarize_events([])` schema (C-12) and `_dprime_rates` with unnormalised correction spellings (M-5).

**T-3 · MED · `unittest discover` does not work**
`tests/` and its subpackages have no `__init__.py`, so `python -m unittest discover -s tests` reports the start directory as not importable (already noted in `CLAUDE.md`). Add the `__init__.py` files, or migrate to pytest; the latter also gives parametrisation, which would compress the detector matrix considerably.

**T-4 · MED · Empty test stubs presented as passing** — `test_event.py::test_make` is `self.assertTrue(True)` under a `# TODO`, and `test_visualization_utils.py` is an empty class. These inflate the apparent test count.

**P-2 · MED · No linter, formatter, or CI** — nothing in the repo would have caught V-1 (argument misalignment), D-9 (missing `f` prefix), or B-1/B-5 (dead comparison branches), all of which `ruff` flags by default. A minimal GitHub Actions job running the test suite on 3.12 plus `ruff check` would be high-leverage.

**P-3 · NIT · `constants.py:47` `TODO: replace these with enum`** — `IMAGE_STR`/`VIDEO_STR`/`MOVING_DOT_STR` are compared as bare strings in `DatasetLoader.__extract_metadata`; a `StrEnum` would make the stimulus-type vocabulary discoverable.

## 7. Upstream issue status

| Issue | Status against this code |
|---|---|
| [#24] `summary()` omits `start_pixel`/`end_pixel` | **Open** — see C-5. |
| [#25] `summarize_events([])` has no columns | **Open** — see C-12; note the downstream `KeyError` in `feature_comparison`. |
| [#26] `get_outlier_reasons` ignores velocity/acceleration/dispersion | **Open** — see C-6; the `TODO` is still at `Event.py:132`. |
| [#27] `summary()` re-derives `outlier_reasons` from mutable config | **Open** — see C-7; docs-only fix. |
| [#18] python-poppler breaks installation | **Appears resolved** — `python-poppler` is no longer in `pyproject.toml` dependencies. Worth closing, or reopening against P-1 if it regressed. |
| [#15] kaleido hangs on `write_image` | **Worked around** — pinned to `kaleido==0.1.0post1` with the `TODO` comment retained. Revisit now that kaleido 1.x exists. |

## 8. Impact on the published results

This section drives the work ordering in §9, which clears §8b (no article impact) first and defers §8a.

Determined by reading `analysis/` for call sites and argument values. The article pipeline runs seven detectors (`ivt, ivvt, idt, idvt, engbert, nh, remodnav`) over **Lund2013** (500 Hz; seven trials at 200 Hz) and **HFC** (300 Hz), with `_default_detector_params = dict(missing_value=np.nan, min_event_duration=4, pad_blinks_time=0)`.

One structural point applies throughout: **a bug that raises cannot have silently corrupted results.** If `create_video`, `M-1` or `D-3` had fired, the pipeline would have died rather than produced a wrong number. Those are severe for users, not for the article.

### 8a. Reaches the published results

| ID | Pri | What it touches | Note |
|---|---|---|---|
| **C-1** | CRIT | All `peak/median/min_velocity` values | `event-features.ipynb` quantiles and the 45/1000 deg/s screens; `event_summary` peak-velocity panel. Confirmable without a re-run (see C-1). |
| **D-2** | HIGH | NH saccade thresholds | NH ran on both datasets; PT always seeds at 300 deg/s. |
| **D-16** | MED | Engbert detections | Ran with `lambda_param=6` on both datasets. |
| **D-5** | MED | NH + REMoDNaV labels | Blink samples inside a saccade/PSO span are relabelled. |
| **D-10** | MED | IDT + IDVT labels | The trailing `window_size - 1` samples of every trial stay UNDEFINED. |
| **D-11** | MED | IDVT labels | NaN-velocity samples (first sample, blink-adjacent) default to SMOOTH_PURSUIT. |
| **D-24** | LOW | REMoDNaV labels | Inclusive spans; each event takes one sample from its predecessor. |
| **C-6, C-7** | MED/LOW | Which events appear in figures | `event_summary(show_outliers=False)` and `main_sequence(include_outliers=False)` filter on `is_outlier`, so the narrow outlier definition selects the plotted set. |
| **V-4, V-5** | MED | AppendixA scarfplot | Colours and colourbar ticks; appearance only, no numbers. |
| **D-20** | MED | Recorded detector metadata | Detectors are built once and reused across all trials, so `_metadata` carries stale keys. Affects the saved metadata, not the detections. |

**Conditional — depends on run configuration, worth checking:**

- **M-6 / M-7** (multi-label d-prime and the all-labels default) matter only if a run passed more than one positive label. The pipeline supports it (`temporal_alignment.py:127`, `sample_metrics.py:31`), so check the `pos_labels` each published run used.
- **D-17** (Engbert negative variance) would have made a trial come out all-saccade — visibly wrong rather than subtly wrong, so it probably did not fire, but it is not provably excluded.

### 8b. Cannot have reached the published results

**Never called** (0 references across `analysis/`):

| ID | Bug | Evidence |
|---|---|---|
| V-1, V-3, V-8, V-12 | all video-path bugs | `create_video` / `create_frames`: 0 files |
| C-2 | `events_to_labels` off-by-one | 0 files |
| C-3, C-4, C-15, C-16 | events branch of `create_boolean_channel` | channel metrics are called with **label** DataFrames (`gt_labels, pred_labels`); `analysis/process/_unused_events_channel_.py` is named for exactly this |
| M-2 | `event_rate` denominator | `event_rate` / `saccade_rate` / `microsaccade`: 0 files |
| C-5, C-8 | missing `start/end_pixel`; `argmin` on NaN | `start_pixel`, `top_pixel` etc.: 0 files |
| M-8 | `transition_matrix` not reindexed | 0 files |

**Trigger condition never met:**

| ID | Bug | Why it did not fire |
|---|---|---|
| D-1 | NH degenerates at 100 Hz | 500/300/200 Hz give `num_edge_sample_to_drop` of 2/1/1, never 0 |
| D-3 | `IndexError` in NH offset search | a crasher; the pipeline completed |
| B-1 | `"offset difference"` alias typo | analysis uses the bare `match_by='offset'`, which hits the correct branch |
| B-2 | tolerance defaults of `0` | every call site passes explicit `max_l2=15`, `max_onset_difference=w`, `min_iou=1/3` |
| M-1 | scalar `threshold` crash | passed `np.arange(21)`, an array |
| M-5 | unnormalised correction spelling | passed `"loglinear"`, already normalised |
| V-2 | `create_image` on `None` | `gaze_heatmap` called with `bg_image=img` |
| V-6 | `feature_comparison` colour fallback | called with explicit `colors=` and `labels=` |
| M-13 | `count` `KeyError` | `event_summary(..., show_outliers=False)` |
| D-21 | asymmetric blink padding | `pad_blinks_time=0`, so the padding loop early-returns |
| D-13 | frozen config-derived defaults | `set_event_configurations` / `set_viewer_distance` / `set_screen_monitor` are never called, so the frozen defaults equal the live config |
| C-12 | `summarize_events([])` | would have raised; no labeler produced zero events |
| S-1 | HFC O(n^2) annotation loop | wasteful, but produces the correct array |

**Outside the result path entirely:** T-1, T-2, T-3, T-4, P-1, P-2, P-3.

## 9. Suggested order of work

Ordered to clear §8b (no effect on the published results) **before** anything in §8a. The article-affecting work is deferred to phase E, where it can be taken as one decision.

Beyond sequencing convenience, this order has a technical advantage: phases A–C build the test suite on code paths where changing behaviour is free, so by the time phase E touches code that moves published numbers, there is a harness to catch regressions.

Two deliberate departures from a strict §8b-then-§8a split: the **C-1 diagnostic** is pulled forward into phase A (it is read-only and changes nothing), and the **article-neutral `NHDetector` findings** are pushed back into phase E to keep that class to a single branch.

**`NHDetector` is exempt from phases B–D.** All NH work is deferred to phase E, including the two article-neutral NH findings (D-1, D-3) — see "The NH exemption" below.

### Phase A — environment, test suite, and the C-1 diagnostic

Nothing else can be verified until this is done; the venv at `<repo>/.venv` currently holds only `pip` and is Python 3.14.

1. Create a working 3.12 environment and `pip install -e .`.
2. **Run the C-1 confirmation test — read-only, commits to nothing.** Check `max(peak_velocity)` across the saccade features already generated for the article. A ceiling at 180 deg/s confirms C-1 from existing outputs, with no re-run and no code change. Running it here rather than in phase E means the phase-E decision is made with the answer already in hand instead of as an open question, and the result may reasonably change how much of phase E you schedule.
3. **T-3** — add `__init__.py` to `tests/` and its subpackages so `unittest discover` runs at all.
4. **T-1** — fix the `"FIXATION(19.00ms)"` expectation so the suite is green and a red result means something.
5. **T-5** — the second red test: replace the stale `0.027844` literal in `test_pixel_utils` with the computed value. **T-6** — drop the stray `print()` at line 74.
6. **T-4** — delete or implement the two empty stubs (`test_make`, `TestVisualizationUtils`).
7. **P-1** — add `requires-python = ">=3.12"`, drop the non-PEP-621 `website` key, tighten the version pins, remove `scikit_posthocs` from runtime deps.
8. **P-2** — add `ruff` and a CI job. Do this *before* phase B: ruff flags V-1, D-9 and B-5 automatically, and will keep catching that class of bug for free.

### Phase B — public entry points that cannot work

Crash-on-first-use bugs. Each is a few lines, each is trivially unit-testable, and none can touch the article because none of it was ever called.

9. **V-1** — `create_video` argument misalignment. `peyes.visualization.create_video` has never worked for anyone.
10. **V-2** — `create_image` dereferencing `image.ndim` before the `None` check. Fixing this unblocks the default path of `gaze_heatmap`, `gaze_trajectory` and `create_frames`.
11. **V-3, V-8, V-12** — the rest of the video path: NaN samples, RGB/BGR swap, `makedirs("")`.
12. **M-1** — normalise `threshold` before `max(threshold)`; widen the check to `numbers.Integral` while you are there.
13. **C-4, C-15, C-16** — the `min_num_samples` / `np.nanmin` / bounds-check cluster in `create_boolean_channel`. Fix as one commit; they are the same three lines of neighbourhood.

### Phase C — silent wrong results in paths the article never used

Live for every downstream user, latent for the article.

14. **C-10 first, and it is a decision, not a fix.** Settle and document whether an n-sample event has duration `n*dt` or `(n-1)*dt`. C-2, C-3 and D-24 all follow from the answer, so fixing them in any other order means doing them twice.
15. **C-2, C-3** — the two off-by-one bugs, once C-10 is settled. Add the labels→events→labels round-trip test here; it is the single highest-value test in the suite.
16. **B-1, B-2** — the `"offset difference"` alias typo and the `0` tolerance defaults. Add a test asserting each documented `match_by` alias reaches the matcher it names.
17. **M-2, M-5, M-8** — `event_rate` denominator, the `corr`/`correction` mix-up, transition-matrix reindexing.
18. **C-5, C-12** — upstream issues #24 and #25 (`start/end_pixel` in `summary()`; empty-frame schema). Small, and they close two open issues.
19. **C-8, C-13, C-14** — nan-aware extremum properties, empty guards, return-type honesty.

### Phase D — robustness and design, still article-neutral

20. **C-9** — replace validation `assert`s with `ValueError`.
21. **C-20** — make `get_chunk_indices` / `merge_chunks` / `reset_short_chunks` agree with `is_one_dimensional`.
22. **M-16, M-17** — `positive_label=None`, and `tp` counting both sides of a match.
23. **S-2, S-4** — annotation index clipping; download timeout and streaming.
24. **D-21, D-22, D-23** — blink-padding asymmetry, REMoDNaV validation, the un-restored logger level. D-21 is in `BaseDetector` and therefore inherited by NH, but its fix is inert at `pad_blinks_time=0` (the article's setting), so it cannot move published numbers and stays in this phase.
25. **C-17, M-10, D-12** — the design cleanups: detector registry, return-type consistency, the IDVT diamond.

### Phase E — deferred: everything that moves published numbers, plus all NH work

Hold until phases A–D are merged, then take as **one** decision rather than piecemeal.

26. **Decide with the maintainer,** informed by the step-2 diagnostic: do the article's values stand with an erratum, or are the affected analyses regenerated? That single answer governs C-1, D-2, D-16, D-5, D-10, D-11, D-24, C-6/C-7 and V-4/V-5.
27. **Check the conditionals** (M-6/M-7 — what `pos_labels` did the published runs use? — and D-17) so the decision is made with the full list in hand.
28. **All `NHDetector` work, in one branch:** D-2 and D-5 (article-affecting) together with D-1, D-3, D-6, D-7, D-8 (article-neutral, deferred only to keep the class untouched until now).
29. **If regenerating,** fix the rest of the §8a set in the same branch and re-run once. **D-20** (metadata accumulation) should go in regardless, since it affects only recorded metadata.

### The NH exemption

D-1 and D-2 sit about 25 lines apart inside `_calculate_saccade_thresholds`, and D-3, D-6, D-7 and D-8 are elsewhere in the same class. D-2 and D-5 move published numbers; the rest do not. Rather than edit `NHDetector` in two phases and split the history, **all** NH findings are deferred to phase E and land in one branch.

The cost is that **D-1 stays open the longest of any CRIT finding.** That is a deliberate trade, but it is worth being explicit about what it means for users in the interim: anyone running `NHDetector` at ~100 Hz gets an all-fixation result with no error and no warning. If phase E ends up scheduled far out, consider carving out the single `if num_edge_sample_to_drop > 0:` guard — or even just a warning when `pt` comes out non-finite — as an isolated commit. It touches no article-affecting line and would remove the silent-failure mode without pre-empting the phase-E decision.

[#24]: https://github.com/huji-hcnl/pEYES/issues/24
[#25]: https://github.com/huji-hcnl/pEYES/issues/25
[#26]: https://github.com/huji-hcnl/pEYES/issues/26
[#27]: https://github.com/huji-hcnl/pEYES/issues/27
[#18]: https://github.com/huji-hcnl/pEYES/issues/18
[#15]: https://github.com/huji-hcnl/pEYES/issues/15

## 10. Verification log

Run against this branch with `PYTHONPATH` forced to the worktree (the editable `peyes` install resolves to a different worktree). Environment: Python 3.14.3, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, plotly 7.0.0, scikit-learn 1.9.0.

| Finding | Observed |
|---|---|
| T-1 | `AssertionError: 'FIXATION(19.0ms)' != 'FIXATION(19.00ms)'` |
| T-5 | `test_calculate_pixel_size` fails: expected `0.027844`, actual `0.0276855` |
| T-2 | 28 tests / 6 modules, **2 failing**, 1 module with zero tests |
| C-1 | 21,000 px/s saccade -> `peak_velocity = 156.7`; linear-correct = `555.5`; `pixels_to_visual_angle(1e9) = 180.000` |
| C-2 | `[F,F,F,S,S]` round-trips to `[1, 1, 0, 2]` — 4 samples, middle sample lost |
| C-3 | offsets from labels at idx `{2, 4}`; from events at idx `{1, 3}` — off by one, and the two paths disagree |
| C-4 | `TypeError: '>' not supported between instances of 'NoneType' and 'int'` on both `create_boolean_channel` and `events_to_labels` |
| C-12 | `summarize_events([]).shape == (0, 0)` |
| C-15 | `AxisError: axis 100 is out of bounds for array of dimension 0` |
| B-1 | `match_by='offset'` returns `FixationEvent`; `match_by='offset difference'` returns `list` (fell through to generic matching). `'onset difference'` is unaffected. |
| B-2 | `match(gt, pred, "onset")` with no tolerance kwarg returns **0 matches** |
| M-1 | `TypeError: 'int' object is not iterable` |
| M-5 | `'log linear'` and `'Log-Linear'` raise `ValueError`; `'loglinear'` and `'log_linear'` work |
| M-13 | `KeyError: 'count'` |
| D-1 | `num_edge_sample_to_drop` = 2/1/1/0/0/0 at 500/300/200/150/100/60 Hz; `ch[0:-0]` empty. Silent NaN band is ~150-166 Hz (below that, D-6 raises first). |
| D-2 | `argmin` seeds PT=300; documented `argmax` would seed PT=100 |
| D-3 | `IndexError: index 5 is out of bounds for axis 0 with size 5` |
| D-6 | `RuntimeError` at 100 Hz (`ws=2` fails the polyorder guard) — NH unusable at low sampling rates |
| D-9 | The D-6 message prints the literal text `{self.sr}Hz` |
| D-10 | 300-sample 500 Hz trace, `ws=28`: trailing **27** samples `UNDEFINED`, before and after post-processing |
| D-27 | Resolves to `dtype=object` on numpy 2.5.2 — a cost, not a crash |
| V-1 | `create_frames` signature confirmed as `(..., bg_image_format, bg_image_alpha, label_colors, gaze_radius, verbose)` |
| V-2 | `AttributeError: 'NoneType' object has no attribute 'ndim'` |
| V-4 | Subset `{FIXATION, SACCADE}` gives bins `[0, 0.5, 0.5, 1.0]`; z-positions 0.2 and 0.4 — both land in bin 0, so saccades render in the fixation colour |
| V-6 | 5 and 6 sequences OK; **7 sequences** raise `AttributeError: 'int' object has no attribute 'strip'` |
| P-1 | `requires-python` confirmed absent; installed numpy 2.5.2 violates the declared `numpy~=1.2` pin |

**Still unverified:** the `authors.website` half of P-1 (`hatchling` is not installed — run `python -m build`), V-5 (needs visual inspection), V-11, and the §8a items that need the article's own data rather than synthetic input (C-1's ceiling on real outputs, D-16, D-17, M-6/M-7).

Not reproduced on a first attempt and then confirmed with a targeted case: **D-10** — a trace ending inside an expanding fixation window has its tail covered, so the bug needs a trace ending in the non-fixation branch. Worth knowing when writing its regression test.
