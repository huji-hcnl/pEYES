# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

pEYES is a Python package for quantitatively comparing eye-tracking event-detection (fixation/saccade/etc.)
algorithms against human-annotated ground truth. See [README.md](README.md) for the full description and citation.
Published on PyPI as `peyes`; source of truth for behavior is [Nir & Deouell (2026)](https://doi.org/10.3758/s13428-026-02983-5).
**v0.2.0 and v0.2.1 are verified to reproduce v0.1.0's exact article results**, bit-for-bit, on the real
Lund2013/HFC datasets and pipeline the article used (see `docs/CODE_REVIEW.md` §13 in the main checkout for
the audit, or `CHANGELOG.md` for the summary). Any change to `peyes/` or `analysis/` that could affect a
computed value needs to be checked against that invariant specifically, not just against the test suite
passing - the existing tests do not cover this on their own (see T-2, and `tests/regression_tests/` for the
one narrower guard that does exist: dependency-version drift, not correctness).

## Environment & commands

- Python 3.12 and 3.14 (per README); dependencies are pinned in [pyproject.toml](pyproject.toml).
  `requires-python` itself has no upper bound - only the floor (3.12) is enforced.
- Use the project's local venv at `C:\Users\nirjo\Documents\University\PhD\Projects\pEYES\.venv` for all Python
  commands in this repo (including from worktrees) — do not create a new venv or use a global/system Python.
- Install (editable, dev): `pip install -e .`
- Run the whole test suite: `python -m unittest discover -s tests`
- Run a single module by dotted path, e.g.: `python -m unittest tests.unit_tests.data_models.test_event`
- Run a single test case/method: append `.ClassName` or `.ClassName.test_method` to the dotted path above, e.g.
  `python -m unittest tests.unit_tests.utils.test_vector_utils.TestVectorUtils.test_some_method`
- Tests use the stdlib `unittest` framework (`unittest.TestCase`), not pytest.
- Lint with `ruff check peyes tests` (configured in [pyproject.toml](pyproject.toml)); CI in
  [.github/workflows/ci.yml](.github/workflows/ci.yml) runs ruff plus the test suite on Python 3.12
  (floor-pinned dependencies) and Python 3.14 (latest dependencies).
- `tests/regression_tests/` guards against dependency-version drift: it runs the real detect → match → metrics
  pipeline (several detector algorithms) on a small, real, checked-in Lund2013 slice and compares the result
  against a golden reference generated under floor-pinned dependencies. See its module docstrings
  (`_harness.py`, `fixtures/build_fixture.py`, `fixtures/generate_golden.py`) for how/when to regenerate the
  fixtures.
- NOTE: the shared venv's editable `peyes` install points at one specific worktree, so `import peyes`
  from a different worktree silently loads that worktree's code. Prefix commands with
  `PYTHONPATH=<this worktree>` and confirm with `python -c "import peyes; print(peyes.__file__)"`.

## Architecture

The installable package lives in `peyes/`; `peyes/__init__.py` is the public API surface — it re-exports a small set
of top-level functions/submodules and hides everything else behind underscore-prefixed internal packages
(`_DataModels`, `_base`, `_utils`). When adding functionality, wire new public entry points through
`peyes/__init__.py` rather than having callers reach into `_DataModels`/`_base`/`_utils` directly.

**Pipeline shape:** raw gaze samples (`t`, `x`, `y`) → per-sample event **labels** (`EventLabelEnum`: UNDEFINED,
FIXATION, SACCADE, PSO, SMOOTH_PURSUIT, BLINK) → **Event objects** (label + its slice of samples) → **matching**
between two label/event sequences (e.g. detector output vs. human rater) → **metrics** computed over labels, events,
or matches.

- `peyes/_DataModels/Detector.py` — `BaseDetector` (ABC) implements the shared `detect()` pipeline (blink detection
  → NaN-out blink samples → algorithm-specific `_detect_impl` → merge/drop short chunks → parse labels). Each
  algorithm (IVT, IVVT, IDT, IDVT, Engbert, NH, REMoDNaV, …) is a subclass implementing `_detect_impl` and
  `get_default_params`. `peyes/_base/create.py::create_detector` is a string-keyed factory over these subclasses —
  add a new algorithm there too when adding a detector class.
- `peyes/_DataModels/Event.py` — `BaseEvent` (ABC) and its per-label subclasses (`FixationEvent`, `SaccadeEvent`,
  …), constructed via `BaseEvent.make`/`make_multiple` from a label + sample arrays.
- `peyes/_DataModels/EventMatcher.py` — static matching strategies (first/last/max overlap, IoU, onset/offset
  difference, window-based, L2 timing, generic) between two event sequences; `peyes/_base/match.py::match` is the
  string-keyed dispatcher over these.
- `peyes/_DataModels/DatasetLoader.py` — one loader class per built-in human-annotated dataset (Lund2013, IRF, HFC,
  GazeCom), each downloading/parsing into a common `pd.DataFrame` shape; dispatched by
  `peyes/datasets/load_dataset.py`.
- `peyes/_DataModels/config.py` — shared defaults (viewer distance, screen/pixel geometry, per-label
  min/max duration and plot color via `EVENT_MAPPING`); `peyes/_base/set_config.py` lets callers override these
  globally.
- Metrics packages (`event_metrics/`, `sample_metrics/`, `match_metrics/`, `alignment_metrics/`) each wrap a private
  `_get_features.py`/similar module with a `__init__.py` of thin, specifically-named public functions (e.g.
  `event_metrics.saccade_rate`, `match_metrics.time_iou`) — follow this wrap-and-re-export pattern when adding a
  metric rather than exposing generic parametrized functions.
- `peyes/visualize/` — Plotly-based figure builders (gaze trajectories/heatmaps, event/feature summaries,
  scarfplots, video export), re-exported as `peyes.visualization`.
- `analysis/` and `docs/User Guide/` are not part of the installed package: `analysis/` holds the scripts/notebooks
  that produced the article's figures and results (organized by dataset/experiment), and `docs/User Guide/` holds
  end-user tutorial notebooks.

## Git workflow

This repo uses a `main` → `dev` → worktree-branch flow. Full details, including the exact command sequence for
each stage, are in [docs/GIT_WORKFLOW.md](docs/GIT_WORKFLOW.md) — read it before merging, rebasing, or pushing
anything in this repo. Summary:

- `main` is off limits except on an explicit, specific user instruction in the current session.
- `dev` is the integration branch; every session/agent works on its own branch (worktree branch where supported)
  and rebases onto `dev`, never `main`.
- Every step that touches shared state — merging into `dev`, pushing to `origin/dev`, opening/merging the
  `dev` → `main` PR — happens only when the user explicitly asks for it, not automatically after a prior step
  succeeds.
- Commit small and often on the working branch; merge into `dev` with `--no-ff` only at meaningful checkpoints; tag
  `backup/<description>` refs before large rebases or after completing a body of work.
