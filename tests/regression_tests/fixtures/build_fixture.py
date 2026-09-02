"""
(Re)builds `lund2013_slice.pkl` from the real, full Lund2013 dataset: the 3 smallest real-sample-count
trials recorded at 500Hz (excluding the 7 trials recorded at 200Hz), trimmed to the columns the regression
harness actually needs.

Run this under the *lowest* numpy/pandas version this package supports (the floor pins in
`pyproject.toml`), not whatever's in the active dev environment - `pandas.to_pickle`/`read_pickle` embed
numpy's internal module layout, and a pickle written under numpy 2.x cannot be read back under numpy 1.x
(confirmed directly while building this fixture: `ModuleNotFoundError: No module named
'numpy._core.numeric'`). Writing it under the floor makes it readable everywhere newer, not just where it
was written - the reverse isn't guaranteed.

Only rerun this deliberately (e.g. to pick a different/larger slice) - it downloads the full public Lund2013
dataset fresh into whatever `PEYES_ANALYSIS_BASE_DIR` resolves to, which defaults to a shared lab network
drive (see `analysis/utils.py::BASE_DIR`) unless overridden. Always override it for this script:

    PEYES_ANALYSIS_BASE_DIR=/some/scratch/dir PYTHONPATH=. python tests/regression_tests/fixtures/build_fixture.py
"""
import os

import peyes
import analysis.utils as u

_NON_500HZ_TRIALS = [33, 34, 39, 44, 54, 58, 63]
_NUM_TRIALS = 3
_KEEP_COLUMNS = [
    peyes.constants.TRIAL_ID_STR, peyes.constants.T, peyes.constants.X, peyes.constants.Y,
    peyes.constants.PUPIL, peyes.constants.VIEWER_DISTANCE_STR, peyes.constants.PIXEL_SIZE_STR, "RA", "MN",
]
_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lund2013_slice.pkl")


def main():
    dataset = u.load_dataset("lund2013", verbose=True)
    trial_col = peyes.constants.TRIAL_ID_STR
    counts = dataset.groupby(trial_col).size()
    # sort_index() then a *stable* sort_values(): ties break deterministically by ascending trial_id
    # regardless of pandas version - plain sort_values() defaults to quicksort, which is not stable.
    eligible = counts[~counts.index.isin(_NON_500HZ_TRIALS)].sort_index()
    eligible = eligible.sort_values(kind="stable")
    chosen_trials = eligible.head(_NUM_TRIALS).index.tolist()
    print(f"Chosen trials (smallest eligible 500Hz, by sample count): {chosen_trials}")

    missing = [c for c in _KEEP_COLUMNS if c not in dataset.columns]
    if missing:
        raise RuntimeError(f"Expected columns missing from dataset: {missing}")

    slice_df = dataset[dataset[trial_col].isin(chosen_trials)][_KEEP_COLUMNS].reset_index(drop=True)
    slice_df.to_pickle(_OUTPUT_PATH)
    print(f"Saved slice (shape={slice_df.shape}) to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
