"""
Regenerates `lund2013_slice_golden.pkl` from `lund2013_slice.pkl` by running the real pipeline
(`tests/regression_tests/_harness.py::run_slice`).

Run this under the floor-pinned dependency versions declared in `pyproject.toml` (not whatever's in the
active dev environment) - e.g. a throwaway venv with Python 3.12 and the exact `numpy`/`pandas`/`scipy`/etc.
floors installed. The golden file is meant to represent "the declared minimum-supported environment", not
"whatever happened to be installed when someone last ran this".

Only rerun this deliberately, after confirming any resulting diff in the golden file is an intentional
change to detection/matching/metrics behavior - not to silence a failing `test_dependency_drift.py`.

Usage (from the repo root, with PYTHONPATH set to the repo root):
    python tests/regression_tests/fixtures/generate_golden.py
"""
import os

import tests.regression_tests._harness as harness
import pandas as pd

_FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_PATH = os.path.join(_FIXTURES_DIR, "lund2013_slice.pkl")
_GOLDEN_PATH = os.path.join(_FIXTURES_DIR, "lund2013_slice_golden.pkl")


def main():
    dataset = pd.read_pickle(_INPUT_PATH)
    golden = harness.run_slice(dataset)
    pd.to_pickle(golden, _GOLDEN_PATH)
    print(f"Wrote golden reference to {_GOLDEN_PATH}")
    for key, df in golden.items():
        print(f"  {key}: shape={df.shape}")


if __name__ == "__main__":
    main()
