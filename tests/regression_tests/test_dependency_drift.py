"""
Part B, Phase 2 (see docs/GIT_WORKFLOW.md-adjacent planning: the future-proofing plan).

Runs the real detect -> events -> match -> metrics pipeline on a small, real, checked-in Lund2013 slice and
compares the result against a golden reference generated once under the floor-pinned dependency versions
declared in `pyproject.toml`. Existing tests give no coverage for this: no test calls `.detect()` on a real
detector or exercises `EventMatcher` numerically (T-2) - this is the first one that does, specifically to
catch a numpy/pandas/scipy/Python bump silently changing detector or matching output on a cold-cache rerun,
which `analysis/process/full_pipeline.py`'s own pickle-caching means the published article's numbers can
never surface on their own.
"""
import os
import unittest

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

import tests.regression_tests._harness as harness

_FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
_INPUT_PATH = os.path.join(_FIXTURES_DIR, "lund2013_slice.pkl")
_GOLDEN_PATH = os.path.join(_FIXTURES_DIR, "lund2013_slice_golden.pkl")


def _normalize_string_dtype(index: pd.Index) -> pd.Index:
    """
    pandas >=3 defaults string-valued index/column levels to its `StringDtype` extension type; the floor
    pin (pandas 2.2, used to generate the golden reference) uses plain `object`. Confirmed empirically
    (2026-09) that this is purely a representation difference - the underlying values are identical either
    way - so it's normalized away here rather than left as a spurious failure. Every genuinely numeric
    level is left untouched, so a real numeric-dtype regression still fails loudly.
    """
    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_arrays(
            [
                pd.Index(np.asarray(lv, dtype=object)) if ptypes.is_string_dtype(lv) else lv
                for lv in (index.get_level_values(i) for i in range(index.nlevels))
            ],
            names=index.names,
        )
    if ptypes.is_string_dtype(index):
        return pd.Index(np.asarray(index, dtype=object), name=index.name)
    return index


def _normalize_int_width(series: pd.Series) -> pd.Series:
    """
    numpy 2.0 (NEP 50) changed the default integer width numpy picks on Windows for values built from
    plain Python ints - floor numpy (1.26) matches the C `long` width (`int32` on 64-bit Windows), latest
    numpy (>=2.0) always picks `int64`. Confirmed empirically (2026-09) that this only ever changes the
    storage width, never the represented value, so it's normalized to a single canonical width (`int64`)
    here. A column that's actually `float`/`bool`/string-valued is left untouched, so a genuine type-kind
    change (e.g. int -> float) still fails loudly.
    """
    if ptypes.is_integer_dtype(series):
        return series.astype(np.int64)
    return series


def _normalized(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = _normalize_string_dtype(df.index)
    df.columns = _normalize_string_dtype(df.columns)
    for col in df.columns:
        df[col] = _normalize_int_width(df[col])
    return df


class TestDependencyDrift(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(_GOLDEN_PATH):
            raise FileNotFoundError(
                f"Golden reference missing at {_GOLDEN_PATH} - regenerate it with "
                f"`tests/regression_tests/fixtures/generate_golden.py` (under floor-pinned dependencies), "
                f"don't skip this test silently."
            )
        cls.dataset = pd.read_pickle(_INPUT_PATH)
        cls.golden = pd.read_pickle(_GOLDEN_PATH)
        cls.actual = harness.run_slice(cls.dataset)

    def test_labels_match_golden(self):
        pd.testing.assert_frame_equal(
            _normalized(self.actual["labels"]), _normalized(self.golden["labels"]), check_exact=True
        )

    def test_matched_features_match_golden(self):
        pd.testing.assert_frame_equal(
            _normalized(self.actual["matched_features"]), _normalized(self.golden["matched_features"]),
            check_exact=True,
        )

    def test_sdt_measures_match_golden(self):
        pd.testing.assert_frame_equal(
            _normalized(self.actual["sdt_measures"]), _normalized(self.golden["sdt_measures"]), check_exact=True
        )


if __name__ == "__main__":
    unittest.main()
