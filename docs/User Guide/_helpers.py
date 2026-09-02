"""
Shared helper for the pEYES User Guide notebooks.

Loads short, pre-selected trials from the built-in datasets so every notebook can get straight to the
functionality it's demonstrating instead of re-deriving the same dataset-column boilerplate. Datasets are
cached under `data/` (next to this file) after the first download, so re-running notebooks is fast.
"""
from pathlib import Path
from typing import Dict, Optional, Sequence

import peyes

_DATA_DIR = Path(__file__).parent / "data"

_DATASET_LOADERS = {
    "lund2013": peyes.datasets.lund2013,
    "irf": peyes.datasets.irf,
    "hfc": peyes.datasets.hfc,
    "gazecom": peyes.datasets.gazecom,
}

# Pre-selected for a short runtime while still containing a rich mix of event types.
_DEFAULT_TRIAL_IDS = {"lund2013": 51, "irf": 1, "hfc": 1, "gazecom": 1}

_NON_RATER_COLUMNS = {
    "trial_id", "subject_id", "stimulus_type", "stimulus_name",
    "t", "x", "y", "pupil", "pixel_size", "viewer_distance",
    "left_x", "left_y", "left_pupil", "right_x", "right_y", "right_pupil",
}


def load_example_trial(
        dataset: str = "lund2013",
        trial_id: Optional[int] = None,
        rater_columns: Optional[Sequence[str]] = None,
        max_samples: Optional[int] = None,
) -> Dict[str, object]:
    """
    Loads one trial from a built-in pEYES dataset, ready to feed into a detector.

    :param dataset: one of "lund2013", "irf", "hfc", "gazecom".
    :param trial_id: which trial to load; defaults to a short, pre-selected trial per dataset.
    :param rater_columns: names of human-rater label columns to include; auto-detected if not given.
    :param max_samples: if given, keeps only the first `max_samples` samples of the trial.
    :return: dict with keys "t", "x", "y", "pupil" (arrays), "pixel_size", "viewer_distance" (floats),
        and "raters" (Dict[str, array] of human-annotated labels).
    """
    if dataset not in _DATASET_LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}', expected one of {sorted(_DATASET_LOADERS)}")
    df = _DATASET_LOADERS[dataset](directory=str(_DATA_DIR), save=True, verbose=False)
    trial_id = _DEFAULT_TRIAL_IDS[dataset] if trial_id is None else trial_id
    trial = df[df["trial_id"] == trial_id].reset_index(drop=True)
    if max_samples is not None:
        trial = trial.iloc[:max_samples]
    if rater_columns is None:
        rater_columns = [c for c in trial.columns if c not in _NON_RATER_COLUMNS and trial[c].notna().any()]
    return {
        "t": trial["t"].values,
        "x": trial["x"].values,
        "y": trial["y"].values,
        "pupil": trial["pupil"].values,
        "pixel_size": float(trial["pixel_size"].values[0]),
        "viewer_distance": float(trial["viewer_distance"].values[0]),
        "raters": {r: trial[r].values for r in rater_columns},
    }
