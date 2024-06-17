import pandas as pd

from src.pEYES._DataModels.DatasetLoader import (
    Lund2013DatasetLoader, IRFDatasetLoader, HFCDatasetLoader, GazeComDatasetLoader
)


def load_dataset(dataset_name: str, directory: str = None, should_save: bool = False) -> pd.DataFrame:
    dataset_name_lower = dataset_name.lower().strip()
    if should_save and not directory:
        raise ValueError("Directory must be specified to save the dataset")
    if dataset_name_lower == "lund2013":
        return Lund2013DatasetLoader.load(directory, should_save)
    if dataset_name_lower == "irf":
        return IRFDatasetLoader.load(directory, should_save)
    if dataset_name_lower == "hfc":
        return HFCDatasetLoader.load(directory, should_save)
    if dataset_name_lower == "gazecom":
        return GazeComDatasetLoader.load(directory, should_save)
    raise NotImplementedError(f"Unknown dataset: {dataset_name}")
