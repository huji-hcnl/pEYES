import pandas as pd

from pEYES._DataModels.DatasetLoader import (
    Lund2013DatasetLoader, IRFDatasetLoader, HFCDatasetLoader, GazeComDatasetLoader
)


def load_dataset(dataset_name: str, directory: str = None, save: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Loads the dataset from the specified directory. If the directory is not specified or the dataset is not found, it is
    downloaded from the internet. If `save` is True and the dataset was downloaded, it is saved to the specified
    directory.

    :param dataset_name: str; the name of the dataset to load
        Must be one of: "Lund2013", "IRF", "HFC", "GazeCom"
    :param directory: str; the directory to load the dataset from or save the dataset to
    :param save: bool; if True, the dataset is saved to the specified directory
    :param verbose: bool; if True, prints additional information during the loading process
    :return: a DataFrame containing the dataset
    :raises ValueError: if `save` is True and `directory` is not specified
    """
    dataset_name_lower = dataset_name.lower().strip()
    if save and not directory:
        raise ValueError("Directory must be specified to save the dataset")
    if dataset_name_lower == "lund2013":
        return Lund2013DatasetLoader.load(directory, save, verbose)
    if dataset_name_lower == "irf":
        return IRFDatasetLoader.load(directory, save, verbose)
    if dataset_name_lower == "hfc":
        return HFCDatasetLoader.load(directory, save, verbose)
    if dataset_name_lower == "gazecom":
        return GazeComDatasetLoader.load(directory, save, verbose)
    raise NotImplementedError(f"Unknown dataset: {dataset_name}")
