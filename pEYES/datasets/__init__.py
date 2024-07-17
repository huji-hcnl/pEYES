import pandas as pd

from pEYES.datasets.load_dataset import load_dataset as _load_dataset
from pEYES.datasets.get_metadata import get_metadata


def lund2013(directory: str = None, save: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Loads the Lund2013 dataset from the specified directory or from the internet if the dataset is not found.

    :param directory: str; the directory to load the dataset from or save the dataset to
    :param save: bool; if True, the dataset is saved to the specified directory
    :param verbose: bool; if True, prints additional information during the loading process
    :return: a DataFrame containing the Lund2013 dataset
    """
    return _load_dataset("Lund2013", directory, save, verbose)


def irf(directory: str = None, save: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Loads the IRF dataset from the specified directory or from the internet if the dataset is not found.

    :param directory: str; the directory to load the dataset from or save the dataset to
    :param save: bool; if True, the dataset is saved to the specified directory
    :param verbose: bool; if True, prints additional information during the loading process
    :return: a DataFrame containing the Lund2013 dataset
    """
    return _load_dataset("IRF", directory, save, verbose)


def hfc(directory: str = None, save: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Loads the HFC dataset from the specified directory or from the internet if the dataset is not found.

    :param directory: str; the directory to load the dataset from or save the dataset to
    :param save: bool; if True, the dataset is saved to the specified directory
    :param verbose: bool; if True, prints additional information during the loading process
    :return: a DataFrame containing the Lund2013 dataset
    """
    return _load_dataset("HFC", directory, save, verbose)


def gazecom(directory: str = None, save: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Loads the GazeCom dataset from the specified directory or from the internet if the dataset is not found.

    :param directory: str; the directory to load the dataset from or save the dataset to
    :param save: bool; if True, the dataset is saved to the specified directory
    :param verbose: bool; if True, prints additional information during the loading process
    :return: a DataFrame containing the Lund2013 dataset
    """
    return _load_dataset("GazeCom", directory, save, verbose)
