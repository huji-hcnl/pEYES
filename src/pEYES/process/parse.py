from typing import Union

import numpy as np
import pandas as pd

import src.pEYES.constants as cnst


def parse(
        data: Union[np.ndarray, pd.DataFrame],
        time_name: str = cnst.T,
        x_name: str = cnst.X,
        y_name: str = cnst.Y,
        pupil_name: str = cnst.PUPIL,
        viewer_distance_name: str = cnst.VIEWER_DISTANCE_STR,
        pixel_size_name: str = cnst.PIXEL_SIZE_STR,
        missing_data_value: float = np.nan,
):
    """
    Parse raw gaze data into a DataFrame, and rename the columns to the standard names: `t`, `x`, `y`, `pupil`,
    `viewer_distance`, and `pixel_size`. Samples of missing data are replaced with NaN.

    :param data: DataFrame or named-array containing the raw gaze data
    :param time_name: name of the time column (ms)
    :param x_name: name of the horizontal gaze position column (pixels)
    :param y_name: name of the vertical gaze position column (pixels)
    :param pupil_name: name of the pupil size column (mm)
    :param viewer_distance_name: name of the viewer distance column (cm)
    :param pixel_size_name: name of the pixel size column (cm)
    :param missing_data_value: the value used to represent missing data in the raw data

    :return: a DataFrame containing the parsed gaze data with renamed to columns to the standard names
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        data = data.copy(deep=True)  # avoid overwriting the original dataset
    else:
        raise TypeError(f"Invalid dataset type: {type(data)}")

    # rename columns
    data.rename(
        columns={
            time_name: cnst.T, x_name: cnst.X, y_name: cnst.Y, pupil_name: cnst.PUPIL,
            viewer_distance_name: cnst.VIEWER_DISTANCE_STR, pixel_size_name: cnst.PIXEL_SIZE_STR
        },
        inplace=True
    )

    # replace missing data with NaN
    if np.isnan(missing_data_value):
        nan_idxs = data[(data[cnst.X].isna()) | (data[cnst.Y].isna())].index
    else:
        nan_idxs = data[(data[cnst.X] == missing_data_value) | (data[cnst.Y] == missing_data_value)].index
    no_status_idxs = data[~data[cnst.STATUS_STR]].index if cnst.STATUS_STR in data.columns else []
    nan_idxs = np.union1d(nan_idxs, no_status_idxs)
    data.loc[nan_idxs, [cnst.X, cnst.Y]] = np.nan
    return data
