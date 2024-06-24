from typing import Union

import numpy as np
import pandas as pd

import src.pEYES.constants as cnst


def parse_raw(
        data: Union[np.ndarray, pd.DataFrame],
        time_column: str = cnst.T,
        x_column: str = cnst.X,
        y_column: str = cnst.Y,
        pupil_column: str = cnst.PUPIL,
        viewer_distance_column: str = cnst.VIEWER_DISTANCE_STR,
        pixel_size_column: str = cnst.PIXEL_SIZE_STR,
        missing_data: float = np.nan,
):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        data = data.copy(deep=True)  # avoid overwriting the original dataset
    else:
        raise TypeError(f"Invalid dataset type: {type(data)}")

    # rename columns
    data.rename(
        columns={
            time_column: cnst.T, x_column: cnst.X, y_column: cnst.Y, pupil_column: cnst.PUPIL,
            viewer_distance_column: cnst.VIEWER_DISTANCE_STR, pixel_size_column: cnst.PIXEL_SIZE_STR
        },
        inplace=True
    )

    # replace missing data with NaN
    if np.isnan(missing_data):
        nan_idxs = data[(data[cnst.X].isna()) | (data[cnst.Y].isna())].index
    else:
        nan_idxs = data[(data[cnst.X] == missing_data) | (data[cnst.Y] == missing_data)].index
    no_status_idxs = data[~data[cnst.STATUS_STR]].index if cnst.STATUS_STR in data.columns else []
    nan_idxs = np.union1d(nan_idxs, no_status_idxs)
    data.loc[nan_idxs, [cnst.X, cnst.Y]] = np.nan
    return data
