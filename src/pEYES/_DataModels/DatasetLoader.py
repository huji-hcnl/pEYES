import os
import io
import itertools
import zipfile as zp
import posixpath as psx
from typing import final, List, Tuple, Dict
from abc import ABC, abstractmethod


import numpy as np
import pandas as pd
import requests as req
from tqdm import tqdm
from scipy.io import loadmat
from scipy.interpolate import interp1d
import arff

import src.pEYES._utils.constants as cnst
from src.pEYES._utils.pixel_utils import calculate_pixel_size, visual_angle_to_pixels
from src.pEYES._utils.event_utils import parse_label
from src.pEYES._DataModels.EventLabelEnum import EventLabelEnum


class BaseDatasetLoader(ABC):
    _NAME: str
    _URL: str
    _ARTICLES: List[str]
    _INDEXERS: List[str] = [
        cnst.TRIAL_ID_STR, cnst.SUBJECT_ID_STR, cnst.STIMULUS_TYPE_STR, cnst.STIMULUS_NAME_STR
    ]

    @classmethod
    @final
    def load(cls, directory: str, save: bool = False, verbose: bool = False) -> pd.DataFrame:
        """
        Loads the dataset from the specified directory. If the dataset is not found, it is downloaded from the internet.
        If `save` is True and the dataset was downloaded, it is saved to the specified directory.
        if `verbose` is True, a progress bar is displayed while parsing the downloaded dataset.
        :return: a DataFrame containing the dataset
        :raises ValueError: if `save` is True and `directory` is not specified
        """
        if save and not directory:
            raise ValueError("Directory must be specified to save the dataset")
        try:
            dataset = pd.read_pickle(os.path.join(directory, f"{cls.name()}.pkl"))
        except (FileNotFoundError, TypeError) as _e:
            if verbose:
                print(f"Dataset {cls.name()} not found in directory {directory}.\nDownloading...")
            dataset = cls.download(verbose)
            if save:
                os.makedirs(directory, exist_ok=True)
                file_path = os.path.join(directory, f"{cls.name()}.pkl")
                if verbose:
                    print(f"Saved dataset to {file_path}")
                dataset.to_pickle(file_path)
        return dataset

    @classmethod
    @final
    def download(cls, verbose: bool = False) -> pd.DataFrame:
        """ Downloads the dataset from the internet, parses it and returns a DataFrame with cleaned data """
        url = cls.url()
        response = req.get(cls._URL)
        code = response.status_code
        if code != 200:
            raise ConnectionError(
                f"HTTP status code {code} when attempting to download dataset {cls.name()} from {url}"
            )
        df = cls._parse_response(response, verbose)
        return cls._reorder_columns(df)

    @classmethod
    @abstractmethod
    def _parse_response(cls, response: req.Response, verbose: bool = False) -> pd.DataFrame:
        """ Parses the downloaded response and returns a DataFrame containing the raw dataset """
        raise NotImplementedError

    @classmethod
    @final
    def _reorder_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        ordered_columns = sorted(df.columns, key=lambda col: cls.column_order().get(col, 10))
        return df[ordered_columns]

    @classmethod
    @final
    def name(cls) -> str:
        """ Name of the dataset """
        if not cls._NAME:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_NAME`")
        return cls._NAME

    @classmethod
    @final
    def articles(cls) -> List[str]:
        """ List of articles that where this dataset was created """
        if not cls._ARTICLES:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_ARTICLES`")
        return cls._ARTICLES

    @classmethod
    @final
    def url(cls) -> str:
        """ URL of the dataset """
        if not cls._URL:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_URL`")
        return cls._URL

    @staticmethod
    def column_order() -> Dict[str, float]:
        return {
            cnst.TRIAL_ID_STR: 0.1, cnst.SUBJECT_ID_STR: 0.2, cnst.STIMULUS_TYPE_STR: 0.3, cnst.STIMULUS_NAME_STR: 0.4,
            cnst.T: 1.0, cnst.X: 1.1, cnst.Y: 1.2, cnst.PUPIL: 1.3,
            cnst.LEFT_X: 2.1, cnst.LEFT_Y: 2.2, cnst.LEFT_PUPIL: 2.3,
            cnst.RIGHT_X: 3.1, cnst.RIGHT_Y: 3.2, cnst.RIGHT_PUPIL: 3.3,
            cnst.PIXEL_SIZE_STR: 4.2, cnst.VIEWER_DISTANCE_STR: 4.3
        }

    @staticmethod
    @final
    def _extract_filename_and_extension(full_path: str) -> (str, str, str):
        """ Splits a full path into its components: path, filename and extension """
        path, extension = os.path.splitext(full_path)
        path, filename = os.path.split(path)
        return path, filename, extension


class Lund2013DatasetLoader(BaseDatasetLoader):
    _NAME = "Lund2013"
    _URL = 'https://github.com/richardandersson/EyeMovementDetectorEvaluation/archive/refs/heads/master.zip'
    _ARTICLES = [
        "Andersson, R., Larsson, L., Holmqvist, K., Stridh, M., & Nyström, M. (2017): One algorithm to rule them " +
        "all? An evaluation and discussion of ten eye movement event-detection algorithms. Behavior Research " +
        "Methods, 49(2), 616-637.",
        "Zemblys, R., Niehorster, D. C., Komogortsev, O., & Holmqvist, K. (2018). Using machine learning to detect " +
        "events in eye-tracking data. Behavior Research Methods, 50(1), 160–181."
    ]

    __PREFIX = 'EyeMovementDetectorEvaluation-master/annotated_data/originally uploaded data/'
    __ERRONEOUS_FILES = ['UH29_img_Europe_labelled_MN.mat']
    __CORRECTION_FILES = [
        'EyeMovementDetectorEvaluation-master/annotated_data/fix_by_Zemblys2018/UH29_img_Europe_labelled_FIX_MN.mat'
    ]

    @classmethod
    def _parse_response(cls, response: req.Response, verbose: bool = False) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))

        # list all files in the zip archive that are relevant to this dataset
        # replaces erroneously labelled files with the corrected ones (see readme.md for more info)
        is_valid_file = lambda f: f.startswith(cls.__PREFIX) and f.endswith('.mat') and f not in cls.__ERRONEOUS_FILES
        file_names = [f for f in zip_file.namelist() if is_valid_file(f)]
        file_names.extend(cls.__CORRECTION_FILES)

        # read all files into a list of dataframes
        trial_id = 0
        dataframes = {}
        for f in tqdm(file_names, desc="Processing Files", disable=not verbose):
            file = zip_file.open(f)
            gaze_data = cls.__read_eyetracker_data(file)
            subject_id, stimulus_type, stimulus_name, rater = cls.__extract_metadata(file)
            gaze_data.rename(columns={cnst.LABEL_STR: rater}, inplace=True)

            # write the DF to a dict based on the subject id, stimulus type, stimulus name, or add to existing DF
            existing_df = dataframes.get((subject_id, stimulus_type, stimulus_name), None)
            if existing_df is None:
                trial_id += 1
                gaze_data[cnst.TRIAL_ID_STR] = trial_id
                gaze_data[cnst.SUBJECT_ID_STR] = subject_id
                gaze_data[cnst.STIMULUS_TYPE_STR] = stimulus_type
                gaze_data[cnst.STIMULUS_NAME_STR] = stimulus_name
                dataframes[(subject_id, stimulus_type, stimulus_name)] = gaze_data
            else:
                existing_df.loc[:, rater] = gaze_data.loc[:, rater]
        return pd.concat(dataframes.values(), ignore_index=True, axis=0)

    @staticmethod
    def __read_eyetracker_data(file) -> pd.DataFrame:
        mat = loadmat(file)
        eyetracking_data = mat["ETdata"]
        eyetracking_data_dict = {name: eyetracking_data[name][0, 0] for name in eyetracking_data.dtype.names}

        # extract singleton values and convert from meters to cm:
        sampling_rate = eyetracking_data_dict['sampFreq'][0, 0]
        view_dist = eyetracking_data_dict['viewDist'][0, 0] * 100
        screen_width, screen_height = eyetracking_data_dict['screenDim'][0] * 100
        screen_res = eyetracking_data_dict['screenRes'][0]  # (1024, 768)
        pixel_size = calculate_pixel_size(screen_width, screen_height, screen_res)

        # extract gaze data:
        samples_data = eyetracking_data_dict['pos']
        right_x, right_y = samples_data[:, 3:5].T  # only recording right eye
        is_missing = (right_x == 0) & (right_y == 0)    # missing samples are marked with (0, 0) coordinates
        right_x[is_missing] = np.nan
        right_y[is_missing] = np.nan
        labels = pd.Series(samples_data[:, 5]).apply(lambda x: parse_label(x, safe=True))
        if np.isnan(samples_data[:, 0]).any():
            # if timestamps are NaN, re-populate them
            timestamps = np.arange(len(right_x)) * cnst.MILLISECONDS_PER_SECOND / sampling_rate
        else:
            # timestamps are available but in microseconds
            timestamps = samples_data[:, 0] - np.nanmin(samples_data[:, 0])  # start timestamps from 0
            timestamps /= cnst.MICROSECONDS_PER_MILLISECOND
        return pd.DataFrame(data={
            cnst.T: timestamps, cnst.X: right_x, cnst.Y: right_y, cnst.PUPIL: np.nan,
            cnst.LABEL_STR: labels, cnst.VIEWER_DISTANCE_STR: view_dist, cnst.PIXEL_SIZE_STR: pixel_size
        })

    @staticmethod
    def __extract_metadata(file) -> Tuple[str, str, str, str]:
        file_name = os.path.basename(file.name)  # remove path
        if not file_name.endswith(".mat"):
            raise ValueError(f"Expected a `.mat` file, got: {file_name}")
        # file_name fmt: `<subject_id>_<stimulus_type>_<stimulus_name_1>_ ... _<stimulus_name_N>_labelled_<rater_name>`
        # moving-dot trials don't contain stimulus names
        file_name = file_name.replace(".mat", "")  # remove extension
        split_name = file_name.split("_")
        subject_id = split_name[0]  # subject id is always 1st in the file name
        stimulus_type = split_name[1]  # stimulus type is always 2nd in the file name
        rater = split_name[-1].upper()  # rater is always last in the file name
        stimulus_name = "_".join(split_name[2:-2]).removesuffix("_labelled")  # everything between stim type and rater
        if stimulus_type.startswith("trial"):
            stimulus_type = cnst.MOVING_DOT_STR  # moving-dot stimulus is labelled as "trial1", "trial2", etc.
        if stimulus_type == "img":
            stimulus_type = cnst.IMAGE_STR
        return subject_id, stimulus_type, stimulus_name, rater


class IRFDatasetLoader(BaseDatasetLoader):
    """
    Loads the dataset from a replication study of the article:
    Using machine learning to detect events in eye-tracking data. Zemblys et al. (2018).
    See also about the repro study: https://github.com/r-zemblys/irf/blob/master/doc/IRF_replication_report.pdf

    Note: binocular data was recorded but only one pair of (x, y) coordinates is provided.

    This loader is based on a previous implementation, see article:
    Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. Behav Res 55, 1653–1714 (2023)
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/humanFixationClassification.py
    """

    _NAME = "IRF"
    _URL = r'https://github.com/r-zemblys/irf/archive/refs/heads/master.zip'
    _ARTICLES = [
        "Zemblys, Raimondas and Niehorster, Diederick C and Komogortsev, Oleg and Holmqvist, Kenneth. Using machine " +
        "learning to detect events in eye-tracking data. Behavior Research Methods, 50(1), 160–181 (2018)."
    ]

    __PREFIX = 'irf-master/etdata/lookAtPoint_EL'
    __STIMULUS_TYPE_VAL = "moving_dot"  # all subjects were shown the same 13-point moving dot stimulus
    __VIEWER_DISTANCE_CM_VAL = 56.5
    __MONITOR_WIDTH_CM_VAL, __MONITOR_HEIGHT_CM_VAL = 37.5, 30.2
    __MONITOR_RESOLUTION_VAL = (1280, 1024)
    __PIXEL_SIZE_CM_VAL = calculate_pixel_size(
        __MONITOR_WIDTH_CM_VAL, __MONITOR_HEIGHT_CM_VAL, __MONITOR_RESOLUTION_VAL
    )
    __RATER_NAME = "RZ"

    @classmethod
    def _parse_response(cls, response: req.Response, verbose: bool = False) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))
        gaze_file_names = [
            f for f in zip_file.namelist() if
            (f.startswith(psx.join(cls.__PREFIX, "lookAtPoint_EL_")) and f.endswith('.npy'))
        ]
        gaze_dfs = []
        for i, f in tqdm(enumerate(gaze_file_names), desc="Processing Files", disable=not verbose):
            file = zip_file.open(f)
            _, file_name, _ = cls._extract_filename_and_extension(f)
            gaze_data = pd.DataFrame(np.load(file))
            gaze_data['evt'] = gaze_data['evt'].apply(lambda x: parse_label(x, safe=True))  # convert labels
            gaze_data[cnst.SUBJECT_ID_STR] = file_name.split('_')[-1]  # format: "lookAtPoint_EL_S<subject_num>"
            gaze_data[cnst.TRIAL_ID_STR] = i + 1
            gaze_data[cnst.PUPIL] = np.nan
            gaze_dfs.append(gaze_data)
        df = pd.concat(gaze_dfs, ignore_index=True, axis=0)

        # add metadata columns:
        df[cnst.STIMULUS_TYPE_STR] = cls.__STIMULUS_TYPE_VAL
        df[cnst.VIEWER_DISTANCE_STR] = cls.__VIEWER_DISTANCE_CM_VAL
        df[cnst.PIXEL_SIZE_STR] = cls.__PIXEL_SIZE_CM_VAL

        # remap columns to correct values
        df.rename(
            columns={"t": cnst.T, "evt": cls.__RATER_NAME, "x": cnst.X, "y": cnst.Y}, inplace=True
        )
        df = cls.__correct_coordinates(df)
        return df

    @classmethod
    def __correct_coordinates(cls, df: pd.DataFrame) -> pd.DataFrame:
        new_df = df.copy()
        nan_idxs = new_df[~new_df[cnst.STATUS_STR]].index
        new_df.loc[nan_idxs, cnst.X] = np.nan
        new_df.loc[nan_idxs, cnst.Y] = np.nan
        new_df.drop(columns=[cnst.STATUS_STR], inplace=True)

        pixel_width_cm = cls.__MONITOR_WIDTH_CM_VAL / cls.__MONITOR_RESOLUTION_VAL[0]
        x = new_df[cnst.X].apply(
            lambda ang: visual_angle_to_pixels(
                angle=ang, d=cls.__VIEWER_DISTANCE_CM_VAL, pixel_size=pixel_width_cm, use_radians=False, keep_sign=True
            )
        )
        x += cls.__MONITOR_RESOLUTION_VAL[0] // 2  # move x=0 coordinate to the left of the screen
        pixel_height = cls.__MONITOR_HEIGHT_CM_VAL / cls.__MONITOR_RESOLUTION_VAL[1]
        y = new_df[cnst.Y].apply(
            lambda ang: visual_angle_to_pixels(
                angle=ang, d=cls.__VIEWER_DISTANCE_CM_VAL, pixel_size=pixel_height, use_radians=False, keep_sign=True
            )
        )
        y += cls.__MONITOR_RESOLUTION_VAL[1] // 2  # move y=0 coordinate to the top of the screen
        new_df.loc[:, cnst.X] = x.astype("float32")
        new_df.loc[:, cnst.Y] = y.astype("float32")
        return new_df


class HFCDatasetLoader(BaseDatasetLoader):
    """
    Loads the two datasets presented in articles:
    - (adults) Is human classification by experienced untrained observers a gold standard in fixation detection?
        Hooge et al. (2018)
    - (infants) An in-depth look at saccadic search in infancy.
        Hessels et al. (2016)

    This loader is based on a previous implementation, see article:
    Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. Behav Res 55, 1653–1714 (2023)
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/humanFixationClassification.py
    """

    _NAME: str = "HFC"
    _URL = r'https://github.com/dcnieho/humanFixationClassification/archive/refs/heads/master.zip'
    _ARTICLES = [
        "Hooge, I.T.C., Niehorster, D.C., Nyström, M., Andersson, R. & Hessels, R.S. (2018). Is human classification " +
        "by experienced untrained observers a gold standard in fixation detection?",
        "Hessels, R.S., Hooge, I.T.C., & Kemner, C. (2016). An in-depth look at saccadic search in infancy. " +
        "Journal of Vision, 16(8), 10.",
        "Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. " +
        "Behav Res 55, 1653–1714 (2023)"
    ]

    __PREFIX = 'humanFixationClassification-master/data'
    __SUBJECT_GROUP_STR = "subject_group"
    __INFANT_STR, __ADULT_STR = "infant", "adult"
    __SEARCH_TASK_STR, __FREE_VIEWING_STR = "search_task", "free_viewing"

    @staticmethod
    def column_order() -> Dict[str, float]:
        return {
            **super(HFCDatasetLoader, HFCDatasetLoader).column_order(), HFCDatasetLoader.__SUBJECT_GROUP_STR: 6.1
        }

    @classmethod
    def _parse_response(cls, response: req.Response, verbose: bool = False) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))
        # extract gaze data:
        gaze_file_names = [
            f for f in zip_file.namelist() if
            (f.startswith(psx.join(cls.__PREFIX, "ETdata")) and f.endswith('.txt'))
        ]
        gaze_dfs = {}
        for i, f in tqdm(enumerate(gaze_file_names), desc="Processing Files", disable=not verbose):
            file = zip_file.open(f)
            gaze_data = pd.read_csv(file, sep='\t')
            gaze_data[cnst.PUPIL] = np.nan  # no pupil data available
            _, file_name, _ = cls._extract_filename_and_extension(file.name)
            subject_group, subject_id = file_name.split('_')  # format: "<subject_type>_<subject_id>"
            gaze_data[cnst.SUBJECT_ID_STR] = subject_id
            gaze_data[cls.__SUBJECT_GROUP_STR] = subject_group
            gaze_data[cnst.STIMULUS_TYPE_STR] = cls.__FREE_VIEWING_STR if subject_group == cls.__ADULT_STR else cls.__SEARCH_TASK_STR
            gaze_data[cnst.TRIAL_ID_STR] = i + 1
            gaze_dfs[file_name] = gaze_data

        # extract annotations:
        coder_file_names = [
            f for f in zip_file.namelist() if
            (f.startswith(psx.join(cls.__PREFIX, "coderSettings")) and f.endswith('.txt'))
        ]
        annotation_dfs = {}
        for f in coder_file_names:
            with zip_file.open(f) as open_file:
                rater_data = pd.read_csv(open_file, sep='\t')
                _, rater_name, _ = cls._extract_filename_and_extension(open_file.name)
                annotation_dfs[rater_name.upper()] = rater_data

        # merge annotations with gaze data:
        merged_dfs = []
        for key, data in gaze_dfs.items():
            if data is None or len(data) == 0 or data.empty:
                continue
            l = len(data)
            for rater_name in annotation_dfs.keys():
                annotations = annotation_dfs.get(rater_name).query("Trial==@key")
                if annotations is None or len(annotations) == 0 or annotations.empty:
                    labels = np.zeros(l, dtype=int)
                else:
                    # reached here if there are annotations from this rater for this trial
                    labels = np.zeros(l, dtype=int)
                    for _, row in annotations.iterrows():
                        f = interp1d(
                            data["time"], range(l), kind="nearest", bounds_error=False, fill_value="extrapolate"
                        )
                        fixation_samples = itertools.chain(
                            *[range(int(s), int(e + 1)) for s, e in zip(
                                f(annotations["FixStart"]), f(annotations["FixEnd"])
                            )]
                        )
                        labels[list(fixation_samples)] = 1
                data[rater_name] = labels
                data[rater_name] = data[rater_name].apply(lambda x: parse_label(x, safe=True))
                merged_dfs.append(data)
        full_dataset = pd.concat(merged_dfs, ignore_index=True, axis=0)
        full_dataset.rename(columns={"time": cnst.T, "x": cnst.X, "y": cnst.Y}, inplace=True)
        return full_dataset


class GazeComDatasetLoader(BaseDatasetLoader):
    """
    Loads a labelled subset of the dataset presented in the article:
    Michael Dorr, Thomas Martinetz, Karl Gegenfurtner, and Erhardt Barth. Variability of eye movements when viewing
    dynamic natural scenes. Journal of Vision, 10(10):1-17, 2010.

    Labels are from the article:
    Agtzidis, I., Startsev, M., & Dorr, M. (2016a). In the pursuit of (ground) truth: A hand-labelling tool for eye
    movements recorded during dynamic scene viewing. In 2016 IEEE second workshop on eye tracking and visualization
    (ETVIS) (pp. 65–68).

    Note 1: This dataset is extremely large and may take a long time to download. It is recommended to save it to local
    storage after downloading it for faster access in the future. The method `load_zipfile` can be used to load the
    dataset from a raw zip file stored in a local directory.
    Note 2: This is only a subset of the full GazeCom Dataset, containing hand-labelled samples. The full dataset with
    documentation can be found in https://www.inb.uni-luebeck.de/index.php?id=515.
    Note 3: binocular data was recorded but only one pair of (x, y) coordinates is provided.

    This loader is based on a previous implementation, see article:
    Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. Behav Res 55, 1653–1714 (2023)
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/tum.py
    """

    _NAME: str = "GazeCom"
    _URL = r'https://gin.g-node.org/ioannis.agtzidis/gazecom_annotations/archive/master.zip'
    _ARTICLES = [
        "Agtzidis, I., Startsev, M., & Dorr, M. (2016a). In the pursuit of (ground) truth: A hand-labelling tool for " +
        "eye movements recorded during dynamic scene viewing. In 2016 IEEE second workshop on eye tracking and" +
        "visualization (ETVIS) (pp. 65–68).",

        "Michael Dorr, Thomas Martinetz, Karl Gegenfurtner, and Erhardt Barth. Variability of eye movements when " +
        "viewing dynamic natural scenes. Journal of Vision, 10(10):1-17, 2010.",
        "Startsev, M., Agtzidis, I., & Dorr, M. (2016). Smooth pursuit. http://michaeldorr.de/smoothpursuit/",

        "Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. " +
        "Behav Res 55, 1653–1714 (2023)"
    ]

    __PREFIX = psx.join('gazecom_annotations', 'ground_truth')
    __ZIPFILE_NAME = "gazecom_annotations-master.zip"
    __HANDLABELLER_PREFIX = "HL"
    __VIEWER_DISTANCE_CM_VAL = 56.5
    __MONITOR_WIDTH_CM_VAL, __MONITOR_HEIGHT_CM_VAL = 40, 22.5
    __MONITOR_RESOLUTION_VAL = (1280, 720)
    __PIXEL_SIZE_CM_VAL = calculate_pixel_size(
        __MONITOR_WIDTH_CM_VAL, __MONITOR_HEIGHT_CM_VAL, __MONITOR_RESOLUTION_VAL
    )

    __LABEL_MAP = {
        0: EventLabelEnum.UNDEFINED,
        1: EventLabelEnum.FIXATION,
        2: EventLabelEnum.SACCADE,
        3: EventLabelEnum.SMOOTH_PURSUIT,
        4: EventLabelEnum.UNDEFINED  # noise
    }
    __COLUMN_MAP = {
        "time": cnst.T, "x": cnst.X, "y": cnst.Y, "handlabeller1": f"{__HANDLABELLER_PREFIX}1",
        "handlabeller2": f"{__HANDLABELLER_PREFIX}2", "handlabeller_final": f"{__HANDLABELLER_PREFIX}_FINAL"
    }

    @classmethod
    def load_zipfile(cls, root: str = None, verbose: bool = False) -> pd.DataFrame:
        """
        Loads the dataset from a zip file stored in a local directory (so it doesn't need to be downloaded again).
        :param root: path to the directory containing the zip file.
        :param verbose: whether to display progress bars.
        :return: DataFrame with the annotated gaze data.
        """
        if not root or not psx.isdir(root):
            raise NotADirectoryError(f"Invalid directory: {root}")
        zip_file = psx.join(root, cls.__ZIPFILE_NAME)
        if not psx.isfile(zip_file):
            raise FileNotFoundError(f"File not found: {zip_file}")
        with zp.ZipFile(zip_file, 'r') as zip_ref:
            df = cls.__read_zipfile(zf=zip_ref, verbose=verbose)
            return cls._reorder_columns(df)

    @staticmethod
    def column_order() -> Dict[str, float]:
        handlabeller_scores = {
            f"{GazeComDatasetLoader.__HANDLABELLER_PREFIX}1": 5.1,
            f"{GazeComDatasetLoader.__HANDLABELLER_PREFIX}2": 5.2,
            f"{GazeComDatasetLoader.__HANDLABELLER_PREFIX}_FINAL": 5.3
        }
        return {**super(GazeComDatasetLoader, GazeComDatasetLoader).column_order(), **handlabeller_scores}

    @classmethod
    def _parse_response(cls, response: req.Response, verbose: bool = False) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))
        return cls.__read_zipfile(zf=zip_file, verbose=verbose)

    @classmethod
    def __read_zipfile(cls, zf: zp.ZipFile, verbose: bool = False) -> pd.DataFrame:
        """
        Reads the contents of a zip file and returns a DataFrame with the annotated data.
        :param zf: ZipFile object
        :param verbose: whether to display progress bars
        :return: DataFrame with annotated gaze data
        """
        annotated_file_names = [f for f in zf.namelist() if (f.endswith('.arff') and cls.__PREFIX in f)]
        gaze_dfs = []
        for i, f in tqdm(enumerate(annotated_file_names), desc="Processing Files", disable=not verbose):
            file = zf.open(f)
            data = arff.loads(file.read().decode('utf-8'))

            # parse gaze data:
            df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
            df[cnst.PUPIL] = np.nan  # no pupil data available
            invalid_idxs = np.where(np.all(df[["x", "y"]] == 0, axis=1) | (df["confidence"] < 0.5))[0]
            df.iloc[invalid_idxs, df.columns.get_indexer(["x", "y"])] = np.nan
            df['time'] = df['time'] / cnst.MILLISECONDS_PER_SECOND
            df.drop(columns=['confidence'], inplace=True)
            df.rename(columns=cls.__COLUMN_MAP, inplace=True)
            for col in df.columns:
                if col.startswith(cls.__HANDLABELLER_PREFIX):
                    df[col] = df[col].map(cls.__LABEL_MAP)

            # add metadata columns:
            _, file_name, _ = cls._extract_filename_and_extension(f)
            subj_id = file_name.split('_')[0]  # file_name: <subject_id>_<stimulus>_<name>_<with>_<underscores>.arff
            stimulus = '_'.join(file_name.split('_')[1:])
            df[cnst.SUBJECT_ID_STR] = subj_id
            df[cnst.STIMULUS_NAME_STR] = stimulus
            df[cnst.TRIAL_ID_STR] = i + 1
            gaze_dfs.append(df)

        # merge and add common metadata columns:
        full_dataset = pd.concat(gaze_dfs, ignore_index=True, axis=0)
        full_dataset[cnst.STIMULUS_TYPE_STR] = cnst.VIDEO_STR
        full_dataset[cnst.VIEWER_DISTANCE_STR] = cls.__VIEWER_DISTANCE_CM_VAL
        full_dataset[cnst.PIXEL_SIZE_STR] = cls.__PIXEL_SIZE_CM_VAL
        return full_dataset
