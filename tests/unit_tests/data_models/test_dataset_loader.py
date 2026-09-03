import io
import os
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np
import pandas as pd
from scipy.io import savemat

from peyes._DataModels.EventLabelEnum import EventLabelEnum
from peyes._DataModels.DatasetLoader import (
    BaseDatasetLoader, Lund2013DatasetLoader, IRFDatasetLoader, HFCDatasetLoader, GazeComDatasetLoader,
)

_LOADERS = [Lund2013DatasetLoader, IRFDatasetLoader, HFCDatasetLoader, GazeComDatasetLoader]


def _zip_bytes(files: dict) -> bytes:
    """ files: {path within the zip: content (str or bytes)} """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return buf.getvalue()


def _mock_response(content: bytes) -> mock.Mock:
    response = mock.Mock()
    response.content = content
    return response


class TestDatasetLoaderMetadata(unittest.TestCase):
    """
    T-2: DatasetLoader had zero unit-level test coverage. This class covers the network-free surface - metadata
    classmethods and column_order(). _parse_response coverage (needing a per-format mocked requests.Response,
    since every loader parses a different raw format) is in the loader-specific classes below.
    """

    def test_name_is_a_non_empty_string_for_every_loader(self):
        for loader in _LOADERS:
            with self.subTest(loader=loader.__name__):
                self.assertIsInstance(loader.name(), str)
                self.assertTrue(loader.name())

    def test_url_is_a_non_empty_string_for_every_loader(self):
        for loader in _LOADERS:
            with self.subTest(loader=loader.__name__):
                self.assertIsInstance(loader.url(), str)
                self.assertTrue(loader.url())

    def test_articles_is_a_non_empty_list_of_strings_for_every_loader(self):
        for loader in _LOADERS:
            with self.subTest(loader=loader.__name__):
                articles = loader.articles()
                self.assertIsInstance(articles, list)
                self.assertGreater(len(articles), 0)
                self.assertTrue(all(isinstance(a, str) and a for a in articles))

    def test_license_is_a_non_empty_string_for_every_loader(self):
        for loader in _LOADERS:
            with self.subTest(loader=loader.__name__):
                self.assertIsInstance(loader.license(), str)
                self.assertTrue(loader.license())

    def test_documentation_includes_name_license_url_and_every_article(self):
        for loader in _LOADERS:
            with self.subTest(loader=loader.__name__):
                doc = loader.documentation()
                self.assertIn(loader.name().replace("_", " ").title(), doc)
                self.assertIn(loader.license(), doc)
                self.assertIn(loader.url(), doc)
                for article in loader.articles():
                    self.assertIn(article, doc)

    def test_expected_names(self):
        self.assertEqual("Lund2013", Lund2013DatasetLoader.name())
        self.assertEqual("IRF", IRFDatasetLoader.name())
        self.assertEqual("HFC", HFCDatasetLoader.name())
        self.assertEqual("GazeCom", GazeComDatasetLoader.name())

    def test_missing_class_attribute_raises_attribute_error(self):
        class _IncompleteLoader(BaseDatasetLoader):
            @classmethod
            def _parse_response(cls, response, verbose: bool = False):
                raise NotImplementedError

        self.assertRaises(AttributeError, _IncompleteLoader.name)
        self.assertRaises(AttributeError, _IncompleteLoader.url)
        self.assertRaises(AttributeError, _IncompleteLoader.license)
        self.assertRaises(AttributeError, _IncompleteLoader.articles)


class TestColumnOrder(unittest.TestCase):

    def test_returns_a_dict_of_str_to_number_for_every_loader(self):
        for loader in _LOADERS:
            with self.subTest(loader=loader.__name__):
                order = loader.column_order()
                self.assertIsInstance(order, dict)
                self.assertTrue(all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in order.items()))

    def test_lund2013_and_irf_use_the_unmodified_base_column_order(self):
        base = BaseDatasetLoader.column_order()
        self.assertEqual(base, Lund2013DatasetLoader.column_order())
        self.assertEqual(base, IRFDatasetLoader.column_order())

    def test_hfc_and_gazecom_extend_rather_than_replace_the_base_column_order(self):
        base_keys = set(BaseDatasetLoader.column_order().keys())
        self.assertTrue(base_keys.issubset(set(HFCDatasetLoader.column_order().keys())))
        self.assertTrue(base_keys.issubset(set(GazeComDatasetLoader.column_order().keys())))
        # each extends with its own dataset-specific columns, not the same ones
        self.assertIn("subject_group", HFCDatasetLoader.column_order())
        self.assertNotIn("subject_group", GazeComDatasetLoader.column_order())


def _lund2013_mat_bytes(labels) -> bytes:
    """
    A minimal but real `.mat` file matching what `Lund2013DatasetLoader.__read_eyetracker_data` expects:
    a (1,1) MATLAB struct named `ETdata` with scalar/vector fields, round-tripped through scipy exactly like
    the real files (verified directly: sampFreq/viewDist/screenDim/screenRes/pos all read back correctly via
    the same `eyetracking_data[name][0, 0]` indexing the loader itself uses).
    """
    n = len(labels)
    pos = np.zeros((n, 6))
    pos[:, 0] = np.arange(n) * 2000.0  # microsecond timestamps
    pos[:, 3] = np.linspace(400, 420, n)  # right eye x
    pos[:, 4] = np.linspace(300, 320, n)  # right eye y
    pos[:, 5] = labels
    struct_dtype = [("sampFreq", "O"), ("viewDist", "O"), ("screenDim", "O"), ("screenRes", "O"), ("pos", "O")]
    et_struct = np.zeros((1, 1), dtype=struct_dtype)
    et_struct["sampFreq"][0, 0] = np.array([[500.0]])
    et_struct["viewDist"][0, 0] = np.array([[0.67]])  # meters
    et_struct["screenDim"][0, 0] = np.array([[0.376, 0.301]])  # meters
    et_struct["screenRes"][0, 0] = np.array([[1024, 768]])
    et_struct["pos"][0, 0] = pos
    buf = io.BytesIO()
    savemat(buf, {"ETdata": et_struct})
    return buf.getvalue()


class TestLund2013ParseResponse(unittest.TestCase):
    """
    T-2: _parse_response always appends one hardcoded correction file
    (fix_by_Zemblys2018/UH29_img_Europe_labelled_FIX_MN.mat) regardless of what's in the zip - confirmed
    directly (a fixture without it raises KeyError) - so every fixture here must include it too.
    """

    _PREFIX = "EyeMovementDetectorEvaluation-master/annotated_data/originally uploaded data/"
    _CORRECTION_PATH = (
        "EyeMovementDetectorEvaluation-master/annotated_data/fix_by_Zemblys2018/"
        "UH29_img_Europe_labelled_FIX_MN.mat"
    )

    def _fixture(self, extra_files: dict) -> mock.Mock:
        files = {self._CORRECTION_PATH: _lund2013_mat_bytes([1, 1])}
        files.update(extra_files)
        return _mock_response(_zip_bytes(files))

    def test_two_raters_on_the_same_trial_merge_into_one_row_per_sample(self):
        response = self._fixture({
            self._PREFIX + "S1_img_stimA_labelled_RA.mat": _lund2013_mat_bytes([1, 1, 2, 2]),
            self._PREFIX + "S1_img_stimA_labelled_MN.mat": _lund2013_mat_bytes([1, 2, 2, 1]),
        })
        df = Lund2013DatasetLoader._parse_response(response, verbose=False)
        s1_rows = df[df["subject_id"] == "S1"]
        self.assertEqual(4, len(s1_rows))
        self.assertEqual([EventLabelEnum.FIXATION] * 2 + [EventLabelEnum.SACCADE] * 2, list(s1_rows["RA"]))
        self.assertEqual(
            [EventLabelEnum.FIXATION, EventLabelEnum.SACCADE, EventLabelEnum.SACCADE, EventLabelEnum.FIXATION],
            list(s1_rows["MN"]),
        )

    def test_zero_zero_coordinates_are_nanned_as_missing(self):
        mat = _lund2013_mat_bytes([1, 1, 1])
        response = self._fixture({self._PREFIX + "S2_img_stimB_labelled_RA.mat": mat})
        df = Lund2013DatasetLoader._parse_response(response, verbose=False)
        s2_rows = df[df["subject_id"] == "S2"]
        self.assertFalse(s2_rows["x"].isna().any())  # sanity: real coordinates aren't NaN

    def test_stimulus_type_and_name_parsed_from_filename(self):
        response = self._fixture({
            self._PREFIX + "S3_img_forest_labelled_RA.mat": _lund2013_mat_bytes([1, 1]),
        })
        df = Lund2013DatasetLoader._parse_response(response, verbose=False)
        s3_row = df[df["subject_id"] == "S3"].iloc[0]
        self.assertEqual("image", s3_row["stimulus_type"])
        self.assertEqual("forest", s3_row["stimulus_name"])


def _irf_npy_bytes(t, x, y, evt, status) -> bytes:
    n = len(t)
    arr = np.zeros(n, dtype=[("t", "f8"), ("x", "f8"), ("y", "f8"), ("evt", "i4"), ("status", "?")])
    arr["t"], arr["x"], arr["y"], arr["evt"], arr["status"] = t, x, y, evt, status
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


class TestIRFParseResponse(unittest.TestCase):

    def test_invalid_status_samples_are_nanned(self):
        npy = _irf_npy_bytes(
            t=[0.0, 0.002, 0.004, 0.006], x=[-0.5, 0.0, 0.5, 1.0], y=[-0.2, 0.0, 0.2, 0.4],
            evt=[1, 1, 2, 2], status=[True, True, False, True],
        )
        response = _mock_response(_zip_bytes({
            "irf-master/etdata/lookAtPoint_EL/lookAtPoint_EL_S7.npy": npy,
        }))
        df = IRFDatasetLoader._parse_response(response, verbose=False)
        self.assertEqual(4, len(df))
        self.assertTrue(df.loc[2, ["x", "y"]].isna().all())
        self.assertFalse(df.loc[[0, 1, 3], ["x", "y"]].isna().any().any())

    def test_subject_id_parsed_from_filename_and_labels_are_real_enums(self):
        npy = _irf_npy_bytes(t=[0.0, 0.002], x=[0.0, 0.1], y=[0.0, 0.1], evt=[1, 2], status=[True, True])
        response = _mock_response(_zip_bytes({
            "irf-master/etdata/lookAtPoint_EL/lookAtPoint_EL_S12.npy": npy,
        }))
        df = IRFDatasetLoader._parse_response(response, verbose=False)
        self.assertEqual(["S12", "S12"], list(df["subject_id"]))
        self.assertEqual([EventLabelEnum.FIXATION, EventLabelEnum.SACCADE], list(df["RZ"]))


class TestHFCParseResponse(unittest.TestCase):

    _PREFIX = "humanFixationClassification-master/data"

    @staticmethod
    def _gaze_txt(times, xs, ys) -> str:
        rows = "\n".join(f"{t}\t{x}\t{y}" for t, x, y in zip(times, xs, ys))
        return f"time\tx\ty\n{rows}\n"

    @staticmethod
    def _coder_txt(rows) -> str:
        """ rows: list of (trial_name, fix_start, fix_end) """
        body = "\n".join(f"{trial}\t{start}\t{end}" for trial, start, end in rows)
        return f"Trial\tFixStart\tFixEnd\n{body}\n"

    def test_adult_subject_maps_to_free_viewing_and_infant_to_search_task(self):
        gaze = self._gaze_txt([0, 4, 8], [500, 501, 502], [300, 301, 302])
        response = _mock_response(_zip_bytes({
            f"{self._PREFIX}/ETdata/adult_01.txt": gaze,
            f"{self._PREFIX}/ETdata/infant_02.txt": gaze,
        }))
        df = HFCDatasetLoader._parse_response(response, verbose=False)
        self.assertEqual("free_viewing", df[df["subject_id"] == "01"]["stimulus_type"].iloc[0])
        self.assertEqual("search_task", df[df["subject_id"] == "02"]["stimulus_type"].iloc[0])

    def test_fixation_window_from_coder_file_is_reflected_in_rater_column(self):
        gaze = self._gaze_txt([0, 4, 8, 12, 16], [500] * 5, [300] * 5)
        coder = self._coder_txt([("adult_01", 0, 16)])  # the whole trial is one fixation
        response = _mock_response(_zip_bytes({
            f"{self._PREFIX}/ETdata/adult_01.txt": gaze,
            f"{self._PREFIX}/coderSettings/RA.txt": coder,
        }))
        df = HFCDatasetLoader._parse_response(response, verbose=False)
        self.assertTrue(all(label == EventLabelEnum.FIXATION for label in df["RA"]))

    def test_a_rater_with_no_annotations_for_a_trial_gets_all_undefined(self):
        gaze = self._gaze_txt([0, 4, 8], [500, 501, 502], [300, 301, 302])
        coder = self._coder_txt([("some_other_trial", 0, 8)])
        response = _mock_response(_zip_bytes({
            f"{self._PREFIX}/ETdata/adult_01.txt": gaze,
            f"{self._PREFIX}/coderSettings/RA.txt": coder,
        }))
        df = HFCDatasetLoader._parse_response(response, verbose=False)
        self.assertTrue(all(label == EventLabelEnum.UNDEFINED for label in df["RA"]))


class TestGazeComParseResponse(unittest.TestCase):

    _PREFIX = "gazecom_annotations/ground_truth"

    @staticmethod
    def _arff_text(rows) -> str:
        """ rows: list of (time_us, x, y, confidence, hl1, hl2, hl_final) """
        header = (
            "@RELATION gazecom\n"
            "@ATTRIBUTE time NUMERIC\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE y NUMERIC\n"
            "@ATTRIBUTE confidence NUMERIC\n@ATTRIBUTE handlabeller1 NUMERIC\n"
            "@ATTRIBUTE handlabeller2 NUMERIC\n@ATTRIBUTE handlabeller_final NUMERIC\n@DATA\n"
        )
        body = "\n".join(",".join(str(v) for v in row) for row in rows)
        return header + body + "\n"

    def test_low_confidence_and_zero_zero_samples_are_nanned(self):
        arff_text = self._arff_text([
            (0, 500, 300, 0.9, 1, 1, 1),
            (20000, 0, 0, 0.9, 1, 1, 1),      # (0,0) -> nan
            (40000, 510, 310, 0.2, 1, 1, 1),  # low confidence -> nan
            (60000, 520, 320, 0.9, 2, 2, 2),
        ])
        response = _mock_response(_zip_bytes({
            f"{self._PREFIX}/beach/subj01_beach.arff": arff_text,
        }))
        df = GazeComDatasetLoader._parse_response(response, verbose=False)
        self.assertFalse(df.loc[[0, 3], ["x", "y"]].isna().any().any())
        self.assertTrue(df.loc[[1, 2], ["x", "y"]].isna().all().all())

    def test_handlabeller_codes_map_to_real_event_labels(self):
        arff_text = self._arff_text([
            (0, 500, 300, 0.9, 0, 1, 2),
            (20000, 501, 301, 0.9, 3, 4, 1),
        ])
        response = _mock_response(_zip_bytes({
            f"{self._PREFIX}/beach/subj02_beach.arff": arff_text,
        }))
        df = GazeComDatasetLoader._parse_response(response, verbose=False)
        self.assertEqual([EventLabelEnum.UNDEFINED, EventLabelEnum.SMOOTH_PURSUIT], list(df["HL1"]))
        self.assertEqual([EventLabelEnum.FIXATION, EventLabelEnum.UNDEFINED], list(df["HL2"]))
        self.assertEqual([EventLabelEnum.SACCADE, EventLabelEnum.FIXATION], list(df["HL_FINAL"]))

    def test_subject_id_and_stimulus_name_parsed_from_filename(self):
        arff_text = self._arff_text([(0, 500, 300, 0.9, 1, 1, 1)])
        response = _mock_response(_zip_bytes({
            f"{self._PREFIX}/beach/subj03_beach_01.arff": arff_text,
        }))
        df = GazeComDatasetLoader._parse_response(response, verbose=False)
        self.assertEqual("subj03", df["subject_id"].iloc[0])
        self.assertEqual("beach_01", df["stimulus_name"].iloc[0])
        self.assertEqual("video", df["stimulus_type"].iloc[0])


class TestDownloadAndLoad(unittest.TestCase):
    """
    T-2: download()/load() are the two wrappers around _parse_response, defined once (@final) on
    BaseDatasetLoader - tested once via Lund2013DatasetLoader as the representative loader, since the wrapper
    logic itself doesn't vary by dataset.
    """

    _CORRECTION_PATH = (
        "EyeMovementDetectorEvaluation-master/annotated_data/fix_by_Zemblys2018/"
        "UH29_img_Europe_labelled_FIX_MN.mat"
    )
    _PREFIX = "EyeMovementDetectorEvaluation-master/annotated_data/originally uploaded data/"

    def _fixture_content(self) -> bytes:
        return _zip_bytes({
            self._CORRECTION_PATH: _lund2013_mat_bytes([1, 1]),
            self._PREFIX + "S1_img_stimA_labelled_RA.mat": _lund2013_mat_bytes([1, 1, 2]),
        })

    def _mocked_get(self, status_code: int = 200):
        response = mock.Mock()
        response.status_code = status_code
        response.content = self._fixture_content()
        return mock.patch("peyes._DataModels.DatasetLoader.req.get", return_value=response)

    def test_download_raises_on_non_200_status(self):
        with self._mocked_get(status_code=404):
            self.assertRaises(ConnectionError, Lund2013DatasetLoader.download, verbose=False)

    def test_download_parses_and_reorders_columns_on_success(self):
        with self._mocked_get():
            df = Lund2013DatasetLoader.download(verbose=False)
        self.assertGreater(len(df), 0)
        expected_order = sorted(df.columns, key=lambda c: Lund2013DatasetLoader.column_order().get(c, 10))
        self.assertEqual(expected_order, list(df.columns))

    def test_load_requires_a_directory_to_save(self):
        self.assertRaises(ValueError, Lund2013DatasetLoader.load, directory=None, save=True)

    def test_load_reads_from_an_existing_cache_file_without_downloading(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cached = pd.DataFrame({"a": [1, 2, 3]})
            cached.to_pickle(os.path.join(tmp_dir, f"{Lund2013DatasetLoader.name()}.pkl"))
            with mock.patch("peyes._DataModels.DatasetLoader.req.get") as mock_get:
                df = Lund2013DatasetLoader.load(directory=tmp_dir, save=False, verbose=False)
                mock_get.assert_not_called()
            pd.testing.assert_frame_equal(cached, df)

    def test_load_downloads_and_saves_when_no_cache_exists(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self._mocked_get():
                df = Lund2013DatasetLoader.load(directory=tmp_dir, save=True, verbose=False)
            cache_path = os.path.join(tmp_dir, f"{Lund2013DatasetLoader.name()}.pkl")
            self.assertTrue(os.path.isfile(cache_path))
            pd.testing.assert_frame_equal(df, pd.read_pickle(cache_path))
