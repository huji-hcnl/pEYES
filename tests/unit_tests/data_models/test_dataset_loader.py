import unittest

from peyes._DataModels.DatasetLoader import (
    BaseDatasetLoader, Lund2013DatasetLoader, IRFDatasetLoader, HFCDatasetLoader, GazeComDatasetLoader,
)

_LOADERS = [Lund2013DatasetLoader, IRFDatasetLoader, HFCDatasetLoader, GazeComDatasetLoader]


class TestDatasetLoaderMetadata(unittest.TestCase):
    """
    T-2: DatasetLoader had zero unit-level test coverage. Scoped to the network-free surface only - metadata
    classmethods and column_order(). _parse_response/download/load each need a per-format mocked
    requests.Response (every loader parses a different raw format) and no fixture for that currently exists
    (tests/regression_tests/fixtures/lund2013_slice.pkl is a post-parse DataFrame, not a raw response) - left
    as a follow-up rather than attempted here.
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
