import pathlib
import tomllib
import unittest

import peyes


class TestVersion(unittest.TestCase):

    def test_dunder_version_matches_pyproject(self):
        """ The version is declared in two places; this keeps them from drifting apart. """
        pyproject = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
        declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
        self.assertEqual(declared, peyes.__version__)

    def test_version_is_exposed(self):
        self.assertIsInstance(peyes.__version__, str)
        self.assertTrue(peyes.__version__)
