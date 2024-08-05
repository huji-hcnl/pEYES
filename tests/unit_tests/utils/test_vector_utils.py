import unittest

import numpy as np

from peyes._utils.vector_utils import *


class TestVectorUtils(unittest.TestCase):

    def test_is_one_dimensional(self):
        self.assertTrue(is_one_dimensional([1, 2, 3]))
        self.assertTrue(is_one_dimensional([[1], [2], [3]]))
        self.assertTrue(is_one_dimensional([[1, 2, 3]]))
        self.assertFalse(is_one_dimensional([[1, 2], [3, 4]]))
        self.assertRaises(ValueError, is_one_dimensional, [[1, 2], [3]])

    def test_normalize(self):
        self.assertTrue(np.array_equal(normalize(np.array([1, 2, 3])), [0, 0.5, 1]))
        self.assertTrue(np.array_equal(normalize(np.array([1, 2, np.nan])), [0, 1, np.nan], equal_nan=True))
        self.assertRaises(ValueError, normalize, [[1, 2], [3, 4]])

    def test_pair_boolean_arrays(self):
        arr1 = np.array([True, False, True, False, True])
        arr2 = np.array([False, True, True, False, True])
        obs = pair_boolean_arrays(arr1, arr2)
        exp = np.array([[0, 1], [2, 2], [4, 4]])
        self.assertTrue(np.array_equal(obs, exp))
        arr2 = np.array([False, True, False, False, True])
        obs = pair_boolean_arrays(arr1, arr2)
        exp = np.array([[0, 1], [4, 4]])
        self.assertTrue(np.array_equal(obs, exp))
        arr2 = np.array([True, True, False, False, True])
        obs = pair_boolean_arrays(arr1, arr2)
        exp = np.array([[0, 0], [4, 4]])
        self.assertTrue(np.array_equal(obs, exp))
        arr2 = np.array([False, False, False, False, False])
        obs = pair_boolean_arrays(arr1, arr2)
        exp = np.empty((0, 2))
        self.assertTrue(np.array_equal(obs, exp))
        arr2 = np.array([True, False, True])
        obs = pair_boolean_arrays(arr1, arr2)
        exp = np.array([[0, 0], [2, 2]])
        self.assertTrue(np.array_equal(obs, exp))
        self.assertRaises(ValueError, pair_boolean_arrays, [[True, False], [True, False]], [True, False])
        self.assertRaises(ValueError, pair_boolean_arrays, [True, False], [[True, False], [True, False]])

    def test_get_chunk_indices(self):
        arr = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        obs = get_chunk_indices(arr)
        exp = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))
        arr[2] = np.nan
        obs = get_chunk_indices(arr)
        exp = [np.array([0, 1]), np.array([2]), np.array([3, 4]), np.array([5, 6, 7, 8])]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))
        arr = np.arange(-5, 5)
        obs = get_chunk_indices(arr)
        exp = [np.array([i]) for i in range(10)]
        self.assertTrue(all([np.array_equal(o, e, equal_nan=True) for o, e in zip(obs, exp)]))

    def test_merge_chunks(self):
        arr = np.array([1, 1, 1, 1, 2, 2, 1, 1, 1])
        self.assertTrue(np.array_equal(merge_chunks(arr, 2), np.ones_like(arr)))
        self.assertTrue(np.array_equal(merge_chunks(arr, 1), arr))
        arr = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3])
        self.assertTrue(np.array_equal(merge_chunks(arr, 2), arr))
        self.assertTrue(np.array_equal(merge_chunks(arr, 1), arr))
        self.assertRaises(ValueError, merge_chunks, np.array([[1, 2], [3, 4]]), 2)
        self.assertRaises(ValueError, merge_chunks, np.array([1, 2, 3]), -1)

    def test_reset_short_chunks(self):
        arr = np.array([1, 1, 1, 1, 2, 2, 1, 1, 1])
        self.assertTrue(np.array_equal(reset_short_chunks(arr, 3, 0), np.array([1, 1, 1, 1, 0, 0, 1, 1, 1])))
        self.assertTrue(np.array_equal(reset_short_chunks(arr, 2, 0), arr))
        arr = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3])
        self.assertTrue(np.array_equal(reset_short_chunks(arr, 3, 0), np.array([1, 1, 1, 1, 0, 0, 3, 3, 3])))
        self.assertTrue(np.array_equal(reset_short_chunks(arr, 2, 0), arr))
        self.assertRaises(ValueError, reset_short_chunks, np.array([[1, 2], [3, 4]]), 2, 0)
        self.assertRaises(ValueError, reset_short_chunks, np.array([1, 2, 3]), -1, 0)
