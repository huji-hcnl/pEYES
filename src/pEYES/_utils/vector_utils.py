import copy
from typing import List

import numpy as np


def is_one_dimensional(arr) -> bool:
    """ Returns true if the array's shape is (n,) or (1, n) or (n, 1) """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return True
    if arr.ndim == 2 and min(arr.shape) == 1:
        return True
    return False


def get_chunk_indices(arr) -> List[np.ndarray]:
    """
    Given a 1D array with multiple values, returns a list of arrays, where each array contains the indices of
    a different "chunk", i.e. a sequence of the same value.
    """
    if not is_one_dimensional(arr):
        raise ValueError("arr must be one-dimensional")
    indices = np.arange(len(arr))
    split_on = np.nonzero(np.diff(arr))[0] + 1  # +1 because we want to include the last index of each chunk
    chunk_indices = np.split(indices, split_on)
    return chunk_indices


def merge_chunks(arr: np.ndarray, max_samples_between: int) -> np.ndarray:
    """
    Splits the input array into chunks of identical values. If two chunks of the same value are separated by a chunk
    of a different value, and the middle chunk is shorter/equal to `max_samples_between`, it is set to the same value as
    the other two chunks.
    Returns a new array with the merged chunks.
    """
    if not is_one_dimensional(arr):
        raise ValueError("input array must be one-dimensional")
    if max_samples_between < 0:
        raise ValueError("max_samples_between must be non-negative")
    arr_copy = copy.deepcopy(arr)
    chunk_indices = get_chunk_indices(arr_copy)
    for i, middle_chunk in enumerate(chunk_indices[1:-1], start=1):
        if len(middle_chunk) > max_samples_between:
            # skip if the chunk is too long
            continue
        left_chunk_val = arr_copy[chunk_indices[i - 1][0]]
        right_chunk_val = arr_copy[chunk_indices[i + 1][0]]
        if left_chunk_val != right_chunk_val:
            # skip if the left and right chunks have different values
            continue
        arr_copy[middle_chunk] = left_chunk_val
    return arr_copy


def reset_short_chunks(arr: np.ndarray, min_samples: int, default_value: int) -> np.ndarray:
    """
    Splits the input array into chunks of identical values. If a chunk is shorter than `min_samples`, it is set to the
    `default_value`. Returns a new array with the reset chunks.
    """
    if not is_one_dimensional(arr):
        raise ValueError("input array must be one-dimensional")
    if min_samples < 0:
        raise ValueError("min_samples must be non-negative")
    arr_copy = copy.deepcopy(arr)
    chunk_indices = get_chunk_indices(arr_copy)
    for chunk in chunk_indices:
        if len(chunk) < min_samples:
            arr_copy[chunk] = default_value
    return arr_copy

