import pytest
import numpy as np
from mlir_graphblas import utils


def test_pick_and_renumber_indices():
    rows = np.array([0, 0, 1, 1, 1, 1], dtype=np.uint64)
    cols = np.array([1, 2, 0, 1, 2, 3], dtype=np.uint64)
    vals = np.array([1, 2, 6, 7, 8, 9])
    selected = np.array([1, 1, 0, 0, 1])
    rows, cols, vals = utils.pick_and_renumber_indices(selected, rows, cols, vals)
    np.testing.assert_equal(rows, [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4])
    np.testing.assert_equal(cols, [0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 1, 2, 0, 1, 2, 3])
    np.testing.assert_equal(vals, [6, 7, 8, 9, 6, 7, 8, 9, 1, 2, 1, 2, 6, 7, 8, 9])
