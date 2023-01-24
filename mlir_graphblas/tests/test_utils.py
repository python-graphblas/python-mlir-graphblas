import pytest
import numpy as np
from mlir_graphblas import utils


def test_renumber_indices():
    a = np.array([1, 1, 1, 3, 5], dtype=np.uint64)
    b = np.array([1, 2, 5, 3], dtype=np.uint64)
    c = utils.renumber_indices(a, b)
    assert c.dtype == np.uint64
    np.testing.assert_equal(c, [0, 0, 0, 3, 2])

    d = np.array([1, 2, 5, 47, 48, 49, 3], dtype=np.uint64)
    e = utils.renumber_indices(a, d)
    np.testing.assert_equal(e, [0, 0, 0, 6, 2])


def test_renumber_indices_errors():
    with pytest.raises(ValueError, match="4"):
        utils.renumber_indices(np.array([1, 1, 1, 3, 5]), np.array([1, 4, 2, 5, 3, 4]))
    with pytest.raises(KeyError, match="11"):
        utils.renumber_indices(np.array([1, 2, 5, 11]), np.array([1, 2, 5, 3, 4]))
