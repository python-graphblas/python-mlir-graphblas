import numpy as np
from numpy.testing import assert_equal as np_assert_equal
from numpy.testing import assert_allclose as np_assert_allclose


def vector_compare(vec, i, v):
    assert vec.ndims == 1
    idx, vals = vec.extract_tuples()
    assert vals.dtype == vec.dtype.np_type
    np_assert_equal(idx, i)
    np_assert_allclose(vals, v)


def matrix_compare(mat, r, c, v):
    assert mat.ndims == 2
    rows, cols, vals = mat.extract_tuples()
    assert vals.dtype == mat.dtype.np_type
    if mat.is_rowwise():
        sort_order = np.argsort(r)
    else:
        sort_order = np.argsort(c)
    np_assert_equal(rows, np.array(r)[sort_order])
    np_assert_equal(cols, np.array(c)[sort_order])
    np_assert_allclose(vals, np.array(v)[sort_order])
