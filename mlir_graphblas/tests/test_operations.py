import pytest
import operator
import functools
import numpy as np
from numpy.testing import assert_equal as np_assert_equal
from numpy.testing import assert_allclose as np_assert_allclose
from ..tensor import Scalar, Vector, Matrix
from ..types import BOOL, INT16, INT32, INT64, FP32, FP64
from .. import operations, descriptor as desc
from ..operators import UnaryOp, BinaryOp, SelectOp, IndexUnaryOp, Monoid, Semiring


@pytest.fixture
def vs():
    x = Vector.new(FP32, 5)
    x.build([1, 2, 3], [10., 20., 30.])
    y = Vector.new(FP32, 5)
    y.build([0, 2, 3], [1., 2., 3.])
    return x, y


@pytest.fixture
def ms():
    x = Matrix.new(INT16, 2, 5)
    x.build([0, 0, 1, 1, 1], [1, 3, 0, 1, 4], [-1, -2, -3, -4, -5])
    y = Matrix.new(INT16, 2, 5)
    y.build([0, 0, 0, 1], [0, 1, 3, 3], [10, 20, 30, 40])
    return x, y


@pytest.fixture
def mm():
    x = Matrix.new(FP32, 5, 6)
    x.build([0, 0, 1, 2, 2, 4], [3, 5, 3, 0, 1, 2], [1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    y = Matrix.new(FP32, 6, 5)
    y.build([0, 1, 1, 2, 3, 5], [4, 0, 4, 3, 0, 0], [6., 1., 8., 2., 5., 7.])
    return x, y


def test_transpose_op(mm):
    x, _ = mm
    xrows, xcols, xvals = x.extract_tuples()
    z = Matrix.new(x.dtype, x.shape[1], x.shape[0])
    operations.transpose(z, x)
    assert x.is_rowwise()
    assert z.is_colwise()
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, xcols)
    np_assert_equal(colidx, xrows)
    np_assert_equal(vals, xvals)

    # Transpose of a transpose is no-op
    z = Matrix.new(x.dtype, *x.shape)
    operations.transpose(z, x, desc=desc.T0)
    assert x.is_rowwise()
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, xrows)
    np_assert_equal(colidx, xcols)
    np_assert_equal(vals, xvals)


def test_ewise_add_vec(vs):
    x, y = vs
    z = Vector.new(x.dtype, x.size())
    operations.ewise_add(z, BinaryOp.plus, x, y)
    idx, vals = z.extract_tuples()
    np_assert_equal(idx, [0, 1, 2, 3])
    np_assert_allclose(vals, [1., 10., 22., 33.])


def test_ewise_add_mat(ms):
    x, y = ms
    z = Matrix.new(x.dtype, *x.shape)
    operations.ewise_add(z, BinaryOp.times, x, y)
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 0, 1, 1, 1, 1])
    np_assert_equal(colidx, [0, 1, 3, 0, 1, 3, 4])
    np_assert_allclose(vals, [10, -20, -60, -3, -4, 40, -5])


def test_ewise_mult_vec(vs):
    x, y = vs
    z = Vector.new(x.dtype, x.size())
    operations.ewise_mult(z, BinaryOp.plus, x, y)
    idx, vals = z.extract_tuples()
    np_assert_equal(idx, [2, 3])
    np_assert_allclose(vals, [22., 33.])


def test_ewise_mult_mat(ms):
    x, y = ms
    z = Matrix.new(x.dtype, *x.shape)
    operations.ewise_mult(z, BinaryOp.first, x, y)
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, [0, 0])
    np_assert_equal(colidx, [1, 3])
    np_assert_allclose(vals, [-1, -2])


def test_mxm(mm):
    x, y = mm
    z = Matrix.new(x.dtype, x.shape[0], y.shape[1])
    operations.mxm(z, Semiring.plus_times, x, y)
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, [0, 1, 2, 2, 4])
    np_assert_equal(colidx, [0, 0, 0, 4, 3])
    np_assert_allclose(vals, [20.9, 16.5, 5.5, 70.4, 13.2])


def test_mxv(vs, mm):
    _, v = vs
    _, m = mm
    z = Vector.new(m.dtype, m.shape[0])
    operations.mxv(z, Semiring.plus_times, m, v)
    idx, vals = z.extract_tuples()
    try:
        np_assert_equal(idx, [1, 2, 3, 5])
        np_assert_allclose(vals, [1., 6., 5., 7.])
    except AssertionError:
        # Check for dense return, indicating lack of lex insert fix
        np_assert_equal(idx, [0, 1, 2, 3, 4, 5])
        np_assert_allclose(vals, [0., 1., 6., 5., 0., 7.])
        pytest.xfail("Waiting for lex insert fix")


def test_vxm(vs, mm):
    _, v = vs
    m, _ = mm
    z = Vector.new(m.dtype, m.shape[1])
    operations.vxm(z, Semiring.plus_times, v, m)
    idx, vals = z.extract_tuples()
    np_assert_equal(idx, [0, 1, 3, 5])
    np_assert_allclose(vals, [8.8, 11., 1.1, 2.2])


def test_apply_mat(ms):
    x, _ = ms
    xrows, xcols, xvals = x.extract_tuples()

    # UnaryOp.abs
    z = Matrix.new(x.dtype, *x.shape)
    operations.apply(z, UnaryOp.abs, x)
    zrows, zcols, zvals = z.extract_tuples()
    np_assert_equal(zrows, xrows)
    np_assert_equal(zcols, xcols)
    np_assert_allclose(zvals, np.abs(xvals))

    # BinaryOp.minus left=2
    z2 = Matrix.new(x.dtype, *x.shape)
    operations.apply(z2, BinaryOp.minus, x, left=2)
    z2rows, z2cols, z2vals = z2.extract_tuples()
    np_assert_equal(z2rows, xrows)
    np_assert_equal(z2cols, xcols)
    np_assert_allclose(z2vals, 2 - xvals)

    # BinaryOp.gt right=-2
    z3 = Matrix.new(BOOL, *x.shape)
    operations.apply(z3, BinaryOp.ge, x, right=-2)
    z3rows, z3cols, z3vals = z3.extract_tuples()
    np_assert_equal(z3rows, xrows)
    np_assert_equal(z3cols, xcols)
    np_assert_allclose(z3vals, xvals >= -2)

    # IndexUnaryOp.rowindex thunk=1
    z4 = Matrix.new(INT64, *x.shape)
    operations.apply(z4, IndexUnaryOp.rowindex, x, thunk=1)
    z4rows, z4cols, z4vals = z4.extract_tuples()
    np_assert_equal(z4rows, xrows)
    np_assert_equal(z4cols, xcols)
    np_assert_allclose(z4vals, xrows + 1)


def test_apply_indexunary_transposed(ms):
    x, _ = ms
    xrows, xcols, xvals = x.extract_tuples()

    # IndexUnaryOp.rowindex thunk=1
    z = Matrix.new(INT64, x.shape[1], x.shape[0])
    operations.apply(z, IndexUnaryOp.rowindex, x, thunk=1, desc=desc.T0)
    zrows, zcols, zvals = z.extract_tuples()
    np_assert_equal(zrows, xcols)
    np_assert_equal(zcols, xrows)
    np_assert_allclose(zvals, xcols + 1)


def test_apply_inplace():
    v = Vector.new(INT32, 6)
    v.build([0, 3, 4], [15, 16, 17])
    operations.apply(v, BinaryOp.times, v, right=3)
    idx, vals = v.extract_tuples()
    np_assert_equal(idx, [0, 3, 4])
    np_assert_equal(vals, [45, 48, 51])

    m = Matrix.new(FP64, 2, 5)
    m.build([0, 0, 1, 1, 1], [1, 3, 0, 2, 3], [2., 50., 25., 20., 15.])
    operations.apply(m, UnaryOp.minv, m)
    rows, cols, vals = m.extract_tuples()
    np_assert_equal(rows, [0, 0, 1, 1, 1])
    np_assert_equal(cols, [1, 3, 0, 2, 3])
    np_assert_allclose(vals, [.5, .02, .04, .05, .06666666666667])


def test_select_vec(vs):
    x, _ = vs

    # Select by index
    z = Vector.new(x.dtype, x.size())
    operations.select(z, SelectOp.rowgt, x, 2)
    idx, vals = z.extract_tuples()
    np_assert_equal(idx, [3])
    np_assert_allclose(vals, [30.])

    # Select by value
    z = Vector.new(x.dtype, x.size())
    operations.select(z, SelectOp.valuegt, x, 10.)
    idx, vals = z.extract_tuples()
    np_assert_equal(idx, [2, 3])
    np_assert_allclose(vals, [20., 30.])


def test_select_mat(mm):
    _, y = mm
    z = Matrix.new(y.dtype, *y.shape)
    operations.select(z, SelectOp.triu, y, -1)
    assert z.is_rowwise()
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, [0, 1, 1, 2])
    np_assert_equal(colidx, [4, 0, 4, 3])
    np_assert_allclose(vals, [6., 1., 8., 2.])

    # Transposed
    z = Matrix.new(y.dtype, y.shape[1], y.shape[0])
    operations.select(z, SelectOp.triu, y, 0, desc=desc.T0)
    assert z.is_colwise()
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 0])
    np_assert_equal(colidx, [1, 3, 5])
    np_assert_allclose(vals, [1., 5., 7.])


def test_empty_select():
    v = Vector.new(FP32, 16)  # don't build; keep empty
    z = Vector.new(FP32, 16)
    operations.select(z, SelectOp.rowle, v, 4)
    assert z._obj is None


def test_reduce_rowwise(mm):
    x, _ = mm
    z = Vector.new(x.dtype, x.shape[0])
    operations.reduce_to_vector(z, Monoid.plus, x)
    idx, vals = z.extract_tuples()
    try:
        np_assert_equal(idx, [0, 1, 2, 4])
        np_assert_allclose(vals, [3.3, 3.3, 9.9, 6.6])
    except AssertionError:
        # Check for dense return, indicating lack of lex insert fix
        np_assert_equal(idx, [0, 1, 2, 3, 4])
        np_assert_allclose(vals, [3.3, 3.3, 9.9, 0.0, 6.6])
        pytest.xfail("Waiting for lex insert fix")


def test_reduce_colwise(mm):
    x, _ = mm
    z = Vector.new(x.dtype, x.shape[1])
    operations.reduce_to_vector(z, Monoid.times, x, desc=desc.T0)
    idx, vals = z.extract_tuples()
    np_assert_equal(idx, [0, 1, 2, 3, 5])
    np_assert_allclose(vals, [4.4, 5.5, 6.6, 3.63, 2.2])


def test_reduce_scalar_mat(mm):
    x, _ = mm
    _, _, xvals = x.extract_tuples()
    s = Scalar.new(x.dtype)
    operations.reduce_to_scalar(s, Monoid.times, x)
    np_assert_allclose(s.extract_element(), functools.reduce(operator.mul, xvals))

    # Verify transpose has no effect on scalar reduction
    operations.reduce_to_scalar(s, Monoid.plus, x, desc=desc.T0)
    np_assert_allclose(s.extract_element(), functools.reduce(operator.add, xvals))


def test_reduce_scalar_vec(vs):
    x, _ = vs
    _, xvals = x.extract_tuples()
    s = Scalar.new(x.dtype)
    operations.reduce_to_scalar(s, Monoid.times, x)
    np_assert_allclose(s.extract_element(), functools.reduce(operator.mul, xvals))


def test_extract_vec(vs):
    x, _ = vs
    xidx, xvals = x.extract_tuples()
    z = Vector.new(x.dtype, 3)
    operations.extract(z, x, [0, 1, 3])
    idx, vals = z.extract_tuples()
    np_assert_equal(idx, [1, 2])
    np_assert_allclose(vals, [10., 30.])

    # Extract all
    z2 = Vector.new(x.dtype, *x.shape)
    operations.extract(z2, x, None)
    idx, vals = z2.extract_tuples()
    np_assert_equal(idx, xidx)
    np_assert_allclose(vals, xvals)


def test_extract_mat(mm):
    x, _ = mm
    xrows, xcols, xvals = x.extract_tuples()

    # Extract all rows, all cols
    z = Matrix.new(x.dtype, *x.shape)
    operations.extract(z, x, None, None)
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, xrows)
    np_assert_equal(colidx, xcols)
    np_assert_allclose(vals, xvals)

    # Extract some rows, some cols
    z2 = Matrix.new(x.dtype, 2, 4)
    operations.extract(z2, x, [0, 4], [1, 2, 3, 5])
    rowidx, colidx, vals = z2.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 1])
    np_assert_equal(colidx, [2, 3, 1])
    np_assert_allclose(vals, [1.1, 2.2, 6.6])

    # Extract some rows, all cols
    z3 = Matrix.new(x.dtype, 2, x.shape[1])
    operations.extract(z3, x, [0, 4], None)
    rowidx, colidx, vals = z3.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 1])
    np_assert_equal(colidx, [3, 5, 2])
    np_assert_allclose(vals, [1.1, 2.2, 6.6])

    # Extract all rows, some cols
    z4 = Matrix.new(x.dtype, x.shape[0], 4)
    operations.extract(z4, x, None, [1, 5, 3, 2])
    rowidx, colidx, vals = z4.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 1, 2, 4])
    np_assert_equal(colidx, [1, 2, 2, 0, 3])
    np_assert_allclose(vals, [2.2, 1.1, 3.3, 5.5, 6.6])


def test_extract_vec_from_mat(mm):
    x, _ = mm
    # Extract partial column
    z = Vector.new(x.dtype, 3)
    operations.extract(z, x, [0, 1, 4], 2)
    idx, vals = z.extract_tuples()
    np_assert_equal(idx, [2])
    np_assert_allclose(vals, [6.6])

    # Extract full column
    z1 = Vector.new(x.dtype, x.shape[0])
    operations.extract(z1, x, None, 3)
    idx, vals = z1.extract_tuples()
    np_assert_equal(idx, [0, 1])
    np_assert_allclose(vals, [1.1, 3.3])

    # Extract partial row
    z2 = Vector.new(x.dtype, 5)
    operations.extract(z2, x, 0, [0, 1, 3, 4, 5])
    idx, vals = z2.extract_tuples()
    np_assert_equal(idx, [2, 4])
    np_assert_allclose(vals, [1.1, 2.2])

    # Extract full row
    z3 = Vector.new(x.dtype, x.shape[1])
    operations.extract(z3, x, 0, None)
    idx, vals = z3.extract_tuples()
    np_assert_equal(idx, [3, 5])
    np_assert_allclose(vals, [1.1, 2.2])

    # Extract partial column via transposed input
    z3 = Vector.new(x.dtype, 3)
    operations.extract(z3, x, 2, [0, 1, 4], desc=desc.T0)
    idx, vals = z3.extract_tuples()
    np_assert_equal(idx, [2])
    np_assert_allclose(vals, [6.6])


def test_assign_vec(vs):
    x, y = vs

    # Assign all
    operations.assign(y, x, accum=BinaryOp.plus)
    idx, vals = y.extract_tuples()
    np_assert_equal(idx, [0, 1, 2, 3])
    np_assert_allclose(vals, [1., 10., 22., 33.])

    # Expand
    z = Vector.new(x.dtype, 16)
    operations.assign(z, x, [1, 3, 13, 10, 2])
    idx, vals = z.extract_tuples()
    assert z.size() == 16
    np_assert_equal(idx, [3, 10, 13])
    np_assert_allclose(vals, [10., 30., 20.])


def test_assign_mat(ms):
    x, y = ms

    # Assign identical rows, identical cols
    z = y.dup()
    operations.assign(z, x, accum=BinaryOp.plus)
    rowidx, colidx, vals = z.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 0, 1, 1, 1, 1])
    np_assert_equal(colidx, [0, 1, 3, 0, 1, 3, 4])
    np_assert_allclose(vals, [10, 19, 28, -3, -4, 40, -5])

    # Assign new rows, new cols
    z2 = Matrix.new(x.dtype, 4, 8)
    operations.assign(z2, x, [3, 0], [0, 3, 4, 1, 7])
    rowidx, colidx, vals = z2.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 0, 3, 3])
    np_assert_equal(colidx, [0, 3, 7, 1, 3])
    np_assert_allclose(vals, [-3, -4, -5, -2, -1])

    # Assign identical rows, new cols
    z3 = Matrix.new(x.dtype, x.shape[0], 8)
    operations.assign(z3, x, None, [0, 3, 4, 1, 7])
    rowidx, colidx, vals = z3.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 1, 1, 1])
    np_assert_equal(colidx, [1, 3, 0, 3, 7])
    np_assert_allclose(vals, [-2, -1, -3, -4, -5])

    # Assign new rows, identical cols
    z4 = Matrix.new(x.dtype, 4, x.shape[1])
    operations.assign(z4, x, [3, 0], None)
    rowidx, colidx, vals = z4.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 0, 3, 3])
    np_assert_equal(colidx, [0, 1, 4, 1, 3])
    np_assert_allclose(vals, [-3, -4, -5, -1, -2])


def test_assign_vec_to_mat(ms):
    x, _ = ms

    # Assign row with identical indices
    z1 = x.dup()
    r0 = Vector.new(x.dtype, x.shape[1])
    r0.build([0, 2, 3, 4], [5, 4, 3, 2])
    operations.assign(z1, r0, 0, None, accum=BinaryOp.plus)
    rowidx, colidx, vals = z1.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 0, 0, 0, 1, 1, 1])
    np_assert_equal(colidx, [0, 1, 2, 3, 4, 0, 1, 4])
    np_assert_allclose(vals, [5, -1, 4, 1, 2, -3, -4, -5])

    # Assign row with new indices
    z2 = x.dup()
    r1 = Vector.new(x.dtype, 3)
    r1.build([0, 2], [100, 150])
    operations.assign(z2, r1, 0, [4, 0, 2], accum=BinaryOp.plus)
    rowidx, colidx, vals = z2.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 0, 0, 1, 1, 1])
    np_assert_equal(colidx, [1, 2, 3, 4, 0, 1, 4])
    np_assert_allclose(vals, [-1, 150, -2, 100, -3, -4, -5])

    # Assign col with identical indices
    z3 = x.dup()
    c0 = Vector.new(x.dtype, x.shape[0])
    c0.build([0, 1], [97, 99])
    operations.assign(z3, c0, None, 3, accum=BinaryOp.plus)
    rowidx, colidx, vals = z3.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 1, 1, 1, 1])
    np_assert_equal(colidx, [1, 3, 0, 1, 3, 4])
    np_assert_allclose(vals, [-1, 95, -3, -4, 99, -5])

    # Assign col with new indices
    z4 = x.dup()
    c1 = Vector.new(x.dtype, 1)
    c1.build([0], [101])
    operations.assign(z4, c1, [1], 3, accum=BinaryOp.plus)
    rowidx, colidx, vals = z4.extract_tuples()
    np_assert_equal(rowidx, [0, 0, 1, 1, 1, 1])
    np_assert_equal(colidx, [1, 3, 0, 1, 3, 4])
    np_assert_allclose(vals, [-1, -2, -3, -4, 101, -5])


# TODO: test assign with scalar input
