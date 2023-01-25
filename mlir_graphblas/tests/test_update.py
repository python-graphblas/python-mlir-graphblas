import pytest
import numpy as np
from numpy.testing import assert_equal as np_assert_equal
from numpy.testing import assert_allclose as np_assert_allclose
from ..tensor import Scalar, Vector, Matrix
from ..types import BOOL, INT16
from .. import operations
import pytest
from ..operators import UnaryOp, BinaryOp, SelectOp, IndexUnaryOp, Monoid, Semiring
from .. import descriptor
from ..tensor import TransposedMatrix


@pytest.fixture
def mats():
    xdata = [0, 0, 1, 1, 1], [0, 3, 0, 1, 4], [-1., -2., -3., -4., -5.]
    ydata = [0, 0, 0, 1], [0, 1, 3, 3], [10., 20., 30., 40.]
    x = Matrix.new(INT16, 2, 5)
    x.build(*xdata)
    xT = Matrix.new(INT16, 5, 2)
    xT.build(xdata[1], xdata[0], xdata[-1])
    y = Matrix.new(INT16, 2, 5)
    y.build(*ydata)
    yT = Matrix.new(INT16, 5, 2)
    yT.build(ydata[1], ydata[0], ydata[-1])
    mask = Matrix.new(BOOL, 2, 5)
    mask.build([0, 1, 1, 1, 1], [0, 1, 2, 3, 4], [1, 1, 1, 1, 1])
    out = Matrix.new(INT16, 2, 5)
    out.build([0, 0, 1, 1], [0, 1, 0, 4], [100., 200., 300., 400.])
    return x, xT, y, yT, mask, out


def test_no_transpose_out_mask(mats):
    x, xT, y, yT, mask, out = mats
    out = TransposedMatrix.wrap(out)
    with pytest.raises(TypeError, match="TransposedMatrix"):
        operations.ewise_add(out, BinaryOp.plus, xT, yT)
    with pytest.raises(TypeError, match="TransposedMatrix"):
        operations.ewise_add(out, BinaryOp.plus, x, y, desc=descriptor.T0T1)

    outT = Matrix.new(INT16, 5, 2)
    mask = TransposedMatrix.wrap(mask)
    with pytest.raises(TypeError, match="TransposedMatrix"):
        operations.ewise_add(outT, BinaryOp.plus, xT, yT, mask=mask)
    with pytest.raises(TypeError, match="TransposedMatrix"):
        operations.ewise_add(outT, BinaryOp.plus, x, y, mask=mask, desc=descriptor.T0T1)


def test_mask_accum_replace(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, accum=BinaryOp.plus, desc=descriptor.RS)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 1, 1, 1])
    np_assert_equal(cols, [0, 1, 3, 4])
    np_assert_allclose(vals, [109, -4, 40, 395])


# TODO: add tests with transposed inputs and all combinations of mask, accum, replace


def test_mask_complement_accum_replace(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, accum=BinaryOp.plus, desc=descriptor.RSC)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 1])
    np_assert_equal(cols, [1, 3, 0])
    np_assert_allclose(vals, [220, 28, 297])


def test_mask_accum(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, accum=BinaryOp.plus, desc=descriptor.S)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 1, 1, 1, 1])
    np_assert_equal(cols, [0, 1, 0, 1, 3, 4])
    np_assert_allclose(vals, [109, 200, 300, -4, 40, 395])


def test_mask_complement_accum(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, accum=BinaryOp.plus, desc=descriptor.SC)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 0, 1, 1])
    np_assert_equal(cols, [0, 1, 3, 0, 4])
    np_assert_allclose(vals, [100, 220, 28, 297, 400])


def test_accum(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, accum=BinaryOp.plus)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 0, 1, 1, 1, 1])
    np_assert_equal(cols, [0, 1, 3, 0, 1, 3, 4])
    np_assert_allclose(vals, [109, 220, 28, 297, -4, 40, 395])


def test_mask_replace(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, desc=descriptor.RS)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 1, 1, 1])
    np_assert_equal(cols, [0, 1, 3, 4])
    np_assert_allclose(vals, [9, -4, 40, -5])


def test_mask_complement_replace(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, desc=descriptor.RSC)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 1])
    np_assert_equal(cols, [1, 3, 0])
    np_assert_allclose(vals, [20, 28, -3])


def test_mask(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, desc=descriptor.S)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 1, 1, 1, 1])
    np_assert_equal(cols, [0, 1, 0, 1, 3, 4])
    np_assert_allclose(vals, [9, 200, 300, -4, 40, -5])


def test_mask_complement(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, desc=descriptor.SC)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 0, 1, 1])
    np_assert_equal(cols, [0, 1, 3, 0, 4])
    np_assert_allclose(vals, [100, 20, 28, -3, 400])


def test_plain_update(mats):
    x, _, y, _, mask, out = mats
    operations.ewise_add(out, BinaryOp.plus, x, y)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 0, 1, 1, 1, 1])
    np_assert_equal(cols, [0, 1, 3, 0, 1, 3, 4])
    np_assert_allclose(vals, [9, 20, 28, -3, -4, 40, -5])


def test_value_mask(mats):
    x, _, y, _, mask, out = mats
    mask = Matrix.new(BOOL, 2, 5)
    mask.build([0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 4, 1, 2, 3, 4], [1, 0, 0, 0, 0, 1, 1, 1, 1])
    operations.ewise_add(out, BinaryOp.plus, x, y, mask=mask, desc=descriptor.NULL)
    rows, cols, vals = out.extract_tuples()
    np_assert_equal(rows, [0, 0, 1, 1, 1, 1])
    np_assert_equal(cols, [0, 1, 0, 1, 3, 4])
    np_assert_allclose(vals, [9, 200, 300, -4, 40, -5])


# TODO: verify all output arguments with assign
