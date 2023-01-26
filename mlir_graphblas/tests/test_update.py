import pytest
from ..tensor import Scalar, Vector, Matrix
from ..types import BOOL, INT16, INT32
from .. import operations
import pytest
from ..operators import UnaryOp, BinaryOp, SelectOp, IndexUnaryOp, Monoid, Semiring
from .. import descriptor
from ..tensor import TransposedMatrix
from .utils import matrix_compare, vector_compare


@pytest.fixture
def mats():
    xdata = [0, 0, 1, 1, 1], [0, 3, 0, 1, 4], [-1, -2, -3, -4, -5]
    ydata = [0, 0, 0, 1], [0, 1, 3, 3], [10, 20, 30, 40]
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
    out.build([0, 0, 1, 1], [0, 1, 0, 4], [100, 200, 300, 400])
    return x, xT, y, yT, mask, out


@pytest.fixture
def assign_mats():
    x = Matrix.new(INT32, 3, 3)
    x.build([0, 0, 1, 1, 2, 2, 2], [0, 1, 1, 2, 0, 1, 2], [1, 2, 5, 6, 7, 8, 9])
    y = Matrix.new(INT32, 2, 2)
    y.build([0, 0, 1], [0, 1, 0], [9, 13, 11])
    mask = Matrix.new(BOOL, 3, 3)
    mask.build([0, 1, 1, 2, 2, 2], [2, 1, 2, 0, 1, 2], [1, 1, 1, 1, 1, 1])
    return x, y, mask


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
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.RS),
                    (x, yT, descriptor.RST1),
                    (xT, y, descriptor.RST0),
                    (xT, yT, descriptor.RST0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, accum=BinaryOp.plus, desc=d)
        matrix_compare(z,
                       [0, 1, 1, 1],
                       [0, 1, 3, 4],
                       [109, -4, 40, 395])


def test_mask_complement_accum_replace(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.RSC),
                    (x, yT, descriptor.RSCT1),
                    (xT, y, descriptor.RSCT0),
                    (xT, yT, descriptor.RSCT0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, accum=BinaryOp.plus, desc=d)
        matrix_compare(z,
                       [0, 0, 1],
                       [1, 3, 0],
                       [220, 28, 297])


def test_mask_accum(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.S),
                    (x, yT, descriptor.ST1),
                    (xT, y, descriptor.ST0),
                    (xT, yT, descriptor.ST0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, accum=BinaryOp.plus, desc=d)
        matrix_compare(z,
                       [0, 0, 1, 1, 1, 1],
                       [0, 1, 0, 1, 3, 4],
                       [109, 200, 300, -4, 40, 395])


def test_mask_complement_accum(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.SC),
                    (x, yT, descriptor.SCT1),
                    (xT, y, descriptor.SCT0),
                    (xT, yT, descriptor.SCT0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, accum=BinaryOp.plus, desc=d)
        matrix_compare(z,
                       [0, 0, 0, 1, 1],
                       [0, 1, 3, 0, 4],
                       [100, 220, 28, 297, 400])


def test_accum(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.NULL),
                    (x, yT, descriptor.T1),
                    (xT, y, descriptor.T0),
                    (xT, yT, descriptor.T0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, accum=BinaryOp.plus, desc=d)
        matrix_compare(z,
                       [0, 0, 0, 1, 1, 1, 1],
                       [0, 1, 3, 0, 1, 3, 4],
                       [109, 220, 28, 297, -4, 40, 395])


def test_mask_replace(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.RS),
                    (x, yT, descriptor.RST1),
                    (xT, y, descriptor.RST0),
                    (xT, yT, descriptor.RST0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, desc=d)
        matrix_compare(z,
                       [0, 1, 1, 1],
                       [0, 1, 3, 4],
                       [9, -4, 40, -5])


def test_mask_complement_replace(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.RSC),
                    (x, yT, descriptor.RSCT1),
                    (xT, y, descriptor.RSCT0),
                    (xT, yT, descriptor.RSCT0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, desc=d)
        matrix_compare(z,
                       [0, 0, 1],
                       [1, 3, 0],
                       [20, 28, -3])


def test_mask(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.S),
                    (x, yT, descriptor.ST1),
                    (xT, y, descriptor.ST0),
                    (xT, yT, descriptor.ST0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, desc=d)
        matrix_compare(z,
                       [0, 0, 1, 1, 1, 1],
                       [0, 1, 0, 1, 3, 4],
                       [9, 200, 300, -4, 40, -5])


def test_mask_complement(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.SC),
                    (x, yT, descriptor.SCT1),
                    (xT, y, descriptor.SCT0),
                    (xT, yT, descriptor.SCT0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, desc=d)
        matrix_compare(z,
                       [0, 0, 0, 1, 1],
                       [0, 1, 3, 0, 4],
                       [100, 20, 28, -3, 400])


def test_plain_update(mats):
    x, xT, y, yT, mask, out = mats
    for a, b, d in [(x, y, descriptor.NULL),
                    (x, yT, descriptor.T1),
                    (xT, y, descriptor.T0),
                    (xT, yT, descriptor.T0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, desc=d)
        matrix_compare(z,
                       [0, 0, 0, 1, 1, 1, 1],
                       [0, 1, 3, 0, 1, 3, 4],
                       [9, 20, 28, -3, -4, 40, -5])


def test_value_mask(mats):
    x, xT, y, yT, _, out = mats
    mask = Matrix.new(BOOL, 2, 5)
    mask.build([0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 3, 4, 1, 2, 3, 4], [1, 0, 0, 0, 0, 1, 1, 1, 1])
    for a, b, d in [(x, y, descriptor.NULL),
                    (x, yT, descriptor.T1),
                    (xT, y, descriptor.T0),
                    (xT, yT, descriptor.T0T1)]:
        z = out.dup()
        operations.ewise_add(z, BinaryOp.plus, a, b, mask=mask, desc=d)
        matrix_compare(z,
                       [0, 0, 1, 1, 1, 1],
                       [0, 1, 0, 1, 3, 4],
                       [9, 200, 300, -4, 40, -5])


def test_vector_mask():
    x = Vector.new(INT16, 5)
    x.build([0, 2, 3], [1, 2, 3])
    y = Vector.new(INT16, 5)
    y.build([0, 1, 2], [10, 20, 30])
    mask = Vector.new(BOOL, 5)
    mask.build([1, 2, 3, 4], [1, 1, 1, 1])
    z = Vector.new(INT16, 5)
    z.build([0, 2, 4], [100, 200, 300])
    operations.ewise_add(z, BinaryOp.plus, x, y, mask=mask, desc=descriptor.S)
    vector_compare(z, [0, 1, 2, 3], [100, 20, 32, 3])


def test_assign_mask_accum_replace(assign_mats):
    x, y, mask = assign_mats
    operations.assign(x, y, [0, 2], [0, 2], mask=mask, accum=BinaryOp.plus, desc=descriptor.RS)
    matrix_compare(x, [0, 1, 1, 2, 2, 2], [2, 1, 2, 0, 1, 2], [13, 5, 6, 18, 8, 9])


def test_assign_mask_complement_accum_replace(assign_mats):
    x, y, mask = assign_mats
    operations.assign(x, y, [0, 2], [0, 2], mask=mask, accum=BinaryOp.plus, desc=descriptor.RSC)
    matrix_compare(x, [0, 0], [0, 1], [10, 2])


def test_assign_mask_accum(assign_mats):
    x, y, mask = assign_mats
    operations.assign(x, y, [0, 2], [0, 2], mask=mask, accum=BinaryOp.plus, desc=descriptor.S)
    matrix_compare(x, [0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 1, 2, 0, 1, 2], [1, 2, 13, 5, 6, 18, 8, 9])


def test_assign_mask_complement_accum(assign_mats):
    x, y, mask = assign_mats
    operations.assign(x, y, [0, 2], [0, 2], mask=mask, accum=BinaryOp.plus, desc=descriptor.SC)
    matrix_compare(x, [0, 0, 1, 1, 2, 2, 2], [0, 1, 1, 2, 0, 1, 2], [10, 2, 5, 6, 7, 8, 9])


def test_assign_mask_replace(assign_mats):
    x, y, mask = assign_mats
    operations.assign(x, y, [0, 2], [0, 2], mask=mask, desc=descriptor.RS)
    matrix_compare(x, [0, 1, 1, 2, 2], [2, 1, 2, 0, 1], [13, 5, 6, 11, 8])


def test_assign_mask_complement_replace(assign_mats):
    x, y, mask = assign_mats
    operations.assign(x, y, [0, 2], [0, 2], mask=mask, desc=descriptor.RSC)
    matrix_compare(x, [0, 0], [0, 1], [9, 2])


def test_assign_mask(assign_mats):
    x, y, mask = assign_mats
    operations.assign(x, y, [0, 2], [0, 2], mask=mask, desc=descriptor.S)
    matrix_compare(x, [0, 0, 0, 1, 1, 2, 2], [0, 1, 2, 1, 2, 0, 1], [1, 2, 13, 5, 6, 11, 8])


def test_assign_mask_complement_zz(assign_mats):
    x, y, mask = assign_mats
    x1 = x.dup()
    operations.assign(x1, y, [0, 2], [0, 2], mask=mask, desc=descriptor.SC)
    matrix_compare(x1, [0, 0, 1, 1, 2, 2, 2], [0, 1, 1, 2, 0, 1, 2], [9, 2, 5, 6, 7, 8, 9])

    x2 = x.dup()
    alt_mask = Matrix.new(BOOL, 3, 3)
    alt_mask.build([0, 1, 1, 2], [2, 1, 2, 0], [1, 1, 1, 1])
    operations.assign(x2, y, [0, 2], [0, 2], mask=alt_mask, desc=descriptor.SC)
    matrix_compare(x2, [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 1], [9, 2, 5, 6, 7, 8])


def test_assign_accum(assign_mats):
    x, y, _ = assign_mats
    operations.assign(x, y, [0, 2], [0, 2], accum=BinaryOp.plus)
    matrix_compare(x, [0, 0, 0, 1, 1, 2, 2, 2], [0, 1, 2, 1, 2, 0, 1, 2], [10, 2, 13, 5, 6, 18, 8, 9])


def test_assign(assign_mats):
    x, y, _ = assign_mats
    operations.assign(x, y, [0, 2], [0, 2])
    matrix_compare(x, [0, 0, 0, 1, 1, 2, 2], [0, 1, 2, 1, 2, 0, 1], [9, 2, 13, 5, 6, 11, 8])


def test_assign_vector_mask():
    x = Vector.new(INT32, 8)
    x.build([0, 1, 3, 4, 5, 7], [1, 2, 3, 4, 5, 6])
    y = Vector.new(INT32, 4)
    y.build([0, 2, 3], [15, 17, 19])
    mask = Vector.new(BOOL, 8)
    mask.build([1, 2, 3, 4, 6], [1, 1, 1, 1, 1])
    operations.assign(x, y, [3, 4, 6, 7], mask=mask)
    vector_compare(x, [0, 1, 3, 5, 6, 7], [1, 2, 15, 5, 17, 6])
