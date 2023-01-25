import pytest
from ..tensor import Scalar, Vector, Matrix
from ..types import BOOL, INT16
from .. import operations
import pytest
from ..operators import UnaryOp, BinaryOp, SelectOp, IndexUnaryOp, Monoid, Semiring
from .. import descriptor
from ..tensor import TransposedMatrix
from .utils import matrix_compare


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


# TODO: verify all output arguments with assign
