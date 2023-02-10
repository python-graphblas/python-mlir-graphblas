import pytest
from mlir_graphblas import implementations as impl
from ..tensor import Scalar, Vector, Matrix, TransposedMatrix
from ..types import BOOL, INT16, INT32, INT64, FP32, FP64
from .. import operations, descriptor as desc
from ..operators import UnaryOp, BinaryOp, SelectOp, IndexUnaryOp, Monoid, Semiring
from .utils import vector_compare, matrix_compare


def test_select_by_indices_vec():
    v = Vector.new(INT16, 10)
    v.build([0, 2, 3, 4, 5, 8, 9], [1, 2, 3, 4, 5, 6, 7])
    w1 = impl.select_by_indices(v, [0, 1, 2, 3, 4, 9, 10])
    vector_compare(w1, [0, 2, 3, 4, 9], [1, 2, 3, 4, 7])
    w2 = impl.select_by_indices(v, [0, 1, 2, 3, 4, 9, 10], complement=True)
    vector_compare(w2, [5, 8], [5, 6])
    w3 = impl.select_by_indices(v, None)
    vector_compare(w3, [0, 2, 3, 4, 5, 8, 9], [1, 2, 3, 4, 5, 6, 7])
    w4 = impl.select_by_indices(v, None, complement=True)
    assert w4._obj is None

    with pytest.raises(AssertionError):
        impl.select_by_indices(v, [0, 1, 2], [2, 3, 4])


def test_select_by_indices_mat():
    m = Matrix.new(INT32, 3, 5)
    m.build([0, 0, 0, 2, 2, 2], [1, 2, 4, 0, 2, 4], [1, 2, 3, 4, 5, 6])
    z1 = impl.select_by_indices(m, [1, 0], [0, 2, 3, 4])
    matrix_compare(z1, [0, 0], [2, 4], [2, 3])
    z2 = impl.select_by_indices(m, [0, 1], [0, 2, 3, 4], complement=True)
    matrix_compare(z2, [0, 2, 2, 2], [1, 0, 2, 4], [1, 4, 5, 6])
    z3 = impl.select_by_indices(m, None, [0, 2, 4])
    matrix_compare(z3, [0, 0, 2, 2, 2], [2, 4, 0, 2, 4], [2, 3, 4, 5, 6])
    z4 = impl.select_by_indices(m, [0, 1], None)
    matrix_compare(z4, [0, 0, 0], [1, 2, 4], [1, 2, 3])
    z5 = impl.select_by_indices(m, None, [0, 2, 4], complement=True)
    matrix_compare(z5, [0], [1], [1])
    z6 = impl.select_by_indices(m, [0, 1], None, complement=True)
    matrix_compare(z6, [2, 2, 2], [0, 2, 4], [4, 5, 6])


def test_flip_layout():
    rows, cols, vals = [0, 0, 0, 2, 2, 2], [0, 1, 3, 0, 2, 3], [11., 10., -4., -1., 2., 3.]
    # Rowwise
    m1 = Matrix.new(FP64, 3, 4)
    m1.build(rows, cols, vals, colwise=False)
    f1 = impl.flip_layout(m1)
    matrix_compare(f1, rows, cols, vals)
    assert f1.is_colwise()
    # Colwise
    m2 = Matrix.new(FP32, 3, 4)
    m2.build(rows, cols, vals, colwise=True)
    f2 = impl.flip_layout(m2)
    matrix_compare(f2, rows, cols, vals)
    assert f2.is_rowwise()
    # Rowwise transposed
    m3 = TransposedMatrix.wrap(m1)
    assert m3.is_colwise()
    f3 = impl.flip_layout(m3)
    matrix_compare(f3, cols, rows, vals)
    assert f3.is_rowwise()
    # Colwise transposed
    m4 = TransposedMatrix.wrap(m2)
    assert m4.is_rowwise()
    f4 = impl.flip_layout(m4)
    matrix_compare(f4, cols, rows, vals)
    assert f4.is_colwise()
