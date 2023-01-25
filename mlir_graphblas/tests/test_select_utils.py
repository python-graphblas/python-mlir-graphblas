import pytest
from mlir_graphblas.implementations import select_by_mask, select_by_indices
from ..tensor import Scalar, Vector, Matrix
from ..types import BOOL, INT16, INT32, INT64, FP32, FP64
from .. import operations, descriptor as desc
from ..operators import UnaryOp, BinaryOp, SelectOp, IndexUnaryOp, Monoid, Semiring
from .utils import vector_compare, matrix_compare


def test_select_by_indices_vec():
    v = Vector.new(INT16, 10)
    v.build([0, 2, 3, 4, 5, 8, 9], [1, 2, 3, 4, 5, 6, 7])
    w1 = select_by_indices(v, [0, 1, 2, 3, 4, 9, 10])
    vector_compare(w1, [0, 2, 3, 4, 9], [1, 2, 3, 4, 7])
    w2 = select_by_indices(v, [0, 1, 2, 3, 4, 9, 10], complement=True)
    vector_compare(w2, [5, 8], [5, 6])
    w3 = select_by_indices(v, None)
    vector_compare(w3, [0, 2, 3, 4, 5, 8, 9], [1, 2, 3, 4, 5, 6, 7])
    w4 = select_by_indices(v, None, complement=True)
    assert w4._obj is None

    with pytest.raises(AssertionError):
        select_by_indices(v, [0, 1, 2], [2, 3, 4])


def test_select_by_indices_mat():
    m = Matrix.new(INT32, 3, 5)
    m.build([0, 0, 0, 2, 2, 2], [1, 2, 4, 0, 2, 4], [1, 2, 3, 4, 5, 6])
    z1 = select_by_indices(m, [1, 0], [0, 2, 3, 4])
    matrix_compare(z1, [0, 0], [2, 4], [2, 3])
    z2 = select_by_indices(m, [0, 1], [0, 2, 3, 4], complement=True)
    matrix_compare(z2, [0, 2, 2, 2], [1, 0, 2, 4], [1, 4, 5, 6])
    z3 = select_by_indices(m, None, [0, 2, 4])
    matrix_compare(z3, [0, 0, 2, 2, 2], [2, 4, 0, 2, 4], [2, 3, 4, 5, 6])
    z4 = select_by_indices(m, [0, 1], None)
    matrix_compare(z4, [0, 0, 0], [1, 2, 4], [1, 2, 3])
    z5 = select_by_indices(m, None, [0, 2, 4], complement=True)
    matrix_compare(z5, [0], [1], [1])
    z6 = select_by_indices(m, [0, 1], None, complement=True)
    matrix_compare(z6, [2, 2, 2], [0, 2, 4], [4, 5, 6])
