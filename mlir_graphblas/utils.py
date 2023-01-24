import ctypes
import numpy as np
from enum import Enum
from mlir import ir
from .exceptions import (
    GrbNullPointer, GrbInvalidValue, GrbInvalidIndex, GrbDomainMismatch,
    GrbDimensionMismatch, GrbOutputNotEmpty, GrbIndexOutOfBounds, GrbEmptyObject
)


LLVMPTR = ctypes.POINTER(ctypes.c_char)
# TODO: fix this once a proper package exists
import os
LIBMLIR_C_RUNNER_UTILS = f"{os.environ['LLVM_BUILD_DIR']}/lib/libmlir_c_runner_utils.dylib"

c_lib = ctypes.CDLL(LIBMLIR_C_RUNNER_UTILS)


def get_sparse_output_pointer():
    return ctypes.pointer(ctypes.pointer(ctypes.c_char(0)))


def get_scalar_output_pointer(dtype):
    return ctypes.pointer(dtype.c_type(0))


def ensure_scalar_of_type(obj, dtype):
    from .tensor import Scalar

    if type(obj) is Scalar:
        if obj.dtype != dtype:
            raise GrbDomainMismatch(f"Expected Scalar of type {dtype}, but found {obj.dtype}")
        return obj
    if type(obj) not in {bool, int, float}:
        raise TypeError(f"Expected Scalar, bool, int, or float, but found {type(obj)}")
    # Create a Scalar with correct dtype
    s = Scalar.new(dtype)
    s.set_element(obj)
    return s


def renumber_indices(indices, selected):
    """
    Given a set of non-unique `indices`, returns an array of the same size
    as `indices` with values renumbered according to the positions in `selected`.

    All values in indices must also be found in selected.

    If these were Python lists instead of numpy arrays, this would be
    equivalent to calling `[selected.index(x) for x in indices]`.
    However, this will be much faster as it uses numpy to perform
    the lookups.

    :param indices: ndarray of non-unique positive integers
    :param selected: ndarray of unique positive integers
    :return: ndarray of same length as indices

    Example
    -------
    >>> a = np.array([1, 1, 1, 3, 5])
    >>> b = np.array([1, 2, 5, 3])
    >>> renumber_indices(a, b)
    array([0, 0, 0, 3, 2])
    """
    # Check that values in selected are unique
    unique = np.unique(selected)
    if unique.size < selected.size:
        unique, counts = np.unique(selected, return_counts=True)
        raise ValueError(f"Found duplicate values in `selected`: {unique[counts > 1]}")

    # Check for required inclusion criteria
    not_found = np.setdiff1d(indices, selected)
    if not_found.size > 0:
        raise KeyError(f"Found values in `indices` not contained in `selected`: {not_found}")

    # To be efficient, the searching must be done on a sorted array
    # Build the sort_order to map back to the original order
    sort_order = np.argsort(selected)
    renumbered_indices = np.arange(len(selected), dtype=indices.dtype)[sort_order]
    pos = np.searchsorted(selected[sort_order], indices)
    return renumbered_indices[pos]


# https://mlir.llvm.org/docs/Dialects/ArithOps/#arithcmpi-mlirarithcmpiop
class CmpIPredicate(Enum):
    eq = 0  # equal
    ne = 1  # not equal
    slt = 2  # signed less than
    sle = 3  # signed less than or equal
    sgt = 4  # signed greater than
    sge = 5  # signed greater than or equal
    ult = 6  # unsigned less than
    ule = 7  # unsigned less than or equal
    ugt = 8  # unsigned greater than
    uge = 9  # unsigned greater than or equal

    def build(self):
        i64 = ir.IntegerType.get_signless(64)
        return ir.IntegerAttr.get(i64, self.value)


# https://llvm.org/docs/LangRef.html#fcmp-instruction
class CmpFPredicate(Enum):
    false = 0  # no comparison, always returns false
    oeq = 1  # ordered and equal
    ogt = 2  # ordered and greater than
    oge = 3  # ordered and greater than or equal
    olt = 4  # ordered and less than
    ole = 5  # ordered and less than or equal
    one = 6  # ordered and not equal
    ord = 7  # ordered (no nans)
    ueq = 8  # unordered or equal
    ugt = 9  # unordered or greater than
    uge = 10  # unordered or greater than or equal
    ult = 11  # unordered or less than
    ule = 12  # unordered or less than or equal
    une = 13  # unordered or not equal
    uno = 14  # unordered (either nans)
    true = 15  # no comparison, always returns true

    def build(self):
        i64 = ir.IntegerType.get_signless(64)
        return ir.IntegerAttr.get(i64, self.value)
