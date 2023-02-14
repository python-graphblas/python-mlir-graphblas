import ctypes
import numpy as np
from enum import Enum
from mlir import ir
from mlir.dialects.sparse_tensor import DimLevelType
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


def get_scalar_input_arg(s):
    return ctypes.byref(s.dtype.c_type(s._obj))


def ensure_scalar_of_type(obj, dtype):
    from .tensor import Scalar

    if type(obj) is Scalar:
        if obj.dtype != dtype:
            raise GrbDomainMismatch(f"Expected Scalar of type {dtype}, but found {obj.dtype}")
        return obj
    if type(obj) not in {bool, int, float}:
        raise TypeError(f"Expected Scalar, bool, int, or float, but found {type(obj)}")
    # Create a Scalar with correct dtype
    return Scalar.new(dtype, obj)


def ensure_unique(indices, name=None):
    unique = np.unique(indices)
    if len(unique) < len(indices):
        unique, counts = np.unique(indices, return_counts=True)
        name_str = f" in {name}" if name is not None else ""
        raise ValueError(f"Found duplicate indices{name_str}: {unique[counts > 1]}")


def determine_sparsity(left, right, union=False):
    """
    Returns the sparsity appropriate for the two inputs based on `union`.
    If union == True, finds a sparsity that anticipates more values than either input.
    If union == False, finds a sparsity that anticipates the same or fewer values than either input.
    """
    assert left.ndims == right.ndims
    assert left.ndims > 0, "Not allowed for Scalars"
    if left._sparsity == right._sparsity:
        return left._sparsity

    dense = DimLevelType.dense
    comp = DimLevelType.compressed

    if left.ndims == 1:  # Vector
        levels = ([comp], [dense])
    else:  # Matrix
        levels = ([comp, comp], [comp, dense], [dense, comp], [dense, dense])

    if union:
        levels = reversed(levels)
    for lvl in levels:
        if left._sparsity == lvl or right._sparsity == lvl:
            return lvl

    raise Exception("something went wrong finding the sparsity")


def pick_and_renumber_indices(selected, indices, *related):
    """
    This function is used by the `extract` operation.

    Picks elements from indices as indicated in `selected`. The new indices will
    be renumbered according to the position in `selected`. Related ndarrays will
    maintain matching order during the picking, but values will not be changed.

    :param selected: values to pick from indices in the order indicated
    :param indices: ndarray representing indices
    :param related: list of ndarray of the same length as indices that must maintain
                    proper relationship as indices are picked and rearranged
    :return: tuple of modified ndarrays -- (indices, *related)
    """
    for r in related:
        assert len(r) == len(indices)
    if len(indices) == 0:
        return indices, *related
    # Ensure indices are sorted (this is needed for np.searchsorted)
    sort_order = np.argsort(indices)
    indices = indices[sort_order]
    related = [r[sort_order] for r in related]
    # Find which `selected` exist in indices
    is_found = np.isin(selected, indices)
    selfound = selected[is_found]
    # Perform sorted search of first and last occurrence
    # Duplicates are handled by building a ramp of indexes between first and last
    pick_start = np.searchsorted(indices, selfound, side='left')
    pick_end = np.searchsorted(indices, selfound, side='right')
    pick_delta = pick_end - pick_start
    expanded_pick = np.ones(pick_delta.sum(), dtype=np.int64)
    expanded_pick[0] = pick_start[0]
    expanded_pick[np.cumsum(pick_delta)[:-1]] += pick_start[1:] - pick_end[:-1]
    pick_index = np.cumsum(expanded_pick)
    # Use pick_index to index into related arrays
    new_related = [r[pick_index] for r in related]
    # Build new indices by repeating range
    new_indices = np.repeat(np.arange(len(selected), dtype=np.uint64)[is_found], pick_delta)
    return new_indices, *new_related


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
