import math
import numpy as np

from mlir import ir
from mlir.dialects import arith, sparse_tensor
from mlir.dialects.sparse_tensor import DimLevelType


__all__ = ["BOOL", "INT8", "INT16", "INT32", "INT64", "FP32", "FP64"]


class DType:
    _by_gbname = {}
    _by_mlirname = {}
    _by_nptype = {}

    def __init__(self):
        raise NotImplementedError("Use DType._register to create new dtypes")

    def build_mlir_type(self):
        return self._mlir_func()

    def __repr__(self):
        return f"DType<{self.gb_name}>"

    def __hash__(self):
        return hash(self.gb_name)

    def is_float(self):
        return self.mlir_name[0] == 'f'

    def is_int(self):
        return self.mlir_name[0] == 'i'

    def bitwidth(self):
        return int(self.mlir_name[1:])

    def build_max(self):
        if self.is_float():
            return arith.ConstantOp(self.build_mlir_type(), math.inf)
        if self.gb_name == 'BOOL':
            return 1
        if self.gb_name == 'INT8':
            return arith.ConstantOp(self.build_mlir_type(), 2**7 - 1)
        if self.gb_name == 'INT16':
            return arith.ConstantOp(self.build_mlir_type(), 2**15 - 1)
        if self.gb_name == 'INT32':
            return arith.ConstantOp(self.build_mlir_type(), 2**31 - 1)
        if self.gb_name == 'INT64':
            return arith.ConstantOp(self.build_mlir_type(), 2**63 - 1)
        raise TypeError(f"cannot call build_max for type {self}")

    def build_min(self):
        if self.is_float():
            return arith.ConstantOp(self.build_mlir_type(), -math.inf)
        if self.gb_name == 'BOOL':
            return 0
        if self.mlir_name == 'i8':
            return arith.ConstantOp(self.build_mlir_type(), 2**7)
        if self.mlir_name == 'i16':
            return arith.ConstantOp(self.build_mlir_type(), 2**15)
        if self.mlir_name == 'i32':
            return arith.ConstantOp(self.build_mlir_type(), 2**31)
        if self.mlir_name == 'i64':
            return arith.ConstantOp(self.build_mlir_type(), 2**63)
        raise TypeError(f"cannot call build_min for type {self}")

    def build_constant(self, val):
        if self.is_float():
            return arith.ConstantOp(self.build_mlir_type(), float(val))
        return arith.ConstantOp(self.build_mlir_type(), int(val))

    @staticmethod
    def _register(gb_name, mlir_name, np_type, mlir_func):
        gb_name = gb_name.upper()
        if gb_name in DType._by_gbname:
            raise KeyError(f"{gb_name} is already registered.")
        typ = DType.__new__(DType)
        typ.gb_name = gb_name
        typ.mlir_name = mlir_name
        typ.np_type = np_type
        typ._mlir_func = mlir_func
        typ.c_type = np.ctypeslib.as_ctypes_type(np_type)
        # Register for static lookup
        DType._by_gbname[gb_name] = typ
        DType._by_mlirname[mlir_name] = typ
        DType._by_nptype[np_type] = typ
        return typ

    @classmethod
    def from_gb(cls, name):
        return cls._by_gbname[name.upper()]

    @classmethod
    def from_mlir(cls, name):
        return cls._by_mlirname[name]

    @classmethod
    def from_np(cls, nptype):
        # Convert dtype into proper numpy type
        if type(nptype) is not type and hasattr(nptype, "type"):
            nptype = nptype.type
        return cls._by_nptype[nptype]


# Populate builtin types
BOOL = DType._register('BOOL', 'i8', np.int8, lambda: ir.IntegerType.get_signless(8))
INT8 = DType._register('INT8', 'i8', np.int8, lambda: ir.IntegerType.get_signless(8))
INT16 = DType._register('INT16', 'i16', np.int16, lambda: ir.IntegerType.get_signless(16))
INT32 = DType._register('INT32', 'i32', np.int32, lambda: ir.IntegerType.get_signless(32))
INT64 = DType._register('INT64', 'i64', np.int64, lambda: ir.IntegerType.get_signless(64))
FP32 = DType._register('FP32', 'f32', np.float32, ir.F32Type.get)
FP64 = DType._register('FP64', 'f64', np.float64, ir.F64Type.get)
# TODO: should we handle complex types?
# TODO: should we handle unsigned ints? If so, utility functions need to change.


class RankedTensorType:
    """Represents an MLIR RankedTensorType
    with Dynamic Size and Sparse Encoding"""
    def __init__(self, dtype, sparsity, ordering):
        assert len(sparsity) == len(ordering)
        self.rank = len(sparsity)
        self.dtype = dtype
        self.sparsity = tuple(sparsity)
        self.ordering = tuple(ordering)

    def as_mlir_type(self):
        sizes = [ir.ShapedType.get_dynamic_size()]*self.rank
        dim_ordering = ir.AffineMap.get_permutation(self.ordering)
        pointer_bit_width = index_bit_width = 0
        sp_encoding = sparse_tensor.EncodingAttr.get(
            self.sparsity, dim_ordering, None, pointer_bit_width, index_bit_width
        )
        return ir.RankedTensorType.get(sizes, self.dtype.build_mlir_type(), sp_encoding)

    def copy(self, dtype=None, sparsity=None, ordering=None):
        if dtype is None:
            dtype = self.dtype
        if sparsity is None:
            sparsity = self.sparsity
        if ordering is None:
            ordering = self.ordering
        return RankedTensorType(dtype, sparsity, ordering)


def find_common_dtype(left: DType, right: DType):
    """
    Follow C convention for unifying types
    """
    if left == right:
        return left

    bits = max(left.bitwidth(), right.bitwidth())
    # There are no unsigned types; if that changes, this will need to change as well
    if left.is_float() or right.is_float():
        return DType.from_gb(f"FP{bits}")
    return DType.from_gb(f"INT{bits}")


def cast(val, intype: DType, outtype: DType):
    if intype == outtype:
        return val

    out_mlir = outtype.build_mlir_type()
    if outtype.is_float():
        if intype.is_int():
            return arith.SIToFPOp(out_mlir, val)
        if intype.bitwidth() < outtype.bitwidth():
            return arith.ExtFOp(out_mlir, val)
        return arith.TruncFOp(out_mlir, val)
    # Output is int
    if intype.is_float():
        return arith.FPToSIOp(out_mlir, val)
    if intype.bitwidth() < outtype.bitwidth():
        return arith.ExtSIOp(out_mlir, val)
    return arith.TruncIOp(out_mlir, val)
