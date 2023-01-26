import ctypes
import numpy as np

from .utils import c_lib, LLVMPTR
from .types import RankedTensorType
from mlir.dialects.sparse_tensor import DimLevelType
from .exceptions import (
    GrbNullPointer, GrbInvalidValue, GrbInvalidIndex, GrbDomainMismatch,
    GrbDimensionMismatch, GrbOutputNotEmpty, GrbIndexOutOfBounds, GrbEmptyObject
)

__all__ = ["Scalar", "Vector", "Matrix"]

_dimlevel_lookup = {
    'dense': DimLevelType.dense,
    'compressed': DimLevelType.compressed,
}

_converters = {}


def _find_converter(dtype, direction='from'):
    if dtype.mlir_name not in _converters:
        suffix = dtype.mlir_name.upper()
        from_ = getattr(c_lib, f"convertFromMLIRSparseTensor{suffix}")
        to_ = getattr(c_lib, f"convertToMLIRSparseTensor{suffix}")
        from_.restype = ctypes.c_void_p
        to_.restype = ctypes.c_void_p
        _converters[dtype.mlir_name] = (from_, to_)
    if direction == 'from':
        return _converters[dtype.mlir_name][0]
    elif direction == 'to':
        return _converters[dtype.mlir_name][1]
    raise ValueError(f'Invalid direction: {direction}; must be either "from" or "to"')


class SparseObject:
    ndims = -1  # filled in by subclass

    def __init__(self, dtype, shape, obj=None):
        self.dtype = dtype
        assert len(shape) == self.ndims
        self.shape = tuple(shape)
        self._obj = obj

    def __del__(self):
        self.clear()

    def clear(self):
        raise NotImplementedError()

    @property
    def baseclass(self):
        if self.ndims == 0:
            return Scalar
        if self.ndims == 1:
            return Vector
        if self.ndims == 2:
            return Matrix
        raise ValueError(f"Unknown baseclass for object with ndims={self.ndims}")


class SparseTensorBase(SparseObject):
    permutation = None  # linalg.generic indexing_maps

    def __init__(self, dtype, shape, obj=None, sparsity=None, ordering=None, *, intermediate_result=False):
        super().__init__(dtype, shape, obj=obj)
        self._sparsity = sparsity  # sparse_tensor.encoding dimLevelType
        self._ordering = ordering  # sparse_tensor.encoding dimOrdering
        self._rtt = None
        self._intermediate_result = intermediate_result

    def dup(self):
        f"""Returns a copy of the {self.baseclass}"""
        from . import implementations as impl

        return impl.dup(self, intermediate=False)

    @property
    def rtt(self):
        if self._rtt is None:
            if self._sparsity is None:
                raise ValueError("Sparsity has not been set; unable to build RankedTensorType")
            ordering = self._ordering
            if ordering is None:
                ordering = list(range(self.ndims))
            self._rtt = RankedTensorType(self.dtype, self._sparsity, ordering)
        return self._rtt

    @property
    def perceived_ordering(self):
        if self.ndims == 1:
            return self._ordering
        if tuple(self._ordering) == self.permutation:
            return [0, 1]
        return [1, 0]

    def get_loop_key(self):
        """
        Returns a hashable tuple containing all the properties of the object
        relevant for iteration looping. This is used to cache execution contexts
        and avoid duplicate JIT'ing when the same structure of inputs is used
        in future computations
        """
        return self.dtype, self.permutation, self.rtt.sparsity, self.rtt.ordering

    def _permute(self, vals: list):
        """
        Returns a new list by applying self.permutation to vals

        vals must be the same length as self.permutation
        """
        return [vals[i] for i in self.permutation]

    def clear(self):
        self._obj = None
        self._sparsity = None
        self._ordering = None
        self._rtt = None
        self._intermediate_result = False

    def extract_tuples(self):
        raise NotImplementedError()


class SparseTensor(SparseTensorBase):
    def clear(self):
        if self._obj is not None:
            c_lib.delSparseTensor(self._obj[0])
        super().clear()

    def _replace(self, tensor: SparseObject):
        if not tensor._intermediate_result and tensor._obj is not None:
            raise ValueError("Can only replace using intermediate values")
        self.clear()
        self._obj = tensor._obj
        self._sparsity = tensor._sparsity
        self._ordering = tensor._ordering
        # The pointer has been moved from tensor to self
        # Ensure it is not double-deleted
        tensor._obj = None

    def _to_sparse_tensor(self, np_indices, np_values, sparsity, ordering):
        assert np_values.dtype.type == self.dtype.np_type
        assert len(sparsity) == self.ndims
        assert len(ordering) == self.ndims
        sparsity = [_dimlevel_lookup.get(sp, sp) for sp in sparsity]
        np_shape = np.array(self.shape, dtype=np.uint64)
        np_sparse = np.array(sparsity, dtype=np.uint8)
        np_perm = np.array(ordering, dtype=np.uint64)

        # Validate indices are within range
        if np_indices.min() < 0:
            raise GrbIndexOutOfBounds(f"negative indices not allowed: {np_indices.min()}")
        if self.ndims == 2:
            max_row = np_indices[:, 0].max()
            max_col = np_indices[:, 1].max()
            if max_row >= self.shape[0]:
                raise GrbIndexOutOfBounds(f"row index out of bounds: {max_row} >= {self.shape[0]}")
            if max_col >= self.shape[1]:
                raise GrbIndexOutOfBounds(f"col index out of bound: {max_col} >= {self.shape[1]}")
        else:
            if np_indices.max() >= self.shape[0]:
                raise GrbIndexOutOfBounds(f"index out of bounds: {np_indices.max()} >= {self.shape[0]}")

        rank = ctypes.c_ulonglong(len(np_shape))
        nse = ctypes.c_ulonglong(len(np_values))
        shape = np_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        values = np_values.ctypes.data_as(ctypes.POINTER(self.dtype.c_type))
        indices = np_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))

        perm = np_perm.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong))
        sparse = np_sparse.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        convert_to = _find_converter(self.dtype, 'to')
        ptr = convert_to(rank, nse, shape, values, indices, perm, sparse)
        self._obj = ctypes.pointer(ctypes.cast(ptr, LLVMPTR))
        self._sparsity = sparsity
        self._ordering = ordering

    def _from_sparse_tensor(self):
        if self._obj is None:
            raise ValueError('SparseTensor obj is None')
        convert_from = _find_converter(self.dtype, 'from')
        rank = ctypes.c_ulonglong(0)
        nse = ctypes.c_ulonglong(0)
        shape = ctypes.POINTER(ctypes.c_ulonglong)()
        values = ctypes.POINTER(self.dtype.c_type)()
        indices = ctypes.POINTER(ctypes.c_ulonglong)()
        convert_from(self._obj[0], ctypes.byref(rank), ctypes.byref(nse),
                     ctypes.byref(shape), ctypes.byref(values), ctypes.byref(indices))
        shape = np.ctypeslib.as_array(shape, shape=[rank.value])
        ishape = [nse.value, rank.value] if rank.value == 2 else [nse.value]
        if nse.value == 0:
            # Creating arrays from empty pointers raises warnings and may break in the future
            # Handle explicitly to avoid this possibility
            values = np.empty((0, ), dtype=np.dtype(values.contents))
            indices = np.empty(ishape, dtype=np.dtype(indices.contents))
        else:
            values = np.ctypeslib.as_array(values, shape=[nse.value])
            indices = np.ctypeslib.as_array(indices, shape=ishape)
        if tuple(shape) != self.shape:
            raise GrbDimensionMismatch(f"extracted shape does not match declared shape {tuple(shape)} != {self.shape}")
        return indices, values, shape, rank.value, nse.value


class Scalar(SparseObject):
    ndims = 0

    def __repr__(self):
        if self._obj is None:
            return f'Scalar<{self.dtype.gb_name}, value=EMPTY>'
        return f'Scalar<{self.dtype.gb_name}, value={self._obj}>'

    @classmethod
    def new(cls, dtype):
        return cls(dtype, ())

    def clear(self):
        self._obj = None

    def dup(self):
        s = Scalar.new(self.dtype)
        s.set_element(self._obj)
        return s

    def nvals(self):
        if self._obj is None:
            return 0
        return 1

    def set_element(self, val):
        self._obj = self.dtype.np_type(val)

    def extract_element(self):
        if self._obj is None:
            return None
        return self._obj.item()


class Vector(SparseTensor):
    ndims = 1
    permutation = (0,)  # linalg.generic indexing_maps

    def __repr__(self):
        return f'Vector<{self.dtype.gb_name}, size={self.shape[0]}>'

    @classmethod
    def new(cls, dtype, size: int):
        return cls(dtype, (size,))

    def resize(self, size: int):
        raise NotImplementedError()

    def size(self):
        if self._obj is None:
            return self.shape[0]
        return c_lib.sparseDimSize(self._obj[0], 0)

    def nvals(self):
        if self._obj is None:
            return 0

        from . import implementations as impl

        return impl.nvals(self)

    def build(self, indices, values, *, dup=None, sparsity=None):
        """
        Build the underlying MLIRSparseTensor structure from COO.

        indices: list or numpy array of int
        values: list or numpy array with matching dtype as declared in `.new()`
                can also be a scalar value to make the Vector iso-valued
        dup: BinaryOp used to combined entries with the same index
             NOTE: this is currently not support; passing dup will raise an error
        sparsity: list of string or DimLevelType
                  allowable strings: "dense", "compressed"
        """
        if self._obj is not None:
            raise GrbOutputNotEmpty("Before rebuilding, call `clear`")
        # TODO: check domain matching (self.dtype, value.dtype, dup.dtype)
        # TODO: if dup is None, check that no duplicates exist or raise GrbInvalidValue
        #       alternatively, require that dup is a Python operator; use numpy.diff
        #       and np.argsort to find and clean up duplicates
        if dup is not None:
            raise NotImplementedError()
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices, dtype=np.uint64)
        if not isinstance(values, np.ndarray):
            if hasattr(values, '__len__'):
                values = np.array(values, dtype=self.dtype.np_type)
            else:
                values = np.ones(indices.shape, dtype=self.dtype.np_type) * values
        if sparsity is None:
            sparsity = [DimLevelType.compressed]
        self._to_sparse_tensor(indices, values, sparsity=sparsity, ordering=[0])

    def set_element(self, val: Scalar, index: int):
        raise NotImplementedError()

    def remove_element(self, index: int):
        raise NotImplementedError()

    def extract_element(self, index: int):
        raise NotImplementedError()

    def extract_tuples(self):
        if self._obj is None:
            return np.array([], dtype=np.uint64), np.array([], dtype=self.dtype.np_type)
        indices, values, shape, rank, nse = self._from_sparse_tensor()
        return indices, values


class Matrix(SparseTensor):
    ndims = 2
    permutation = (0, 1)  # linalg.generic indexing_maps

    def __repr__(self):
        return f'Matrix<{self.dtype.gb_name}, shape={self.shape}, format={self._compute_format()}>'

    def _compute_format(self):
        if self._sparsity == [DimLevelType.dense, DimLevelType.compressed]:
            return 'CSR' if self.is_rowwise() else 'CSC'
        if self._sparsity == [DimLevelType.compressed, DimLevelType.compressed]:
            return 'DCSR' if self.is_rowwise() else 'DCSC'
        return '?'

    def is_rowwise(self):
        return tuple(self._ordering) == self.permutation

    def is_colwise(self):
        return tuple(self._ordering) != self.permutation

    @classmethod
    def new(cls, dtype, nrows: int, ncols: int):
        return cls(dtype, (nrows, ncols))

    def diag(self, k: int):
        raise NotImplementedError()

    def resize(self, nrows: int, ncols: int):
        raise NotImplementedError()

    def nrows(self):
        if self._obj is None:
            return self.shape[0]
        return c_lib.sparseDimSize(self._obj[0], self._ordering[0])

    def ncols(self):
        if self._obj is None:
            return self.shape[1]
        return c_lib.sparseDimSize(self._obj[0], self._ordering[1])

    def nvals(self):
        if self._obj is None:
            return 0

        from .implementations import nvals

        return nvals(self)

    def build(self, row_indices, col_indices, values, *, dup=None, sparsity=None, colwise=False):
        """
        Build the underlying MLIRSparseTensor structure from COO.

        row_indices: list or numpy array of int
        col_indices: list or numpy array of int
        values: list or numpy array with matching dtype as declared in `.new()`
                can also be a scalar value to make the Vector iso-valued
        dup: BinaryOp used to combined entries with the same (row, col) coordinate
             NOTE: this is currently not support; passing dup will raise an error
        sparsity: list of string or DimLevelType
                  allowable strings: "dense", "compressed"
        colwise: (default False) Whether to store the data column-wise
        """
        if self._obj is not None:
            raise GrbOutputNotEmpty("Before rebuilding, call `clear`")
        # TODO: check domain matching (self.dtype, value.dtype, dup.dtype)
        # TODO: if dup is None, check that no duplicates exist or raise GrbInvalidValue
        #       alternatively, require that dup is a Python operator; use numpy.diff
        #       and np.argsort to find and clean up duplicates
        if dup is not None:
            raise NotImplementedError()
        if not isinstance(row_indices, np.ndarray):
            row_indices = np.array(row_indices, dtype=np.uint64)
        if not isinstance(col_indices, np.ndarray):
            col_indices = np.array(col_indices, dtype=np.uint64)
        indices = np.stack([row_indices, col_indices], axis=1)
        if not isinstance(values, np.ndarray):
            if hasattr(values, '__len__'):
                values = np.array(values, dtype=self.dtype.np_type)
            else:
                values = np.ones(indices.shape, dtype=self.dtype.np_type) * values
        ordering = [1, 0] if colwise else [0, 1]
        if sparsity is None:
            sparsity = [DimLevelType.dense, DimLevelType.compressed]
        self._to_sparse_tensor(indices, values, sparsity=sparsity, ordering=ordering)

    def set_element(self, val: Scalar, row: int, col: int):
        raise NotImplementedError()

    def remove_element(self, row: int, col: int):
        raise NotImplementedError()

    def extract_element(self, row: int, col: int):
        raise NotImplementedError()

    def extract_tuples(self):
        if self._obj is None:
            return (
                np.array([], dtype=np.uint64),  # row_indices
                np.array([], dtype=np.uint64),  # col_indices
                np.array([], dtype=self.dtype.np_type)  # values
            )
        indices, values, shape, rank, nse = self._from_sparse_tensor()
        # Expand indices into separate arrays
        row_indices = indices[:, 0]
        col_indices = indices[:, 1]
        return row_indices, col_indices, values


class TransposedMatrix(SparseTensorBase):
    ndims = 2
    permutation = (1, 0)  # linalg.generic indexing_maps

    @classmethod
    def wrap(cls, m: Matrix):
        """
        Creates a TransposedMatrix view of `m`.
        If the input is already a TransposedMatrix,
            this will return the underlying untransposed Matrix.
        """
        if type(m) is TransposedMatrix:
            return m._referenced_matrix
        if type(m) is not Matrix:
            raise TypeError(f"Only Matrix can be transposed, not {type(m)}")
        # Reverse the shape
        shape = [m.shape[1], m.shape[0]]
        trans = cls(m.dtype, shape, m._obj, m._sparsity, m._ordering)
        # Keep a reference to matrix
        trans._referenced_matrix = m
        return trans

    def __repr__(self):
        return f'TransposedMatrix<{self.dtype.gb_name}, shape={self.shape}, format={self._compute_format()}>'

    def _compute_format(self):
        if self._sparsity == [DimLevelType.dense, DimLevelType.compressed]:
            return 'CSR' if self.is_rowwise() else 'CSC'
        if self._sparsity == [DimLevelType.compressed, DimLevelType.compressed]:
            return 'DCSR' if self.is_rowwise() else 'DCSC'
        return '?'

    def is_rowwise(self):
        return tuple(self._ordering) == self.permutation

    def is_colwise(self):
        return tuple(self._ordering) != self.permutation

    def nrows(self):
        return self._referenced_matrix.ncols

    def ncols(self):
        return self._referenced_matrix.nrows

    def nvals(self):
        return self._referenced_matrix.nvals()

    def clear(self):
        super().clear()
        self._referenced_matrix = None

    def extract_tuples(self):
        col_indices, row_indices, values = self._referenced_matrix.extract_tuples()
        return row_indices, col_indices, values
