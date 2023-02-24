import ctypes
import numpy as np
from typing import Union, Optional
from mlir import ir
from mlir.dialects import arith
from mlir.dialects import bufferization
from mlir.dialects import func
from mlir.dialects import linalg
from mlir.dialects import sparse_tensor
from mlir.dialects import tensor
from mlir.dialects import scf
from mlir.dialects import memref
from mlir.dialects.sparse_tensor import DimLevelType

from .tensor import SparseTensorBase, SparseTensor, Matrix, Vector, Scalar, TransposedMatrix
from .operators import UnaryOp, BinaryOp, SelectOp, IndexUnaryOp, Monoid, Semiring
from .compiler import compile, engine_cache
from . descriptor import Descriptor, NULL as NULL_DESC
from .utils import (get_sparse_output_pointer, get_scalar_output_pointer,
                    get_scalar_input_arg, pick_and_renumber_indices, determine_sparsity)
from .types import DType, RankedTensorType, BOOL, INT64, FP64, cast
from .exceptions import GrbError, GrbIndexOutOfBounds, GrbDimensionMismatch


# TODO: vec->matrix broadcasting as builtin param in select_by_mask (rowwise/colwise)
def select_by_mask(sp: SparseTensorBase, mask: SparseTensor, desc: Descriptor = NULL_DESC):
    """
    The only elements which survive in `sp` are those with corresponding elements in
    `mask` which are "truthy" (i.e. non-zero).
    If desc.mask_structure is True, then all elements found in the mask are treated as
    being "truthy" (e.g. only the structure is considered, not the actual values).

    If desc.mask_complement is True, the inverse logic is applied and the surviving elements
    in `sp` correspond to missing or "falsy" elements in the mask.
    """
    assert mask.ndims == sp.ndims
    if mask.shape != sp.shape:
        raise GrbDimensionMismatch(f"Mask shape mismatch: {mask.shape} != {sp.shape}")

    rank = sp.ndims
    if rank == 0:  # Scalar
        s = Scalar.new(sp.dtype)
        if mask._obj is not None:
            s.set_element(sp.extract_element())
        return s

    if not isinstance(mask, SparseTensor):
        raise TypeError(f"Mask must be Vector or Matrix, not {type(mask)}")

    # Convert value mask to structural mask
    if not desc.mask_structure:
        zero = Scalar.new(mask.dtype, 0)
        mask = select(mask.dtype, SelectOp.valuene, mask, thunk=zero)

    # Build and compile if needed
    key = ('select_by_mask', *sp.get_loop_key(), *mask.get_loop_key(), desc.mask_complement)
    if key not in engine_cache:
        engine_cache[key] = _build_select_by_mask(mask, sp, desc.mask_complement)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [mask._obj, sp._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return mask.baseclass(sp.dtype, mask.shape, mem_out, determine_sparsity(mask, sp),
                          mask.perceived_ordering, intermediate_result=True)


def _build_select_by_mask(mask: SparseTensor, sp: SparseTensorBase, complement: bool):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rank = sp.ndims
            index = ir.IndexType.get()
            dtype_sp = sp.dtype.build_mlir_type()
            dtype_mask = mask.dtype.build_mlir_type()
            dtype_out = dtype_sp
            perm_sp = ir.AffineMap.get_permutation(sp.permutation)
            perm_mask = ir.AffineMap.get_permutation(mask.permutation)
            perm_out = ir.AffineMap.get_permutation(range(rank))
            rtt_sp = sp.rtt.as_mlir_type()
            rtt_mask = mask.rtt.as_mlir_type()
            rtt_out = mask.rtt.copy(dtype=sp.dtype,
                                    sparsity=determine_sparsity(mask, sp)).as_mlir_type()

            @func.FuncOp.from_py_func(rtt_mask, rtt_sp)
            def main(msk, x):
                c = [arith.ConstantOp(index, i) for i in range(rank)]
                dims = [tensor.DimOp(msk, c[i]).result for i in mask.permutation]
                out = bufferization.AllocTensorOp(rtt_out, dims, None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [msk, x],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm_mask, perm_sp, perm_out)]),
                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*rank)
                )
                block = generic_op.regions[0].blocks.append(dtype_mask, dtype_sp, dtype_out)
                with ir.InsertionPoint(block):
                    m, a, o = block.arguments
                    res = sparse_tensor.BinaryOp(dtype_out, m, a, right_identity=complement)
                    if not complement:
                        overlap = res.regions[0].blocks.append(dtype_mask, dtype_sp)
                        with ir.InsertionPoint(overlap):
                            _, arg0 = overlap.arguments
                            sparse_tensor.YieldOp(result=arg0)
                    linalg.YieldOp([res])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def select_by_indices(sp: SparseTensorBase,
                      row_indices: Optional[list[int]] = None,
                      col_indices: Optional[list[int]] = None,
                      complement: bool = False):
    """
    Returns a new sparse tensor with the same dtype and shape as `sp`.

    The only elements copied over from `sp` are those with indices found in
    row_indices and col_indices. Vectors ignore col_indices.

    If complement is True, the inverse logic is applied and indices *not*
    found in row_indices and col_indices are copied to the output.

    If row_indices is None, all rows are selected (i.e. GrB_ALL).
    If col_indices is None, all columns are selected (i.e. GrB_ALL).
    This allows, for example, selecting all rows for a subset of columns
    without needing to list every possible row index.
    """
    if sp.ndims == 1:
        # Vector
        assert col_indices is None
        if row_indices is None:
            if complement:
                return Vector.new(sp.dtype, *sp.shape)
            return dup(sp.dtype, sp)

        idx, vals = sp.extract_tuples()
        row_indices = np.array(row_indices, dtype=np.uint64)
        selected = np.isin(idx, row_indices, invert=complement)
        v = Vector.new(sp.dtype, *sp.shape, intermediate_result=True)
        v.build(idx[selected], vals[selected])
        return v

    # Matrix
    if row_indices is None and col_indices is None:
        if complement:
            return Matrix.new(sp.dtype, *sp.shape)
        return dup(sp.dtype, sp)

    rowidx, colidx, vals = sp.extract_tuples()
    if row_indices is not None:
        row_indices = np.array(row_indices, dtype=np.uint64)
        rowsel = np.isin(rowidx, row_indices, invert=complement)
    if col_indices is not None:
        col_indices = np.array(col_indices, dtype=np.uint64)
        colsel = np.isin(colidx, col_indices, invert=complement)

    if row_indices is not None and col_indices is not None:
        sel = (rowsel | colsel) if complement else (rowsel & colsel)
    elif row_indices is not None:
        sel = rowsel
    else:
        sel = colsel
    m = Matrix.new(sp.dtype, *sp.shape, intermediate_result=True)
    m.build(rowidx[sel], colidx[sel], vals[sel])
    return m


def build_iso_vector_from_indices(dtype,
                                  size: int,
                                  indices: Optional[list[int]] = None,
                                  value=1):
    """
    Returns a new sparse Vector of size `size` with all
    elements in indices set to `value`.
    """
    v = Vector.new(dtype, size, intermediate_result=True)
    if indices is None:
        indices = np.arange(size)
    if not hasattr(indices, '__len__'):
        raise TypeError(f"indices must be a tuple/list/array, not {type(indices)}")
    v.build(indices, value)
    return v


def build_iso_matrix_from_indices(dtype,
                                  nrows: int,
                                  ncols: int,
                                  row_indices: Optional[list[int]] = None,
                                  col_indices: Optional[list[int]] = None,
                                  value=1,
                                  *,
                                  colwise=False):
    """
    Returns a new sparse Matrix of shape (nrows, ncols) with all
    elements in (row_indices, col_indices) pairs set to `value`.
    """
    m = Matrix.new(dtype, nrows, ncols, intermediate_result=True)
    if row_indices is None:
        row_indices = np.arange(nrows)
    if col_indices is None:
        col_indices = np.arange(ncols)
    if not hasattr(row_indices, '__len__'):
        raise TypeError(f"row_indices must be a tuple/list/array, not {type(row_indices)}")
    if not hasattr(col_indices, '__len__'):
        raise TypeError(f"col_indices must be a tuple/list/array, not {type(col_indices)}")
    # Build all combinations of indices
    ridx = np.repeat(row_indices, len(col_indices))
    cidx = np.tile(col_indices, len(row_indices))
    m.build(ridx, cidx, value, colwise=colwise)
    return m


def nvals(sp: SparseTensorBase):
    # Build and compile if needed
    key = ('nvals', *sp.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_nvals(sp)

    # Call the compiled function
    mem_out = ctypes.pointer(ctypes.c_long(0))
    arg_pointers = [sp._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return mem_out.contents.value


def _build_nvals(sp: SparseTensorBase):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rtt = sp.rtt.as_mlir_type()

            @func.FuncOp.from_py_func(rtt)
            def main(x):
                nvals = sparse_tensor.NumberOfEntriesOp(x)
                return arith.IndexCastOp(INT64.build_mlir_type(), nvals)

            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        return compile(module)


def dup(out_type: DType, sp: SparseTensorBase, intermediate: bool = True):
    if sp._obj is None:
        return sp.baseclass(out_type, sp.shape, intermediate_result=intermediate)

    if sp.ndims == 0:  # Scalar
        return Scalar.new(out_type, sp._obj)

    # Build and compile if needed
    key = ('dup', out_type, *sp.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_dup(out_type, sp)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [sp._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return sp.baseclass(out_type, sp.shape, mem_out, sp._sparsity,
                        sp.perceived_ordering, intermediate_result=intermediate)


def _build_dup(out_type: DType, sp: SparseTensorBase):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rank = sp.ndims
            index = ir.IndexType.get()
            dtype = sp.dtype.build_mlir_type()
            dtype_out = out_type.build_mlir_type()
            perm = ir.AffineMap.get_permutation(sp.permutation)
            perm_out = ir.AffineMap.get_permutation(range(rank))
            rtt = sp.rtt.as_mlir_type()
            rtt_out = sp.rtt.copy(dtype=out_type,
                                  ordering=sp.perceived_ordering).as_mlir_type()

            @func.FuncOp.from_py_func(rtt)
            def main(x):
                c = [arith.ConstantOp(index, i) for i in range(rank)]
                dims = [tensor.DimOp(x, c[i]).result for i in sp.permutation]
                out = bufferization.AllocTensorOp(rtt_out, dims, None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm, perm_out)]),
                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*rank)
                )
                block = generic_op.regions[0].blocks.append(dtype, dtype_out)
                with ir.InsertionPoint(block):
                    a, _ = block.arguments
                    result = cast(a, sp.dtype, out_type)
                    linalg.YieldOp([result])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def flip_layout(m: Union[Matrix, TransposedMatrix]):
    if m._obj is None:
        return m

    trans = type(m) is TransposedMatrix
    if trans:
        m = m._referenced_matrix

    # Build and compile if needed
    key = ('flip_layout', *m.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_flip_layout(m)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [m._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)

    flipped = Matrix(m.dtype, m.shape, mem_out, m._sparsity,
                     list(reversed(m._ordering)), intermediate_result=True)
    if trans:
        return TransposedMatrix.wrap(flipped)
    return flipped


def _build_flip_layout(m: Union[Matrix, TransposedMatrix]):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rtt = m.rtt.as_mlir_type()
            rev_order = tuple(reversed(m.rtt.ordering))
            rtt_out = m.rtt.copy(ordering=rev_order).as_mlir_type()

            @func.FuncOp.from_py_func(rtt)
            def main(x):
                return sparse_tensor.ConvertOp(rtt_out, x)
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def _build_scalar_binop(out_type: DType, op: BinaryOp, left: Scalar, right: Scalar):
    # Both scalars are non-empty
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            dtype_left = left.dtype.build_mlir_type()
            dtype_right = right.dtype.build_mlir_type()

            @func.FuncOp.from_py_func(dtype_left, dtype_right)
            def main(x, y):
                result = op(out_type, x, y)
                return result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def ewise_add(out_type: DType, op: BinaryOp, left: SparseTensorBase, right: SparseTensorBase):
    assert left.ndims == right.ndims
    assert left.dtype == right.dtype

    if left._obj is None:
        if right.dtype == out_type:
            return right
        return dup(out_type, right)
    if right._obj is None:
        if left.dtype == out_type:
            return left
        return dup(out_type, left)

    rank = left.ndims
    if rank == 0:  # Scalar
        key = ('scalar_binop', op.name, out_type, left.dtype, right.dtype)
        if key not in engine_cache:
            engine_cache[key] = _build_scalar_binop(out_type, op, left, right)
        mem_out = get_scalar_output_pointer(left.dtype)
        arg_pointers = [get_scalar_input_arg(left), get_scalar_input_arg(right), mem_out]
        engine_cache[key].invoke('main', *arg_pointers)
        return Scalar(out_type, (), out_type.np_type(mem_out.contents.value))

    # Build and compile if needed
    key = ('ewise_add', op.name, out_type, *left.get_loop_key(), *right.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_ewise_add(out_type, op, left, right)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [left._obj, right._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return left.baseclass(out_type, left.shape, mem_out,
                          determine_sparsity(left, right, union=True), left.perceived_ordering,
                          intermediate_result=True)


def _build_ewise_add(out_type: DType, op: BinaryOp, left: SparseTensorBase, right: SparseTensorBase):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rank = left.ndims
            index = ir.IndexType.get()
            dtype_left = left.dtype.build_mlir_type()
            dtype_right = right.dtype.build_mlir_type()
            dtype_out = out_type.build_mlir_type()
            perm_left = ir.AffineMap.get_permutation(left.permutation)
            perm_right = ir.AffineMap.get_permutation(right.permutation)
            perm_out = ir.AffineMap.get_permutation(range(rank))
            rtt_left = left.rtt.as_mlir_type()
            rtt_right = right.rtt.as_mlir_type()
            rtt_out = RankedTensorType(dtype=out_type,
                                       sparsity=determine_sparsity(left, right, union=True),
                                       ordering=left.perceived_ordering).as_mlir_type()

            @func.FuncOp.from_py_func(rtt_left, rtt_right)
            def main(x, y):
                c = [arith.ConstantOp(index, i) for i in range(rank)]
                dims = [tensor.DimOp(x, c[i]).result for i in left.permutation]
                out = bufferization.AllocTensorOp(rtt_out, dims, None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x, y],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm_left, perm_right, perm_out)]),
                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*rank)
                )
                block = generic_op.regions[0].blocks.append(dtype_left, dtype_right, dtype_out)
                with ir.InsertionPoint(block):
                    a, b, o = block.arguments
                    res = sparse_tensor.BinaryOp(dtype_out, a, b)
                    overlap = res.regions[0].blocks.append(dtype_left, dtype_right)
                    with ir.InsertionPoint(overlap):
                        arg0, arg1 = overlap.arguments
                        overlap_res = op(out_type, arg0, arg1)
                        sparse_tensor.YieldOp(result=overlap_res)
                    left_region = res.regions[1].blocks.append(dtype_left)
                    with ir.InsertionPoint(left_region):
                        arg0, = left_region.arguments
                        left_res = cast(arg0, left.dtype, out_type)
                        sparse_tensor.YieldOp(result=left_res)
                    right_region = res.regions[2].blocks.append(dtype_right)
                    with ir.InsertionPoint(right_region):
                        arg0, = right_region.arguments
                        right_res = cast(arg0, right.dtype, out_type)
                        sparse_tensor.YieldOp(result=right_res)
                    linalg.YieldOp([res])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def ewise_mult(out_type: DType, op: BinaryOp, left: SparseTensorBase, right: SparseTensorBase):
    assert left.ndims == right.ndims

    if left._obj is None or right._obj is None:
        return left.baseclass(out_type, left.shape)

    rank = left.ndims
    if rank == 0:  # Scalar
        key = ('scalar_binop', op.name, out_type, left.dtype, right.dtype)
        if key not in engine_cache:
            engine_cache[key] = _build_scalar_binop(out_type, op, left, right)
        mem_out = get_scalar_output_pointer(out_type)
        arg_pointers = [get_scalar_input_arg(left), get_scalar_input_arg(right), mem_out]
        engine_cache[key].invoke('main', *arg_pointers)
        return Scalar(out_type, (), out_type.np_type(mem_out.contents.value))

    # Build and compile if needed
    key = ('ewise_mult', op.name, out_type, *left.get_loop_key(), *right.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_ewise_mult(out_type, op, left, right)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [left._obj, right._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return left.baseclass(out_type, left.shape, mem_out,
                          determine_sparsity(left, right), left.perceived_ordering,
                          intermediate_result=True)


def _build_ewise_mult(out_type: DType, op: BinaryOp, left: SparseTensorBase, right: SparseTensorBase):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rank = left.ndims
            index = ir.IndexType.get()
            dtype_left = left.dtype.build_mlir_type()
            dtype_right = right.dtype.build_mlir_type()
            dtype_out = out_type.build_mlir_type()
            perm_left = ir.AffineMap.get_permutation(left.permutation)
            perm_right = ir.AffineMap.get_permutation(right.permutation)
            perm_out = ir.AffineMap.get_permutation(range(rank))
            rtt_left = left.rtt.as_mlir_type()
            rtt_right = right.rtt.as_mlir_type()
            rtt_out = RankedTensorType(dtype=out_type,
                                       sparsity=determine_sparsity(left, right),
                                       ordering=left.perceived_ordering).as_mlir_type()

            @func.FuncOp.from_py_func(rtt_left, rtt_right)
            def main(x, y):
                c = [arith.ConstantOp(index, i) for i in range(rank)]
                dims = [tensor.DimOp(x, c[i]).result for i in left.permutation]
                out = bufferization.AllocTensorOp(rtt_out, dims, None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x, y],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm_left, perm_right, perm_out)]),
                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*rank)
                )
                block = generic_op.regions[0].blocks.append(dtype_left, dtype_right, dtype_out)
                with ir.InsertionPoint(block):
                    a, b, o = block.arguments
                    res = sparse_tensor.BinaryOp(dtype_out, a, b)
                    overlap = res.regions[0].blocks.append(dtype_left, dtype_right)
                    with ir.InsertionPoint(overlap):
                        arg0, arg1 = overlap.arguments
                        overlap_res = op(out_type, arg0, arg1)
                        sparse_tensor.YieldOp(result=overlap_res)
                    linalg.YieldOp([res])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


# TODO: pass the mask to mxm
def mxm(out_type: DType, op: Semiring, left: Union[Matrix, TransposedMatrix], right: Union[Matrix, TransposedMatrix]):
    assert left.ndims == right.ndims == 2

    if left._obj is None or right._obj is None:
        return Matrix.new(out_type, left.shape[0], right.shape[1])

    # Build and compile if needed
    key = ('mxm', op.name, out_type, *left.get_loop_key(), *right.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_mxm(out_type, op, left, right)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [left._obj, right._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return Matrix(out_type, [left.shape[0], right.shape[1]], mem_out,
                  determine_sparsity(left, right), left.perceived_ordering, intermediate_result=True)


def _build_mxm(out_type: DType,
               op: Semiring,
               left: Union[Matrix, TransposedMatrix],
               right: Union[Matrix, TransposedMatrix]):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index = ir.IndexType.get()
            dtype_left = left.dtype.build_mlir_type()
            dtype_right = right.dtype.build_mlir_type()
            dtype_out = out_type.build_mlir_type()
            perm_left = ir.AffineMap.get(3, 0, left._permute([ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(2)]))
            perm_right = ir.AffineMap.get(3, 0, right._permute([ir.AffineDimExpr.get(2), ir.AffineDimExpr.get(1)]))
            perm_out = ir.AffineMap.get(3, 0, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
            rtt_left = left.rtt.as_mlir_type()
            rtt_right = right.rtt.as_mlir_type()
            rtt_out = RankedTensorType(dtype=out_type,
                                       sparsity=determine_sparsity(left, right),
                                       ordering=left.perceived_ordering).as_mlir_type()

            @func.FuncOp.from_py_func(rtt_left, rtt_right)
            def main(x, y):
                c = [arith.ConstantOp(index, i) for i in range(2)]
                nrows = tensor.DimOp(x, c[left.permutation[0]]).result
                ncols = tensor.DimOp(y, c[right.permutation[1]]).result
                out = bufferization.AllocTensorOp(rtt_out, [nrows, ncols], None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x, y],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm_left, perm_right, perm_out)]),
                    ir.ArrayAttr.get([
                        ir.Attribute.parse('#linalg.iterator_type<parallel>'),
                        ir.Attribute.parse('#linalg.iterator_type<parallel>'),
                        ir.Attribute.parse('#linalg.iterator_type<reduction>'),
                    ])
                )
                block = generic_op.regions[0].blocks.append(dtype_left, dtype_right, dtype_out)
                with ir.InsertionPoint(block):
                    a, b, o = block.arguments
                    bin_result = sparse_tensor.BinaryOp(dtype_out, a, b)
                    overlap = bin_result.regions[0].blocks.append(dtype_left, dtype_right)
                    with ir.InsertionPoint(overlap):
                        arg0, arg1 = overlap.arguments
                        overlap_res = op.binop(out_type, arg0, arg1)
                        sparse_tensor.YieldOp(result=overlap_res)
                    ident = op.monoid.identity(out_type)
                    red_result = sparse_tensor.ReduceOp(bin_result, o, ident)
                    reduce = red_result.regions[0].blocks.append(dtype_out, dtype_out)
                    with ir.InsertionPoint(reduce):
                        arg0, arg1 = reduce.arguments
                        reduce_res = op.monoid.binop(out_type, arg0, arg1)
                        sparse_tensor.YieldOp(result=reduce_res)
                    linalg.YieldOp([red_result])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


# TODO: pass the mask to mxv
def mxv(out_type: DType, op: Semiring, left: Union[Matrix, TransposedMatrix], right: Vector):
    assert left.ndims == 2
    assert right.ndims == 1

    if left._obj is None or right._obj is None:
        return Vector.new(out_type, left.shape[0])

    # Build and compile if needed
    key = ('mxv', op.name, out_type, *left.get_loop_key(), *right.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_mxv(out_type, op, left, right)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [left._obj, right._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return Vector(out_type, [left.shape[0]], mem_out,
                  right._sparsity, right.perceived_ordering, intermediate_result=True)


def _build_mxv(out_type: DType, op: Semiring, left: Union[Matrix, TransposedMatrix], right: Vector):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index = ir.IndexType.get()
            dtype_left = left.dtype.build_mlir_type()
            dtype_right = right.dtype.build_mlir_type()
            dtype_out = out_type.build_mlir_type()
            perm_left = ir.AffineMap.get(2, 0, left._permute([ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)]))
            perm_right = ir.AffineMap.get(2, 0, right._permute([ir.AffineDimExpr.get(1)]))
            perm_out = ir.AffineMap.get(2, 0, [ir.AffineDimExpr.get(0)])
            rtt_left = left.rtt.as_mlir_type()
            rtt_right = right.rtt.as_mlir_type()
            rtt_out = right.rtt.copy(dtype=out_type).as_mlir_type()

            @func.FuncOp.from_py_func(rtt_left, rtt_right)
            def main(x, y):
                c = [arith.ConstantOp(index, i) for i in range(2)]
                size = tensor.DimOp(x, c[left.permutation[0]]).result
                out = bufferization.AllocTensorOp(rtt_out, [size], None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x, y],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm_left, perm_right, perm_out)]),
                    ir.ArrayAttr.get([
                        ir.Attribute.parse('#linalg.iterator_type<parallel>'),
                        ir.Attribute.parse('#linalg.iterator_type<reduction>'),
                    ])
                )
                block = generic_op.regions[0].blocks.append(dtype_left, dtype_right, dtype_out)
                with ir.InsertionPoint(block):
                    a, b, o = block.arguments
                    bin_result = sparse_tensor.BinaryOp(dtype_out, a, b)
                    overlap = bin_result.regions[0].blocks.append(dtype_left, dtype_right)
                    with ir.InsertionPoint(overlap):
                        arg0, arg1 = overlap.arguments
                        overlap_res = op.binop(out_type, arg0, arg1)
                        sparse_tensor.YieldOp(result=overlap_res)
                    ident = op.monoid.identity(out_type)
                    red_result = sparse_tensor.ReduceOp(bin_result, o, ident)
                    reduce = red_result.regions[0].blocks.append(dtype_out, dtype_out)
                    with ir.InsertionPoint(reduce):
                        arg0, arg1 = reduce.arguments
                        reduce_res = op.monoid.binop(out_type, arg0, arg1)
                        sparse_tensor.YieldOp(result=reduce_res)
                    linalg.YieldOp([red_result])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


# TODO: pass the mask to vxm
def vxm(out_type: DType, op: Semiring, left: Vector, right: Union[Matrix, TransposedMatrix]):
    assert left.ndims == 1
    assert right.ndims == 2

    if left._obj is None or right._obj is None:
        return Vector.new(out_type, right.shape[1])

    # Build and compile if needed
    key = ('vxm', op.name, out_type, *left.get_loop_key(), *right.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_vxm(out_type, op, left, right)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [left._obj, right._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return Vector(out_type, [right.shape[1]], mem_out,
                  left._sparsity, left.perceived_ordering, intermediate_result=True)


def _build_vxm(out_type: DType, op: Semiring, left: Vector, right: Union[Matrix, TransposedMatrix]):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index = ir.IndexType.get()
            dtype_left = left.dtype.build_mlir_type()
            dtype_right = right.dtype.build_mlir_type()
            dtype_out = out_type.build_mlir_type()
            perm_left = ir.AffineMap.get(2, 0, left._permute([ir.AffineDimExpr.get(0)]))
            perm_right = ir.AffineMap.get(2, 0, right._permute([ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)]))
            perm_out = ir.AffineMap.get(2, 0, [ir.AffineDimExpr.get(1)])
            rtt_left = left.rtt.as_mlir_type()
            rtt_right = right.rtt.as_mlir_type()
            rtt_out = left.rtt.copy(dtype=out_type).as_mlir_type()

            @func.FuncOp.from_py_func(rtt_left, rtt_right)
            def main(x, y):
                c = [arith.ConstantOp(index, i) for i in range(2)]
                size = tensor.DimOp(y, c[right.permutation[1]]).result
                out = bufferization.AllocTensorOp(rtt_out, [size], None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x, y],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm_left, perm_right, perm_out)]),
                    ir.ArrayAttr.get([
                        ir.Attribute.parse('#linalg.iterator_type<reduction>'),
                        ir.Attribute.parse('#linalg.iterator_type<parallel>'),
                    ])
                )
                block = generic_op.regions[0].blocks.append(dtype_left, dtype_right, dtype_out)
                with ir.InsertionPoint(block):
                    a, b, o = block.arguments
                    bin_result = sparse_tensor.BinaryOp(dtype_out, a, b)
                    overlap = bin_result.regions[0].blocks.append(dtype_left, dtype_right)
                    with ir.InsertionPoint(overlap):
                        arg0, arg1 = overlap.arguments
                        overlap_res = op.binop(out_type, arg0, arg1)
                        sparse_tensor.YieldOp(result=overlap_res)
                    ident = op.monoid.identity(out_type)
                    red_result = sparse_tensor.ReduceOp(bin_result, o, ident)
                    reduce = red_result.regions[0].blocks.append(dtype_out, dtype_out)
                    with ir.InsertionPoint(reduce):
                        arg0, arg1 = reduce.arguments
                        reduce_res = op.monoid.binop(out_type, arg0, arg1)
                        sparse_tensor.YieldOp(result=reduce_res)
                    linalg.YieldOp([red_result])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def apply(out_type: DType, op: Union[UnaryOp, BinaryOp, IndexUnaryOp],
          sp: SparseTensorBase,
          left: Optional[Scalar] = None,
          right: Optional[Scalar] = None,
          thunk: Optional[Scalar] = None,
          inplace: bool = False):
    if sp._obj is None:
        return sp.baseclass(out_type, sp.shape)

    optype = type(op)
    rank = sp.ndims
    if rank == 0:  # Scalar
        if optype is UnaryOp:
            key = ('scalar_apply_unary', op.name, out_type, sp.dtype)
        elif optype is BinaryOp:
            if left is not None:
                key = ('scalar_apply_bind_first', op.name, out_type, sp.dtype, left._obj)
            else:
                key = ('scalar_apply_bind_second', op.name, out_type, sp.dtype, right._obj)
        else:
            raise GrbError("apply scalar not supported for IndexUnaryOp")

        if key not in engine_cache:
            engine_cache[key] = _build_scalar_apply(out_type, op, sp, left, right)
        mem_out = get_scalar_output_pointer(out_type)
        arg_pointers = [get_scalar_input_arg(sp), mem_out]
        engine_cache[key].invoke('main', *arg_pointers)
        return Scalar.new(out_type, mem_out.contents.value)

    # Build and compile if needed
    # Note that Scalars are included in the key because they are inlined in the compiled code
    if optype is UnaryOp:
        key = ('apply_unary', op.name, out_type, *sp.get_loop_key(), inplace)
    elif optype is BinaryOp:
        if left is not None:
            key = ('apply_bind_first', op.name, out_type, *sp.get_loop_key(), left._obj, inplace)
        else:
            key = ('apply_bind_second', op.name, out_type, *sp.get_loop_key(), right._obj, inplace)
    else:
        if inplace:
            raise TypeError("apply inplace not supported for IndexUnaryOp")
        key = ('apply_indexunary', op.name, out_type, *sp.get_loop_key(), thunk._obj)
    if key not in engine_cache:
        if inplace:
            engine_cache[key] = _build_apply_inplace(op, sp, left, right)
        else:
            engine_cache[key] = _build_apply(out_type, op, sp, left, right, thunk)

    # Call the compiled function
    if inplace:
        engine_cache[key].invoke('main', sp._obj)
        return sp
    mem_out = get_sparse_output_pointer()
    arg_pointers = [sp._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return sp.baseclass(out_type, sp.shape, mem_out,
                        sp._sparsity, sp.perceived_ordering, intermediate_result=True)


def _build_scalar_apply(out_type: DType,
                        op: Union[UnaryOp, BinaryOp],
                        sp: SparseTensorBase,
                        left: Optional[Scalar],
                        right: Optional[Scalar]):
    optype = type(op)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            dtype = sp.dtype.build_mlir_type()

            @func.FuncOp.from_py_func(dtype)
            def main(x):
                if optype is BinaryOp:
                    if left is not None:
                        left_val = arith.ConstantOp(left.dtype.build_mlir_type(), left.extract_element())
                        result = op(out_type, left_val, x)
                    else:
                        right_val = arith.ConstantOp(right.dtype.build_mlir_type(), right.extract_element())
                        result = op(out_type, x, right_val)
                else:
                    result = op(out_type, x)
                return result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def _build_apply(out_type: DType,
                 op: Union[UnaryOp, BinaryOp, IndexUnaryOp],
                 sp: SparseTensorBase,
                 left: Optional[Scalar],
                 right: Optional[Scalar],
                 thunk: Optional[Scalar]):
    optype = type(op)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rank = sp.ndims
            index = ir.IndexType.get()
            i64 = ir.IntegerType.get_signless(64)
            dtype = sp.dtype.build_mlir_type()
            dtype_out = out_type.build_mlir_type()
            perm = ir.AffineMap.get_permutation(sp.permutation)
            perm_out = ir.AffineMap.get_permutation(range(rank))
            rtt = sp.rtt.as_mlir_type()
            rtt_out = sp.rtt.copy(dtype=out_type, ordering=sp.perceived_ordering).as_mlir_type()

            @func.FuncOp.from_py_func(rtt)
            def main(x):
                c = [arith.ConstantOp(index, i) for i in range(rank)]
                dims = [tensor.DimOp(x, c[i]).result for i in sp.permutation]
                out = bufferization.AllocTensorOp(rtt_out, dims, None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm, perm_out)]),
                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*rank)
                )
                block = generic_op.regions[0].blocks.append(dtype, dtype_out)
                with ir.InsertionPoint(block):
                    a, o = block.arguments
                    if optype is IndexUnaryOp:
                        rowidx = linalg.IndexOp(ir.IntegerAttr.get(i64, 0))
                        if rank == 2:
                            colidx = linalg.IndexOp(ir.IntegerAttr.get(i64, 1))
                        else:
                            colidx = arith.ConstantOp(index, 0)
                    res = sparse_tensor.UnaryOp(dtype_out, a)
                    present = res.regions[0].blocks.append(dtype)
                    with ir.InsertionPoint(present):
                        arg0,  = present.arguments
                        if optype is IndexUnaryOp:
                            if op.thunk_as_index:
                                thunk_val = arith.ConstantOp(index, thunk.extract_element())
                            else:
                                thunk_val = arith.ConstantOp(thunk.dtype.build_mlir_type(), thunk.extract_element())
                            val = op(out_type, arg0, rowidx, colidx, thunk_val)
                        elif optype is BinaryOp:
                            if left is not None:
                                left_val = arith.ConstantOp(left.dtype.build_mlir_type(), left.extract_element())
                                val = op(out_type, left_val, arg0)
                            else:
                                right_val = arith.ConstantOp(right.dtype.build_mlir_type(), right.extract_element())
                                val = op(out_type, arg0, right_val)
                        else:
                            val = op(out_type, arg0)
                        sparse_tensor.YieldOp(result=val)
                    linalg.YieldOp([res])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def _build_apply_inplace(op: Union[UnaryOp, BinaryOp],
                         sp: SparseTensorBase,
                         left: Optional[Scalar],
                         right: Optional[Scalar]):
    optype = type(op)
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index = ir.IndexType.get()
            dtype = sp.dtype.build_mlir_type()
            rtt = sp.rtt.as_mlir_type()
            memref_dtype = ir.MemRefType.get([ir.MemRefType.get_dynamic_size()], dtype)

            @func.FuncOp.from_py_func(rtt)
            def main(x):
                c0 = arith.ConstantOp(index, 0)
                c1 = arith.ConstantOp(index, 1)
                c_len = sparse_tensor.NumberOfEntriesOp(x)
                vals = sparse_tensor.ToValuesOp(memref_dtype, x)
                for_loop = scf.ForOp(c0, c_len, c1)
                with ir.InsertionPoint(for_loop.body):
                    x = for_loop.induction_variable
                    val = memref.LoadOp(vals, [x])
                    if optype is BinaryOp:
                        if left is not None:
                            left_val = arith.ConstantOp(left.dtype.build_mlir_type(), left.extract_element())
                            result = op(sp.dtype, left_val, val)
                        else:
                            right_val = arith.ConstantOp(right.dtype.build_mlir_type(), right.extract_element())
                            result = op(sp.dtype, val, right_val)
                    else:
                        result = op(sp.dtype, val)
                    memref.StoreOp(result, vals, [x])
                    scf.YieldOp([])
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def select(out_type: DType, op: SelectOp, sp: SparseTensor, thunk: Scalar):
    # Handle case of empty tensor
    if sp._obj is None:
        return sp.__class__(out_type, sp.shape)

    rank = sp.ndims
    if rank == 0:  # Scalar
        key = ('scalar_select', op.name, sp.dtype, thunk._obj)
        if key not in engine_cache:
            engine_cache[key] = _build_scalar_select(op, sp, thunk)
        mem_out = get_scalar_output_pointer(sp.dtype)
        arg_pointers = [get_scalar_input_arg(sp), mem_out]
        engine_cache[key].invoke('main', *arg_pointers)
        # Invocation returns True/False for whether to keep value
        if mem_out.contents.value:
            return Scalar.new(out_type, sp._obj)
        else:
            return Scalar.new(out_type)

    # Build and compile if needed
    # Note that thunk is included in the key because it is inlined in the compiled code
    key = ('select', op.name, *sp.get_loop_key(), thunk._obj)
    if key not in engine_cache:
        engine_cache[key] = _build_select(op, sp, thunk)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [sp._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    res = sp.baseclass(sp.dtype, sp.shape, mem_out,
                       sp._sparsity, sp.perceived_ordering, intermediate_result=True)

    # _build_select cannot change output dtype; handle that now
    if out_type != sp.dtype:
        res = dup(out_type, res, intermediate=True)

    return res


def _build_scalar_select(op: SelectOp, sp: SparseTensorBase, thunk: Scalar):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index = ir.IndexType.get()
            dtype = sp.dtype.build_mlir_type()

            @func.FuncOp.from_py_func(dtype)
            def main(x):
                c0 = arith.ConstantOp(index, 0)
                if op.thunk_as_index:
                    thunk_val = arith.ConstantOp(index, thunk.extract_element())
                else:
                    thunk_val = arith.ConstantOp(thunk.dtype.build_mlir_type(), thunk.extract_element())
                cmp = op(BOOL, x, c0, c0, thunk_val)
                return cmp
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def _build_select(op: SelectOp, sp: SparseTensorBase, thunk: Scalar):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rank = sp.ndims
            index = ir.IndexType.get()
            i64 = ir.IntegerType.get_signless(64)
            dtype = sp.dtype.build_mlir_type()
            perm = ir.AffineMap.get_permutation(sp.permutation)
            perm_out = ir.AffineMap.get_permutation(range(rank))
            rtt = sp.rtt.as_mlir_type()
            rtt_out = sp.rtt.copy(ordering=sp.perceived_ordering).as_mlir_type()

            @func.FuncOp.from_py_func(rtt)
            def main(x):
                c = [arith.ConstantOp(index, i) for i in range(rank)]
                dims = [tensor.DimOp(x, c[i]).result for i in sp.permutation]
                out = bufferization.AllocTensorOp(rtt_out, dims, None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm, perm_out)]),
                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')]*rank)
                )
                block = generic_op.regions[0].blocks.append(dtype, dtype)
                with ir.InsertionPoint(block):
                    a, o = block.arguments
                    rowidx = linalg.IndexOp(ir.IntegerAttr.get(i64, 0))
                    if rank == 2:
                        colidx = linalg.IndexOp(ir.IntegerAttr.get(i64, 1))
                    else:
                        colidx = arith.ConstantOp(index, 0)
                    res = sparse_tensor.SelectOp(a)
                    region = res.regions[0].blocks.append(dtype)
                    with ir.InsertionPoint(region):
                        arg0, = region.arguments
                        if op.thunk_as_index:
                            thunk_val = arith.ConstantOp(index, thunk.extract_element())
                        else:
                            thunk_val = arith.ConstantOp(thunk.dtype.build_mlir_type(), thunk.extract_element())
                        cmp = op(BOOL, arg0, rowidx, colidx, thunk_val)
                        sparse_tensor.YieldOp(result=cmp)
                    linalg.YieldOp([res])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def reduce_to_vector(out_type: DType, op: Monoid, mat: Union[Matrix, TransposedMatrix]):
    if mat._obj is None:
        return Vector.new(out_type, mat.shape[0])

    # Build and compile if needed
    key = ('reduce_to_vector', op.name, *mat.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_reduce_to_vector(op, mat)

    # Call the compiled function
    mem_out = get_sparse_output_pointer()
    arg_pointers = [mat._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    res = Vector(mat.dtype, [mat.shape[0]], mem_out,
                 [DimLevelType.compressed], [0], intermediate_result=True)

    # _build_reduce_to_vector cannot change output dtype; handle that now
    if out_type != mat.dtype:
        res = dup(out_type, res, intermediate=True)

    return res


def _build_reduce_to_vector(op: Monoid, mat: Union[Matrix, TransposedMatrix]):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            index = ir.IndexType.get()
            dtype = mat.dtype.build_mlir_type()
            perm = ir.AffineMap.get_permutation(mat.permutation)
            perm_out = ir.AffineMap.get(2, 0, [ir.AffineDimExpr.get(0)])
            rtt = mat.rtt.as_mlir_type()
            rtt_out = mat.rtt.copy(ordering=[0], sparsity=[DimLevelType.compressed]).as_mlir_type()

            @func.FuncOp.from_py_func(rtt)
            def main(x):
                c = [arith.ConstantOp(index, i) for i in range(2)]
                size = tensor.DimOp(x, c[mat.permutation[0]]).result
                out = bufferization.AllocTensorOp(rtt_out, [size], None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm, perm_out)]),
                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>'),
                                      ir.Attribute.parse('#linalg.iterator_type<reduction>')])
                )
                block = generic_op.regions[0].blocks.append(dtype, dtype)
                with ir.InsertionPoint(block):
                    a, o = block.arguments
                    ident = op.identity(mat.dtype)
                    res = sparse_tensor.ReduceOp(o, a, ident)
                    region = res.regions[0].blocks.append(dtype, dtype)
                    with ir.InsertionPoint(region):
                        arg0, arg1 = region.arguments
                        reduce_res = op.binop(mat.dtype, arg0, arg1)
                        sparse_tensor.YieldOp(result=reduce_res)
                    linalg.YieldOp([res])
                return generic_op.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def reduce_to_scalar(out_type: DType, op: Monoid, sp: SparseTensorBase):
    if sp._obj is None:
        return Scalar.new(out_type)

    # Build and compile if needed
    key = ('reduce_to_scalar', op.name, out_type, *sp.get_loop_key())
    if key not in engine_cache:
        engine_cache[key] = _build_reduce_to_scalar(out_type, op, sp)

    # Call the compiled function
    mem_out = get_scalar_output_pointer(out_type)
    arg_pointers = [sp._obj, mem_out]
    engine_cache[key].invoke('main', *arg_pointers)
    return Scalar.new(out_type, mem_out.contents.value)


def _build_reduce_to_scalar(out_type: DType, op: Monoid, sp: SparseTensorBase):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            rank = sp.ndims
            dtype = sp.dtype.build_mlir_type()
            perm = ir.AffineMap.get_permutation(sp.permutation)
            perm_out = ir.AffineMap.get(rank, 0, [])
            rtt = sp.rtt.as_mlir_type()
            rtt_out = ir.RankedTensorType.get([], dtype)

            @func.FuncOp.from_py_func(rtt)
            def main(x):
                out = bufferization.AllocTensorOp(rtt_out, [], None, None, False)
                generic_op = linalg.GenericOp(
                    [rtt_out],
                    [x],
                    [out],
                    ir.ArrayAttr.get([ir.AffineMapAttr.get(p) for p in (perm, perm_out)]),
                    ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<reduction>')]*rank)
                )
                block = generic_op.regions[0].blocks.append(dtype, dtype)
                with ir.InsertionPoint(block):
                    a, o = block.arguments
                    ident = op.identity(sp.dtype)
                    res = sparse_tensor.ReduceOp(o, a, ident)
                    region = res.regions[0].blocks.append(dtype, dtype)
                    with ir.InsertionPoint(region):
                        arg0, arg1 = region.arguments
                        reduce_res = op.binop(sp.dtype, arg0, arg1)
                        sparse_tensor.YieldOp(result=reduce_res)
                    linalg.YieldOp([res])
                s = tensor.ExtractOp(generic_op, [])
                s = cast(s, sp.dtype, out_type)
                return s.result
            main.func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        return compile(module)


def extract(out_type: DType, tensor: SparseTensorBase, row_indices, col_indices, row_size, col_size):
    # There may be a way to do this in MLIR, but for now we use numpy
    if tensor.ndims == 1:
        # Vector
        assert col_indices is None
        assert col_size is None

        if row_indices is None:  # None indicates GrB_ALL
            return dup(out_type, tensor)

        idx, vals = tensor.extract_tuples()
        if out_type != tensor.dtype:
            vals = vals.astype(out_type.np_type)
        pick_list = np.array(row_indices, dtype=np.uint64)
        idx, vals = pick_and_renumber_indices(pick_list, idx, vals)
        v = Vector.new(out_type, row_size)
        v.build(idx, vals)
        return v

    # Matrix
    if row_indices is None and col_indices is None:
        return dup(out_type, tensor)

    rowidx, colidx, vals = tensor.extract_tuples()
    if out_type != tensor.dtype:
        vals = vals.astype(out_type.np_type)
    if row_indices is not None:
        if type(row_indices) is int:
            pick_list = np.array([row_indices], dtype=np.uint64)
        else:
            pick_list = np.array(row_indices, dtype=np.uint64)
        rowidx, colidx, vals = pick_and_renumber_indices(pick_list, rowidx, colidx, vals)
    if col_indices is not None:
        if type(col_indices) is int:
            pick_list = np.array([col_indices], dtype=np.uint64)
        else:
            pick_list = np.array(col_indices, dtype=np.uint64)
        colidx, rowidx, vals = pick_and_renumber_indices(pick_list, colidx, rowidx, vals)
    if type(row_indices) is int:
        # Extract row as Vector
        assert np.all(rowidx == 0)
        v = Vector.new(out_type, col_size)
        v.build(colidx, vals)
        return v
    if type(col_indices) is int:
        # Extract col as Vector
        assert np.all(colidx == 0)
        v = Vector.new(out_type, row_size)
        v.build(rowidx, vals)
        return v
    m = Matrix.new(out_type, row_size, col_size)
    m.build(rowidx, colidx, vals)
    return m


def assign(out_type: DType, tensor: SparseTensorBase, row_indices, col_indices, row_size, col_size=None):
    # There may be a way to do this in MLIR, but for now we use numpy
    if tensor.ndims == 1:
        # Vector input
        if row_indices is None and col_size is None:
            # Vector output with GrB_ALL
            return dup(out_type, tensor)

        idx, vals = tensor.extract_tuples()
        if out_type != tensor.dtype:
            vals = vals.astype(out_type.np_type)

        if col_size is None:
            # Vector output
            v = Vector.new(out_type, row_size)
            # Map idx to output indices
            idx = np.array(row_indices, dtype=np.uint64)[idx]
            v.build(idx, vals, sparsity=tensor._sparsity)
            return v
        # Assign Vector as row or column of Matrix
        m = Matrix.new(out_type, row_size, col_size)
        if type(row_indices) is int:
            # Map idx to output cols
            colidx = idx if col_indices is None else np.array(col_indices, dtype=np.uint64)[idx]
            m.build([row_indices]*len(vals), colidx, vals, sparsity=["compressed", "compressed"])
        if type(col_indices) is int:
            # Map idx to output rows
            rowidx = idx if row_indices is None else np.array(row_indices, dtype=np.uint64)[idx]
            m.build(rowidx, [col_indices]*len(vals), vals, sparsity=["compressed", "compressed"])
        return m

    # Matrix input
    if row_indices is None and col_indices is None:
        return dup(out_type, tensor)

    rowidx, colidx, vals = tensor.extract_tuples()
    if out_type != tensor.dtype:
        vals = vals.astype(out_type.np_type)

    # Map indices to output
    if row_indices is not None:
        rowidx = np.array(row_indices, dtype=np.uint64)[rowidx]
    if col_indices is not None:
        colidx = np.array(col_indices, dtype=np.uint64)[colidx]
    m = Matrix.new(out_type, row_size, col_size)
    m.build(rowidx, colidx, vals, sparsity=["compressed", "compressed"])
    return m
