import numpy as np
from typing import Optional, Union
from .tensor import SparseObject, SparseTensor, SparseTensorBase, Matrix, Vector, Scalar, TransposedMatrix
from .operators import UnaryOp, BinaryOp, SelectOp, IndexUnaryOp, Monoid, Semiring
from . descriptor import Descriptor, NULL as NULL_DESC
from .exceptions import (
    GrbError, GrbNullPointer, GrbInvalidValue, GrbInvalidIndex, GrbDomainMismatch,
    GrbDimensionMismatch, GrbOutputNotEmpty, GrbIndexOutOfBounds, GrbEmptyObject
)
from . import implementations as impl
from .utils import ensure_scalar_of_type, ensure_unique
from .types import BOOL, INT64
from .operators import BinaryOp, SelectOp


__all__ = ["transpose", "ewise_add", "ewise_mult", "mxm", "apply", "select",
           "reduce_to_vector", "reduce_to_scalar", "extract", "assign"]


# TODO: at some point, it may make sense to make this logic embeddable within
#       the MLIR-generating code; that could get complicated fast, so for now
#       this will be kept separate
def update(output: SparseObject,
           tensor: SparseObject,
           mask: Optional[SparseTensor] = None,
           accum: Optional[BinaryOp] = None,
           desc: Descriptor = NULL_DESC,
           *,
           row_indices: Optional[list[int]] = None,
           col_indices: Optional[list[int]] = None):
    """
    This function assumes that if a mask is supplied, it has already been applied to `tensor`

    There are six possible results based on presence of mask, accum, replace:

    Mask | Accum | Replace | Result
    -----|-------|---------|-------
      Y  |   Y   |    Y    | Mask the output, then perform eWiseAdd using accum
      Y  |   Y   |    N    | Perform eWiseAdd using accum
      N  |   Y   |    ?    | Perform eWiseAdd using accum
      Y  |   N   |    Y    | Set input as output
      Y  |   N   |    N    | Apply inverted mask to output, then perform eWiseAdd using any op
      N  |   N   |    ?    | Set input as output

    When row and column indices are provided (for assign), the behavior becomes:

    Mask | Accum | Replace | Result
    -----|-------|---------|-------
      ?  |   Y   |    ?    | < Same as above >
      Y  |   N   |    Y    | Mask the output, drop indices in output, then perform eWiseAdd
      Y  |   N   |    N    | Select indices in mask, apply inverted mask to output, then perform eWiseAdd
      N  |   N   |    ?    | Drop indices in output, then perform eWiseAdd

    """
    if output.shape != tensor.shape:
        raise ValueError(f"shape mismatch in update: {output.shape} != {tensor.shape}")

    if isinstance(output, Scalar):
        if mask is not None:
            raise ValueError("mask not allowed for Scalar update")
        if accum is None or output._obj is None:
            output.set_element(tensor.extract_element())
        else:
            output._obj = impl.ewise_add(output.dtype, accum, output, tensor)._obj
        return

    if not isinstance(output, SparseTensor):
        raise TypeError(f"output must be Scalar, Vector or Matrix, not {type(output)}")

    if output._obj is None:
        # accum, mask, replace are meaningless if output is empty
        result = tensor
    elif accum is None:
        # Build inverted descriptor; used below
        desc_inverted = Descriptor(mask_complement=not desc.mask_complement,
                                   mask_structure=desc.mask_structure)
        if row_indices is None and col_indices is None:
            if mask is None or desc.replace:  # mask=N, accum=N, replace=? -or- mask=Y, accum=N, replace=Y
                result = tensor
            else:  # mask=Y, accum=N, replace=N
                # Apply inverted mask, then eWiseAdd
                output._replace(impl.select_by_mask(output, mask, desc_inverted))
                result = impl.ewise_add(output.dtype, BinaryOp.oneb, output, tensor)
        else:
            if mask is None:  # mask=N, accum=N, replace=?, w/ indices
                # Drop indices in output, then eWiseAdd
                output._replace(impl.select_by_indices(output, row_indices, col_indices, complement=True))
            elif desc.replace:  # mask=Y, accum=N, replace=Y, w/ indices
                # Apply mask to output, drop row/col indices from output, then eWiseAdd
                output._replace(impl.select_by_mask(output, mask, desc))
                output._replace(impl.select_by_indices(output, row_indices, col_indices, complement=True))
            else:  # mask=Y, accum=N, replace=N, w/ indices
                if desc.mask_complement:
                    # Need to select the row/col indices that aren't in the mask,
                    #   apply it complemented to the output, then eWiseAdd
                    # However, you can't select elements that aren't there, so we
                    # use `build_structural_mask_from_indices` to build all possible assigned indices
                    # and filter those using the mask.
                    if output.ndims == 1:
                        all_indices = impl.build_iso_vector_from_indices(BOOL, *output.shape, row_indices)
                    else:
                        all_indices = impl.build_iso_matrix_from_indices(BOOL, *output.shape, row_indices, col_indices)
                    new_mask = impl.select_by_mask(all_indices, mask, desc)
                    output._replace(impl.select_by_mask(output, new_mask, desc))
                else:
                    # Select the row/col indices in the mask, apply it inverted to the output, then eWiseAdd
                    new_mask = impl.select_by_indices(mask, row_indices, col_indices)
                    output._replace(impl.select_by_mask(output, new_mask, desc_inverted))
            result = impl.ewise_add(output.dtype, BinaryOp.oneb, output, tensor)
    elif mask is None or not desc.replace:
        # eWiseAdd using accum
        result = impl.ewise_add(output.dtype, accum, output, tensor)
    else:
        # Mask the output, then perform eWiseAdd using accum
        output._replace(impl.select_by_mask(output, mask, desc))
        result = impl.ewise_add(output.dtype, accum, output, tensor)

    if result is output:
        # This can happen if empty tensors are used as input
        return output

    # If not an intermediate result or wrong dtype, make a copy
    if not result._intermediate_result or result.dtype != output.dtype:
        result = impl.dup(output.dtype, result)

    output._replace(result)


def transpose(out: Matrix,
              tensor: Matrix,
              *,
              mask: Optional[SparseTensor] = None,
              accum: Optional[BinaryOp] = None,
              desc: Descriptor = NULL_DESC):
    # Apply descriptor transpose
    if tensor.ndims != 2:
        raise TypeError(f"transpose requires Matrix, not {type(tensor)}")
    if desc.transpose0:
        tensor = TransposedMatrix.wrap(tensor)

    # Compare shapes
    expected_out_shape = (tensor.shape[1], tensor.shape[0])
    if out.shape != expected_out_shape:
        raise GrbDimensionMismatch(f"output shape mismatch: {out.shape} != {expected_out_shape}")

    result = TransposedMatrix.wrap(tensor)

    if mask is not None:
        result = impl.select_by_mask(result, mask, desc)

    update(out, result, mask, accum, desc)


def ewise_add(out: SparseTensor,
              op: BinaryOp,
              left: SparseTensor,
              right: SparseTensor,
              *,
              mask: Optional[SparseTensor] = None,
              accum: Optional[BinaryOp] = None,
              desc: Descriptor = NULL_DESC):
    # Normalize op
    if type(op) is Semiring:
        op = op.monoid
    if type(op) is Monoid:
        op = op.binop
    if type(op) is not BinaryOp:
        raise TypeError(f"op must be BinaryOp, Monoid, or Semiring")

    # Apply transposes
    if desc.transpose0 and left.ndims == 2:
        left = TransposedMatrix.wrap(left)
    if desc.transpose1 and right.ndims == 2:
        right = TransposedMatrix.wrap(right)

    # Compare shapes
    if left.shape != right.shape:
        raise GrbDimensionMismatch(f"inputs shape mismatch: {left.shape} != {right.shape}")
    if out.shape != left.shape:
        raise GrbDimensionMismatch(f"output shape mismatch: {out.shape} != {left.shape}")

    if mask is not None:
        left = impl.select_by_mask(left, mask, desc)
        right = impl.select_by_mask(right, mask, desc)

    result = impl.ewise_add(out.dtype, op, left, right)
    update(out, result, mask, accum, desc)


def ewise_mult(out: SparseTensor,
               op: BinaryOp,
               left: SparseTensor,
               right: SparseTensor,
               *,
               mask: Optional[SparseTensor] = None,
               accum: Optional[BinaryOp] = None,
               desc: Descriptor = NULL_DESC):
    # Normalize op
    if type(op) is not BinaryOp:
        if hasattr(op, 'binop'):
            op = op.binop
        else:
            raise TypeError(f"op must be BinaryOp, Monoid, or Semiring")

    # Apply transposes
    if desc.transpose0 and left.ndims == 2:
        left = TransposedMatrix.wrap(left)
    if desc.transpose1 and right.ndims == 2:
        right = TransposedMatrix.wrap(right)

    # Compare shapes
    if left.shape != right.shape:
        raise GrbDimensionMismatch(f"inputs shape mismatch: {left.shape} != {right.shape}")
    if out.shape != left.shape:
        raise GrbDimensionMismatch(f"output shape mismatch: {out.shape} != {left.shape}")

    if mask is not None:
        # Only need to apply mask to one of the inputs
        left = impl.select_by_mask(left, mask, desc)

    result = impl.ewise_mult(out.dtype, op, left, right)
    update(out, result, mask, accum, desc)


def mxm(out: Matrix,
        op: Semiring,
        left: Matrix,
        right: Matrix,
        *,
        mask: Optional[SparseTensor] = None,
        accum: Optional[BinaryOp] = None,
        desc: Descriptor = NULL_DESC):
    # Verify op
    if type(op) is not Semiring:
        raise TypeError(f"op must be Semiring, not {type(op)}")

    # Apply transposes
    if left.ndims != right.ndims != 2:
        raise GrbDimensionMismatch("mxm requires rank 2 tensors")
    if desc.transpose0:
        left = TransposedMatrix.wrap(left)
    if desc.transpose1:
        right = TransposedMatrix.wrap(right)

    # Compare shapes
    if left.shape[1] != right.shape[0]:
        raise GrbDimensionMismatch("incompatible input shapes for mxm")
    expected_out_shape = (left.shape[0], right.shape[1])
    if out.shape != expected_out_shape:
        raise GrbDimensionMismatch(f"output shape mismatch: {out.shape} != {expected_out_shape}")

    # Check for compatible iteration schemes
    #  - rowwise x rowwise => rowwise expanded access pattern
    #  - rowwise x colwise => rowwise lex insert
    #  - colwise x colwise => colwise expanded access pattern
    #  - colwise x rowwise => <illegal>
    if left.is_colwise() and right.is_rowwise():
        # Need to flip one of the matrices to make iteration valid
        right = impl.flip_layout(right)

    # TODO: apply the mask during the computation, not at the end
    result = impl.mxm(out.dtype, op, left, right)
    if mask is not None:
        result = impl.select_by_mask(result, mask, desc)
    update(out, result, mask, accum, desc)


def mxv(out: Vector,
        op: Semiring,
        left: Matrix,
        right: Vector,
        *,
        mask: Optional[SparseTensor] = None,
        accum: Optional[BinaryOp] = None,
        desc: Descriptor = NULL_DESC):
    # Verify op
    if type(op) is not Semiring:
        raise TypeError(f"op must be Semiring, not {type(op)}")

    # Apply transpose
    if left.ndims != 2:
        raise GrbDimensionMismatch("mxv requires matrix as first input")
    if desc.transpose0:
        left = TransposedMatrix.wrap(left)

    # Compare shapes
    if right.ndims != 1:
        raise GrbDimensionMismatch("mxv requires vector as second input")
    if left.shape[1] != right.shape[0]:
        raise GrbDimensionMismatch("incompatible input shapes for mxv")
    if out.shape != (left.shape[0],):
        raise GrbDimensionMismatch(f"output size should be {left.shape[0]} not {out.shape[0]}")

    # TODO: apply the mask during the computation, not at the end
    result = impl.mxv(out.dtype, op, left, right)
    if mask is not None:
        result = impl.select_by_mask(result, mask, desc)
    update(out, result, mask, accum, desc)


def vxm(out: Vector,
        op: Semiring,
        left: Vector,
        right: Matrix,
        *,
        mask: Optional[SparseTensor] = None,
        accum: Optional[BinaryOp] = None,
        desc: Descriptor = NULL_DESC):
    # Verify op
    if type(op) is not Semiring:
        raise TypeError(f"op must be Semiring, not {type(op)}")

    # Apply transpose
    if right.ndims != 2:
        raise GrbDimensionMismatch("vxm requires matrix as second input")
    if desc.transpose1:
        right = TransposedMatrix.wrap(right)

    # Compare shapes
    if left.ndims != 1:
        raise GrbDimensionMismatch("vxm requires vector as first input")
    if left.shape[0] != right.shape[0]:
        raise GrbDimensionMismatch("incompatible input shapes for vxm")
    if out.shape != (right.shape[1],):
        raise GrbDimensionMismatch(f"output size should be {right.shape[1]} not {out.shape[0]}")

    # TODO: apply the mask during the computation, not at the end
    result = impl.vxm(out.dtype, op, left, right)
    if mask is not None:
        result = impl.select_by_mask(result, mask, desc)
    update(out, result, mask, accum, desc)


def apply(out: SparseTensor,
          op: Union[UnaryOp, BinaryOp, IndexUnaryOp],
          tensor: SparseTensor,
          *,
          left: Optional[Scalar] = None,
          right: Optional[Scalar] = None,
          thunk: Optional[Scalar] = None,
          mask: Optional[SparseTensor] = None,
          accum: Optional[BinaryOp] = None,
          desc: Descriptor = NULL_DESC):
    # Validate op and required scalars
    optype = type(op)
    if optype is UnaryOp:
        if thunk is not None or left is not None or right is not None:
            raise TypeError("UnaryOp does not accept thunk, left, or right")
    elif optype is BinaryOp:
        if thunk is not None:
            raise TypeError("BinaryOp accepts left or thing, not thunk")
        if left is None and right is None:
            raise TypeError("BinaryOp requires either left or right")
        if left is not None and right is not None:
            raise TypeError("Cannot provide both left and right")
        if left is not None:
            left = ensure_scalar_of_type(left, tensor.dtype)
        else:
            right = ensure_scalar_of_type(right, tensor.dtype)
    elif optype is IndexUnaryOp:
        if left is not None or right is not None:
            raise TypeError("IndexUnaryOp accepts thunk, not left or right")
        thunk_dtype = INT64 if op.thunk_as_index else tensor.dtype
        thunk = ensure_scalar_of_type(thunk, thunk_dtype)
    else:
        raise TypeError(f"op must be UnaryOp, BinaryOp, or IndexUnaryOp, not {type(op)}")

    # Apply transpose
    if desc.transpose0 and tensor.ndims == 2:
        tensor = TransposedMatrix.wrap(tensor)

    # Compare shapes
    if out.shape != tensor.shape:
        raise GrbDimensionMismatch(f"output shape must match input shape: {out.shape} != {tensor.shape}")

    if mask is not None:
        tensor = impl.select_by_mask(tensor, mask, desc)

    # Check for inplace apply (out == tensor, Unary/Binary, no masks, no accum, etc)
    if (
            out is tensor
            and optype is not IndexUnaryOp
            and mask is None
            and accum is None
            and desc is NULL_DESC
            and not tensor._intermediate_result
    ):
        impl.apply(out.dtype, op, tensor, left, right, None, inplace=True)
    else:
        result = impl.apply(out.dtype, op, tensor, left, right, thunk)
        update(out, result, mask, accum, desc)


def select(out: SparseTensor,
           op: SelectOp,
           tensor: SparseTensor,
           thunk: Scalar,
           *,
           mask: Optional[SparseTensor] = None,
           accum: Optional[BinaryOp] = None,
           desc: Descriptor = NULL_DESC):
    # Verify op
    if type(op) is not SelectOp:
        raise TypeError(f"op must be SelectOp, not {type(op)}")

    # Verify dtypes
    thunk_dtype = INT64 if op.thunk_as_index else tensor.dtype
    thunk = ensure_scalar_of_type(thunk, thunk_dtype)

    # Apply transpose
    if desc.transpose0 and tensor.ndims == 2:
        tensor = TransposedMatrix.wrap(tensor)

    # Compare shapes
    if out.shape != tensor.shape:
        raise GrbDimensionMismatch(f"output shape must match input shape: {out.shape} != {tensor.shape}")

    if mask is not None:
        tensor = impl.select_by_mask(tensor, mask, desc)

    result = impl.select(out.dtype, op, tensor, thunk)
    update(out, result, mask, accum, desc)


def reduce_to_vector(out: Vector,
                     op: Monoid,
                     tensor: Matrix,
                     *,
                     mask: Optional[Vector] = None,
                     accum: Optional[BinaryOp] = None,
                     desc: Descriptor = NULL_DESC):
    # Verify op
    if type(op) is not Monoid:
        raise TypeError(f"op must be Monoid, not {type(op)}")

    # Apply transpose
    if tensor.ndims != 2:
        raise GrbDimensionMismatch("reduce_to_vector requires matrix input")
    if desc.transpose0:
        tensor = TransposedMatrix.wrap(tensor)

    # Compare shapes
    if out.ndims != 1:
        raise GrbDimensionMismatch("reduce_to_vector requires vector output")
    if out.shape != (tensor.shape[0],):
        raise GrbDimensionMismatch(f"output size should be {tensor.shape[0]} not {out.shape[0]}")

    # TODO: apply the mask during the computation, not at the end
    result = impl.reduce_to_vector(out.dtype, op, tensor)
    if mask is not None:
        result = impl.select_by_mask(result, mask, desc)
    update(out, result, mask, accum, desc)


def reduce_to_scalar(out: Scalar,
                     op: Monoid,
                     tensor: SparseTensor,
                     *,
                     accum: Optional[BinaryOp] = None,
                     desc: Descriptor = NULL_DESC):
    # Verify op
    if type(op) is not Monoid:
        raise TypeError(f"op must be Monoid, not {type(op)}")

    # Compare shapes
    if out.ndims != 0:
        raise GrbDimensionMismatch("reduce_to_scalar requires scalar output")

    result = impl.reduce_to_scalar(out.dtype, op, tensor)
    update(out, result, accum=accum, desc=desc)


def extract(out: SparseTensor,
            tensor: SparseTensorBase,
            row_indices=None,
            col_indices=None,
            *,
            mask: Optional[Vector] = None,
            accum: Optional[BinaryOp] = None,
            desc: Descriptor = NULL_DESC):
    """
    Setting row_indices or col_indices to `None` is the equivalent of GrB_ALL
    """
    # Apply transpose
    if desc.transpose0 and tensor.ndims == 2:
        tensor = TransposedMatrix.wrap(tensor)

    # Check indices
    if tensor.ndims == 0:  # Scalar input
        raise TypeError("Use `extract_element` rather than `extract` for Scalars")
    elif tensor.ndims == 1:  # Vector input
        if col_indices is not None:
            raise ValueError("col_indices not allowed for Vector, use row_indices")
        if type(row_indices) is int:
            raise TypeError("Use extract_element to get a single element from the Vector")
    else:  # Matrix input
        if type(row_indices) is int and type(col_indices) is int:
            raise TypeError("Use extract_element to get a single element from the Matrix")

    # Compute output sizes
    if type(row_indices) is int:
        row_size = None
    elif row_indices is None:
        row_size = tensor.shape[0]
    else:
        row_size = len(row_indices)

    if type(col_indices) is int or tensor.ndims < 2:
        col_size = None
    elif col_indices is None:
        col_size = tensor.shape[1]
    else:
        col_size = len(col_indices)

    # Compare shapes
    if tensor.ndims == 1:  # Vector input
        expected_out_shape = (row_size,)
    else:  # Matrix input
        if type(row_indices) is int:
            expected_out_shape = (col_size,)
        elif type(col_indices) is int:
            expected_out_shape = (row_size,)
        else:
            expected_out_shape = (row_size, col_size)
    if out.shape != expected_out_shape:
        raise GrbDimensionMismatch(f"output shape mismatch: {out.shape} != {expected_out_shape}")

    result = impl.extract(out.dtype, tensor, row_indices, col_indices, row_size, col_size)
    if mask is not None:
        result = impl.select_by_mask(result, mask, desc)
    update(out, result, mask, accum, desc)


def assign(out: SparseTensor,
           tensor: SparseObject,
           row_indices=None,
           col_indices=None,
           *,
           mask: Optional[Vector] = None,
           accum: Optional[BinaryOp] = None,
           desc: Descriptor = NULL_DESC):
    """
    Setting row_indices or col_indices to `None` is the equivalent of GrB_ALL
    """
    # Handle pure-Python scalar
    if not isinstance(tensor, SparseObject):
        if not isinstance(tensor, (int, float, bool)):
            raise TypeError(f"tensor must be a SparseObject or Python scalar, not {type(tensor)}")
        tensor = ensure_scalar_of_type(tensor, out.dtype)

    # Apply transpose
    if desc.transpose0 and tensor.ndims == 2:
        tensor = TransposedMatrix.wrap(tensor)

    # Check indices
    if out.ndims == 0:  # Scalar output
        raise TypeError("Use `set_element` rather than `assign` for Scalars")
    if out.ndims == 1:  # Vector output
        if col_indices is not None:
            raise ValueError("col_indices not allowed for Vector, use row_indices")
        if type(row_indices) is int:
            raise TypeError("Use `set_element` rather than `assign` to set a single element in the Vector")
    else:  # Matrix output
        if type(row_indices) is int and type(col_indices) is int:
            raise TypeError("Use `set_element` rather than `assign` to set a single element in the Matrix")

    # Compute output sizes
    if type(row_indices) is int:
        row_size = None
    elif row_indices is None:
        row_size = out.shape[0]
    else:
        ensure_unique(row_indices, "row_indices")
        row_size = len(row_indices)

    if type(col_indices) is int or out.ndims < 2:
        col_size = None
    elif col_indices is None:
        col_size = out.shape[1]
    else:
        ensure_unique(col_indices, "col_indices")
        col_size = len(col_indices)

    if tensor.ndims == 0:  # Scalar input
        if row_indices is None and col_indices is None and out.ndims == 2:
            # Scalar input with GrB_ALL
            if mask is None:
                raise GrbError("This will create a dense matrix. Please provide a mask or indices.")
            # Use mask to build an iso-valued Matrix
            result = impl.apply(out.dtype, BinaryOp.second, mask, right=tensor)
        else:
            if out.ndims == 1:  # Vector output
                result = impl.build_iso_vector_from_indices(out.dtype, *out.shape, row_indices, tensor)
            else:  # Matrix output
                result = impl.build_iso_matrix_from_indices(out.dtype, *out.shape, row_indices, col_indices, tensor, colwise=out.is_colwise())
            if mask is not None:
                result = impl.select_by_mask(result, mask, desc)
    else:  # Vector/Matrix input
        # Verify expected input shape
        if out.ndims == 1:  # Vector output
            expected_input_shape = (row_size,)
        else:  # Matrix output
            if type(row_indices) is int:
                expected_input_shape = (col_size,)
            elif type(col_indices) is int:
                expected_input_shape = (row_size,)
            else:
                expected_input_shape = (row_size, col_size)
        if tensor.shape != expected_input_shape:
            raise GrbDimensionMismatch(f"input shape mismatch: {tensor.shape} != {expected_input_shape}")

        result = impl.assign(out.dtype, tensor, row_indices, col_indices, *out.shape)
        if mask is not None:
            result = impl.select_by_mask(result, mask, desc)
    update(out, result, mask, accum, desc, row_indices=row_indices, col_indices=col_indices)
