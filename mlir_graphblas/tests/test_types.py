import pytest
import numpy as np
from ..types import (
    DType, BOOL, INT8, INT16, INT32, INT64, FP32, FP64, RankedTensorType,
    find_common_dtype
)
from mlir.dialects.sparse_tensor import DimLevelType
from mlir import ir


def test_lookup_gb():
    assert DType.from_gb('BOOL') == BOOL
    assert DType.from_gb('BOOL') != INT8
    assert DType.from_gb('INT8') == INT8
    assert DType.from_gb('INT16') == INT16
    assert DType.from_gb('INT32') == INT32
    assert DType.from_gb('INT64') == INT64
    assert DType.from_gb('FP32') == FP32
    assert DType.from_gb('FP64') == FP64


def test_lookup_mlir():
    assert DType.from_mlir('i8') == INT8
    assert DType.from_mlir('i16') == INT16
    assert DType.from_mlir('i32') == INT32
    assert DType.from_mlir('i64') == INT64
    assert DType.from_mlir('f32') == FP32
    assert DType.from_mlir('f64') == FP64


def test_lookup_np():
    assert DType.from_np(np.int8) == INT8
    assert DType.from_np(np.int16) == INT16
    assert DType.from_np(np.int32) == INT32
    assert DType.from_np(np.int64) == INT64
    assert DType.from_np(np.float32) == FP32
    assert DType.from_np(np.float64) == FP64


def test_sets():
    for t in (INT8, INT16, INT32, INT64):
        assert t.is_int()
        assert not t.is_float()
    for t in (FP32, FP64):
        assert t.is_float()
        assert not t.is_int()


def test_rtt():
    rtt = RankedTensorType(INT16, [DimLevelType.dense, DimLevelType.compressed], [1, 0])
    with ir.Context(), ir.Location.unknown():
        rtt_str = str(rtt.as_mlir_type())
    assert rtt_str == (
        'tensor<?x?xi16, #sparse_tensor.encoding<{ '
        'dimLevelType = [ "dense", "compressed" ], '
        'dimOrdering = affine_map<(d0, d1) -> (d1, d0)> '
        '}>>'
    )


def test_find_common_dtype():
    assert find_common_dtype(FP64, FP64) == FP64
    assert find_common_dtype(FP64, FP32) == FP64
    assert find_common_dtype(FP64, INT8) == FP64
    assert find_common_dtype(INT8, FP64) == FP64
    assert find_common_dtype(INT16, FP32) == FP32
    assert find_common_dtype(INT16, INT16) == INT16
    assert find_common_dtype(INT8, INT64) == INT64
    assert find_common_dtype(INT32, BOOL) == INT32
    assert find_common_dtype(BOOL, BOOL) == BOOL
