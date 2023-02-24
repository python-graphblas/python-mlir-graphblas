import ctypes
import numbers
import math
import functools
import mlir
import numpy as np
from mlir import ir
from mlir import passmanager
from mlir import execution_engine
from mlir import runtime
from mlir import dialects
from mlir.dialects import arith

from .exceptions import GrbDomainMismatch
from .types import DType, BOOL, INT8, INT64, FP64, cast, find_common_dtype
from .utils import CmpFPredicate, CmpIPredicate

__all__ = ["UnaryOp", "BinaryOp", "IndexUnaryOp", "SelectOp", "Monoid", "Semiring"]


class Op:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def _register(cls, op):
        # Make the name an attribute on the class
        setattr(cls, op.name, op)

    @staticmethod
    def _mlirtype_of(var):
        if isinstance(var, ir.BlockArgument) or isinstance(var, ir.OpResult):
            return str(var.type)
        elif hasattr(var, "result"):
            return str(var.result.type)
        else:
            raise TypeError(f"Unable to determine type for {var}")

    @staticmethod
    def _dtype_of(var):
        return DType.from_mlir(Op._mlirtype_of(var))


class _FuncOp(Op):
    _type_convert = {
        bool: BOOL,
        int: INT64,
        float: FP64,
        None: None,
    }

    def __init__(self, func, *, input=None, output=None):
        """
        If input is defined, it must be one of (bool, int, float) to
            indicate the restricted allowable input dtypes
        If output is defined, it must be one of (bool, int, float) to
            indicate the output will always be of that type or an
            an integer (0, 1, ...) to indicate that the output dtype
            will match the dtype of argument 0, 1, ...
        """
        super().__init__(func.__name__)
        self.func = func
        # Validate input
        if input is not None:
            assert input in {bool, int, float}
        self.input = input
        # Validate output
        if output is not None:
            assert output in {bool, int}
        self.output = output

    @classmethod
    def _register(cls, func=None, **kwargs):
        if func is None:
            return functools.partial(cls._register, **kwargs)

        op = cls(func, **kwargs)
        op.__doc__ = func.__doc__
        super()._register(op)
        return op

    def validate_input(self, input_type):
        if self.input is None:
            return

        if self.input is bool:
            if input_type not in {BOOL, INT8}:
                raise GrbDomainMismatch("input must be boolean type")
        elif self.input is int:
            if not input_type.is_int():
                raise GrbDomainMismatch("input must be int type")
        elif self.input is float:
            if not input_type.is_float():
                raise GrbDomainMismatch("input must be float type")

    def validate_output(self, output_type):
        if self.output is None:
            return

        if self.output is bool:
            if output_type != BOOL:
                raise GrbDomainMismatch("output must be BOOL type")
        elif self.output is int:
            if output_type != INT64:
                raise GrbDomainMismatch("output must be INT64 type")


class UnaryOp(_FuncOp):
    """
    Operation taking one argument and returning
    one result, possibly of a different dtype.

    Create a UnaryOp by decorating a function
    which returns an MLIR operation.

    @UnaryOp._register
    def name_of_op(x, dtype):
        return ...
    """
    def __call__(self, out_type: DType, x):
        x, xtype = self._validate(out_type, x)
        return self.func(x, xtype)

    def _validate(self, out_type, x):
        self.validate_output(out_type)
        xtype = self._dtype_of(x)
        if self.output is None:
            x = cast(x, xtype, out_type)
            xtype = out_type
        self.validate_input(xtype)
        return x, xtype


class BinaryOp(_FuncOp):
    """
    Operation taking two arguments (both with identical dtypes)
    and returning one result, possibly of a different dtype.

    Create a BinaryOp by decorating a function
    which returns an MLIR operation.

    @BinaryOp._register
    def name_of_op(x, y, input_dtype):
        return ...
    """
    def __call__(self, out_type: DType, x, y):
        x, y, dtype = self._validate(out_type, x, y)
        return self.func(x, y, dtype)

    def _validate(self, out_type, x, y):
        self.validate_output(out_type)
        xtype = self._dtype_of(x)
        ytype = self._dtype_of(y)
        if self.output is None:
            x = cast(x, xtype, out_type)
            y = cast(y, ytype, out_type)
            dtype = out_type
        else:
            dtype = find_common_dtype(xtype, ytype)
            x = cast(x, xtype, dtype)
            y = cast(y, ytype, dtype)
        self.validate_input(dtype)
        return x, y, dtype


class IndexUnaryOp(_FuncOp):
    """
    Operation taking four arguments and returning
    one result. The inputs are:
     - value
     - row_index
     - col_index
     - thunk

    Create an IndexUnaryOp by decorating a function
    which returns an MLIR operation.

    @IndexUnaryOp._register
    def name_of_op(val, row, col, thunk, val_dtype):
        return ...
    """
    def __init__(self, func, *, input=None, output=None, thunk_as_index=False):
        """
        If input is defined, it must be one of (bool, int, float) to
            indicate the restricted allowable input dtypes
        If output is defined, it must be one of (bool, int, float) to
            indicate the output will always be of that type
        """
        super().__init__(func, input=input, output=output)
        self.thunk_as_index = thunk_as_index

    def __call__(self, out_type: DType, val, row, col, thunk):
        if self.thunk_as_index:
            # Ensure thunk is an index
            thunk_type_str = self._mlirtype_of(thunk)
            if thunk_type_str != "index":
                raise GrbDomainMismatch("thunk must be index type")
            val, dtype = UnaryOp._validate(self, out_type, val)
        else:
            # Thunk dtype should make val dtype
            val, thunk, dtype = BinaryOp._validate(self, out_type, val, thunk)
        return self.func(val, row, col, thunk, dtype)


class SelectOp(IndexUnaryOp):
    """
    Operation taking four arguments and returning
    one boolean result. The inputs are:
     - value
     - row_index
     - col_index
     - thunk

    Create a SelectOp by decorating a function
    which returns an MLIR operation. The output must
    be `bool` and this must be called out during
    the registration.

    @SelectOp._register(output=bool)
    def name_of_op(val, row, col, thunk, val_dtype):
        return ...
    """
    def __init__(self, func, *, input=None, output=None, thunk_as_index=False):
        """
        If input is defined, it must be one of (bool, int, float) to
            indicate the restricted allowable input dtypes
        output must be defined and it must be `bool`
        """
        if output is not bool:
            raise TypeError("output must be `bool`")
        super().__init__(func, input=input, output=output, thunk_as_index=thunk_as_index)


class Monoid(Op):
    """
    Operation expanding upon a BinaryOp. Restricts the
    input and output dtypes to all be the same.

    An identity value is used as the starting value
    for a reduction of a set of values.

    Create a Monoid by decorating a function
    which returns the identity and passing the
    BinaryOp into the decorator.

    @Monoid._register(BinaryOp.some_func)
    def some_monoid(dtype):
        identity_for_dtype = ...
        return identity_for_dtype
    """
    def __init__(self, binop: BinaryOp, identity_func):
        super().__init__(identity_func.__name__)
        self.binop = binop
        self.identity = identity_func

    @classmethod
    def _register(cls, binop):
        if binop.output is not None:
            raise ValueError("The BinaryOp associated with a Monoid must have the same output and input dtype")
        def monoid_wrapper(func):
            op = cls(binop, func)
            op.__doc__ = f"Monoid for BinaryOp.{binop.name}"
            super(Monoid, cls)._register(op)
            return op
        return monoid_wrapper


class Semiring(Op):
    def __init__(self, monoid: Monoid, binop: BinaryOp, *, name=None):
        if name is None:
            name = f"{monoid.name}_{binop.name}"
        super().__init__(name)
        self.monoid = monoid
        self.binop = binop

    @classmethod
    def _register(cls, monoid: Monoid, binop: BinaryOp, name=None):
        op = cls(monoid, binop, name=name)
        super()._register(op)


############################
# Unary Op Definitions
############################

@UnaryOp._register
def identity(x, dtype):
    return x


@UnaryOp._register
def abs(x, dtype):
    op = dialects.math.AbsFOp if dtype.is_float() else dialects.math.AbsIOp
    return op(x)


@UnaryOp._register
def ainv(x, dtype):
    if dtype.is_float():
        return arith.NegFOp(x)
    return arith.SubIOp(dtype.build_constant(0), x)


@UnaryOp._register(input=float)
def minv(x, dtype):
    return arith.DivFOp(dtype.build_constant(1), x)


@UnaryOp._register(input=bool)
def lnot(x, dtype):
    # Only first bit should be set for boolean
    return arith.XOrIOp(x, dtype.build_constant(1))


@UnaryOp._register(input=int)
def bnot(x, dtype):
    # -1 is equivalent to all 1-bits
    return arith.XOrIOp(x, dtype.build_constant(-1))

############################
# Binary Op Definitions
############################

@BinaryOp._register(input=bool)
def lor(x, y, dtype):
    # TODO: do we need to AND with 1 to ensure a 0/1 result?
    return arith.OrIOp(x, y)


@BinaryOp._register(input=bool)
def land(x, y, dtype):
    # TODO: do we need to AND with 1 to ensure a 0/1 result?
    return arith.AndIOp(x, y)


@BinaryOp._register(input=bool)
def lxor(x, y, dtype):
    # TODO: do we need to AND with 1 to ensure a 0/1 result?
    return arith.XOrIOp(x, y)


@BinaryOp._register(input=bool)
def lxnor(x, y, dtype):
    # TODO: do we need to AND with 1 to ensure a 0/1 result?
    xor_result = arith.XOrIOp(x, y)
    return arith.XOrIOp(xor_result, dtype.build_constant(1))


@BinaryOp._register(input=int)
def bor(x, y, dtype):
    return arith.OrIOp(x, y)


@BinaryOp._register(input=int)
def band(x, y, dtype):
    return arith.AndIOp(x, y)


@BinaryOp._register(input=int)
def bxor(x, y, dtype):
    return arith.XOrIOp(x, y)


@BinaryOp._register(input=int)
def bxnor(x, y, dtype):
    xor_result = arith.XOrIOp(x, y)
    return arith.XOrIOp(xor_result, dtype.build_constant(1))


@BinaryOp._register(output=bool)
def eq(x, y, dtype):
    i1 = ir.IntegerType.get_signless(1)
    if dtype.is_float():
        cmp = arith.CmpFOp(CmpFPredicate.oeq.build(), x, y)
    else:
        cmp = arith.CmpIOp(CmpIPredicate.eq.build(), x, y)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@BinaryOp._register(output=bool)
def ne(x, y, dtype):
    i1 = ir.IntegerType.get_signless(1)
    if dtype.is_float():
        cmp = arith.CmpFOp(CmpFPredicate.one.build(), x, y)
    else:
        cmp = arith.CmpIOp(CmpIPredicate.ne.build(), x, y)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)

@BinaryOp._register(output=bool)
def gt(x, y, dtype):
    i1 = ir.IntegerType.get_signless(1)
    if dtype.is_float():
        cmp = arith.CmpFOp(CmpFPredicate.ogt.build(), x, y)
    else:
        cmp = arith.CmpIOp(CmpIPredicate.sgt.build(), x, y)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@BinaryOp._register(output=bool)
def lt(x, y, dtype):
    i1 = ir.IntegerType.get_signless(1)
    if dtype.is_float():
        cmp = arith.CmpFOp(CmpFPredicate.olt.build(), x, y)
    else:
        cmp = arith.CmpIOp(CmpIPredicate.slt.build(), x, y)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@BinaryOp._register(output=bool)
def ge(x, y, dtype):
    i1 = ir.IntegerType.get_signless(1)
    if dtype.is_float():
        cmp = arith.CmpFOp(CmpFPredicate.oge.build(), x, y)
    else:
        cmp = arith.CmpIOp(CmpIPredicate.sge.build(), x, y)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@BinaryOp._register(output=bool)
def le(x, y, dtype):
    i1 = ir.IntegerType.get_signless(1)
    if dtype.is_float():
        cmp = arith.CmpFOp(CmpFPredicate.ole.build(), x, y)
    else:
        cmp = arith.CmpIOp(CmpIPredicate.sle.build(), x, y)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@BinaryOp._register
def oneb(x, y, dtype):
    return dtype.build_constant(1)
# Alias
BinaryOp.pair = BinaryOp.oneb


@BinaryOp._register
def first(x, y, dtype):
    return x


@BinaryOp._register
def second(x, y, dtype):
    return y


@BinaryOp._register
def min(x, y, dtype):
    op = arith.MinFOp if dtype.is_float() else arith.MinSIOp
    return op(x, y)


@BinaryOp._register
def max(x, y, dtype):
    op = arith.MaxFOp if dtype.is_float() else arith.MaxSIOp
    return op(x, y)


@BinaryOp._register
def plus(x, y, dtype):
    op = arith.AddFOp if dtype.is_float() else arith.AddIOp
    return op(x, y)


@BinaryOp._register
def minus(x, y, dtype):
    op = arith.SubFOp if dtype.is_float() else arith.SubIOp
    return op(x, y)


@BinaryOp._register
def times(x, y, dtype):
    op = arith.MulFOp if dtype.is_float() else arith.MulIOp
    return op(x, y)


@BinaryOp._register
def div(x, y, dtype):
    op = arith.DivFOp if dtype.is_float() else arith.FloorDivSIOp
    return op(x, y)


############################
# Select Op Definitions
############################

@SelectOp._register(output=bool, thunk_as_index=True)
def tril(val, row, col, thunk, val_dtype):
    row_plus = arith.AddIOp(row, thunk)
    return arith.CmpIOp(CmpIPredicate.sle.build(), col, row_plus)


@SelectOp._register(output=bool, thunk_as_index=True)
def triu(val, row, col, thunk, val_dtype):
    row_plus = arith.AddIOp(row, thunk)
    return arith.CmpIOp(CmpIPredicate.sge.build(), col, row_plus)


@SelectOp._register(output=bool, thunk_as_index=True)
def diag(val, row, col, thunk, val_dtype):
    row_plus = arith.AddIOp(row, thunk)
    return arith.CmpIOp(CmpIPredicate.eq.build(), col, row_plus)


@SelectOp._register(output=bool, thunk_as_index=True)
def offdiag(val, row, col, thunk, val_dtype):
    row_plus = arith.AddIOp(row, thunk)
    return arith.CmpIOp(CmpIPredicate.ne.build(), col, row_plus)


@SelectOp._register(output=bool, thunk_as_index=True)
def colle(val, row, col, thunk, val_dtype):
    return arith.CmpIOp(CmpIPredicate.sle.build(), col, thunk)


@SelectOp._register(output=bool, thunk_as_index=True)
def colgt(val, row, col, thunk, val_dtype):
    return arith.CmpIOp(CmpIPredicate.sgt.build(), col, thunk)


@SelectOp._register(output=bool, thunk_as_index=True)
def rowle(val, row, col, thunk, val_dtype):
    return arith.CmpIOp(CmpIPredicate.sle.build(), row, thunk)


@SelectOp._register(output=bool, thunk_as_index=True)
def rowgt(val, row, col, thunk, val_dtype):
    return arith.CmpIOp(CmpIPredicate.sgt.build(), row, thunk)


@SelectOp._register(output=bool)
def valueeq(val, row, col, thunk, val_dtype):
    if val_dtype.is_float():
        return arith.CmpFOp(CmpFPredicate.oeq.build(), val, thunk)
    else:
        return arith.CmpIOp(CmpIPredicate.eq.build(), val, thunk)


@SelectOp._register(output=bool)
def valuene(val, row, col, thunk, val_dtype):
    if val_dtype.is_float():
        return arith.CmpFOp(CmpFPredicate.one.build(), val, thunk)
    else:
        return arith.CmpIOp(CmpIPredicate.ne.build(), val, thunk)


@SelectOp._register(output=bool)
def valuelt(val, row, col, thunk, val_dtype):
    if val_dtype.is_float():
        return arith.CmpFOp(CmpFPredicate.olt.build(), val, thunk)
    else:
        return arith.CmpIOp(CmpIPredicate.slt.build(), val, thunk)


@SelectOp._register(output=bool)
def valuele(val, row, col, thunk, val_dtype):
    if val_dtype.is_float():
        return arith.CmpFOp(CmpFPredicate.ole.build(), val, thunk)
    else:
        return arith.CmpIOp(CmpIPredicate.sle.build(), val, thunk)


@SelectOp._register(output=bool)
def valuegt(val, row, col, thunk, val_dtype):
    if val_dtype.is_float():
        return arith.CmpFOp(CmpFPredicate.ogt.build(), val, thunk)
    else:
        return arith.CmpIOp(CmpIPredicate.sgt.build(), val, thunk)


@SelectOp._register(output=bool)
def valuege(val, row, col, thunk, val_dtype):
    if val_dtype.is_float():
        return arith.CmpFOp(CmpFPredicate.oge.build(), val, thunk)
    else:
        return arith.CmpIOp(CmpIPredicate.sge.build(), val, thunk)


############################
# Index Unary Op Definitions
############################

@IndexUnaryOp._register(output=int, thunk_as_index=True)
def rowindex(val, row, col, thunk, val_dtype):
    idx = arith.AddIOp(row, thunk)
    # idx is index; cast to i64
    return arith.IndexCastOp(INT64.build_mlir_type(), idx)


@IndexUnaryOp._register(output=int, thunk_as_index=True)
def colindex(val, row, col, thunk, val_dtype):
    idx = arith.AddIOp(col, thunk)
    # idx is index; cast to i64
    return arith.IndexCastOp(INT64.build_mlir_type(), idx)


@IndexUnaryOp._register(output=int, thunk_as_index=True)
def diagindex(val, row, col, thunk, val_dtype):
    diag = arith.SubIOp(col, row)
    idx = arith.AddIOp(diag, thunk)
    # idx is index; cast to i64
    return arith.IndexCastOp(INT64.build_mlir_type(), idx)


@IndexUnaryOp._register(output=bool, thunk_as_index=True)
def tril(val, row, col, thunk, val_dtype):
    cmp = SelectOp.tril.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool, thunk_as_index=True)
def triu(val, row, col, thunk, val_dtype):
    cmp = SelectOp.triu.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool, thunk_as_index=True)
def diag(val, row, col, thunk, val_dtype):
    cmp = SelectOp.diag.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool, thunk_as_index=True)
def offdiag(val, row, col, thunk, val_dtype):
    cmp = SelectOp.offdiag.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool, thunk_as_index=True)
def colle(val, row, col, thunk, val_dtype):
    cmp = SelectOp.colle.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool, thunk_as_index=True)
def colgt(val, row, col, thunk, val_dtype):
    cmp = SelectOp.colgt.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool, thunk_as_index=True)
def rowle(val, row, col, thunk, val_dtype):
    cmp = SelectOp.rowle.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool, thunk_as_index=True)
def rowgt(val, row, col, thunk, val_dtype):
    cmp = SelectOp.rowgt.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool)
def valueeq(val, row, col, thunk, val_dtype):
    cmp = SelectOp.valueeq.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool)
def valuene(val, row, col, thunk, val_dtype):
    cmp = SelectOp.valuene.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool)
def valuelt(val, row, col, thunk, val_dtype):
    cmp = SelectOp.valuelt.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool)
def valuele(val, row, col, thunk, val_dtype):
    cmp = SelectOp.valuele.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool)
def valuegt(val, row, col, thunk, val_dtype):
    cmp = SelectOp.valuegt.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


@IndexUnaryOp._register(output=bool)
def valuege(val, row, col, thunk, val_dtype):
    cmp = SelectOp.valuege.func(val, row, col, thunk, val_dtype)
    # cmp is i1; cast to i8 (BOOL)
    return arith.ExtUIOp(BOOL.build_mlir_type(), cmp)


############################
# Monoid Definitions
############################

@Monoid._register(BinaryOp.plus)
def plus(dtype):
    return dtype.build_constant(0)


@Monoid._register(BinaryOp.times)
def times(dtype):
    return dtype.build_constant(1)


@Monoid._register(BinaryOp.min)
def min(dtype):
    return dtype.build_max()


@Monoid._register(BinaryOp.max)
def max(dtype):
    return dtype.build_min()


@Monoid._register(BinaryOp.lor)
def lor(dtype):
    return BOOL.build_constant(0)


@Monoid._register(BinaryOp.land)
def land(dtype):
    return BOOL.build_constant(1)


@Monoid._register(BinaryOp.lxor)
def lxor(dtype):
    return BOOL.build_constant(0)


@Monoid._register(BinaryOp.lxnor)
def lxnor(dtype):
    return BOOL.build_constant(1)


############################
# Semiring Definitions
############################

for monoid in (Monoid.plus, Monoid.min, Monoid.max):
    for binop in (BinaryOp.plus, BinaryOp.minus, BinaryOp.times, BinaryOp.div,
                  BinaryOp.max, BinaryOp.min, BinaryOp.first, BinaryOp.second):
        Semiring._register(monoid, binop)

for monoid in (Monoid.lor, Monoid.land, Monoid.lxor, Monoid.lxnor):
    for binop in (BinaryOp.land, BinaryOp.lor, BinaryOp.eq, BinaryOp.ne,
                  BinaryOp.gt, BinaryOp.ge, BinaryOp.lt, BinaryOp.le):
        Semiring._register(monoid, binop)
