from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.builtin import IntegerType, Float32Type, IntegerAttr, FlatSymbolRefAttr


@dataclass
class IET:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(Constant)

        self.ctx.register_op(Addi)
        self.ctx.register_op(Muli)
        self.ctx.register_op(Subi)
        self.ctx.register_op(FloordiviSigned)
        self.ctx.register_op(RemiSigned)

        self.ctx.register_op(Addf)
        self.ctx.register_op(Mulf)

        self.ctx.register_op(Call)
        self.ctx.register_op(Return)

        self.ctx.register_op(And)
        self.ctx.register_op(Or)
        self.ctx.register_op(Xor)

        self.ctx.register_op(Cmpi)

        self.f32 = Float32Type.get()
        self.i64 = IntegerType.get(64)
        self.i32 = IntegerType.get(32)
        self.i1 = IntegerType.get(1)

    # TODO make this generic in the type
    def constant(self, val: int, typ: Attribute) -> Operation:
        return Operation.with_result_types(
            Constant, [], [typ],
            attributes={"value": IntegerAttr.get(val, typ)})

    def constant_from_attr(self, attr: Attribute, typ: Attribute) -> Operation:
        return Operation.with_result_types(Constant, [], [typ],
                                           attributes={"value": attr})

    def mulf(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            self.ctx.get_op("iet.mulf"),
            [get_ssa_value(x), get_ssa_value(y)], [self.f32], {})

    def addf(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            self.ctx.get_op("iet.addf"),
            [get_ssa_value(x), get_ssa_value(y)], [self.f32], {})

    def call(self, callee: str, ops: List[OpOrBlockArg],
             return_types: List[Attribute]) -> Operation:
        return Operation.with_result_types(
            Call, [get_ssa_value(op) for op in ops],
            return_types,
            attributes={"callee": FlatSymbolRefAttr.get(callee)})

    def return_(self, *ops: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(Return,
                                           [get_ssa_value(op) for op in ops],
                                           [], {})

    # TODO these operations should support all kinds of integer types
    def muli(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            Muli, [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})

    def addi(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            Addi, [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})

    def subi(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            Subi, [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})

    def floordivi_signed(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            FloordiviSigned,
            [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})

    def remi_signed(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            RemiSigned, [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})

    # TODO these operations should support all kinds of integer types
    def and_(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            And, [get_ssa_value(x), get_ssa_value(y)], [self.i1], {})

    def or_(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            Or, [get_ssa_value(x), get_ssa_value(y)], [self.i1], {})

    def xor_(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            Xor, [get_ssa_value(x), get_ssa_value(y)], [self.i1], {})

    def cmpi(self, x: OpOrBlockArg, y: OpOrBlockArg, arg: int) -> Operation:
        return Operation.with_result_types(
            Cmpi, [get_ssa_value(x), get_ssa_value(y)], [self.i1],
            attributes={"predicate": IntegerAttr.get(arg, self.i64)})


@irdl_op_definition
class Constant:
    name: str = "iet.constant"
    output = ResultDef(AnyAttr())
    value = AttributeDef(AnyAttr())

    # TODO verify that the output and value type are equal
    def verify_(self) -> None:
        # TODO how to force the attr to have a type? and how to query it?
        pass


@irdl_op_definition
class Addi:
    name: str = "iet.addi"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class Muli:
    name: str = "iet.muli"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class Subi:
    name: str = "iet.subi"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class FloordiviSigned:
    name: str = "iet.floordivi_signed"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class RemiSigned:
    name: str = "iet.remi_signed"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class Call:
    name: str = "iet.call"
    arguments = VarOperandDef(AnyAttr())
    callee = AttributeDef(FlatSymbolRefAttr)

    # Note: naming this results triggers an ArgumentError
    res = VarResultDef(AnyAttr())
    # TODO how do we verify that the types are correct?


@irdl_op_definition
class Return:
    name: str = "iet.return"
    arguments = VarOperandDef(AnyAttr())


@irdl_op_definition
class And:
    name: str = "iet.and"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class Or:
    name: str = "iet.or"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class Xor:
    name: str = "iet.xor"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class Cmpi:
    name: str = "iet.cmpi"
    predicate = AttributeDef(IntegerAttr)
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType.get(1))


@irdl_op_definition
class Addf:
    name: str = "iet.addf"
    input1 = OperandDef(Float32Type)
    input2 = OperandDef(Float32Type)
    output = ResultDef(Float32Type)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")


@irdl_op_definition
class Mulf:
    name: str = "iet.mulf"
    input1 = OperandDef(Float32Type)
    input2 = OperandDef(Float32Type)
    output = ResultDef(Float32Type)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")
