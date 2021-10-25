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

        self.i32 = IntegerType.get(32)

    # TODO make this generic in the type
    def constant(self, val: int, typ: Attribute) -> Operation:
        return Operation.with_result_types(
            Constant, [], [typ],
            attributes={"value": IntegerAttr.get(val, typ)})

    def constant_from_attr(self, attr: Attribute, typ: Attribute) -> Operation:
        return Operation.with_result_types(Constant, [], [typ],
                                           attributes={"value": attr})

    # TODO these operations should support all kinds of integer types
    def muli(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            Muli, [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})

    def addi(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
        return Operation.with_result_types(
            Addi, [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})

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
