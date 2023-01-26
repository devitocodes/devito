from sympy import Mod
from typing import Tuple, List, Annotated, Union
from dataclasses import dataclass

from xdsl.dialects.builtin import (IntegerType, StringAttr,
                                   ArrayAttr, OpAttr, FunctionType,
                                   ContainerOf, IndexType,
                                   Float32Type, IntegerAttr, FloatAttr)
from xdsl.dialects.arith import Constant

from xdsl.irdl import (ResultDef, OperandDef, AttributeDef, AnyAttr,
                       irdl_op_definition, Operand, AnyOf, RegionDef,
                       VarOperand, OptOpAttr)
from xdsl.ir import MLContext, Operation, Block, Region, OpResult, SSAValue, Attribute


signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))


@dataclass
class IET:
    ctx: MLContext

    def __post_init__(self):
        # TODO add all operations
        self.ctx.register_op(FloatConstant)
        self.ctx.register_op(Powi)
        self.ctx.register_op(Modi)
        self.ctx.register_op(Iteration)
        self.ctx.register_op(IterationWithSubIndices)
        self.ctx.register_op(Callable)
        self.ctx.register_op(Idx)
        self.ctx.register_op(Assign)
        self.ctx.register_op(Initialise)
        self.ctx.register_op(PointerCast)
        self.ctx.register_op(Statement)
        self.ctx.register_op(StructDecl)
        self.f32 = Float32Type()


@irdl_op_definition
class FloatConstant(Operation):
    name: str = "iet.floatconstant"
    output = ResultDef(AnyAttr())
    value = AttributeDef(AnyAttr())

    # TODO verify that the output and value type are equal
    def verify_(self) -> None:
        # TODO how to force the attr to have a type? and how to query it?
        pass

    @staticmethod
    def get(value):
        attr = FloatAttr.from_float_and_width(value, 32)
        res = Constant.build(operands=[],
                             attributes={"value": attr},
                             result_types=[Float32Type()])
        return res


@irdl_op_definition
class Modi(Operation):
    name: str = "iet.modi"
    input1 = Annotated[Operand, signlessIntegerLike]
    input2 = Annotated[Operand, signlessIntegerLike]
    output = Annotated[OpResult, signlessIntegerLike]

    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(lhs, rhs):
        res = Modi.build(operands=[lhs, rhs],
                         result_types=[IntegerType.build(32)])
        return res


@irdl_op_definition
class Powi(Operation):
    name: str = "iet.Powi"
    base: Annotated[Operand, signlessIntegerLike]
    exp: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    def verify_(self) -> None:
        if self.base.typ != self.exp.typ or self.exp.typ != self.result.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(base: Union[Operation, SSAValue],
            exp: Union[Operation, SSAValue]):
        base = SSAValue.get(base)
        return Powi.build(operands=[base, exp],
                          result_types=[base.typ])


@irdl_op_definition
class Idx(Operation):
    name: str = "iet.idx"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    @staticmethod
    def get(operand1: Union[Operation, SSAValue],
            operand2: Union[Operation, SSAValue]):
        operand1 = SSAValue.get(operand1)
        return Idx.build(operands=[operand1, operand2],
                         result_types=[operand1.typ])


@irdl_op_definition
class Assign(Operation):
    name: str = "iet.assign"
    lhs: Annotated[Operand, signlessIntegerLike]
    rhs: Annotated[Operand, signlessIntegerLike]


@irdl_op_definition
class PointerCast(Operation):
    name: str = "iet.pointercast"
    statement = AttributeDef(StringAttr)

    @staticmethod
    def get(statement):
        return PointerCast.build(
            operands=[],
            attributes={"statement": StringAttr.from_str(str(statement))},
            result_types=[])


@irdl_op_definition
class Statement(Operation):
    name: str = "iet.comment"
    statement = AttributeDef(StringAttr)

    @staticmethod
    def get(statement):
        return Statement.build(
            operands=[],
            attributes={"statement": StringAttr.from_str(str(statement))},
            result_types=[])


@irdl_op_definition
class StructDecl(Operation):
    name: str = "iet.structdecl"
    id = AttributeDef(StringAttr)
    fields = AttributeDef(ArrayAttr)
    declname = AttributeDef(StringAttr)
    padbytes = AttributeDef(AnyAttr())

    @staticmethod
    def get(name: str, fields: List[str], declname: str, padbytes: int = 0):
        padb = IntegerAttr.from_int_and_width(padbytes, 32)
        return StructDecl.build(
            operands=[],
            attributes={
                "id":
                StringAttr.from_str(name),
                "fields":
                ArrayAttr.from_list([StringAttr.from_str(str(f)) for f in fields]),
                "declname":
                StringAttr.from_str(str(declname)),
                "padbytes":
                padb
            },
            result_types=[])


@irdl_op_definition
class Initialise(Operation):
    name: str = "iet.initialise"
    id = AttributeDef(StringAttr)
    rhs = OperandDef(Float32Type)
    lhs = ResultDef(Float32Type)

    @staticmethod
    def get(lhs, rhs, id):
        res = Initialise.build(attributes={"id": StringAttr.from_str(id)},
                               operands=[lhs],
                               result_types=[Float32Type()])
        return res


@irdl_op_definition
class Callable(Operation):
    name: str = "iet.callable"

    body: Region
    callable_name = OpAttr[StringAttr]
    parameters = Annotated[VarOperand, AnyAttr()]
    header_parameters = Annotated[VarOperand, AnyAttr()]
    types = Annotated[VarOperand, AnyAttr()]
    qualifiers = Annotated[VarOperand, AnyAttr()]
    body = Region

    @staticmethod
    def get(name: str,
            parameters: Attribute,
            header_parameters: Attribute,
            types: Attribute,
            qualifiers: Attribute,
            body: Block = []):
        return Callable.build(attributes={
            "callable_name": StringAttr.from_str(name),
            "parameters":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in parameters]),
            "header_parameters":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in header_parameters]),
            "types":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in types]),
            "qualifiers":
            ArrayAttr.from_list([StringAttr.from_str(q) for q in qualifiers])
        }, regions=[Region.from_block_list([body])])


@irdl_op_definition
class Iteration(Operation):
    name: str = "iet.iteration"

    arg_name = OpAttr[StringAttr]
    body: Region
    limits = OpAttr[Attribute]
    properties = OpAttr[Attribute]
    pragmas = OpAttr[Attribute]

    @staticmethod
    def get(properties: List[str | StringAttr],
            limits: Tuple[str, str, str],
            arg_name: str | StringAttr,
            body: Block,
            pragmas: List[str | StringAttr] = []):
        return Iteration.build(attributes={
            "limits":
            ArrayAttr.from_list([
                StringAttr.from_str(str(limits[0])),
                StringAttr.from_str(str(limits[1])),
                StringAttr.from_str(str(limits[2]))
            ]),
            "properties":
            ArrayAttr.from_list([StringAttr.from_str(str(p)) for p in properties]),
            "pragmas":
            ArrayAttr.from_list([StringAttr.from_str(str(p)) for p in pragmas]),
            "arg_name": StringAttr.from_str(str(arg_name))
        }, regions=[Region.from_block_list([body])])


@irdl_op_definition
class IterationWithSubIndices(Operation):
    name: str = "iet.iteration_with_subindices"
    arg_name = AttributeDef(StringAttr)
    body = RegionDef()
    limits = AttributeDef(ArrayAttr)
    uindices_names = AttributeDef(ArrayAttr)
    uindices_symbmins_dividends = AttributeDef(ArrayAttr)
    uindices_symbmins_divisors = AttributeDef(ArrayAttr)
    properties = AttributeDef(ArrayAttr)
    pragmas = AttributeDef(ArrayAttr)

    @staticmethod
    def get(properties: List[str],
            limits: Tuple[str, str, str],
            uindices_names: Tuple[str, ...],
            uindices_symbmins: Tuple[Mod, ...],
            arg: str,
            body: Block,
            pragmas: List[str] = []):
        return IterationWithSubIndices.build(attributes={
            "limits":
            ArrayAttr.from_list([
                StringAttr.from_str(str(limits[0])),
                StringAttr.from_str(str(limits[1])),
                StringAttr.from_str(str(limits[2]))
            ]),
            "uindices_names":
            ArrayAttr.from_list(
                [StringAttr.from_str(u) for u in uindices_names]),
            # TODO make a "ModAttr"??
            "uindices_symbmins_dividends":
            ArrayAttr.from_list([StringAttr.from_str(str(u.args[0]))
                                 for u in uindices_symbmins]),
            "uindices_symbmins_divisors":
                ArrayAttr.from_list([StringAttr.from_str(str(u.args[1]))
                                     for u in uindices_symbmins]),
            "properties":
            ArrayAttr.from_list([StringAttr.from_str(str(p)) for p in properties]),
            "pragmas":
            ArrayAttr.from_list([StringAttr.from_str(str(p)) for p in pragmas]),
            "arg_name":
            arg
        }, regions=[Region.from_block_list([body])])
