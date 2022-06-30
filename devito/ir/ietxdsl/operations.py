from sympy import Mod
from xdsl.dialects.builtin import *


@dataclass
class IET:
    ctx: MLContext

    def __post_init__(self):
        # TODO add all operations
        self.ctx.register_op(Constant)
        self.ctx.register_op(Addi)
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
        self.i32 = IntegerType.from_width(32)


@irdl_op_definition
class Constant(Operation):
    name: str = "iet.constant"
    output = ResultDef(AnyAttr())
    value = AttributeDef(AnyAttr())

    # TODO verify that the output and value type are equal
    def verify_(self) -> None:
        # TODO how to force the attr to have a type? and how to query it?
        pass

    @staticmethod
    def get(value):
        attr = IntegerAttr.from_int_and_width(value, 32)
        res = Constant.build(operands=[],
                             attributes={"value": attr},
                             result_types=[IntegerType.from_width(32)])
        return res


@irdl_op_definition
class Addi(Operation):
    name: str = "iet.addi"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    # TODO replace with trait
    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(lhs, rhs):
        res = Addi.build(operands=[lhs, rhs],
                         result_types=[IntegerType.build(32)])
        return res


@irdl_op_definition
class Modi(Operation):
    name: str = "iet.modi"
    input1 = OperandDef(IntegerType)
    input2 = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(lhs, rhs):
        res = Modi.build(operands=[lhs, rhs],
                         result_types=[IntegerType.build(32)])
        return res


@irdl_op_definition
class Idx(Operation):
    name: str = "iet.idx"
    array = OperandDef(IntegerType)
    index = OperandDef(IntegerType)
    output = ResultDef(IntegerType)

    @staticmethod
    def get(array, index):
        return Idx.build(operands=[array, index],
                         result_types=[IntegerType.build(32)])


@irdl_op_definition
class Assign(Operation):
    name: str = "iet.assign"
    lhs = OperandDef(IntegerType)
    rhs = OperandDef(IntegerType)


@irdl_op_definition
class PointerCast(Operation):
    name: str = "iet.pointercast"
    statement = AttributeDef(StringAttr)

    @staticmethod
    def get(statement):
        return PointerCast.build(
            operands=[],
            attributes={"statement": StringAttr.from_str(statement)},
            result_types=[])


@irdl_op_definition
class Statement(Operation):
    name: str = "iet.comment"
    statement = AttributeDef(StringAttr)

    @staticmethod
    def get(statement):
        return Statement.build(
            operands=[],
            attributes={"statement": StringAttr.from_str(statement)},
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
                ArrayAttr.from_list([StringAttr.from_str(f) for f in fields]),
                "declname":
                StringAttr.from_str(declname),
                "padbytes":
                padb
            },
            result_types=[])


@irdl_op_definition
class Initialise(Operation):
    name: str = "iet.initialise"
    id = AttributeDef(StringAttr)
    rhs = OperandDef(IntegerType)
    lhs = ResultDef(IntegerType)

    @staticmethod
    def get(lhs, rhs, id):
        res = Initialise.build(attributes={"id": StringAttr.from_str(id)},
                               operands=[lhs],
                               result_types=[IntegerType.build(32)])
        return res


@irdl_op_definition
class Callable(Operation):
    name: str = "iet.callable"
    callable_name = AttributeDef(StringAttr)
    parameters = AttributeDef(ArrayAttr)
    header_parameters = AttributeDef(ArrayAttr)
    types = AttributeDef(ArrayAttr)
    body = RegionDef()

    @staticmethod
    def get(name: str, params: List[str], header_params: List[str],
            types: List[str], body: Block):
        return Callable.build(attributes={
            "callable_name":
            StringAttr.from_str(name),
            "parameters":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in params]),
            "header_parameters":
            ArrayAttr.from_list(
                [StringAttr.from_str(p) for p in header_params]),
            "types":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in types])
        },
            regions=[Region.from_block_list([body])])


@irdl_op_definition
class Iteration(Operation):
    name: str = "iet.iteration"
    arg_name = AttributeDef(StringAttr)
    body = RegionDef()
    limits = AttributeDef(ArrayAttr)
    properties = AttributeDef(ArrayAttr)
    pragmas = AttributeDef(ArrayAttr)

    @staticmethod
    def get(properties: List[str],
            limits: Tuple[str, str, str],
            arg: str,
            body: Block,
            pragmas: List[str] = []):
        return Iteration.build(attributes={
            "limits":
            ArrayAttr.from_list([
                StringAttr.from_str(limits[0]),
                StringAttr.from_str(limits[1]),
                StringAttr.from_str(limits[2])
            ]),
            "properties":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in properties]),
            "pragmas":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in pragmas]),
            "arg_name":
            arg
        },
            regions=[Region.from_block_list([body])])


@irdl_op_definition
class IterationWithSubIndices(Operation):
    name: str = "iet.iteration_with_subindices"
    arg_name = AttributeDef(StringAttr)
    body = RegionDef()
    limits = AttributeDef(ArrayAttr)
    uindices_names = AttributeDef(ArrayAttr)
    uindices_symbmins = AttributeDef(ArrayAttr)
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
                StringAttr.from_str(limits[0]),
                StringAttr.from_str(limits[1]),
                StringAttr.from_str(limits[2])
            ]),
            "uindices_names":
            ArrayAttr.from_list(
                [StringAttr.from_str(u) for u in uindices_names]),
            "uindices_symbmins":
            ArrayAttr.from_list([u for u in uindices_symbmins]),
            "properties":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in properties]),
            "pragmas":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in pragmas]),
            "arg_name":
            arg
        },
            regions=[
            Region.from_block_list([body])
        ])
