from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.builtin import *


@dataclass
class IET:
    ctx: MLContext

    def __post_init__(self):
        # TODO add all operations
        self.ctx.register_op(Constant)
        self.ctx.register_op(Addi)
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
class Callable(Operation):
    name: str = "iet.callable"
    callable_name = AttributeDef(StringAttr)
    parameters = AttributeDef(ArrayAttr)
    body = RegionDef()

    @staticmethod
    def get(name: str, params: List[str], body: Block):
        return Callable.build(attributes={
            "callable_name":
            StringAttr.from_str(name),
            "parameters":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in params])
        },
                              regions=[Region.from_block_list([body])])


@irdl_op_definition
class Iteration(Operation):
    name: str = "iet.iteration"
    body = RegionDef()
    limits = AttributeDef(ArrayAttr)
    properties = AttributeDef(ArrayAttr)

    @staticmethod
    def get(properties: List[str], limits: Tuple[str, str, str], body: Block):
        return Iteration.build(attributes={
            "limits":
            ArrayAttr.from_list([
                StringAttr.from_str(limits[0]),
                StringAttr.from_str(limits[1]),
                StringAttr.from_str(limits[2])
            ]),
            "properties":
            ArrayAttr.from_list([StringAttr.from_str(p) for p in properties])
        },
                               regions=[Region.from_block_list([body])])
