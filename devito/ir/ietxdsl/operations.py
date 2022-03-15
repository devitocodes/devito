from xdsl.ir import *
from xdsl.irdl import *
from xdsl.util import *
from xdsl.dialects.builtin import *

#<<<<<<< HEAD
#
#@dataclass
#class IET:
#    ctx: MLContext
#
#    def __post_init__(self):
#        self.ctx.register_op(Constant)
#
#        self.ctx.register_op(Addi)
#
#        self.i32 = IntegerType.get(32)
#
#    # TODO make this generic in the type
#    def constant(self, val: int) -> Operation:
#        return Operation.with_result_types(
#            Constant, [], [self.i32],
#            attributes={"value": IntegerAttr.get(val, self.i32)})
#
#    # TODO these operations should support all kinds of integer types
#    def addi(self, x: OpOrBlockArg, y: OpOrBlockArg) -> Operation:
#        return Operation.with_result_types(
#            Addi, [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})
#
#    def idx(self, array: OpOrBlockArg, index: OpOrBlockArg):
#        return Operation.with_result_types(
#            Idx, [get_ssa_value(array), get_ssa_value(index)], [self.i32], {})
#
#    @staticmethod
#    def assign(lhs: OpOrBlockArg, rhs: OpOrBlockArg):
#        return Operation.with_result_types(
#            Assign, [get_ssa_value(lhs), get_ssa_value(rhs)], [], {})
#
#    @staticmethod
#    def callable(name: str, params: List[str], body: Block):
#        op = Operation.with_result_types(Callable, [], [],
#                                         attributes={
#                                             "callable_name": StringAttr.get(name),
#                                             "parameters": ArrayAttr.get([StringAttr.get(p) for p in params]),
#                                         })
#        r = Region()
#        r.add_block(body)
#        op.add_region(r)
#        return op
#
#    @staticmethod
#    def iteration(properties: List[str], limits: Tuple[str, str, str], body: Block):
#        op = Operation.with_result_types(Iteration, [], [],
#                                         attributes={
#                                             "limits": ArrayAttr.get([StringAttr.get(limits[0]),
#                                                                      StringAttr.get(limits[1]),
#                                                                      StringAttr.get(limits[0])]),
#                                             "properties": ArrayAttr.get([StringAttr.get(p) for p in properties])
#                                         })
#        r = Region()
#        r.add_block(body)
#        op.add_region(r)
#        return op
#
#
#=======
#>>>>>>> 976a2532f (Update to xdsl 0.3.0)
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
                            result_types=[IntegerType.build(32)])
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

#<<<<<<< HEAD
#
#@irdl_op_definition
#class Idx:
#    name: str = "iet.idx"
#    array = OperandDef(IntegerType)
#    index = OperandDef(IntegerType)
#    output = ResultDef(IntegerType)
#
#
#@irdl_op_definition
#class Assign:
#    name: str = "iet.assign"
#    lhs = OperandDef(IntegerType)
#    rhs = OperandDef(IntegerType)
#
#
#@irdl_op_definition
#class Callable:
#    name: str = "iet.callable"
#    callable_name = AttributeDef(StringAttr)
#    parameters = AttributeDef(ArrayAttr)
#    body = RegionDef()
#
#
#@irdl_op_definition
#class Iteration:
#    name: str = "iet.iteration"
#    body = RegionDef()
#    limits = AttributeDef(ArrayAttr)
#    properties = AttributeDef(ArrayAttr)
#=======

#@dataclass
#class IET:
#    ctx: MLContext
#
#    def __post_init__(self):
#        self.ctx.register_op(Constant)
#
#        self.ctx.register_op(Addi)
#
#        self.i32 = IntegerType.get(32)
#
#    # TODO make this generic in the type
#    def constant(self, val: int) -> Operation:
#        return Operation.with_result_types(
#            Constant, [], [self.i32],
#            attributes={"value": IntegerAttr.get(val, self.i32)})
#
#    # TODO these operations should support all kinds of integer types
#    #def addi(self, x, y) -> Operation:
#    #    return Operation.with_result_types(
#    #        Addi, [get_ssa_value(x), get_ssa_value(y)], [self.i32], {})


#
#
