from __future__ import annotations

from sympy import Mod
from typing import Iterable, Tuple, List, Annotated, Union
from dataclasses import dataclass

from xdsl.dialects.builtin import (IntegerType, StringAttr, ArrayAttr, OpAttr,
                                   ContainerOf, IndexType, Float16Type, Float32Type,
                                   Float64Type, AnyIntegerAttr, FloatAttr, f32, IntAttr)
from xdsl.dialects.arith import Constant
from xdsl.dialects.func import Return

from xdsl.dialects import arith, builtin, memref, llvm

from xdsl.irdl import irdl_op_definition, Operand, AnyOf, SingleBlockRegion, irdl_attr_definition, Attribute, ParametrizedAttribute
from xdsl.ir import MLContext, Operation, Block, Region, OpResult, SSAValue, Attribute, Dialect


signlessIntegerLike = ContainerOf(AnyOf([IntegerType, IndexType]))
floatingPointLike = ContainerOf(AnyOf([Float16Type, Float32Type, Float64Type]))

# TODO: remove
@dataclass
class IET:
    ctx: MLContext

    def __post_init__(self):
        # TODO add all operations
        self.ctx.register_op(Powi)
        self.ctx.register_op(Modi)
        self.ctx.register_op(Iteration)
        self.ctx.register_op(IterationWithSubIndices)
        self.ctx.register_op(Callable)
        self.ctx.register_op(Initialise)
        self.ctx.register_op(PointerCast)
        self.ctx.register_op(Statement)
        self.ctx.register_op(StructDecl)
        self.ctx.register_op(For)
        self.f32 = floatingPointLike

@irdl_attr_definition
class Profiler(ParametrizedAttribute):
    name = "iet.profiler"


# TODO: might be replacable by `llvm.LLVMStruct`?
@irdl_attr_definition
class Dataobj(ParametrizedAttribute):
    """
    The Dataobject type represents a pointer to a struct with this layout:

    struct dataobj
    {
        void *restrict data;
        unsigned long * size;
        unsigned long * npsize;
        unsigned long * dsize;
        int * hsize;
        int * hofs;
        int * oofs;
        void * dmap;
    };
    """
    name = "iet.dataobj"

    @staticmethod
    def get_llvm_struct_type():
        unsigned_long = builtin.IntegerType.from_width(32, builtin.Signedness.UNSIGNED)
        return llvm.LLVMStructType.from_type_list([
            llvm.LLVMPointerType.opaque(),              # data
            llvm.LLVMPointerType.typed(unsigned_long),  # size
            llvm.LLVMPointerType.typed(unsigned_long),  # npsize
            llvm.LLVMPointerType.typed(unsigned_long),  # dsize
            llvm.LLVMPointerType.typed(builtin.i32),    # hsize
            llvm.LLVMPointerType.typed(builtin.i32),    # hofs
            llvm.LLVMPointerType.typed(builtin.i32),    # oofs
            llvm.LLVMPointerType.opaque(),              # dmap
        ])


@irdl_op_definition
class Modi(Operation):
    name: str = "iet.modi"
    input1: Annotated[Operand, signlessIntegerLike]
    input2: Annotated[Operand, signlessIntegerLike]
    output: Annotated[OpResult, signlessIntegerLike]

    def verify_(self) -> None:
        if self.input1.typ != self.input2.typ or self.input2.typ != self.output.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(lhs: Union[Operation, SSAValue],
            rhs: Union[Operation, SSAValue]):
        res = Modi.build(operands=[lhs, rhs],
                         result_types=[IntegerType.build(32)])
        return res


@irdl_op_definition
class Powi(Operation):
    name: str = "iet.Powi"
    base: Annotated[Operand, signlessIntegerLike]
    exponent: Annotated[Operand, signlessIntegerLike]
    result: Annotated[OpResult, signlessIntegerLike]

    def verify_(self) -> None:
        if self.base.typ != self.exp.typ or self.exp.typ != self.result.typ:
            raise Exception("expect all input and output types to be equal")

    @staticmethod
    def get(base: Union[Operation, SSAValue],
            exponent: Union[Operation, SSAValue]):
        base = SSAValue.get(base)
        return Powi.build(operands=[base, exponent],
                          result_types=[f32])


@irdl_op_definition
class Initialise(Operation):
    name: str = "iet.initialise"

    lhs: Annotated[OpResult, Attribute]
    rhs: Annotated[Operand, Attribute]
    namet: OpAttr[StringAttr]

    @staticmethod
    def get(lhs: Union[Operation, SSAValue],
            rhs: Union[Operation, SSAValue],
            name: str):
        attributes = {"name": StringAttr(name)}
        res = Initialise.build(attributes=attributes,
                               operands=[rhs],
                               result_types=[lhs.typ])
        return res

@irdl_op_definition
class PointerCast(Operation):
    name: str = "iet.pointercast"
    statement: OpAttr[StringAttr]
    shape_indices: OpAttr[ArrayAttr[IntAttr]]
    
    arg: Annotated[Operand, Dataobj]  # TOOD: Make it Dataobj()!
    result: Annotated[OpResult, memref.MemRefType[Attribute]]

    @staticmethod
    def get(arg: SSAValue, statement, shape: tuple[int, ...], return_type: Attribute):
        return PointerCast.build(
            operands=[arg],
            attributes={
                "statement": StringAttr(str(statement)),
                "shape_indices": ArrayAttr([IntAttr(x) for x in shape])
            },
            result_types=[return_type])


@irdl_op_definition
class Statement(Operation):
    name: str = "iet.comment"
    statement: OpAttr[StringAttr]

    @staticmethod
    def get(statement: str):
        return Statement.build(
            operands=[],
            attributes={"statement": StringAttr(str(statement))},
            result_types=[])


@irdl_op_definition
class StructDecl(Operation):
    name: str = "iet.structdecl"
    id: OpAttr[StringAttr]
    fields: OpAttr[Attribute]
    declname: OpAttr[StringAttr]
    padbytes: OpAttr[Attribute]

    @staticmethod
    def get(name: str, fields: List[str], declname: str, padbytes: int = 0):
        padb = AnyIntegerAttr.from_int_and_width(padbytes, 32)
        return StructDecl.build(
            operands=[],
            attributes={
                "id":
                StringAttr(name),
                "fields":
                ArrayAttr([StringAttr(str(f)) for f in fields]),
                "declname":
                StringAttr(str(declname)),
                "padbytes":
                padb
            },
            result_types=[])


@irdl_op_definition
class Callable(Operation):
    name: str = "iet.callable"

    body: Region
    callable_name: OpAttr[StringAttr]
    parameters: OpAttr[Attribute]
    header_parameters: OpAttr[Attribute]
    types: OpAttr[Attribute]
    qualifiers: OpAttr[Attribute]
    retval: OpAttr[Attribute]
    prefix: OpAttr[Attribute]

    @staticmethod
    def get(name: str,
            parameters: Attribute,
            header_parameters: Attribute,
            types: Attribute,
            qualifiers: Attribute,
            retval: Attribute,
            prefix: Attribute,
            body: Block = []):
        return Callable.build(attributes={
            "callable_name": StringAttr(name),
            "parameters":
            ArrayAttr([StringAttr(p) for p in parameters]),
            "header_parameters":
            ArrayAttr([StringAttr(p) for p in header_parameters]),
            "types":
            ArrayAttr([StringAttr(p) for p in types]),
            "qualifiers":
            ArrayAttr([StringAttr(q) for q in qualifiers]),
            # It should be only one though
            "retval":
            StringAttr(retval),
            "prefix":
            StringAttr(prefix)
        }, regions=[Region.from_block_list([body])])


@irdl_op_definition
class Call(Operation):
    name: str = "iet.call"

    call_name: OpAttr[StringAttr]
    c_names: OpAttr[Attribute]
    c_typenames: OpAttr[Attribute]
    c_typeqs: OpAttr[Attribute]
    prefix: OpAttr[Attribute]
    ret_type: OpAttr[Attribute]

    @staticmethod
    def get(name: str,
            c_names: Attribute,
            c_typenames: Attribute,
            c_typeqs: Attribute,
            prefix: Attribute,
            ret_type: Attribute):
        return Call.build(attributes={
            "callable_name": StringAttr(name),
            "c_names":
            ArrayAttr([StringAttr(p) for p in c_names]),
            "c_typenames":
            ArrayAttr([StringAttr(p) for p in c_typenames]),
            "c_typeqs":
            ArrayAttr([StringAttr(p) for p in c_typeqs]),
            "prefix":
            StringAttr(prefix),
            # It should be only one though
            "ret_type":
            StringAttr(ret_type)
        })


# TODO: remove
@irdl_op_definition
class Iteration(Operation):
    name: str = "iet.iteration"

    body: Region
    arg_name: OpAttr[StringAttr]
    limits: OpAttr[Attribute]
    properties: OpAttr[Attribute]
    pragmas: OpAttr[Attribute]

    @staticmethod
    def get(properties: List[str | StringAttr],
            limits: Tuple[str, str, str],
            arg_name: str | StringAttr,
            body: Block,
            pragmas: List[str | StringAttr] = []):
        return Iteration.build(attributes={
            "limits":
            ArrayAttr([
                StringAttr(str(limits[0])),
                StringAttr(str(limits[1])),
                StringAttr(str(limits[2]))
            ]),
            "properties":
            ArrayAttr([StringAttr(str(p)) for p in properties]),
            "pragmas":
            ArrayAttr([StringAttr(str(p)) for p in pragmas]),
            "arg_name": StringAttr(str(arg_name))
        }, regions=[Region.from_block_list([body])])


# TODO: remove
@irdl_op_definition
class IterationWithSubIndices(Operation):
    name: str = "iet.iteration_with_subindices"

    arg_name: OpAttr[StringAttr]
    body: Region
    limits: OpAttr[Attribute]
    uindices_names: OpAttr[Attribute]
    uindices_symbmins_dividends: OpAttr[Attribute]
    uindices_symbmins_divisors: OpAttr[Attribute]
    properties: OpAttr[Attribute]
    pragmas: OpAttr[Attribute]

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
            ArrayAttr([
                StringAttr(str(limits[0])),
                StringAttr(str(limits[1])),
                StringAttr(str(limits[2]))
            ]),
            "uindices_names":
            ArrayAttr(
                [StringAttr(u) for u in uindices_names]),
            # TODO make a "ModAttr"??
            "uindices_symbmins_dividends":
            ArrayAttr([StringAttr(str(u.args[0]))
                       for u in uindices_symbmins]),
            "uindices_symbmins_divisors":
                ArrayAttr([StringAttr(str(u.args[1]))
                           for u in uindices_symbmins]),
            "properties":
            ArrayAttr([StringAttr(str(p)) for p in properties]),
            "pragmas":
            ArrayAttr([StringAttr(str(p)) for p in pragmas]),
            "arg_name":
            StringAttr(str(arg))
        }, regions=[Region.from_block_list([body])])



@irdl_op_definition
class For(Operation):
    name: str = "iet.for"

    lb: Annotated[Operand, IndexType]
    ub: Annotated[Operand, IndexType]
    step: Annotated[Operand, IndexType]

    body: SingleBlockRegion

    subindices: OpAttr[IntAttr]

    properties: OpAttr[ArrayAttr[builtin.StringAttr]]
    pragmas: OpAttr[ArrayAttr[builtin.StringAttr]]

    @property
    def block(self) -> Block:
        return self.body.blocks[0]
    
    @property
    def parallelism_property(self) -> str | None:
        """
        Return either "parallel" or "sequential" (or None),
        depending on the properties present
        """
        for attr in self.properties.data:
            if attr.data in ('parallel', 'sequential'):
                return attr.data
        return None

    @staticmethod
    def get(
        lb: SSAValue | Operation,
        ub: SSAValue | Operation,
        step: SSAValue | Operation,
        subindices: int = 0,
        properties: Iterable[str] = None,
        pragmas: Iterable[str] = None,
    ) -> For:
        if subindices is None:
            subindices = []
        if pragmas is None:
            pragmas = []

        body = Region.from_block_list([Block.from_arg_types(
            [builtin.i32] * (subindices + 1)
        )])

        return For.build(
            operands=[lb, ub, step],
            attributes={
                'subindices': IntAttr(subindices),
                'pragmas': ArrayAttr([StringAttr(pragma) for pragma in pragmas]),
                'properties': ArrayAttr([StringAttr(prop) for prop in properties]),
            },
            result_types=[],
            regions=[body],
        )


IET_SSA = Dialect([
    Statement,
    PointerCast,
    For,
    Call,
    Callable,
    StructDecl,
    Initialise,
    Modi,
    Powi
], [
    Profiler,
    Dataobj
])