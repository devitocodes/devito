from __future__ import annotations

from sympy import Mod
from typing import Iterable, Tuple, List, Union, Sequence
from dataclasses import dataclass

from xdsl.dialects.builtin import (IntegerType, StringAttr, ArrayAttr,
                                   ContainerOf, IndexType, Float16Type, Float32Type,
                                   Float64Type, AnyIntegerAttr, f32, IntAttr)

from xdsl.dialects import builtin, memref, llvm
from xdsl.dialects import stencil

from xdsl.irdl import (irdl_op_definition, operand_def, result_def, attr_def, region_def,
                       var_operand_def, Operand, AnyOf, irdl_attr_definition, Attribute,
                       ParametrizedAttribute, VarOperand, IRDLOperation)
from xdsl.ir import (MLContext, Block, Region, OpResult, SSAValue,
                     Dialect, Operation)


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
        # unsigned_long = builtin.IntegerType(32, builtin.Signedness.UNSIGNED)
        return llvm.LLVMStructType.from_type_list([
            llvm.LLVMPointerType.opaque(),              # data
            llvm.LLVMPointerType.typed(builtin.i32),  # size
            # llvm.LLVMPointerType.typed(builtin.i32),  # npsize
            # llvm.LLVMPointerType.typed(builtin.i32),  # dsize
            # llvm.LLVMPointerType.typed(builtin.i32),    # hsize
            # llvm.LLVMPointerType.typed(builtin.i32),    # hofs
            # llvm.LLVMPointerType.typed(builtin.i32),    # oofs
            # llvm.LLVMPointerType.opaque(),              # dmap
        ])


@irdl_op_definition
class Modi(IRDLOperation):
    name: str = "iet.modi"
    input1: Operand = operand_def(signlessIntegerLike)
    input2: Operand = operand_def(signlessIntegerLike)
    output: OpResult = result_def(signlessIntegerLike)

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
class Powi(IRDLOperation):
    name: str = "iet.Powi"
    base: Operand = operand_def(signlessIntegerLike)
    exponent: Operand = operand_def(signlessIntegerLike)
    result: OpResult = result_def(signlessIntegerLike)

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
class Initialise(IRDLOperation):
    name: str = "iet.initialise"

    lhs: OpResult = result_def(Attribute)
    rhs: Operand = operand_def(Attribute)
    namet: StringAttr = attr_def(StringAttr)

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
class PointerCast(IRDLOperation):
    name: str = "iet.pointercast"
    statement: StringAttr = attr_def(StringAttr)
    shape_indices: ArrayAttr[IntAttr] = attr_def(ArrayAttr[IntAttr])

    arg: Operand = operand_def(Dataobj)  # TOOD: Make it Dataobj()!
    result: OpResult = result_def(memref.MemRefType[Attribute])

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
class Statement(IRDLOperation):
    name: str = "iet.comment"
    statement: StringAttr = attr_def(StringAttr)

    @staticmethod
    def get(statement: str):
        return Statement.build(
            operands=[],
            attributes={"statement": StringAttr(str(statement))},
            result_types=[])


@irdl_op_definition
class StructDecl(IRDLOperation):
    name: str = "iet.structdecl"
    id: StringAttr = attr_def(StringAttr)
    fields: Attribute = attr_def(Attribute)
    declname: StringAttr = attr_def(StringAttr)
    padbytes: Attribute = attr_def(Attribute)

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
class Callable(IRDLOperation):
    name: str = "iet.callable"

    body: Region = region_def()
    callable_name: StringAttr = attr_def(StringAttr)
    parameters: Attribute = attr_def(Attribute)
    header_parameters: Attribute = attr_def(Attribute)
    types: Attribute = attr_def(Attribute)
    qualifiers: Attribute = attr_def(Attribute)
    retval: Attribute = attr_def(Attribute)
    prefix: Attribute = attr_def(Attribute)

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
        }, regions=[Region([body])])


@irdl_op_definition
class Call(IRDLOperation):
    name: str = "iet.call"

    call_name: StringAttr = attr_def(StringAttr)
    c_names: Attribute = attr_def(Attribute)
    c_typenames: Attribute = attr_def(Attribute)
    c_typeqs: Attribute = attr_def(Attribute)
    prefix: Attribute = attr_def(Attribute)
    ret_type: Attribute = attr_def(Attribute)

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
class Iteration(IRDLOperation):
    name: str = "iet.iteration"

    body: Region = region_def()
    arg_name: StringAttr = attr_def(StringAttr)
    limits: Attribute = attr_def(Attribute)
    properties: Attribute = attr_def(Attribute)
    pragmas: Attribute = attr_def(Attribute)

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
        }, regions=[Region([body])])


# TODO: remove
@irdl_op_definition
class IterationWithSubIndices(IRDLOperation):
    name: str = "iet.iteration_with_subindices"

    arg_name: StringAttr = attr_def(StringAttr)
    body: Region = region_def()
    limits: Attribute = attr_def(Attribute)
    uindices_names: Attribute = attr_def(Attribute)
    uindices_symbmins_dividends: Attribute = attr_def(Attribute)
    uindices_symbmins_divisors: Attribute = attr_def(Attribute)
    properties: Attribute = attr_def(Attribute)
    pragmas: Attribute = attr_def(Attribute)

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
        }, regions=[Region([body])])


@irdl_op_definition
class For(IRDLOperation):
    name: str = "iet.for"

    lb: Operand = operand_def(IndexType)
    ub: Operand = operand_def(IndexType)
    step: Operand = operand_def(IndexType)

    result: OpResult = result_def(IndexType)

    body: Region = region_def("single_block")

    subindices: IntAttr = attr_def(IntAttr)

    _properties: ArrayAttr[builtin.StringAttr] = attr_def(ArrayAttr[builtin.StringAttr])
    pragmas: ArrayAttr[builtin.StringAttr] = attr_def(ArrayAttr[builtin.StringAttr])

    def subindice_ssa_vals(self) -> tuple[SSAValue, ...]:
        return self.block.args[1:]

    @property
    def block(self) -> Block:
        return self.body.blocks[0]

    @property
    def parallelism_property(self) -> str | None:
        """
        Return either "parallel" or "sequential" (or None),
        depending on the properties present
        """
        for attr in self._properties.data:
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
        loop_var_name: str = 'time',
    ) -> For:
        if pragmas is None:
            pragmas = []

        body = Region([Block(
            arg_types=[builtin.IndexType()] * (subindices + 1)
        )])
        body.blocks[0].args[0].name_hint = loop_var_name
        for i in range(subindices):
            body.blocks[0].args[i+1].name_hint = f"{loop_var_name[0]}{i}"

        return For.build(
            operands=[lb, ub, step],
            attributes={
                'subindices': IntAttr(subindices),
                'pragmas': ArrayAttr([StringAttr(pragma) for pragma in pragmas]),
            },
            result_types=[IndexType()],
            regions=[body],
        )


@irdl_op_definition
class Stencil(IRDLOperation):
    """
    Represents a cluster of expressions that should then be translated to a stencil.
    """
    name = "devito.stencil"

    input_indices: VarOperand = var_operand_def(AnyIntegerAttr)
    output: Operand = operand_def(AnyIntegerAttr)

    shape: ArrayAttr[IntAttr] = attr_def(ArrayAttr[IntAttr])
    """
    shape without halo (int, int, int)
    """
    halo: ArrayAttr[ArrayAttr[IntAttr]] = attr_def(ArrayAttr[ArrayAttr[IntAttr]])
    """
    how far each dimension is expanded
    ((int, int), (int, int), (int, int))
    """

    field_type: Attribute = attr_def(Attribute)

    grid_name: StringAttr = attr_def(StringAttr)

    body: Region = region_def("single_block")

    @property
    def block(self):
        return self.body.blocks[0]

    @staticmethod
    def get(
        time_indices: Sequence[SSAValue | Operation],
        shape: Sequence[int],
        halo: Sequence[Sequence[int]],
        time_buffers: int,
        typ: Attribute,
        grid_name: str
    ) -> Stencil:
        assert len(halo) == len(shape)
        assert all(len(inner) == 2 for inner in halo)

        *inputs, output = time_indices
        assert len(time_indices) == time_buffers
        block = Block(
            arg_types=[
                stencil.TempType(len(shape), typ)
            ] * (time_buffers - 1))

        for block_arg, idx_arg in zip(block.args, reversed(inputs)):
            name = SSAValue.get(idx_arg).name_hint
            if name is None:
                continue
            block_arg.name_hint = f"{name}_buff"

        return Stencil.build(
            operands=[inputs, output],
            attributes={
                'shape': ArrayAttr(IntAttr(x) for x in shape),
                'halo': ArrayAttr(ArrayAttr(IntAttr(x) for x in inner) for inner in halo),
                'field_type': typ,
                'grid_name': StringAttr(grid_name)
            },
            regions=[Region([block])]
        )


@irdl_op_definition
class LoadSymbolic(IRDLOperation):
    name = "devito.load_symbolic"
    symbol_name: StringAttr = attr_def(StringAttr)

    result: OpResult = result_def()

    @staticmethod
    def get(name: str, typ: Attribute):
        op = LoadSymbolic.build(
            attributes={'symbol_name': StringAttr(name)},
            result_types=[typ],
        )
        op.result.name_hint = name
        return op


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

DEVITO_SSA = Dialect([
    Stencil,
    LoadSymbolic
], [
])
