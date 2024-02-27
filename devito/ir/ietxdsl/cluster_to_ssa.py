# ------------- General imports -------------#

from typing import Any, Iterable
from dataclasses import dataclass, field
from sympy import Add, Expr, Float, Indexed, Integer, Mod, Mul, Pow, Symbol
from devito.tools.data_structures import OrderedSet
from devito.types.dense import DiscreteFunction
from devito.types.equation import Eq

# ------------- xdsl imports -------------#
from xdsl.dialects import (arith, builtin, func, memref, scf,
                           stencil, gpu, llvm)
from xdsl.dialects.experimental import math
from xdsl.ir import Block, Operation, OpResult, Region, SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.builder import ImplicitBuilder

# ------------- devito imports -------------#
from devito import Grid, SteppingDimension
from devito.ir.equations import LoweredEq
from devito.symbolics import retrieve_indexed, retrieve_function_carriers
from devito.logger import perf

# ------------- devito-xdsl SSA imports -------------#
from devito.ir.ietxdsl import iet_ssa
from devito.ir.ietxdsl.utils import is_int, is_float
import numpy as np

dtypes_to_xdsltypes = {
    np.float32: builtin.f32,
    np.float64: builtin.f64,
    np.int32: builtin.i32,
    np.int64: builtin.i64,
}

# flake8: noqa

def field_from_function(f: DiscreteFunction) -> stencil.FieldType:
    halo = [f.halo[d] for d in f.grid.dimensions]
    shape = f.grid.shape
    bounds = [(-h[0], s+h[1]) for h, s in zip(halo, shape)]
    return stencil.FieldType(bounds, element_type=dtypes_to_xdsltypes[f.dtype])


class ExtractDevitoStencilConversion:
    """
    Lower Devito equations to the stencil dialect
    """

    eqs: list[LoweredEq]
    block: Block
    loaded_values: dict[tuple[int, ...], SSAValue]
    time_offs: int

    def _convert_eq(self, eq: LoweredEq):
        # Convert a Devito equation to a func.func op
        # an equation may look like this:
        #  u[x+1,y+1,z] = (u[x,y,z+1] + u[x+2,y+2,z+1]) / 2
        if isinstance(eq.lhs, Symbol):
            return xdsl_func.FuncOp.external(eq.lhs.name, [], [builtin.i32])
        assert isinstance(eq.lhs, Indexed)

        # Get the left hand side, called "output function" here because it tells us
        # Where to write the results of each step.
        output_function = eq.lhs.function
        # Get its grid: contains all necessary discretization information
        # (Grid size, halo width, ...)
        grid: Grid = output_function.grid
        # Get the stepping dimension. It's usually time, and usually the first one.
        # Getting it here; more readable and less input assumptions :)
        step_dim = grid.stepping_dim

        # Get all functions used in the equation
        functions = OrderedSet(*(f.function for f in retrieve_function_carriers(eq)))
        # We identify time buffers by their function and positive time offset.
        # e.g., u(t+2, x-1, y) would be indentified as (u, 2)
        # NB. We map time offsets to positive with simple modulo arithmetic: if u has a
        # a time size of 3, then u(t-2, ...) is identified as u(t, 1).

        # We store a list of those here to help the following steps.
        self.time_buffers = [(f, i) for f in functions for i in range(f.time_size)]

        # Also, store the output function's time buffer of this equation
        output_time_offset = (eq.lhs.indices[step_dim] - step_dim) % eq.lhs.function.time_size
        self.out_time_buffer = (output_function, output_time_offset)

        # For each used time_buffer, define a stencil.field type for the function .
        fields_types = [field_from_function(f) for (f, _) in self.time_buffers]

        # Create a function with the fields as arguments
        xdsl_func = func.FuncOp("apply_kernel", (fields_types, []))

        # Define nice argument names to try and stay sane while debugging
        # And store in self.function_args a mapping from time_buffers to their
        # corresponding function arguments.
        self.function_args = {}
        for i, (f, t) in enumerate(self.time_buffers):
            xdsl_func.body.block.args[i].name_hint = f"{f.name}_vec_{t}"
            self.function_args[(f,t)] = xdsl_func.body.block.args[i]

        with ImplicitBuilder(xdsl_func.body.block):
            self._build_step_loop(step_dim, eq)
            # func wants a return
            func.Return()

        return xdsl_func

    def _visit_math_nodes(self, dim: SteppingDimension, node: Expr, output_indexed:Indexed) -> SSAValue:
        # Handle Indexeds
        if isinstance(node, Indexed):
            space_offsets = [node.indices[d] - output_indexed.indices[d] for d in node.function.space_dimensions]
            # import pdb; pdb.set_trace()
            temp = self.apply_temps[(node.function, (node.indices[dim] - dim) % node.function.time_size)]
            access = stencil.AccessOp.get(temp, space_offsets)
            return access.res
        # Handle Integers
        elif isinstance(node, Integer):
            cst = arith.Constant.from_int_and_width(int(node), builtin.i64)
            return cst.result
        # Handle Floats
        elif isinstance(node, Float):
            cst = arith.Constant(builtin.FloatAttr(float(node), builtin.f32))
            return cst.result
        # Handle Symbols
        elif isinstance(node, Symbol):
            symb = iet_ssa.LoadSymbolic.get(node.name, builtin.f32)
            return symb.result     
        # Handle Add Mul
        elif isinstance(node, (Add, Mul)):
            args = [self._visit_math_nodes(dim, arg, output_indexed) for arg in node.args]
            # add casts when necessary
            # get first element out, store the rest in args
            # this makes the reduction easier
            carry, *args = self._ensure_same_type(*args)
            # select the correct op from arith.addi, arith.addf, arith.muli, arith.mulf
            if isinstance(carry.type, builtin.IntegerType):
                op_cls = arith.Addi if isinstance(node, Add) else arith.Muli
            elif isinstance(carry.type, builtin.Float32Type):
                op_cls = arith.Addf if isinstance(node, Add) else arith.Mulf
            else:
                raise("Add support for another type")
            for arg in args:
                op = op_cls(carry, arg)
                carry = op.result
            return carry
        # Handle Pow
        elif isinstance(node, Pow):
            args = [self._visit_math_nodes(dim, arg, output_indexed) for arg in node.args]
            assert len(args) == 2, "can't pow with != 2 args!"
            base, ex = args
            if is_int(base):
                if is_int(ex):
                    op_cls = math.IpowIOP
                else:
                    raise ValueError("no IPowFOp yet!")
            elif is_float(base):
                if is_float(ex):
                    op_cls = math.PowFOp
                elif is_int(ex):
                    op_cls = math.FPowIOp
                else:
                    raise ValueError("Expected float or int as pow args!")
            else:
                raise ValueError("Expected float or int as pow args!")

            op = op_cls(base, ex)
            return op.result
        # Handle Mod
        elif isinstance(node, Mod):
            raise NotImplementedError("Go away, no mod here. >:(")
        else:
            raise NotImplementedError(f"Unknown math: {node}", node)

    def _build_step_body(self, dim: SteppingDimension, eq:LoweredEq) -> None:
        loop_temps = {
            (f, t): stencil.LoadOp.get(a).res
            for (f, t), a in self.block_args.items()
            if (f, t) != self.out_time_buffer
        }
        for (f,t), a in loop_temps.items():
            a.name_hint = f"{f.name}_t{t}_temp"        

        output_function = self.out_time_buffer[0]
        shape = output_function.grid.shape_local
        apply = stencil.ApplyOp.get(
            loop_temps.values(),
            Block(arg_types=[a.type for a in loop_temps.values()]),
            result_types=[stencil.TempType(len(shape), element_type=dtypes_to_xdsltypes[output_function.dtype])]
        )

        #Give names to stencil.apply's block arguments
        for apply_arg, apply_op in zip(apply.region.block.args, apply.operands):
            # Just reuse the corresponding operand name
            # i.e. %v_t1_temp -> %v_t1_blk
            apply_arg.name_hint = apply_op.name_hint[:-5]+"_blk"

        self.apply_temps = {k:v for k,v in zip(loop_temps.keys(), apply.region.block.args)}

        with ImplicitBuilder(apply.region.block):
            stencil.ReturnOp.get([self._visit_math_nodes(dim, eq.rhs, eq.lhs)])
        # TODO Think about multiple outputs
        stencil.StoreOp.get(
            apply.res[0],
            self.block_args[self.out_time_buffer],
            stencil.IndexAttr.get(*([0] * len(shape))),
            stencil.IndexAttr.get(*shape),
        )

    def _build_step_loop(
        self,
        dim: SteppingDimension,
        eq: LoweredEq,
    ) -> scf.For:
        # Bounds and step boilerpalte
        lb = iet_ssa.LoadSymbolic.get(dim.symbolic_min._C_name, builtin.IndexType())
        ub = iet_ssa.LoadSymbolic.get(dim.symbolic_max._C_name, builtin.IndexType())
        one = arith.Constant.from_int_and_width(1, builtin.IndexType())
        try:
            step = arith.Constant.from_int_and_width(
                int(dim.symbolic_incr), builtin.IndexType()
            )
            step.result.name_hint = "step"
        except:
            raise ValueError("step must be int!")

        iter_args = list(self.function_args.values())
        # Create the for loop
        loop = scf.For(lb, arith.Addi(ub, one), step, iter_args, Block(arg_types=[builtin.IndexType(), *(a.type for a in iter_args)]))
        loop.body.block.args[0].name_hint = "time"

        self.block_args = {(f,t) : loop.body.block.args[1+i] for i, (f,t) in enumerate(self.time_buffers)}
        for ((f,t), arg) in self.block_args.items():
            arg.name_hint = f"{f.name}_t{t}"

        with ImplicitBuilder(loop.body.block):
            self._build_step_body(dim, eq)
            # Swap buffers through scf.yield
            yield_args = [self.block_args[(f, (t+1)%f.time_size)] for (f, t) in self.block_args.keys()]
            scf.Yield(*yield_args)

    def convert(self, eqs: Iterable[Eq]) -> builtin.ModuleOp:
        # Lower equations to a ModuleOp
        return builtin.ModuleOp(
            Region([Block([self._convert_eq(eq) for eq in eqs])])
        )

    def _ensure_same_type(self, *vals: SSAValue):
        if all(isinstance(val.type, builtin.IntegerAttr) for val in vals):
            return vals
        if all(is_float(val) for val in vals):
            return vals
        # not everything homogeneous
        processed = []
        for val in vals:
            if is_float(val):
                processed.append(val)
                continue
            # if the val is the result of a arith.constant with no uses,
            # we change the type of the arith.constant to our desired type
            if (
                isinstance(val, OpResult)
                and isinstance(val.op, arith.Constant)
                and val.uses == 0
            ):
                val.type = builtin.f32
                val.op.attributes["value"] = builtin.FloatAttr(
                    float(val.op.value.value.data), builtin.f32
                )
                processed.append(val)
                continue
            # insert an integer to float cast op
            conv = arith.SIToFPOp(val, builtin.f32)
            processed.append(conv.result)
        return processed

# -------------------------------------------------------- ####
#                                                          ####
#           devito.stencil  ---> stencil dialect           ####
#                                                          ####
# -------------------------------------------------------- ####

@dataclass
class WrapFunctionWithTransfers(RewritePattern):
    func_name: str
    done: bool = field(default=False)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.sym_name.data != self.func_name or self.done:
            return
        self.done = True

        op.sym_name = builtin.StringAttr("gpu_kernel")
        print("Doing GPU STUFF")
        # GPU STUFF
        wrapper = func.FuncOp(self.func_name, op.function_type, Region(Block([func.Return()], arg_types=op.function_type.inputs)))
        body = wrapper.body.block
        wrapper.body.block.insert_op_before(func.Call("gpu_kernel", body.args, []), body.last_op)
        for arg in wrapper.args:
            shapetype = arg.type
            if isinstance(shapetype, stencil.FieldType):
                memref_type = memref.MemRefType.from_element_type_and_shape(shapetype.get_element_type(), shapetype.get_shape())
                alloc = gpu.AllocOp(memref.MemRefType.from_element_type_and_shape(shapetype.get_element_type(), shapetype.get_shape()))
                outcast = builtin.UnrealizedConversionCastOp.get(alloc, shapetype)
                arg.replace_by(outcast.results[0])
                incast = builtin.UnrealizedConversionCastOp.get(arg, memref_type)
                copy = gpu.MemcpyOp(source=incast, destination=alloc)
                body.insert_ops_before([alloc, outcast, incast, copy], body.ops.first)

                copy_out = gpu.MemcpyOp(source=alloc, destination=incast)
                dealloc = gpu.DeallocOp(alloc)
                body.insert_ops_before([copy_out, dealloc], body.ops.last)
        rewriter.insert_op_after_matched_op(wrapper)

@dataclass
class MakeFunctionTimed(RewritePattern):
    """
    Populate the section0 devito timer with the total runtime of the function
    """
    func_name: str
    seen_ops: set[func.Func] = field(default_factory=set)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.sym_name.data != self.func_name or op in self.seen_ops:
            return

        # only apply once
        self.seen_ops.add(op)

        rewriter.insert_op_at_start([
            t0 := func.Call('timer_start', [], [builtin.f64])
        ], op.body.block)

        ret = op.get_return_op()
        assert ret is not None

        rewriter.insert_op_before([
            timers := iet_ssa.LoadSymbolic.get('timers', llvm.LLVMPointerType.opaque()),
            t1 := func.Call('timer_end', [t0], [builtin.f64]),
            llvm.StoreOp(t1, timers),
        ], ret)

        rewriter.insert_op_after_matched_op([
            func.FuncOp.external('timer_start', [], [builtin.f64]),
            func.FuncOp.external('timer_end', [builtin.f64], [builtin.f64])
        ])


def get_containing_func(op: Operation) -> func.FuncOp | None:
    while op is not None and not isinstance(op, func.FuncOp):
        op = op.parent_op()
    return op


@dataclass
class _InsertSymbolicConstants(RewritePattern):
    known_symbols: dict[str, int | float]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.LoadSymbolic, rewriter: PatternRewriter, /):
        symb_name = op.symbol_name.data
        if symb_name not in self.known_symbols:
            return

        if op.result.type in (builtin.f32, builtin.f64):
            rewriter.replace_matched_op(
                arith.Constant(builtin.FloatAttr
                    (
                        float(self.known_symbols[symb_name]), op.result.type
                    )
                )
            )
            return

        if isinstance(op.result.type, (builtin.IntegerType, builtin.IndexType)):
            rewriter.replace_matched_op(
                arith.Constant.from_int_and_width(
                    int(self.known_symbols[symb_name]), op.result.type
                )
            )


class _LowerLoadSymbolidToFuncArgs(RewritePattern):

    func_to_args: dict[func.FuncOp, dict[str, SSAValue]]

    def __init__(self):
        from collections import defaultdict

        self.func_to_args = defaultdict(dict)

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.LoadSymbolic, rewriter: PatternRewriter, /):
        parent: func.FuncOp | None = get_containing_func(op)
        assert parent is not None
        args = list(parent.body.block.args)
        symb_name = op.symbol_name.data

        try:
            arg_index = [a.name_hint for a in args].index(symb_name)
        except ValueError:
            arg_index = -1

        if arg_index == -1:
            body = parent.body.block
            args.append(body.insert_arg(op.result.type, len(body.args)))
            arg_index = len(args) - 1
            

        op.result.replace_by(args[arg_index])

        rewriter.erase_matched_op()
        parent.update_function_type()


def convert_devito_stencil_to_xdsl_stencil(module, timed: bool = True):
    """
    TODO: Add docstring
    """

    if timed:
        grpa = GreedyRewritePatternApplier([MakeFunctionTimed('apply_kernel')])
        PatternRewriteWalker(grpa, walk_regions_first=True).rewrite_module(module)


def finalize_module_with_globals(module: builtin.ModuleOp, known_symbols: dict[str, Any],
                                 gpu_boilerplate):
    """
    TODO: Add docstring
    """
    patterns = [
        _InsertSymbolicConstants(known_symbols),
        _LowerLoadSymbolidToFuncArgs(),
    ]
    rewriter = GreedyRewritePatternApplier(patterns)
    PatternRewriteWalker(rewriter).rewrite_module(module)

    # GPU boilerplate
    if gpu_boilerplate:
        walker = PatternRewriteWalker(GreedyRewritePatternApplier([WrapFunctionWithTransfers('apply_kernel')]))
        walker.rewrite_module(module)
