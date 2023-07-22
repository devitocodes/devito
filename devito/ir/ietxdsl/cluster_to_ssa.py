# ------------- devito import -------------#

from sympy import Add, Expr, Float, Indexed, Integer, Mod, Mul, Pow, Symbol
from xdsl.dialects import arith, builtin, func, memref, scf, stencil
from xdsl.dialects.experimental import dmp, math
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue
from typing import Any

from devito import Grid, SteppingDimension
from devito.ir.clusters import Cluster
from devito.ir.equations import LoweredEq
from devito.ir.ietxdsl import iet_ssa
from devito.ir.ietxdsl.ietxdsl_functions import dtypes_to_xdsltypes
from devito.symbolics import retrieve_indexed
from devito.logger import perf

# ----------- devito ssa import -----------#


# -------------- xdsl import --------------#


default_int = builtin.i64


class ExtractDevitoStencilConversion:
    """
    Lower Devito equations to the stencil dialect
    """

    eqs: list[LoweredEq]
    block: Block
    loaded_values: dict[tuple[int, ...], SSAValue]
    time_offs: int

    def __init__(self, eqs: list[LoweredEq]) -> None:
        self.eqs = eqs
        self.loaded_values = dict()
        self.time_offs = 0

    def _convert_eq(self, eq: LoweredEq):
        # Convert a Devito equation to a func.func op
        # an equation may look like this:
        #  u[x+1,y+1,z] = (u[x,y,z+1] + u[x+2,y+2,z+1]) / 2
        if isinstance(eq.lhs, Symbol):
            return func.FuncOp.external(eq.lhs.name, [], [builtin.i32])
        assert isinstance(eq.lhs, Indexed)

        outermost_block = Block([])
        self.block = outermost_block

        # u(t, x, y)
        function = eq.lhs.function
        mlir_type = dtypes_to_xdsltypes[function.dtype]
        grid: Grid = function.grid
        # get the halo of the space dimensions only e.g [(2, 2), (2, 2)] for the 2d case
        # Do not forget the issue with Devito adding an extra point!
        # (for derivative regions)
        halo = [function.halo[function.dimensions.index(d)] for d in grid.dimensions]

        # shift all time values so that for all accesses at t + n, n>=0.
        self.time_offs = min(
            int(idx.indices[0] - grid.stepping_dim) for idx in retrieve_indexed(eq)
        )
        # calculate the actual size of our time dimension
        actual_time_size = (
            max(int(idx.indices[0] - grid.stepping_dim) for idx in retrieve_indexed(eq))
            - self.time_offs
            + 1
        )

        # build the for loop
        perf("Build Time Loop")
        loop = self._build_iet_for(grid.stepping_dim, ["sequential"], actual_time_size)

        # build stencil
        perf("Init out stencil Op")
        stencil_op = iet_ssa.Stencil.get(
            loop.subindice_ssa_vals(),
            grid.shape_local,
            halo,
            actual_time_size,
            mlir_type,
            eq.lhs.function._C_name,
        )
        self.block.add_op(stencil_op)
        self.block = stencil_op.block

        # dims -> ssa vals
        perf("Apply time offsets")
        time_offset_to_field: dict[str, SSAValue] = {
            i: stencil_op.block.args[i] for i in range(actual_time_size - 1)
        }

        # reset loaded values
        self.loaded_values: dict[tuple[int, ...], SSAValue] = dict()

        # add all loads into the stencil
        perf("Add stencil Loads")
        self._add_access_ops(retrieve_indexed(eq.rhs), time_offset_to_field)

        # add math
        perf("Visiting math equations")
        rhs_result = self._visit_math_nodes(eq.rhs)

        # emit return
        offsets = _get_dim_offsets(eq.lhs, self.time_offs)
        assert (
            offsets[0] == actual_time_size - 1
        ), "result should be written to last time buffer"
        assert all(
            o == 0 for o in offsets[1:]
        ), f"can only write to offset [0,0,0], given {offsets[1:]}"

        self.block.add_op(stencil.ReturnOp.get([rhs_result]))
        outermost_block.add_op(func.Return())

        return func.FuncOp.from_region(
            "apply_kernel", [], [], Region([outermost_block])
        )

    def _visit_math_nodes(self, node: Expr) -> SSAValue:
        if isinstance(node, Indexed):
            offsets = _get_dim_offsets(node, self.time_offs)
            return self.loaded_values[offsets]
        if isinstance(node, Integer):
            cst = arith.Constant.from_int_and_width(int(node), builtin.i64)
            self.block.add_op(cst)
            return cst.result
        if isinstance(node, Float):
            cst = arith.Constant.from_float_and_width(float(node), builtin.f32)
            self.block.add_op(cst)
            return cst.result
        # if isinstance(math, Constant):
        #    symb = iet_ssa.LoadSymbolic.get(math.name, dtypes_to_xdsltypes[math.dtype])
        #    self.block.add_op(symb)
        #    return symb.result
        if isinstance(node, Symbol):
            symb = iet_ssa.LoadSymbolic.get(node.name, builtin.f32)
            self.block.add_op(symb)
            return symb.result

        # handle all of the math
        if not isinstance(node, (Add, Mul, Pow, Mod)):
            raise ValueError(f"Unknown math: {node}", node)

        args = [self._visit_math_nodes(arg) for arg in node.args]

        # make sure all args are the same type:
        if isinstance(node, (Add, Mul)):
            # add casts when necessary
            # get first element out, store the rest in args
            # this makes the reduction easier
            carry, *args = self._ensure_same_type(*args)
            # select the correct op from arith.addi, arith.addf, arith.muli, arith.mulf
            if isinstance(carry.type, builtin.IntegerType):
                op_cls = arith.Addi if isinstance(node, Add) else arith.Muli
            else:
                op_cls = arith.Addf if isinstance(node, Add) else arith.Mulf

            for arg in args:
                op = op_cls(carry, arg)
                self.block.add_op(op)
                carry = op.result
            return carry

        if isinstance(node, Pow):
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

            op = op_cls.get(base, ex)
            self.block.add_op(op)
            return op.result

        if isinstance(node, Mod):
            raise ValueError("Go away, no mod here. >:(")

        raise ValueError("Unknown math!")

    def _add_access_ops(
        self, reads: list[Indexed], time_offset_to_field: dict[int, SSAValue]
    ):
        for read in reads:
            """
            AccessOp:
                name: str = "stencil.access"
                temp: Operand = operand_def(TempType)
                offset: IndexAttr = attr_def(IndexAttr)
                res: OpResult = result_def(Attribute)
            """
            # get the compile time constant offsets for this read
            offsets = _get_dim_offsets(read, self.time_offs)
            if offsets in self.loaded_values:
                continue

            # assume time dimension is first dimension
            t_offset = offsets[0]
            space_offsets = offsets[1:]

            # use time offset to find correct stencil field to read
            field = time_offset_to_field[t_offset]

            # use space offsets in the field
            access_op = stencil.AccessOp.get(field, space_offsets)
            self.block.add_op(access_op)
            # cache the resulting ssa value for later use
            # by the offsets (same offset = same value)
            self.loaded_values[offsets] = access_op.res

    def _build_iet_for(
        self, dim: SteppingDimension, props: list[str], subindices: int
    ) -> iet_ssa.For:
        # Build a for loop in the custom iet_ssa.py using lower-upper bound and step
        lb = iet_ssa.LoadSymbolic.get(dim.symbolic_min._C_name, builtin.IndexType())
        ub = iet_ssa.LoadSymbolic.get(dim.symbolic_max._C_name, builtin.IndexType())
        try:
            step = arith.Constant.from_int_and_width(
                int(dim.symbolic_incr), builtin.IndexType()
            )
            step.result.name_hint = "step"
        except:
            raise ValueError("step must be int!")

        loop = iet_ssa.For.get(lb, ub, step, subindices, props)

        self.block.add_ops([lb, ub, step, loop])
        self.block = loop.block

        return loop

    def convert(self) -> builtin.ModuleOp:
        return builtin.ModuleOp(
            Region([Block([self._convert_eq(eq) for eq in self.eqs])])
        )

    def _ensure_same_type(self, *vals: SSAValue):
        if all(isinstance(val.type, builtin.IntegerAttr) for val in vals):
            return vals
        if all(is_float(val) for val in vals):
            return vals
        # not everything homogeneous
        new_vals = []
        for val in vals:
            if is_float(val):
                new_vals.append(val)
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
                new_vals.append(val)
                continue
            # insert an integer to float cast op
            conv = arith.SIToFPOp.get(val, builtin.f32)
            self.block.add_op(conv)
            new_vals.append(conv.result)
        return new_vals


def _get_dim_offsets(idx: Indexed, t_offset: int) -> tuple:
    # shift all time values so that for all accesses at t + n, n>=0.
    # time_offs = min(int(i - d) for i, d in zip(idx.indices, idx.function.dimensions))
    halo = ((t_offset, 0), *idx.function.halo[1:])
    try:
        return tuple(
            int(i - d - halo_offset)
            for i, d, (halo_offset, _) in zip(
                idx.indices, idx.function.dimensions, halo
            )
        )
    except Exception as ex:
        raise ValueError("Indices must be constant offset from dimension!") from ex


def is_int(val: SSAValue):
    return isinstance(val.type, builtin.IntegerType)


def is_float(val: SSAValue):
    return val.type in (builtin.f32, builtin.f64)


# -------------------------------------------------------- ####
#                                                          ####
#           devito.stencil  ---> stencil dialect           ####
#                                                          ####
# -------------------------------------------------------- ####

from dataclasses import dataclass, field

from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from devito.ir.ietxdsl.lowering import (
    LowerIetForToScfFor,
)

from xdsl.dialects import llvm

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
            t0 := func.Call.get('timer_start', [], [builtin.f64])
        ], op.body.block)

        ret = op.get_return_op()
        assert ret is not None

        rewriter.insert_op_before([
            timers := iet_ssa.LoadSymbolic.get('timers', llvm.LLVMPointerType.typed(builtin.f64)),
            t1 := func.Call.get('timer_end', [t0], [builtin.f64]),
            llvm.StoreOp.get(t1, timers),
        ], ret)

        rewriter.insert_op_after_matched_op([
            func.FuncOp.external('timer_start', [], [builtin.f64]),
            func.FuncOp.external('timer_end', [builtin.f64], [builtin.f64])
        ])


class _DevitoStencilToStencilStencil(RewritePattern):
    """
    This converts the `devito.stencil` op into the following:

    1. change type of %t0, %t1 to stencil.field()
    2. attach info to iet.for where t0, t1 come from
    3. generate the following instructions
    ```
        %t0_temp = stencil.load(%t0) : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>

        // apply with t0 and t1 data to generate t2 data
        %t1_result = stencil.apply(%t0_temp) ({
        ^1(%t0_buff : !stencil.temp<?x?x?xf64>):
            ...
            // stencil.body is placed here
        }) : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>

        // write generated temp back to field
        stencil.store(%t1_result, %t1) [0, 0, 0] : [64, 64, 64]
    ```
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.Stencil, rewriter: PatternRewriter, /):
        perf("Convert devito.stencil op, mainly for subiterators")
        rank = len(op.shape)

        lb = list(-halo_elm.data[0].data for halo_elm in op.halo)
        ub = list(
            shape_elm.data + halo_elm.data[1].data
            for shape_elm, halo_elm in zip(op.shape, op.halo)
        )

        field_t = stencil.FieldType(zip(lb, ub), element_type=op.field_type)

        # change type of iet.for iteration variables to stencil.field
        for val in (*op.input_indices, op.output):
            assert isinstance(val.owner, Block)
            loop = val.owner.parent_op()
            assert isinstance(loop, iet_ssa.For)
            val.type = field_t
            loop.attributes["index_owner"] = op.grid_name

        input_temps = []

        for field in op.input_indices:
            rewriter.insert_op_before_matched_op(load_op := stencil.LoadOp.get(field))
            load_op.res.name_hint = field.name_hint + "_temp"
            input_temps.insert(0, load_op.res)

        rewriter.replace_matched_op(
            [
                out := stencil.ApplyOp.get(
                    input_temps,
                    op.body.detach_block(0),
                    result_types=[stencil.TempType(rank, element_type=op.field_type)],
                ),
                stencil.StoreOp.get(
                    out,
                    op.output,
                    stencil.IndexAttr.get(*([0] * rank)),
                    stencil.IndexAttr.get(*op.shape),
                ),
                scf.Yield.get(op.output, *op.input_indices),
            ]
        )

        out.res[0].name_hint = op.output.name_hint + "_result"


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
                arith.Constant.from_float_and_width(
                    float(self.known_symbols[symb_name]), op.result.type
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
        args = self.func_to_args[func]
        symb_name = op.symbol_name.data

        if symb_name not in args:
            body = parent.body.blocks[0]
            args[symb_name] = body.insert_arg(op.result.type, len(body.args))

        op.result.replace_by(args[symb_name])
        rewriter.erase_matched_op()
        parent.update_function_type()
        # attach information on parameter names to func
        parent.attributes["param_names"] = builtin.ArrayAttr(
            [
                builtin.StringAttr(name)
                for name, _ in sorted(args.items(), key=lambda tpl: tpl[1].index)
            ]
        )


def convert_devito_stencil_to_xdsl_stencil(module):
    grpa = GreedyRewritePatternApplier(
        [
            _DevitoStencilToStencilStencil(),
            LowerIetForToScfFor(),
            MakeFunctionTimed('apply_kernel'),
        ]
    )
    perf("DevitoStencil to stencil.stencil")
    perf("LowerIetForToScfFor")

    PatternRewriteWalker(grpa, walk_regions_first=True).rewrite_module(module)



def finalize_module_with_globals(module: builtin.ModuleOp, known_symbols: dict[str, Any]):
    grpa = GreedyRewritePatternApplier(
        [
            _InsertSymbolicConstants(known_symbols),
            _LowerLoadSymbolidToFuncArgs(),
        ]
    )
    PatternRewriteWalker(grpa).rewrite_module(module)
