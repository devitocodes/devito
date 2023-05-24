# ------------- devito import -------------#

from sympy import Indexed, Integer, Symbol, Add, Mod, Pow, Mul, Float, Expr

from devito import Grid, SteppingDimension
from devito.symbolics import retrieve_indexed

from devito.ir.clusters import Cluster
from devito.ir.equations import LoweredEq
# ----------- devito ssa import -----------#

from devito.ir.ietxdsl import iet_ssa
from devito.ir.ietxdsl.ietxdsl_functions import dtypes_to_xdsltypes

# -------------- xdsl import --------------#

from xdsl.ir import Block, Region, SSAValue, Operation, OpResult
from xdsl.dialects import builtin, func, memref, arith
from xdsl.dialects.experimental import stencil, math
from xdsl.dialects import stencil as stencil_nexp

default_int = builtin.i64


class ExtractDevitoStencilConversion:
    eqs: list[LoweredEq]
    block: Block
    loaded_values: dict[tuple[int, ...], SSAValue]
    time_offs: int

    def __init__(self, eqs: list[LoweredEq]) -> None:
        self.eqs = eqs
        self.loaded_values = dict()
        self.time_offs = 0

    def _convert_eq(self, eq: LoweredEq):
        # Convert a Devito equation
        if isinstance(eq.lhs, Symbol):
            return func.FuncOp.external(eq.lhs.name, [], [builtin.i32])
        assert isinstance(eq.lhs, Indexed)

        outermost_block = Block([])
        self.block = outermost_block

        function = eq.lhs.function
        mlir_type = dtypes_to_xdsltypes[function.dtype]
        grid: Grid = function.grid
        # get the halo of the space dimensions only
        halo = [function.halo[function.dimensions.index(d)] for d in grid.dimensions]

        # shift all time values so that for all accesses at t + n, n>=0.
        self.time_offs = min(
            int(idx.indices[0] - grid.stepping_dim) for idx in retrieve_indexed(eq)
        )
        # calculate the actual size of our time dimension
        actual_time_size = max(
            int(idx.indices[0] - grid.stepping_dim) for idx in retrieve_indexed(eq)
        ) - self.time_offs + 1

        # build the for loop
        loop = self._build_iet_for(grid.stepping_dim, ['sequential'], actual_time_size)

        # build stencil
        stencil_op = iet_ssa.Stencil.get(
            loop.subindice_ssa_vals(),
            grid.shape,
            halo,
            actual_time_size,
            mlir_type,
            eq.lhs.function._C_name,
        )
        self.block.add_op(stencil_op)
        self.block = stencil_op.block

        # dims -> ssa vals
        time_offset_to_field: dict[str, SSAValue] = {
            i: stencil_op.block.args[i] for i in range(actual_time_size-1)
        }

        # reset loaded values
        self.loaded_values: dict[tuple[int, ...], SSAValue] = dict()

        # add all loads into the stencil
        self._add_access_ops(retrieve_indexed(eq.rhs), time_offset_to_field)

        # add math
        rhs_result = self._visit_math_nodes(eq.rhs)

        # emit return
        offsets = _get_dim_offsets(eq.lhs, self.time_offs)
        assert offsets[0] == actual_time_size-1, "result should be written to last time buffer"
        assert all(o == 0 for o in offsets[1:]), f"can only write to offset [0,0,0], given {offsets[1:]}"

        self.block.add_op(
            stencil.ReturnOp.get([rhs_result])
        )
        outermost_block.add_op(
            func.Return.get()
        )

        return func.FuncOp.from_region('myfunc', [], [],
                                       Region([outermost_block]))

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
        #if isinstance(math, Constant):
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
            if isinstance(carry.typ, builtin.IntegerType):
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
                    raise ValueError("no IPowFop yet!")
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

    def _add_access_ops(self, reads: list[Indexed],
                        time_offset_to_field: dict[int, SSAValue]):
        for read in reads:
            """
            AccessOp:
                name: str = "stencil.access"
                temp: Annotated[Operand, TempType]
                offset: OpAttr[IndexAttr]
                res: Annotated[OpResult, Attribute]
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
            access_op = stencil.AccessOp.get(
                field,
                space_offsets
            )
            self.block.add_op(access_op)
            # cache the resulting ssa value for later use
            # by the offsets (same offset = same value)
            self.loaded_values[offsets] = access_op.res

    def _build_iet_for(self, dim: SteppingDimension, props: list[str],
                       subindices: int) -> iet_ssa.For:
        lb = iet_ssa.LoadSymbolic.get(
            str(dim.symbolic_min),
            builtin.IndexType()
        )
        ub = iet_ssa.LoadSymbolic.get(
            str(dim.symbolic_max),
            builtin.IndexType()
        )
        try:
            step = arith.Constant.from_int_and_width(int(dim.symbolic_incr),
                                                     builtin.IndexType())
            step.result.name_hint = "step"
        except:
            raise ValueError("step must be int!")

        loop = iet_ssa.For.get(
            lb, ub, step, subindices, props
        )
        self.block.add_ops([
            lb, ub, step, loop
        ])

        self.block = loop.block

        return loop

    def convert(self) -> builtin.ModuleOp:
        return builtin.ModuleOp(
            Region([
                Block([
                    self._convert_eq(eq) for eq in self.eqs
                ])
            ])
        )

    def _ensure_same_type(self, *vals: SSAValue):
        if all(isinstance(val.typ, builtin.IntegerAttr) for val in vals):
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
            if (isinstance(val, OpResult) and
               isinstance(val.op, arith.Constant) and val.uses == 0):
                val.typ = builtin.f32
                val.op.attributes['value'] = builtin.FloatAttr(
                    float(val.op.value.value.data),
                    builtin.f32
                )
                new_vals.append(val)
                continue
            # insert an integer to float cast op
            conv = arith.SIToFPOp.get(val, builtin.f32)
            self.block.add_op(conv)
            new_vals.append(conv.result)
        return new_vals


def _cluster_shape(cl: Cluster) -> tuple[int, ...]:
    return cl.grid.shape


def _cluster_grid(cl: Cluster) -> Grid:
    return cl.grid


def _cluster_function(cl: Cluster):  # TODO: fix typing here
    return cl.exprs[0].args[0].function


def _get_dim_offsets(idx: Indexed, t_offset: int) -> tuple:
    # shift all time values so that for all accesses at t + n, n>=0.
    # time_offs = min(int(i - d) for i, d in zip(idx.indices, idx.function.dimensions))
    halo = ((t_offset, 0), *idx.function.halo[1:])
    try:
        return tuple(int(i - d - halo_offset) for i, d, (halo_offset, _)
                     in zip(idx.indices, idx.function.dimensions, halo))
    except Exception as ex:
        raise ValueError("Indices must be constant offset from dimension!") from ex


def is_int(val: SSAValue):
    return isinstance(val.typ, builtin.IntegerType)


def is_float(val: SSAValue):
    return val.typ in (builtin.f32, builtin.f64)

# -------------------------------------------------------- ####
#                                                          ####
#           devito.stencil  ---> stencil dialect           ####
#                                                          ####
# -------------------------------------------------------- ####

from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, GreedyRewritePatternApplier, op_type_rewrite_pattern, PatternRewriteWalker

from devito.ir.ietxdsl.lowering import recalc_func_type, DropIetComments, LowerIetForToScfFor, ConvertForLoopVarToIndex, ConvertScfForArgsToIndex, ConvertScfForArgsToIndex

from dataclasses import dataclass


class _DevitoStencilToStencilStencil(RewritePattern):
    """
    This converts the `devito.stencil` op into the following:

    ```
        // get data object
        %data = devito.load_symbolic() {symbol_name = "data"} : () -> memref...
        // we do magic here to accomodate triple buffering
        // this will be tackled in the lowering
        %t0_field = devito.get_field(%data, %t0)
        %t1_field = devito.get_field(%data, %t1)
        %t2_field = devito.get_field(%data, %t2)

        // do cast and load ops
        %t0_w_size = stencil.cast(%t0_field) [-4, -4, -4] to [68, 68, 68]
        %t1_w_size = stencil.cast(%t1_field) [-4, -4, -4] to [68, 68, 68]
        %t2_w_size = stencil.cast(%t1_field) [-4, -4, -4] to [68, 68, 68]
        // put ub - lb here (make the temp ?x?x?)
        %t0_temp = stencil.load(%t0_w_size) : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>
        %t1_temp = stencil.load(%t1_w_size) : (!stencil.field<72x72x72xf64>) -> !stencil.temp<?x?x?xf64>

        // apply with t0 and t1 data to generate t2 data
        %t2_output = stencil.apply(%t0_temp, %t1_temp) ({
        ^1(%t0_buff : !stencil.temp<?x?x?xf64>, %t1_buff : !stencil.temp<?x?x?xf64>):
            ...
            // stencil.body is placed here
        }) : (!stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>

        // write generated temp back to field
        // what are the bound notations here?
        stencil.store(%t2_output, %t2_w_size) [0, 0, 0] : [64, 64, 64]
    ```
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.Stencil, rewriter: PatternRewriter, /):
        rank = len(op.shape.data)

        data = iet_ssa.LoadSymbolic.get('data', memref.MemRefType.from_element_type_and_shape(
            memref.MemRefType.from_element_type_and_shape(op.field_type, [-1] * rank), [op.time_buffers.value.data]
        ))
        lb = stencil.IndexAttr.get(*list(-halo_elm.data[0].value.data for halo_elm in op.halo.data))
        ub = stencil.IndexAttr.get(*list(shape_elm.value.data + halo_elm.data[1].value.data for shape_elm, halo_elm in zip(op.shape.data, op.halo.data)))

        field_t = stencil.FieldType(
            (ub - lb), op.field_type
        )

        fields = list(
            iet_ssa.GetField.get(data, t_idx, field_t, lb, ub)
            for t_idx in (*op.input_indices, op.output)
        )
        # name the resulting fields
        for field in fields:
            field.field.name_hint = f"{field.t_index.name_hint}_w_size"

        loads = list(
            stencil.LoadOp.get(field)
            for field in fields[:-1]
        )
        rewriter.replace_matched_op([
            iet_ssa.Statement.get("// get data obj"),
            data,
            iet_ssa.Statement.get("// get individual fields"),
            *fields,
            iet_ssa.Statement.get("// stencil loads"),
            *loads,
            out := stencil.ApplyOp.get(
                loads, op.body.detach_block(0), result_types=[stencil.TempType([-1] * rank, op.field_type)]
            ),
            stencil.StoreOp.get(
                out,
                fields[-1],
                stencil.IndexAttr.get(*([0] * len(op.halo))),
                stencil.IndexAttr.get(*[x.value.data for x in op.shape.data]),
            )
        ])


class _LowerGetField(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.GetField, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op([
            idx := arith.IndexCastOp.get(op.t_index, builtin.IndexType()),
            ref := memref.Load.get(op.data, [idx]),
            field := stencil.ExternalLoadOp.get(ref, res_type=field_type_to_dynamic_shape_type(op.field.typ)),
            field_w_size := stencil_nexp.CastOp.get(field, op.lb, op.ub, op.field.typ)
        ], [field_w_size.result])


def field_type_to_dynamic_shape_type(t: stencil.FieldType):
    return stencil.FieldType(
        [-1] * len(t.shape), t.element_type
    )


def get_containing_func(op: Operation) -> func.FuncOp | None:
    while op is not None and not isinstance(op, func.FuncOp):
        op = op.parent_op()
    if op is None:
        return None
    return op


@dataclass
class _InsertSymbolicConstants(RewritePattern):
    known_symbols: dict[str, int | float]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.LoadSymbolic, rewriter: PatternRewriter, /):
        symb_name = op.symbol_name.data
        if symb_name not in self.known_symbols:
            return

        if op.result.typ in (builtin.f32, builtin.f64):
            rewriter.replace_matched_op(
                arith.Constant.from_float_and_width(
                    float(self.known_symbols[symb_name]), op.result.typ
                )
            )
            return

        assert op.result.typ in (builtin.i32, builtin.i64, builtin.IndexType())

        rewriter.replace_matched_op(
            arith.Constant.from_int_and_width(
                int(self.known_symbols[symb_name]), op.result.typ
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
            args[symb_name] = body.insert_arg(op.result.typ, 0)

        op.result.replace_by(args[symb_name])
        rewriter.erase_matched_op()
        recalc_func_type(parent)
        # attach information on parameter names to func
        parent.attributes['param_names'] = builtin.ArrayAttr([
            builtin.StringAttr(name) for name, _ in sorted(args.items(), key=lambda tpl: tpl[1].index)
        ])


def convert_devito_stencil_to_xdsl_stencil(module):
    grpa = GreedyRewritePatternApplier([
        _DevitoStencilToStencilStencil(),
        _LowerGetField(),
        DropIetComments(),
        LowerIetForToScfFor(),
        ConvertScfForArgsToIndex(),
        ConvertForLoopVarToIndex()
    ])
    PatternRewriteWalker(grpa).rewrite_module(module)


if __name__ == '__main__':
    import sys
    from xdsl.parser import Parser, MLContext, Source
    ctx = MLContext()
    ctx.register_dialect(stencil.Stencil)
    ctx.register_dialect(builtin.Builtin)
    ctx.register_dialect(arith.Arith)
    ctx.register_dialect(func.Func)
    ctx.register_dialect(iet_ssa.DEVITO_SSA)
    ctx.register_dialect(iet_ssa.IET_SSA)
    ctx.register_dialect(memref.MemRef)
    ctx.register_dialect(math.Math)
    p = Parser(ctx, open(sys.argv[-1], "r").read(), Source.MLIR, filename=sys.argv[-1])

    module = p.parse_module()

    convert_devito_stencil_to_xdsl_stencil(module)

    from xdsl.printer import Printer
    Printer(target=Printer.Target.MLIR).print(module)


def prod(iter):
    """
    Calculate the product over an iterator of numbers
    """
    carry = 1
    for x in iter:
        carry *= x
    return carry


def generate_launcher_base(module: builtin.ModuleOp,
                           known_symbols: dict[str, int | float],
                           dims: tuple[int]) -> str:
    """
    This transforms a module containing a function with symbolic
    loads into a function that is ready to be lowered by xdsl
    It replaces all symbolics with the one in known_symbols
    """
    grpa = GreedyRewritePatternApplier([
        _InsertSymbolicConstants(known_symbols),
        _LowerLoadSymbolidToFuncArgs(),
    ])
    PatternRewriteWalker(grpa).rewrite_module(module)
    f = module.ops.first
    assert isinstance(f, func.FuncOp)
    dtype: str = f.function_type.inputs.data[0].element_type.element_type.name

    t_dims = f.function_type.inputs.data[0].shape.data[0].value.data

    memref_type = "x".join(
        (*(str(x) for x in dims), dtype)
    )

    rank = len(dims)

    size_in_bytes = prod(dims) * int(dtype[1:]) // 8

    # figure out which is the last time buffer we write to
    last_time_m = (int(known_symbols['time_M']) + t_dims - 1) % t_dims

    return f"""
"builtin.module"() ({{
    "func.func"() ({{
        %num_bytes = arith.constant {size_in_bytes} : index
        %byte_ref = func.call @load_input(%num_bytes) : (index) -> memref<{size_in_bytes}xi8>

        %cst0 = arith.constant 0 : index
        %cst1 = arith.constant 1 : index

        %t0 = memref.view %byte_ref[%cst0][] : memref<{size_in_bytes}xi8> to memref<{memref_type}>
        %t1 = memref.alloc() : memref<{memref_type}>
        %ref = memref.alloc() : memref<{t_dims}xmemref<{memref_type}>>

        "memref.store"(%t0, %ref, %cst0) : (memref<{memref_type}>, memref<2xmemref<{memref_type}>>, index) -> ()
        "memref.store"(%t1, %ref, %cst1) : (memref<{memref_type}>, memref<2xmemref<{memref_type}>>, index) -> ()

        %time_start = func.call @timer_start() : () -> i64

        func.call @myfunc(%ref) : (memref<{t_dims}xmemref<{memref_type}>>) -> ()

        func.call @timer_end(%time_start) : (i64) -> ()

        func.call @dump_memref_{dtype}_rank_{rank}(%t{last_time_m}) : (memref<{memref_type}>) -> ()

        func.return %cst0 : index

    }}) {{"function_type" = () -> (index), "sym_name" = "main"}} : () -> ()

    func.func private @dump_memref_{dtype}_rank_{rank}(memref<{memref_type}>) -> ()
    func.func private @load_input(index) -> memref<{size_in_bytes}xi8>

    func.func private @timer_start() -> i64

    func.func private @timer_end(i64) -> ()

    func.func private @myfunc(memref<{t_dims}xmemref<{memref_type}>>) -> ()
}}) : () -> ()
"""