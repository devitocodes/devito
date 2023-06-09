# ------------- devito import -------------#

from sympy import Add, Expr, Float, Indexed, Integer, Mod, Mul, Pow, Symbol
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.dialects import stencil as stencil_nexp
from xdsl.dialects.experimental import dmp, math, stencil
from xdsl.ir import Attribute, Block, Operation, OpResult, Region, SSAValue

from devito import Grid, SteppingDimension
from devito.ir.clusters import Cluster
from devito.ir.equations import LoweredEq
from devito.ir.ietxdsl import iet_ssa
from devito.ir.ietxdsl.ietxdsl_functions import dtypes_to_xdsltypes
from devito.symbolics import retrieve_indexed

# ----------- devito ssa import -----------#


# -------------- xdsl import --------------#


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
        # Convert a Devito equation to a func.func op
        # an equation may look like this:
        #  u[x+1,y+1,z] = (u[x,y,z+1] + u[x+2,y+2,z+1]) / 2
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
        actual_time_size = (
            max(int(idx.indices[0] - grid.stepping_dim) for idx in retrieve_indexed(eq))
            - self.time_offs
            + 1
        )

        # build the for loop
        loop = self._build_iet_for(grid.stepping_dim, ["sequential"], actual_time_size)

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
            i: stencil_op.block.args[i] for i in range(actual_time_size - 1)
        }

        # reset loaded values
        self.loaded_values: dict[tuple[int, ...], SSAValue] = dict()

        # add all loads into the stencil
        self._add_access_ops(retrieve_indexed(eq.rhs), time_offset_to_field)

        # add math
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
        outermost_block.add_op(func.Return.get(loop.result))

        return func.FuncOp.from_region(
            "apply_kernel", [], [loop.result.typ], Region([outermost_block])
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
            access_op = stencil.AccessOp.get(field, space_offsets)
            self.block.add_op(access_op)
            # cache the resulting ssa value for later use
            # by the offsets (same offset = same value)
            self.loaded_values[offsets] = access_op.res

    def _build_iet_for(
        self, dim: SteppingDimension, props: list[str], subindices: int
    ) -> iet_ssa.For:
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
            if (
                isinstance(val, OpResult)
                and isinstance(val.op, arith.Constant)
                and val.uses == 0
            ):
                val.typ = builtin.f32
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
        return tuple(
            int(i - d - halo_offset)
            for i, d, (halo_offset, _) in zip(
                idx.indices, idx.function.dimensions, halo
            )
        )
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

from dataclasses import dataclass

from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from devito.ir.ietxdsl.lowering import (
    ConvertForLoopVarToIndex,
    ConvertScfForArgsToIndex,
    DropIetComments,
    LowerIetForToScfFor,
    recalc_func_type,
)


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
        rank = len(op.shape)

        lb = list(-halo_elm.data[0].data for halo_elm in op.halo)
        ub = list(
            shape_elm.data + halo_elm.data[1].data
            for shape_elm, halo_elm in zip(op.shape, op.halo)
        )

        field_t = stencil.FieldType(zip(lb, ub), typ=op.field_type)

        # change type of iet.for iteration variables to stencil.field
        for val in (*op.input_indices, op.output):
            assert isinstance(val.owner, Block)
            loop = val.owner.parent_op()
            assert isinstance(loop, iet_ssa.For)
            val.typ = field_t
            loop.attributes["index_owner"] = op.grid_name

        input_temps = []

        for field in op.input_indices:
            rewriter.insert_op_before_matched_op(load_op := stencil.LoadOp.get(field))
            input_temps.append(load_op.res)
            load_op.res.name_hint = field.name_hint + "_temp"

        rewriter.replace_matched_op(
            [
                out := stencil.ApplyOp.get(
                    input_temps,
                    op.body.detach_block(0),
                    result_types=[stencil.TempType(rank, typ=op.field_type)],
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


class _LowerGetField(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.GetField, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(
            [
                idx := arith.IndexCastOp.get(op.t_index, builtin.IndexType()),
                ref := memref.Load.get(op.data, [idx]),
                field := stencil.ExternalLoadOp.get(
                    ref, res_type=field_type_to_dynamic_shape_type(op.field.typ)
                ),
                field_w_size := stencil_nexp.CastOp.get(
                    field, op.lb, op.ub, op.field.typ
                ),
            ],
            [field_w_size.result],
        )


def field_type_to_dynamic_shape_type(t: stencil.FieldType):
    return stencil.FieldType([-1] * len(t.shape), t.element_type)


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
            args[symb_name] = body.insert_arg(op.result.typ, len(body.args))

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
        ]
    )
    PatternRewriteWalker(grpa, walk_regions_first=True).rewrite_module(module)


def prod(iter):
    """
    Calculate the product over an iterator of numbers
    """
    carry = 1
    for x in iter:
        carry *= x
    return carry


def translate_signature(
    t: func.FunctionType,
) -> tuple[list[Attribute], list[Attribute]]:
    inputs = []
    outputs = []
    for i in t.inputs:
        if not isinstance(i, stencil.FieldType):
            inputs.append(i)
            break
        inputs.append(
            memref.MemRefType.from_element_type_and_shape(i.element_type, i.get_shape())
        )

    for o in t.outputs:
        if not isinstance(o, stencil.FieldType):
            outputs.append(o)
            break
        outputs.append(
            memref.MemRefType.from_element_type_and_shape(o.element_type, o.get_shape())
        )
    return inputs, outputs


def generate_launcher_base(
    module: builtin.ModuleOp,
    known_symbols: dict[str, int | float],
    dims: tuple[int],
    mpi: bool = False,
    gpu: bool = False,
) -> str:
    """
    This transforms a module containing a function with symbolic
    loads into a function that is ready to be lowered by xdsl
    It replaces all symbolics with the one in known_symbols
    """
    grpa = GreedyRewritePatternApplier(
        [
            _InsertSymbolicConstants(known_symbols),
            _LowerLoadSymbolidToFuncArgs(),
        ]
    )
    PatternRewriteWalker(grpa).rewrite_module(module)
    f = module.ops.first

    assert isinstance(f, func.FuncOp)

    input_types, result_types = translate_signature(f.function_type)

    # grab the signature of the kernel
    ext_func_decl = func.FuncOp.external(f.sym_name.data, input_types, result_types)
    kernel_signature = str(ext_func_decl)

    dtype: str = input_types[0].element_type.name

    memref_type = str(input_types[0])

    dims = input_types[0].get_shape()

    rank = len(dims)

    size_in_bytes = prod(dims) * int(dtype[1:]) // 8

    func_args = ["%t0"]
    alloc_lines = []

    for i, t in enumerate(input_types[1:]):
        alloc_lines.append(
            f'        %t{i+1} = "memref.alloc"() {{"operand_segment_sizes" = array<i32: 0, 0>}} : () -> {str(t)}'
        )
        func_args.append(f"%t{i+1}")

    if gpu:
        func_args = []
        for i, t in enumerate(input_types):
            alloc_lines.append(
                f"""        
        %gt{i} = "gpu.alloc"() {{operand_segment_sizes = array<i32: 0, 0, 0>}} : () -> {str(t)}
        "gpu.memcpy"(%gt{i}, %t{i}) {{"operand_segment_sizes" = array<i32: 0, 1, 1>}} : ({str(t)}, {str(t)}) -> ()
        "memref.dealloc"(%t{i}) {{"operand_segment_sizes" = array<i32: 0, 1>}} : ({str(t)}) -> ()
            """
            )
            func_args.append(f"%gt{i}")

    alloc_lines = "\n".join(alloc_lines)


    # set all allocated memref values to 0

    ubs_def = '\n        '.join(
        f'%loop_ub_{i} = "arith.constant"() {{"value" = {val} : index}} : () -> index' for i, val in enumerate(dims)
    )
    ubs = [
        f'%loop_ub_{i}' for i in range(rank)
    ]
    lbs = [
        '%loop_lb' for _ in range(rank)
    ]
    step = [
        '%loop_step' for _ in range(rank)
    ]
    
    stores = "\n            ".join(
        '"memref.store"(%init_val, {}, {}) : (f32, {}, {}) -> ()'.format(
            ref, 
            ", ".join(f"%i{i}" for i in range(rank)),
            str(memref_type),
            ", ".join(f"index" for _ in range(rank)),
        ) for ref in func_args[1:]
    )

    alloc_lines += f"""
        %init_val = "arith.constant"() {{"value" = 0.0 : f32}} : () -> f32
        %loop_lb = "arith.constant"() {{"value" = 0 : index}} : () -> index
        %loop_step = "arith.constant"() {{"value" = 1 : index}} : () -> index
        {ubs_def}

        "scf.parallel"({", ".join(lbs)}, {", ".join(ubs)}, {", ".join(step)}) ({{
            ^init_loop_body({', '.join(f'%i{i} : index' for i in range(rank))}):
                {stores}
                "scf.yield"() : () -> ()
        }}) {{"operand_segment_sizes" = array<i32: {rank}, {rank}, {rank}, 0>}} : ({', '.join(['index'] * (3 * rank))}) -> ()
    """

    # grab the field type with allocated bounds
    field_type: stencil.FieldType = f.function_type.inputs.data[0]
    assert isinstance(field_type, stencil.FieldType)
    assert isinstance(field_type.bounds, stencil.StencilBoundsAttr)

    bounds = field_type.bounds

    global_shape = dmp.HaloShapeInformation.from_index_attrs(
        buff_lb=bounds.lb,
        buff_ub=bounds.ub,
        core_lb=stencil.IndexAttr.get(*([0] * len(bounds.lb))),
        core_ub=(bounds.ub + bounds.lb),
    )

    teardown = f'"func.call"(%res) {{"callee" = @dump_memref_{dtype}_rank_{rank}}} : ({memref_type}) -> ()'

    setup = ""

    if mpi:
        setup += f"""
        "mpi.init"() : () -> ()

        %rank = "mpi.comm.rank"() : () -> i32
        %rank_idx = "arith.index_cast"(%rank) : (i32) -> index

        "dmp.scatter"(%t0, %rank_idx) {{ "global_shape" = {str(global_shape)} }} : ({memref_type}, index) -> ()
        """
        teardown = f"""
        "dmp.gather"(%res, %rank_idx) ({{
        ^bb0(%global_data: {memref_type}):
            "func.call"(%global_data) {{"callee" = @dump_memref_{dtype}_rank_{rank}}} : ({memref_type}) -> ()
        }}) {{ "root_rank" = 0, "global_shape" = {str(global_shape)} }} : ({memref_type}, index) -> ()

        "mpi.finalize"() : () -> ()
        """
        

    if gpu:
        gpu_deallocs = [
            f"""
        "gpu.dealloc" ({fa}) {{"operand_segment_sizes" = array<i32: 0, 1>}} : ({memref_type}) -> ()
        """
            for fa in func_args
        ]
        gpu_deallocs = "\n".join(gpu_deallocs)
        #setup = ""
        teardown = f"""
        %cpu_res = "memref.alloc"() {{"operand_segment_sizes" = array<i32: 0, 0>}} : () -> {str(t)}
        "gpu.memcpy"(%cpu_res, %res) {{"operand_segment_sizes" = array<i32: 0, 1, 1>}} : ({str(t)}, {str(t)}) -> ()
        {gpu_deallocs}
        "func.call"(%cpu_res) {{"callee" = @dump_memref_{dtype}_rank_{rank}}} : ({memref_type}) -> ()
        """ 

    return f"""
"builtin.module"() ({{
    "func.func"() ({{
        %num_bytes = "arith.constant"() {{"value" = {size_in_bytes} : index}} : () -> index
        %byte_ref = "func.call"(%num_bytes) {{"callee" = @load_input}} : (index) -> memref<{size_in_bytes}xi8>

        %cst0 = "arith.constant"() {{"value" = 0 : index}} : () -> index
        %cst1 = "arith.constant"() {{"value" = 1 : index}} : () -> index

        %t0 = "memref.view"(%byte_ref, %cst0) : (memref<{size_in_bytes}xi8>, index) -> {memref_type}
{alloc_lines}
{setup}
        %time_start = "func.call"() {{"callee" = @timer_start}} : () -> i64

        %res = "func.call"({', '.join(func_args)}) {{"callee" = @{f.sym_name.data}}}  : {str(ext_func_decl.function_type)}

        "func.call"(%time_start) {{"callee" = @timer_end}} : (i64) -> ()
{teardown}
        "func.return" (%cst0) : (index) -> ()

    }}) {{"function_type" = () -> (index), "sym_name" = "main"}} : () -> ()

    func.func private @dump_memref_{dtype}_rank_{rank}({memref_type}) -> ()
    func.func private @load_input(index) -> memref<{size_in_bytes}xi8>

    func.func private @timer_start() -> i64

    func.func private @timer_end(i64) -> ()

    {kernel_signature}
}}) : () -> ()
"""
