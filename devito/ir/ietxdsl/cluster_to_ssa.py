#------------- devito import -------------#

from devito import Operator, Constant
from sympy import Indexed, Integer, Symbol, Add, Eq, Mod, Pow, Mul, Float, Expr

from sympy.core.function import FunctionClass

from devito import SpaceDimension, Grid, SteppingDimension, Dimension
from devito.symbolics import retrieve_indexed


from devito.ir import PointerCast, FindNodes
from devito.ir.clusters import Cluster
from devito.ir.equations import LoweredEq

from devito.ir.iet import FindSymbols, FindNodes
from devito.ir.iet.nodes import CallableBody, MetaCall, Definition, Dereference, Prodder  # noqa

#----------- devito ssa import -----------#

from devito.ir.ietxdsl import iet_ssa
from devito.ir.ietxdsl.ietxdsl_functions import collectStructs, get_arg_types, dtypes_to_xdsltypes

#-------------- xdsl import --------------#

from xdsl.ir import Block, Region, SSAValue, Operation, OpResult
from xdsl.dialects import builtin, func, memref, arith
from xdsl.dialects.experimental import stencil, math



class ExtractDevitoStencilConversion:

    eqs: list[LoweredEq]
    block: Block
    loaded_values: dict[tuple[int,...], SSAValue]#
    time_offs: int

    def __init__(self, eqs: list[LoweredEq]) -> None:
        self.eqs = eqs
        self.loaded_values = dict()
        self.time_offs = 0

    def _convert_eq(self, eq: LoweredEq):
        if isinstance(eq.lhs, Symbol):
            return func.FuncOp.external(eq.lhs.name, [], [builtin.i32])
        assert isinstance(eq.lhs, Indexed)

        outermost_block = Block.from_arg_types([])
        self.block = outermost_block

        function = eq.lhs.function
        grid: Grid = function.grid
        # get the halo of the space dimensions only
        halo = [function.halo[function.dimensions.index(d)] for d in grid.dimensions]

        # shift all time values so that for all accesses at t + n, n>=0.

        self.time_offs = min(
            int(idx.indices[0] - grid.stepping_dim) for idx in retrieve_indexed(eq)
        )

        # build the for loop
        self._build_iet_for(grid.stepping_dim, ['sequential'], function._time_size)

        # build stencil
        stencil_op = iet_ssa.Stencil.get(
            grid.shape,
            halo,
            function.time_order,
            [f't_{i}_buff' for i in range(function.time_order) ]
        )
        self.block.add_op(stencil_op)
        self.block = stencil_op.block

        # dims -> ssa vals
        time_offset_to_field: dict[str, SSAValue] = {
            i: stencil_op.block.args[i] for i in range(function.time_order)
        }

        # reset loaded values
        self.loaded_values: dict[tuple[int, ...], SSAValue] = dict()

        # add all loads into the stencil
        self._add_access_ops(retrieve_indexed(eq.rhs), time_offset_to_field)

        # add math
        rhs_result = self._visit_math_nodes(eq.rhs)

        # emit return
        offsets = _get_dim_offsets(eq.lhs, self.time_offs)
        assert offsets[0] == function._time_size-1, "result should be written to last time buffer"
        assert all(o == 0 for o in offsets[1:]), f"can only write to offset [0,0,0], given {offsets[1:]}"

        self.block.add_op(
            stencil.ReturnOp.get(rhs_result)
        )

        return func.FuncOp.from_region('my-func', [], [], Region.from_block_list([outermost_block]))

    def _visit_math_nodes(self, node: Expr) -> SSAValue:
        if isinstance(node, Indexed):
            offsets = _get_dim_offsets(node, self.time_offs)
            return self.loaded_values[offsets]
        if isinstance(node, Integer):
            cst = arith.Constant.from_int_and_width(int(node), builtin.i32)
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
                op = op_cls.get(carry, arg)
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

    def _add_access_ops(self, reads: list[Indexed], time_offset_to_field: dict[int, SSAValue]):
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
                space_offsets,
                dtypes_to_xdsltypes[read.function.dtype]
            )
            self.block.add_op(access_op)
            # cache the resulting ssa value for later use
            # by the offsets (same offset = same value)
            self.loaded_values[offsets] = access_op.res

    def _build_iet_for(self, dim: SteppingDimension, props: list[str], subindices: int):
        lb = iet_ssa.LoadSymbolic.get(
            str(dim.symbolic_min),
            builtin.IndexType()
        )
        ub = iet_ssa.LoadSymbolic.get(
            str(dim.symbolic_max),
            builtin.IndexType()
        )
        try:
            step = arith.Constant.from_int_and_width(int(dim.symbolic_incr), builtin.IndexType())
        except:
            raise ValueError("step must be int!")
        
        loop = iet_ssa.For.get(
            lb, ub, step, subindices, props
        )
        self.block.add_ops([
            lb, ub, step, loop
        ])
        self.block = loop.block

    def convert(self) -> builtin.ModuleOp:
        return builtin.ModuleOp.from_region_or_ops(
            Region.from_block_list([
                Block.from_ops([
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
            if isinstance(val, OpResult) and isinstance(val.op, arith.Constant) and val.uses == 0:
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
    #time_offs = min(int(i - d) for i, d in zip(idx.indices, idx.function.dimensions))
    halo = ((t_offset, 0), *idx.function.halo[1:])
    try:
        return tuple(int(i - d - halo_offset) for i, d, (halo_offset, _) in zip(idx.indices, idx.function.dimensions, halo))
    except Exception as ex:
        raise ValueError("Indices must be constant offset from dimension!") from ex



def is_int(val: SSAValue):
    return isinstance(val.typ, builtin.IntegerType)

def is_float(val: SSAValue):
    return val.typ in (builtin.f32, builtin.f64)