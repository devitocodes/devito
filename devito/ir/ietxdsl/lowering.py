from dataclasses import dataclass, field
from xdsl.ir import Block, Attribute, BlockArgument, SSAValue, Operation
from xdsl.dialects import builtin, scf, arith, func, llvm, memref
from xdsl.dialects.experimental import math

from devito.ir.ietxdsl import iet_ssa

from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, GreedyRewritePatternApplier, op_type_rewrite_pattern, PatternRewriteWalker

def _generate_subindices(subindices: int, block: Block, rewriter: PatternRewriter):
    # keep track of the what argument we should replace with what
    arg_changes = []

    # keep track of the ops we want to insert
    modulo = arith.Constant.from_int_and_width(subindices, builtin.i32)
    new_ops = [
        modulo
    ]

    # generate the new indices
    for i in range(subindices):
        offset = arith.Constant.from_int_and_width(i, builtin.i32)
        index_off = arith.Addi.get(block.args[0], offset)
        index = arith.RemSI.get(index_off, modulo)

        new_ops += [
            offset, index_off, index
        ]
        # replace block.args[i+1] with (arg0 + i) % n
        arg_changes.append((block.args[i+1], index.result))

    rewriter.insert_op_at_pos(new_ops, block, 0)

    for old, new in arg_changes:
        old.replace_by(new)
        block.erase_arg(old)


class ConvertIetForArgsToIndex(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.For, rewriter: PatternRewriter, /):
        for val in (op.lb, op.ub, op.step):
            if isinstance(val.typ, builtin.IndexType):
                continue
            cast = arith.IndexCastOp.get(val, builtin.IndexType())
            rewriter.insert_op_before_matched_op(cast)
            op.replace_operand(op.operands.index(val), cast.result)



class ConvertForLoopVarToIndex(RewritePattern):
    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
        if not isinstance(op, (scf.For, scf.ParallelOp)):
            return

        block: Block = op.regions[0].blocks[0]
        # change all loop vars to index and immediately cast to i32
        for loop_var in block.args:
            if isinstance(loop_var.typ, builtin.IndexType):
                continue
            loop_var.typ = builtin.IndexType()
            # insert a cast from index to i32 at the start of the loop

            rewriter.insert_op_at_pos(
                i32_val := arith.IndexCastOp.get(loop_var, builtin.i32),
                block,
                0
            )

            loop_var.replace_by(i32_val.result)
            i32_val.replace_operand(0, loop_var)





class LowerIetForToScfFor(RewritePattern):
    """
    This lowers ALL `iet.for` loops it finds to *sequential* scf.for loops

    It does not care if the loop is declared "parellell".
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.For, rewriter: PatternRewriter, /):
        body = op.body.detach_block(0)

        _generate_subindices(op.subindices.data, body, rewriter)

        rewriter.insert_op_at_pos(
            scf.Yield(),
            body,
            len(body.ops)
        )

        rewriter.replace_matched_op(
            scf.For.get(
                op.lb,
                op.ub,
                op.step,
                [],
                body
            )
        )

        

class LowerIetForToScfParallel(RewritePattern):
    """
    This converts all loops with a "parallel" property to `scf.parallel`

    It does not currently fuse together multiple nested parallel runs
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.For, rewriter: PatternRewriter, /):
        if op.parallelism_property != 'parallel':
            return

        body: Block = op.body.detach_block(0)

        assert op.subindices.data == 0

        if len(body.ops) == 2 and isinstance(body.ops[1], iet_ssa.For):
            # TODO: fuse with next parallel
            pass

        rewriter.insert_op_at_pos(
            scf.Yield(),
            body,
            len(body.ops)
        )

        rewriter.replace_matched_op(
            scf.ParallelOp.get(
                [op.lb],
                [op.ub],
                [op.step],
                [body]
            )
        )

class DropIetComments(RewritePattern):
    """
    This drops all iet.comment operations

    TODO: convert iet.comment ops that have timer info into their own nodes
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.Statement, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


@dataclass
class LowerIetPointerCastAndDataObject(RewritePattern):
    dimensions: list[SSAValue] = field(default_factory=list)

    """
    This pass converts the pointer cast into an `getelementptr` operation.
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.PointerCast, rewriter: PatternRewriter, /):
        memref_typ = op.result.typ
        assert isinstance(memref_typ, memref.MemRefType)

        assert len(self.dimensions) == 0, "can only convert one pointer cast!"

        new_ops = []
        for i in range(len(memref_typ.shape)):
            ptr = llvm.GetElementPtrOp.get(
                op.arg,
                [0, 1, i], # (*u_vec).data => two accesses (dereference, first element in struct)
                result_type = llvm.LLVMPointerType.typed(
                    builtin.i32
                )
            )
            val = llvm.LoadOp.get(ptr)
            new_ops += [ptr, val]
            self.dimensions.append(val.dereferenced_value)

        rewriter.insert_op_before_matched_op(
            new_ops
        )

        rewriter.replace_matched_op(
            llvm.GetElementPtrOp.get(
                op.arg,
                [0, 0], # (*u_vec).data => two accesses (dereference, first element in struct)
                result_type = llvm.LLVMPointerType.typed(
                    memref_typ.element_type
                ) # .data is a contigous array
            )
        )


class CleanupDanglingIetDatatypes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        for i, arg_typ in enumerate(op.function_type.inputs.data):
            if isinstance(arg_typ, iet_ssa.Dataobj):
                op.body.blocks[0].args[i].typ = llvm.LLVMPointerType.typed(
                    iet_ssa.Dataobj.get_llvm_struct_type()
                )
            elif isinstance(arg_typ, iet_ssa.Profiler):
                op.body.blocks[0].args[i].typ = llvm.LLVMPointerType.opaque()
            
        op.attributes['function_type'] = builtin.FunctionType.from_lists(
            [arg.typ for arg in op.body.blocks[0].args],
            op.function_type.outputs.data
        )

@dataclass
class LowerMemrefLoadToLLvmPointer(RewritePattern):
    """
    This converts a memref load %ref at %x, %y, %z, ...
    to an llvm pointer based load

    We have the dimensions in variables o1, o2, o3

    %linear_offset = x(o1 * o2) + y(o2) + z
    %elem_ptr = getelementptr %ptr, %linear_offset
    ptr.load %elem_ptr

    Why we need to convert from memref.load to llvm.load?

    Memerefs are much more complex than a pointer to contiguous memory.
    Since we are just given a pointer to contigous memory, we think it's
    easiest to manually compute these linear_offsets for accesses.
    """
    ptr_lowering: LowerIetPointerCastAndDataObject

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Load, rewriter: PatternRewriter, /):
        idx_calc_ops, idx = calc_index(op.indices, self.ptr_lowering.dimensions)

        rewriter.replace_matched_op([
            *idx_calc_ops,
            gep := llvm.GetElementPtrOp.get(op.memref, [], [idx]),
            load := llvm.LoadOp.get(gep)
        ], [load.dereferenced_value])


@dataclass
class LowerMemrefStoreToLLvmPointer(RewritePattern):
    ptr_lowering: LowerIetPointerCastAndDataObject

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.Store, rewriter: PatternRewriter, /):
        idx_calc_ops, idx = calc_index(op.indices, self.ptr_lowering.dimensions)

        rewriter.replace_matched_op([
            *idx_calc_ops,
            gep := llvm.GetElementPtrOp.get(op.memref, [], [idx]),
            store := llvm.StoreOp.get(op.value, gep)
        ], [])



def calc_index(indices: list[SSAValue], offsets: list[SSAValue]) -> list[Operation, SSAValue]:
    assert len(indices) == len(offsets)
    results = []
    # we have to convert the indices in the format
    # prt[x][y][z] with the offsets ox, oy, oz into the format:
    # ptr[
    #   (x * ox * oy) +
    #   (y * oy) +
    #   (z)
    # ]
    # we re-arrange this to minimize multiplications
    # ((x * ox) + y) * oy) + z
    carry: SSAValue = indices[0] # x
    
    for offset, index in zip(offsets[:-1], indices[1:]):
        mul = arith.Muli.get(carry, offset)
        add = arith.Addi.get(mul, index)
        results += [mul, add]
        carry = add.result

    return results, carry



def iet_to_standard_mlir(module: builtin.ModuleOp):

    walk = PatternRewriteWalker(GreedyRewritePatternApplier([
        ConvertIetForArgsToIndex(),
        LowerIetForToScfParallel(),
        LowerIetForToScfFor(),
        DropIetComments(),
        ptr_lower := LowerIetPointerCastAndDataObject(),
        LowerMemrefLoadToLLvmPointer(ptr_lower),
        LowerMemrefStoreToLLvmPointer(ptr_lower),
        CleanupDanglingIetDatatypes(),
    ]))

    walk.rewrite_module(module)
    PatternRewriteWalker(ConvertForLoopVarToIndex()).rewrite_module(module)
