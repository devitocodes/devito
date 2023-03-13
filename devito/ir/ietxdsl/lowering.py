from dataclasses import dataclass
from xdsl.ir import Block, Attribute, BlockArgument
from xdsl.dialects import builtin, scf, arith, func, llvm
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


class LowerIetForToScfFor(RewritePattern):
    """
    This lowers ALL `iet.for` loops it finds to *sequential* scf.for loops

    It does not care if the loop is declared "parellell".
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.For, rewriter: PatternRewriter, /):
        body = op.body.detach_block(0)

        _generate_subindices(op.subindices.data, body, rewriter)

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

        body = op.body.detach_block(0)

        _generate_subindices(op.subindices.data, body, rewriter)

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


class LowerIetPointerCastAndDataObject(RewritePattern):
    """
    This pass converts the pointer cast into an `getelementptr` operation.
    """
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: iet_ssa.PointerCast, rewriter: PatternRewriter, /):
        # TODO
        pass

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


def iet_to_standard_mlir(module: builtin.ModuleOp):

    walk = PatternRewriteWalker(GreedyRewritePatternApplier([
        LowerIetForToScfParallel(),
        LowerIetForToScfFor(),
        DropIetComments(),
        LowerIetPointerCastAndDataObject(),
        CleanupDanglingIetDatatypes(),
    ]))
    walk.rewrite_module(module)
