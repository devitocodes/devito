from dataclasses import dataclass
from xdsl.ir import Block, Attribute
from xdsl.dialects import builtin, scf, arith
from xdsl.dialects.experimental import math

from devito.ir.ietxdsl import iet_ssa

from xdsl.pattern_rewriter import RewritePattern, PatternRewriter, GreedyRewritePatternApplier, op_type_rewrite_pattern

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


@dataclass
class LowerIetForToScfFor(RewritePattern):
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
        




