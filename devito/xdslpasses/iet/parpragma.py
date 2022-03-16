from devito.ir.ietxdsl import *
from xdsl.pattern_rewriter import RewritePattern, GreedyRewritePatternApplier, \
    op_type_rewrite_pattern, PatternRewriteWalker, PatternRewriter
import xdsl.dialects.builtin as builtin


# NOTE: this is WIP and needs refactoring ;)
@dataclass
class MakeSimdPattern(RewritePattern):
    """
    This pattern reproduces the behaviour of PragmaSimdTransformer
    """

    def is_parallel_relaxed(self, iteration: Iteration) -> bool:
        return any([
            prop.data
            in ["parallel", "parallel_if_private", "parallel_if_private"]
            for prop in iteration.properties.data
        ])

    @op_type_rewrite_pattern
    def match_and_rewrite(self, iteration: Iteration,
                          rewriter: PatternRewriter):

        if (not self.is_parallel_relaxed(iteration)):
            return

        # check if parent is parallel as well
        parent_op = iteration.parent.parent.parent
        if (not self.is_parallel_relaxed(parent_op)):
            return

        # TODO how to only check for iteration trees?
        # NOTE: currently only checking the first child
        child_ops = iteration.body.blocks[0].ops

        # check if children is parallel as well
        if (isinstance(child_ops[0], Iteration)
                and self.is_parallel_relaxed(child_ops[0])):
            return

        # TODO: insert additional checks
        iteration.pragmas.data.append(StringAttr.from_str("simd-for"))


def construct_walker() -> PatternRewriteWalker:
    applier = GreedyRewritePatternApplier([MakeSimdPattern()])

    return PatternRewriteWalker(applier,
                                walk_regions_first=False,
                                apply_recursively=False)


def make_simd(ctx, op: builtin.ModuleOp):
    walker = construct_walker()
    walker.rewrite_module(op)
