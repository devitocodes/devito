from devito.iteration import Iteration
from devito.expression import Expression
import cgen

__all__ = ['StencilKernel']


class StencilKernel(object):
    """Code generation class, alternative to Propagator"""

    def __init__(self, stencils):
        # Ensure we always deal with Expression lists
        stencils = stencils if isinstance(stencils, list) else [stencils]
        self.expressions = [Expression(s) for s in stencils]

        # Wrap expressions with Iterations according to dimensions
        for i, expr in enumerate(self.expressions):
            newexpr = expr
            for d in reversed(expr.dimensions):
                newexpr = Iteration(newexpr, d, d.size)
            self.expressions[i] = newexpr

        # TODO: Merge Iterations iff outermost variables agree

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply defined stenicl kernel to a set of data objects"""
        raise NotImplementedError("StencilKernel - Codegen and apply() missing")

    @property
    def ccode(self):
        return cgen.Block([e.ccode for e in self.expressions])
