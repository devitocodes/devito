from collections import Iterable
from itertools import chain

import cgen

from devito.expression import Expression
from devito.tools import filter_ordered

__all__ = ['Iteration']


class Iteration(Expression):
    """Iteration object that encapsualtes a single loop over sympy expressions.

    :param expressions: Single or list of SymPy expressions or :class:`Expression`
                        objects that define the loop body.
    :param variable: Symbol that defines the name of the variable over which
                     to iterate.
    :param limits: Limits for the iteration space, either the loop size or a
                   tuple of the form (start, finish, stepping).
    """

    def __init__(self, expressions, variable, limits):
        # Ensure we deal with a list of Expression objects internally
        self.expressions = expressions if isinstance(expressions, list) else [expressions]
        self.expressions = [e if isinstance(e, Expression) else Expression(e)
                            for e in self.expressions]
        self.variable = str(variable)
        if isinstance(limits, Iterable):
            assert(len(limits) == 3)
            self.limits = limits
        else:
            self.limits = (0, limits, 1)

    def __repr__(self):
        str_expr = "\n\t".join([str(s) for s in self.expressions])
        return "Iteration<%s; %s>::\n\t%s" % (self.variable, self.limits,
                                              str_expr)

    def substitute(self, substitutions):
        """Apply substitutions to expressions in loop body

        :param substitutions: Dict containing the substitutions to apply to
                              the stored loop stencils.
        """
        for expr in self.expressions:
            expr.substitute(substitutions)

    @property
    def ccode(self):
        """Generate C code for the represented stencil loop

        :returns: :class:`cgen.For` object representing the loop
        """
        forward = self.limits[1] >= self.limits[0]
        loop_body = cgen.Block([s.ccode for s in self.expressions])
        loop_init = cgen.InlineInitializer(
            cgen.Value("int", self.variable), self.limits[0])
        loop_cond = '%s %s %s' % (self.variable, '<' if forward else '>', self.limits[1])
        if self.limits[2] == 1:
            loop_inc = '%s%s' % (self.variable, '++' if forward else '--')
        else:
            loop_inc = '%s %s %s' % (self.variable, '+=' if forward else '-=',
                                     self.limits[2])
        return cgen.For(loop_init, loop_cond, loop_inc, loop_body)

    @property
    def signature(self):
        """List of data objects used by the loop and it's body

        :returns: List of unique data objects required by the loop
        """
        signatures = [e.signature for e in self.expressions]
        return filter_ordered(chain(*signatures))
