from collections import Iterable

import cgen
from devito.codeprinter import ccode

__all__ = ['Iteration']


class Iteration(object):
    """Iteration object that encapsualtes a single loop over sympy expressions.

    :param stencils: SymPy equation or list of equations that define the
                     stencil used to create the loop body.
    :param variable: Symbol that defines the name of the variable over which
                     to iterate.
    :param limits: Limits for the iteration space, either the loop size or a
                   tuple of the form (start, finish, stepping).
    """

    def __init__(self, stencils, variable, limits):
        self.stencils = stencils if isinstance(stencils, list) else [stencils]
        self.variable = str(variable)
        if isinstance(limits, Iterable):
            assert(len(limits) == 3)
            self.limits = limits
        else:
            self.limits = (0, limits, 1)

    def __repr__(self):
        str_stencils = "\n\t".join([str(s) for s in self.stencils])
        return "Iteration<%s; %s>::\n\t%s" % (self.variable, self.limits,
                                              str_stencils)

    def substitute(self, substitutions):
        """Apply substitutions to loop stencils via sympy.subs()

        :param substitutions: Dict containing the substitutions to apply to
                              the stored loop stencils.
        """
        self.stencils = [eq.subs(substitutions) for eq in self.stencils]

    @property
    def ccode(self):
        """Generate C code for the represented stencil loop

        :returns: :class:`cgen.For` object representing the loop
        """
        forward = self.limits[1] >= self.limits[0]
        loop_body = cgen.Block([ccode(cgen.Assign(ccode(eq.lhs), ccode(eq.rhs)))
                                for eq in self.stencils])
        loop_init = cgen.InlineInitializer(
            cgen.Value("int", self.variable), self.limits[0])
        loop_cond = '%s %s %s' % (self.variable, '<' if forward else '>', self.limits[1])
        if self.limits[2] == 1:
            loop_inc = '%s%s' % (self.variable, '++' if forward else '--')
        else:
            loop_inc = '%s %s %s' % (self.variable, '+=' if forward else '-=',
                                     self.limits[2])
        return cgen.For(loop_init, loop_cond, loop_inc, loop_body)
