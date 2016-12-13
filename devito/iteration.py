from collections import Iterable, defaultdict
from itertools import chain

import cgen
from sympy import Symbol

from devito.dimension import Dimension
from devito.expression import Expression
from devito.logger import warning
from devito.tools import filter_ordered

__all__ = ['Iteration']


class IterationBound(object):
    """Utility class to encapsulate variable loop bounds and link them
    back to the respective Dimension object.

    :param name: Variable name for the open loop bound variable
    """

    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def __repr__(self):
        return self.name

    @property
    def ccode(self):
        """C code for the variable declaration within a kernel signature"""
        return cgen.Value('const int', self.name)


class Iteration(Expression):
    """Iteration object that encapsualtes a single loop over sympy expressions.

    :param expressions: Single or list of SymPy expressions or :class:`Expression`
                        objects that define the loop body.
    :param index: Symbol that defines the name of the variable over which
                  to iterate.
    :param limits: Limits for the iteration space, either the loop size or a
                   tuple of the form (start, finish, stepping).
    :param offsets: Optional map list of offsets to honour in the loop
    """

    def __init__(self, expressions, dimension, limits, offsets=None):
        # Ensure we deal with a list of Expression objects internally
        self.expressions = expressions if isinstance(expressions, list) else [expressions]
        self.expressions = [e if isinstance(e, Expression) else Expression(e)
                            for e in self.expressions]

        # Generate index variable name and variable substitutions
        self.dim = dimension
        if isinstance(self.dim, Dimension):
            if self.dim.buffered is None:
                # Generate index variable from dimension
                self.index = str(self.dim.get_varname())
                self.subs = {self.dim: Symbol(self.index)}
            else:
                # Generate numbered indices for each buffer
                self.index = self.dim.name
                self.subs = {self.dim: Symbol(self.dim.get_varname())}
                for offset in self.index_offsets[self.dim]:
                    self.subs[self.dim + offset] = Symbol(self.dim.get_varname())
        else:
            warning("Generating Iteration without Dimension object")
            self.index = str(dimension)

        # Propagate variable names to the lower expressions
        self.substitute(self.subs)

        # Generate loop limits
        if isinstance(limits, Iterable):
            assert(len(limits) == 3)
            self.limits = list(limits)
        else:
            self.limits = list((0, limits, 1))

        # Replace open limits with variables names
        if self.limits[1] is None:
            # FIXME: Add dimension size as variable bound.
            # Needs further generalisation to support loop blocking.
            self.limits[1] = IterationBound("%s_size" % self.dim.name, self.dim)

        # Record offsets to later adjust loop limits accordingly
        self.offsets = [0, 0]
        for off in (offsets or {}):
            self.offsets[0] = min(self.offsets[0], int(off))
            self.offsets[1] = max(self.offsets[1], int(off))

    def __repr__(self):
        str_expr = "\n\t".join([str(s) for s in self.expressions])
        return "Iteration<%s; %s>::\n\t%s" % (self.index, self.limits,
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
        loop_body = [s.ccode for s in self.expressions]
        if self.dim.buffered is not None:
            modulo = self.dim.buffered
            v_subs = [cgen.Initializer(cgen.Value('int', v),
                                       "(%s) %% %d" % (s, modulo))
                      for s, v in self.subs.items()]
            loop_body = v_subs + loop_body
        loop_init = cgen.InlineInitializer(
            cgen.Value("int", self.index), "%d + %d" % (self.limits[0], -self.offsets[0]))
        loop_cond = '%s %s %s' % (self.index, '<' if self.limits[2] >= 0 else '>',
                                  "%s - %d" % (self.limits[1], self.offsets[1]))
        loop_inc = '%s += %s' % (self.index, self.limits[2])
        return cgen.For(loop_init, loop_cond, loop_inc, cgen.Block(loop_body))

    @property
    def signature(self):
        """List of data objects used by the loop and it's body

        :returns: List of unique data objects required by the loop
        """
        signature = [e.signature for e in self.expressions]
        signature = filter_ordered(chain(*signature))
        if isinstance(self.limits[1], IterationBound):
            signature += [self.dim]
        return signature

    def indexify(self):
        """Convert all enclosed stencil expressions to "indexed" format"""
        for e in self.expressions:
            e.indexify()

    @property
    def index_offsets(self):
        """Collect all non-zero offsets used with each index in a map

        Note: This assumes we have indexified the stencil expression."""
        offsets = defaultdict(list)
        for expr in self.expressions:
            offsets.update(expr.index_offsets)
        return offsets
